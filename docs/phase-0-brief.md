# Phase 0 implementation brief — contracts and truth model

**Status:** active
**Date:** 2026-09-12
**Authority:** phase-plan.md §Phase 0, ADR-001, ADR-002, ADR-003, ADR-006, ADR-007, ADR-010, ADR-011, ADR-012, ADR-013
**Preconditions met:** determinism ports (`src/apron/domain/ports.py`), canonical digest with cross-implementation vectors (`src/apron/domain/canonical.py`, `tests/vectors/`)
**Precondition not met:** pinned representative source artifacts for every external format being mapped

## Goal

Phase 0 produces the contracts everything else is built on. After this
phase, a contributor can write an adapter, run a conformance suite against
it, and propose it — without ever reading the phase plan or the ADRs.

The tasks below produce three things in order:

1. **Pinned external formats** — real files from the ecosystems Apron
   interoperates with, each with provenance. These are what the schemas
   are written against, so the first schema is a grounded contract, not
   an imagined one.
2. **Frozen schemas and extension-point Protocols** — the Pydantic models,
   the `typing.Protocol` contracts for every extension point, and the
   calculator dispatch contract. These are the permanent API surface.
3. **Tests and fixtures that prove the exit gate** — round-trip, export,
   determinism, negative/rejection scenarios, restart/dedup, and
   promotion-gate tests. When these pass, Phase 0 is done.

## Implementer notes

The following choices are left to the implementer because they have no
architectural consequence:

- **Managed-API golden fixture:** pick any managed API with an immutable
  version identifier and published pricing. The model name is test data.
- **Compound-solution golden fixture:** pick a multi-role shape with at
  least one managed and one self-hosted endpoint and at least one
  cross-modal edge. The exact roles are test data. The separate topology
  fixtures (§2.5) already cover failover, replica, sharded and
  opaque-router independently.
- **InferenceX fixture row:** check for a public `agg_bmk.json` workflow
  artifact and use a real row if available. If not, synthesize from the
  documented schema in `results-and-ingestion.md` and
  `process_result.py`, both pinned.

---

## Part 1 — Pin external formats

### 1.1 Fixture location and provenance format

Location: `tests/fixtures/external-formats/<source-name>/`

Each directory contains:

- `provenance.json` — metadata envelope
- One or more pinned files copied verbatim from the source

Provenance schema (the first contract this brief produces):

```json
{
  "source_name": "vllm-recipes",
  "repository": "https://github.com/vllm-project/recipes",
  "commit": "f050a17eec51c7093727ddbc765c005647bc92f3",
  "retrieval_date": "2026-09-12",
  "license": "Apache-2.0",
  "has_formal_schema": false,
  "schema_description": "Convention-based YAML; no published JSON Schema or validator",
  "pinned_files": [
    {
      "source_path": "models/Qwen/Qwen3-8B.yaml",
      "local_path": "qwen3-8b.yaml",
      "sha256": "<computed at pin time>"
    }
  ],
  "notes": "Representative dense model recipe. Schema derived from convention, not a formal spec."
}
```

### 1.2 Sources to pin

Five sources are exit-gate requirements (the three ADR-002 ecosystem
shapes, the vLLM configuration surface, and HF Hub artifacts). Two
additional sources (Inspect AI, Harbor) are informational references
that inform `EvaluationAdapter` Protocol design; they are not exit-gate
pins and their adapter implementations belong to Phase 2.

The commits below are current HEAD at retrieval date; the implementer
records the digest of each pinned file.

#### 1.2.1 vllm-project/recipes (ADR-002 §4, ecosystem shape 1 of 3)

| field | value |
|---|---|
| repository | `https://github.com/vllm-project/recipes` |
| commit | `f050a17eec51c7093727ddbc765c005647bc92f3` |
| license | Apache-2.0 |
| formal schema | No. Convention-based YAML. |

**Pin two files:**

1. `models/Qwen/Qwen3-8B.yaml` — dense model, single variant (bf16),
   hardware overrides, features, compatible strategies. Proves the common
   case.
2. `models/deepseek-ai/DeepSeek-V3.yaml` — MoE model with multiple
   variants (bf16, fp8), MLA architecture, MoE-specific strategies
   (TEP, DEP, PD), quantization variants with separate `model_id`.
   Proves the complex case.

**What Apron maps:** `deployment-plan.json` round-trips losslessly for
shared fields to a recipes YAML entry. Shared fields: `model_id` ↔
`ArtifactLocator.uri` (a locator, not `ArtifactIdentity`),
`min_vllm_version`, `precision`/dtype, `vram_minimum_gb`, hardware
verification status, `base_args`, compatible strategies and hardware
overrides. Fields recipes has that Apron does not: `guide` (prose),
`features` (CLI toggle), `meta` display fields. Fields Apron has that
recipes does not: execution fingerprint, evidence level, freshness,
measurement provenance, task and serving evidence, economics.

#### 1.2.2 NVIDIA aiconfigurator (ADR-002 §4, ecosystem shape 2 of 3)

| field | value |
|---|---|
| repository | `https://github.com/ai-dynamo/aiconfigurator` |
| commit | `77fd0773407b3683d8a671fe24a30a7110651b64` (already pinned in ADR-002) |
| license | Apache-2.0 |
| formal schema | **Yes.** `estimate-request-v1.schema.json`, version `aic-estimate-request/1.0.0` |

**Pin three files:**

1. `src/aiconfigurator/sdk/config_adapter/schemas/estimate-request-v1.schema.json`
   — the published JSON Schema for estimate requests.
2. `src/aiconfigurator/sdk/config_adapter/schema.py` — Pydantic request
   models (`EstimateRequestV1`, `ModelSettingsV1`, `BackendSettingsV1`,
   `SystemSettingsV1`, `WorkloadSettingsV1`, topology discriminated union,
   `QuantizationSettingsV1`, `RuntimeSettingsV1`, `SourceProvenanceV1`).
3. `src/aiconfigurator/cli/example.yaml` — representative estimate request
   examples (aggregated and disaggregated topologies).

**What Apron maps:** `deployment-plan.json` round-trips losslessly for
shared fields to an aiconfigurator estimate request. Shared fields:
`model.path` ↔ `ArtifactLocator.uri`, `backend.name`/`version` ↔ engine,
`systems.prefill`/`decode` ↔ hardware, `workload` (ISL/OSL/concurrency)
↔ serving workload spec, topology (TP/PP/EP/DP) ↔ deployment plan
parallelism, `quantization` ↔ quantization spec. Fields aiconfigurator returns that Apron consumes as a `PlanningClaim`:
TTFT, TPOT, power, batch size, context budget. The `database_mode`
(SILICON/HYBRID/EMPIRICAL/SOL) is recorded as producer metadata on the
`PlanningClaim`, not mapped to Apron's `EpistemicStatus` — a planning
source's self-reported confidence cannot become Apron's `measured`
status (ADR-002 §10). Required aiconfigurator fields that Apron cannot
populate (`provenance.source_type` closed enum, `batch_size`) are
documented in the shared-field table (§3.5a) with their export
convention.

#### 1.2.3 SemiAnalysis InferenceX (ADR-002 §4, ecosystem shape 3 of 3)

| field | value |
|---|---|
| repository | `https://github.com/SemiAnalysisAI/InferenceX` |
| commit | `900f1989d777abb2c1753e8cfe2241371551158e` |
| license | Apache-2.0 |
| formal schema | No. Schema defined by ingestion code and documentation. |

**Pin two files:**

1. `docs/results-and-ingestion.md` — the throughput row schema
   documentation, including identity keys and field groups.
2. `utils/process_result.py` — the throughput result transformer that
   defines the derived per-GPU metrics, latency conversions and field
   names.

**If a public `agg_bmk.json` workflow artifact is accessible** (owner
input §3), also pin one representative throughput row as
`example-throughput-row.json`.

**What Apron maps:** `deployment-plan.json` round-trips losslessly for
shared fields to an InferenceX-shaped result row. Shared fields: `hw` ↔
hardware, `model` ↔ `ArtifactLocator.uri`, `framework` ↔ engine, `precision` ↔
dtype/quantization, topology fields (TP/PP/EP/DP/disagg) ↔ deployment
plan, `isl`/`osl`/`conc` ↔ serving workload. Metric fields InferenceX
has that Apron consumes as `authoritative-external` evidence:
`tput_per_gpu`, TTFT/TPOT/ITL/E2EL percentiles, power. Fields Apron has
that InferenceX does not: task evaluation, accepted request, economics,
diagnosis, qualification state.

#### 1.2.4 vLLM configuration surface (engine adapter contract)

| field | value |
|---|---|
| repository | `https://github.com/vllm-project/vllm` |
| commit | `a1541f5742a29864a80087af313ad460066a1524` (already pinned in ADR-003, ADR-007) |
| license | Apache-2.0 |
| formal schema | No. Schema defined by Python classes. |

The architecture dispatch proof and ADR-007 are written against this
commit. For Phase 0 schema definition, the configuration surface at this
commit is authoritative.

**Pin five files** (at the pinned commit, not HEAD):

1. `vllm/engine/arg_utils.py` — `EngineArgs` and `AsyncEngineArgs`
   dataclasses defining the complete vLLM configuration surface.
2. `vllm/v1/kv_cache_interface.py` — `FullAttentionSpec`,
   `MLAAttentionSpec`, `SlidingWindowSpec`, `MambaSpec` and the KV cache
   mechanism registry.
3. `vllm/tasks.py` — task registry (`generate`, `embed`, `classify`,
   `score`, `reward`, `transcription`, `transcription_streaming`).
4. `vllm/transformers_utils/model_arch_config_convertor.py` — architecture
   normalization, MLA detection, KV head count extraction.
5. `docs/models/supported_models.md` — the model support matrix with
   capability columns.

**Note:** a shallow clone at HEAD does not contain this commit. The
implementer must either do a full clone or use `git fetch --depth 1
origin <commit>` to retrieve the exact pinned revision. Alternatively,
pin the HEAD files with a note that the dispatch proof references
`a1541f57` and record both commits in provenance.

#### 1.2.5 Hugging Face Hub manifest fields (artifact resolution)

| field | value |
|---|---|
| source | Hugging Face Hub API and model repositories |
| formal schema | No. Validated by Hub server; no published JSON Schema. |

Hugging Face Hub has no single pinnable repository for its metadata
schemas. The fixture uses real model repository files.

**Pin from `Qwen/Qwen3-8B` at its current immutable revision:**

1. `config.json` — transformer model configuration (the fields
   `ModelConfig` reads: `architectures`, `model_type`, `hidden_size`,
   `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`,
   `vocab_size`, `max_position_embeddings`, `torch_dtype`, etc.).
2. `model.safetensors.index.json` — the shard index with
   `metadata.total_size` and `weight_map` (tensor name → shard file).
3. The YAML frontmatter from `README.md` — model card metadata
   (`pipeline_tag`, `library_name`, `license`, `tags`, `base_model`,
   `model-index` with eval results).

**Also pin from a MoE model** (`deepseek-ai/DeepSeek-V3`):

4. `config.json` — proves MLA fields (`kv_lora_rank`, `model_type:
   deepseek_v3`) and MoE fields (`num_experts`, `num_experts_per_tok`).

**Retrieval method:** `huggingface_hub` Python API with `revision` pinned
to the commit SHA visible in the Hub UI at retrieval time. Record the
revision SHA, retrieval date and file SHA-256.

**What Apron maps:** `ArtifactLocator` names the source kind (`huggingface`)
and requested revision. `ArtifactSourceObservation` carries the resolved
immutable revision, manifest (`model.safetensors.index.json`),
per-file/component content digests, license/gating observations and
attributed publisher metadata (model card fields). `ModelSpec` reads
`config.json` fields to build the component/mechanism graph.
`ArtifactIdentity` derives from the resolved contents and component
structure, not from the repository name.

#### 1.2.6 UK AISI Inspect AI (informational reference, not exit-gate)

| field | value |
|---|---|
| repository | `https://github.com/UKGovernmentBEIS/inspect_ai` |
| commit | `8ebe620d74c1eb679438db1b65324e30e2306092` |
| license | MIT |
| formal schema | No. Schema defined by Pydantic models in source. |

**Pin two files:**

1. `src/inspect_ai/log/_log.py` — `EvalLog`, `EvalSpec`, `EvalSample`,
   `EvalResults`, `EvalStats`, `EvalScore`, `EvalMetric`, `EvalConfig`,
   `EvalPlan`, `EvalDataset`, `EvalRevision`.
2. `src/inspect_ai/scorer/_metric.py` — `Score`, `Value`, `ScoreReason`.

**What Apron maps:** Inspect AI's `EvalLog` is the shape an Inspect AI
evaluation adapter ingests. Apron maps it to `TaskAttemptRecord`s and
`EvaluationProtocol` fields. Key correspondences: `EvalSpec.model` ↔
solution endpoint, `EvalSpec.task`/`task_version`/`dataset` ↔
`TaskSuiteSpec`, `EvalSpec.config` ↔ `EvaluationProtocol` parameters,
`EvalSample.scores` ↔ per-attempt criterion outcomes,
`EvalResults.scores` ↔ aggregate evidence, `EvalSpec.revision` ↔
evaluation provenance. Apron preserves the exact `EvalLog` provenance
but does not inherit its evidence authority — an Inspect result enters
at the `authoritative-external` level.

#### 1.2.7 Harbor framework (informational reference, not exit-gate)

| field | value |
|---|---|
| repository | `https://github.com/harbor-framework/harbor` |
| commit | `1e5c5c6db929a10a140d05e606882c671ae20729` |
| license | Apache-2.0 |
| formal schema | No. Schema defined by Pydantic models in source. |

**Pin three files:**

1. `src/harbor/models/trial/result.py` — `TrialResult`, `StepResult`,
   `TimingInfo`, `ExceptionInfo`, `AgentInfo`, `ModelInfo`.
2. `src/harbor/models/trial/config.py` — `TrialConfig`, `AgentConfig`,
   `EnvironmentConfig`, `VerifierConfig`.
3. `src/harbor/models/task/config.py` — `TaskConfig` (the `task.toml`
   schema).

**What Apron maps:** Harbor's `TrialResult` is the shape a Harbor
evaluation adapter ingests. Key correspondences: `TrialResult.task_name`
↔ `TaskSuiteSpec` reference, `TrialResult.agent_info` ↔
`ApplicationSpec` agent identity, `TrialResult.verifier_result` ↔
criterion outcome, `StepResult` ↔ per-step attempt records,
`TrialConfig.environment` ↔ sandbox/execution context,
`TrialResult.task_checksum` ↔ task suite revision integrity.

### 1.3 Implementation order for Part 1

Exit-gate pins (blocking):

1. Create `tests/fixtures/external-formats/` and the `provenance.json`
   schema (a small frozen Pydantic model in the domain layer).
2. Pin vllm-recipes (two YAML files from `.sources/recipes/`).
3. Pin aiconfigurator (three files from `.sources/aiconfigurator/` at the
   already-pinned commit).
4. Pin InferenceX (two files from `.sources/InferenceX/`; add example row
   if available).
5. Pin vLLM configuration surface (five files; resolve the commit
   question per §1.2.4 note).
6. Pin HF Hub artifacts (download via `huggingface_hub` API with pinned
   revisions for Qwen3-8B and DeepSeek-V3).
7. Pin llmcalc legacy catalogue. Source: the 101 unique non-Gemma entries
   (phase-plan.md line 37). Pin the catalogue with provenance as
   `owner_attested_legacy_boot` evidence level. Record original empirical
   VRAM, configuration and provider fields; mark absent artifact revision,
   engine/image digest, exact hardware, workload, log and date explicitly
   unknown. Do not carry forward llmcalc's generic KV, activation, MoE
   or quantization fallback formulas.
8. Write a test that every `provenance.json` is valid, every pinned file
   exists, and every recorded SHA-256 matches the file on disk.

Informational references (not blocking, inform schema design):

9. Pin Inspect AI (two files from `.sources/inspect_ai/`).
10. Pin Harbor (three files from `.sources/harbor/`).

---

## Part 2 — Schema set

### 2.0 Implementation conventions

All schemas are frozen Pydantic models (`model_config =
ConfigDict(frozen=True)`). Alternatives within one contract use
discriminated unions with `Literal` tags and
`Field(discriminator=...)`. Schema version is a monotonic integer
carried in every serialized record.

**Module location:** schemas live in their framework-spec.md §6
directories, not in a flat `schemas/` module. `CapabilitySignature`
goes in `src/apron/domain/capabilities/`, `ArtifactIdentity` in
`src/apron/domain/artifacts/identity/`, `InferenceSolution` and
`EndpointBinding` in `src/apron/domain/solutions/`,
`ComponentMechanism` in `src/apron/domain/mechanisms/`. Schemas
without a named home (primitives, records, authority, reports) go in
`src/apron/domain/schemas/`. Migrations go in
`src/apron/domain/schemas/migrations/`.

**Canonicalization convention:** all serialization for digest
computation uses `model_dump(mode="json")`, which converts `datetime`
to ISO strings and other non-JSON types to their JSON representation.
`canonical.py`'s `canonicalize()` receives only JSON-primitive types.
This is load-bearing — `model_dump()` without `mode="json"` returns
live Python objects and `canonicalize()` raises `TypeError`.

**Cross-layer references are fingerprint strings, not embedded
objects.** When a schema references another schema from a different
layer, it carries the referenced record's `record_digest_hex` as a
`str` field, not an embedded typed object. This eliminates circular
imports between layers. `TaskAttemptRecord` carries "exact
decision/task-suite/application/evaluation-protocol/solution
fingerprints" — these are digest strings. `EvaluationProtocol`
"binds" a `DecisionRequest` by carrying its fingerprint, not by
embedding it. `QualityEvidence` "references" task attempts by their
fingerprints.

**Fingerprint vs digest.** A record's digest (`record_digest_hex`) is
its full canonical form. A record's fingerprint is the digest of its
**identity subset** — the fields that define what the record IS, not
how it is displayed or when it was created. Each schema declares its
identity fields via a `fingerprint_fields()` classmethod that returns
the subset of `model_dump(mode="json")` used for fingerprinting.
Display metadata, timestamps and mutable annotations are excluded.
This is how `test_solution_fingerprint_stable_on_irrelevant_change`
works.

**Migration convention.** Migrations are explicit functions from
version N to N+1 in `src/apron/domain/schemas/migrations/`. Migration
produces a new record version with a new digest; old references (by
digest) remain valid against the old version. INV-5 (append-only) and
INV-12 (correction keeps original visible) are satisfied because
migration never mutates an existing record — it creates a successor.
The migration registry maps `(schema_name, version)` to the forward
migration function. Built and tested with the first schema (Layer 0)
using a synthetic v1→v2 migration that adds one optional field.

**Extension-point Protocols.** The 10 extension-point contracts
(framework-spec.md §1) are `typing.Protocol` definitions, not
Pydantic models. They are defined alongside the data schemas they
consume, in the domain layer (engineering-standards.md §1, binding).
The `EvaluationAdapter` Protocol, for example, sits next to the
evaluation data schemas. Each Protocol is built with at least one
fake-adapter test client. Full conformance suites in `conformance/`
are a prerequisite for Phase 1a (before the first real adapter), not
a Phase 0 exit-gate deliverable — but the Protocol definitions
themselves are Phase 0 contracts.

### 2.1 Dependency-ordered implementation layers

#### Layer 0 — Primitives (no cross-schema dependencies)

Modules: `src/apron/domain/schemas/primitives.py` for general
primitives; `src/apron/domain/capabilities/` for `CapabilitySignature`

| schema | source | key fields |
|---|---|---|
| `HardwareSpec` | ADR-001, ADR-006 | GPU SKU, total memory, compute capability, memory bandwidth, interconnect, topology position |
| `CapabilitySignature` | ADR-007 §2 | operation, input ports with modality combinations (T+I vs T/I), output ports with result representations, cardinality/streaming constraints |
| `ArtifactLocator` | ADR-002 §8 | source kind (`huggingface`, `local`, `oci`, `s3`), source URI, requested revision |
| `EpistemicStatus` | ADR-010 §11 | discriminated union: `derived`, `proven_constraint`, `predicted`, `measured` — each with required provenance fields |
| `ClaimScope` | ADR-002 §7 | `config`, `boot`, `memory`, `remediation`, `serving_performance`, `task_outcome`, `outcome_economics` |
| `ExecutionTarget` | ADR-001 §4 | `typing.Protocol` with `local-container`, `remote-container`, `rented-provider` kinds; prepare/provision, execute, observe, collect, teardown semantics; exact execution fingerprint. Every evidence record carries target kind, operator/source, provider, detected hardware and complete execution fingerprint |

#### Layer 1 — Artifact identity and lineage

Module: `src/apron/domain/artifacts/` (`ArtifactIdentity` in
`identity/`, others alongside)

| schema | source | key fields |
|---|---|---|
| `ArtifactSourceObservation` | ADR-002 §8 | resolved immutable revision, manifest, per-file content digests, sizes, license/gating observations, attributed publisher metadata |
| `ArtifactIdentity` | ADR-002 §8, ADR-006 | derived from resolved file/component contents and structure; not from a registry namespace |
| `ModelLineage` | ADR-006 | logical model family, attributable publishers/derivatives |
| `QuantizationSpec` | ADR-006 | weight/activation/KV formats; linear/MoE schemes; bits, group/block shape, scale/zero-point representation, excluded modules, calibration recipe/dataset, producer version |
| `OfflineTransformSpec` | ADR-006 | producer, version, recipe, calibration data identity, output artifact |
| `RuntimeTransformSpec` | ADR-006 | `none` or exact pinned-engine online conversion, target layer rules, exclusions, producer/version |

#### Layer 2 — Model, execution, evidence and calculator contract

Modules: `src/apron/domain/mechanisms/` for `ComponentMechanism` and
the calculator contract; `src/apron/domain/schemas/models.py` for
model and evidence schemas

| schema | source | key fields |
|---|---|---|
| `ArtifactSpec` | ADR-006 | one `ArtifactIdentity` + locators/observations, exact weight/index manifest and content digests, config and quantization-config digests, tokenizer/chat-template identity, attributed claims, claimed lineage |
| `ArtifactRelation` | ADR-006 | discriminated union: `published_by_owner`, `claimed_derived_from`, `reproducibly_derived_from`, `structurally_compatible_with`, `quality_compared_with`, `tokenizer_compatible_with` |
| `ExecutionSpec` | ADR-006 | engine image digest, resolved checkpoint method, implementation overrides, selected linear/MoE/attention kernels |
| `ComponentMechanism` | ADR-007 §4 | role, typed execution mechanism (`autoregressive_decode`, `discrete_diffusion_decode`, `single_pass_pooling`, `encoder_decoder_generation`, `media_encoder`, `projector`, `latent_denoising`, `vae_decode`, `vocoder`, namespaced extension) |
| `Calculator` | phase-plan, ADR-003 | `typing.Protocol`: dispatches by typed `ComponentMechanism` + `HardwareSpec` + workload shape → `PlanningClaim`. Mechanism registry; unimplemented mechanism returns `unknown` without fallback formula. The calculator is a function, not an object (engineering-standards.md §1) |
| `ModelSpec` | ADR-007 §4, ADR-011 §4 | common identity core (repository, immutable revision, license, files, exact component bytes/dtype, remote-code requirement, lineage, publisher claims) + typed component graph with edges |
| `QualityEvidence` | ADR-006, ADR-011 | carries fingerprints of the task attempts and evaluation protocol that produced it; not a free-standing assertion. Referenced by fingerprint, not embedded |
| `CompatibilityEvidence` | ADR-006 | scoped to a candidate and execution fingerprint |

#### Layer 3 — Task, application and serving

Module: `src/apron/domain/schemas/tasks.py`

| schema | source | key fields |
|---|---|---|
| `TaskSuiteSpec` | ADR-011 §2 | versioned cases or sampling frame, required `CapabilitySignature`s with exact input combinations and expected result representations, tools, environment, task categories, success criteria, rubrics, segment weights, privacy classification, representativeness claims |
| `ApplicationSpec` | ADR-011 §3 | prompt/template revisions, agent/scaffold version, tool definitions/versions, retrieval components, reasoning/sampling settings, turn/tool/retry limits, named logical role graph, invocation conditions |
| `ServingWorkloadSpec` | ADR-011 §2 | request and concurrency distributions, burstiness, input/output shape distributions, prefix/cache behavior, LoRA usage, latency/throughput percentiles, availability, duration, capacity horizon |
| `WorkloadSpec` | ADR-011 §2 | compatibility envelope referencing `TaskSuiteSpec` and `ServingWorkloadSpec` |

#### Layer 4 — Solution, plan and evaluation

Modules: `src/apron/domain/solutions/` for `InferenceSolution` and
`EndpointBinding`; `src/apron/domain/schemas/solutions.py` for
`DeploymentPlan`, `PlanningClaim`, `EvaluationProtocol`

| schema | source | key fields |
|---|---|---|
| `EndpointBinding` | ADR-013 §1 | discriminated union: `direct_endpoint`, `logical_route`, `replica_pool`, `distributed_execution_group` — each with its declared backing deployments |
| `InferenceSolution` | ADR-011 §4, ADR-013 | managed API binding, self-hosted artifact/engine/target binding, concrete role bindings for application roles, resolved routing/fallback/escalation policy; every endpoint retains fingerprints of its own `ModelSpec`, `ArtifactSpec`, `ExecutionSpec`, `CapabilitySignature`s, provider/target identity and `DeploymentPlan` |
| `PlanningClaim` | ADR-002 §10 | producer and version, exact input fingerprint, proposed configuration, claim scope, producer-declared epistemic tier (NOT Apron `EpistemicStatus`), source-record references, calibration domain, uncertainty, unsupported/opaque fields |
| `DeploymentPlan` | phase-plan §Phase 0 | canonical executable plan for a deployable endpoint or coordinated set of endpoints; includes parallelism topology (TP/PP/EP/DP/DCP/PCP), engine configuration, batch size, resource allocation, rendered command strings (literal `vllm serve` and Docker Compose text, not adapter output) |
| `EvaluationProtocol` | ADR-011 §5 | carries fingerprints of `DecisionRequest`, `TaskSuiteSpec`, `ApplicationSpec` and candidate `InferenceSolution` (not embedded — cross-layer references are digest strings per §2.0); harness/version, dataset snapshot, solver/agent, scorer/rubric, deterministic checks, judge model/prompt, sampling parameters, seeds, repetitions, concurrency, stopping rules, aggregation, uncertainty method |

#### Layer 5 — Records

Module: `src/apron/domain/schemas/records.py`

Every record carries `claim_scope`, `production_mode` and a
machine-readable `reason` (INV-20). Every record carries `lifecycle`
state: `observed`, `superseded` or `retracted`; correction keeps
the original visible (INV-12).

| schema | source | key fields |
|---|---|---|
| `TaskAttemptRecord` | ADR-011 §7 | exact decision/task-suite/application/evaluation-protocol/solution fingerprints (digest strings); case/attempt identity; output and permitted artifacts; criterion scores and acceptance; trace references; turns/retries/tool calls; input/cached-input/output usage; time; endpoint/judge/infrastructure cost; failures; raw-observation provenance; `claim_scope`, `production_mode`, `reason`, `lifecycle` |
| `VerificationReport` | ADR-006, phase-plan | target kind, operator/source, provider, detected hardware (from `ExecutionTarget`); initial total/free/requested memory; model/weight memory; persistent consumption; transient peak headroom; non-PyTorch increase; CUDA-graph estimate/applied/actual; available KV-cache memory; safety buffer; profiling shape. Each stored as a separate non-overlapping field; `claim_scope`, `production_mode`, `reason`, `lifecycle` |
| `RemediationRecord` | ADR-006 §Remediation-proof | `mechanism_outcome` (`verified`/`failed`/`not_evaluated`), `request_outcome` (`satisfied`/`violated`/`not_evaluated`), accepted request snapshot fingerprint, task/application/evaluation fingerprints, corrected-solution and plan diff, violated constraints, proving record fingerprints. Public result derived: `Fixed` = mechanism verified + request satisfied; `Alternative with trade-offs` = mechanism works + constraint violated; `Unverified suggestion` = incomplete proof |
| `DiagnosisRule` | ADR-006 §Failure-fingerprint, phase-plan | exception class, engine call-site module, error family; correction spec; status (`hypothesis` or `mechanism_verified`); promoting `VerificationReport` fingerprint when verified |
| `EvidenceReleaseManifest` | ADR-002 §6 | content-addressed, schema version, record digests, signatures/attestations, CDLA-Permissive-2.0 license, reconstructable independently of any mirror |

#### Layer 6 — Authority and automation

Module: `src/apron/domain/schemas/authority.py`

| schema | source | key fields |
|---|---|---|
| `DecisionRequest` | ADR-011 §1 | task/application objective, success criteria, quality floor, serving SLOs, privacy/security/license/data-residency constraints, budget/time bounds, permitted providers/accounts/resources, optimization objective |
| `ActionRequest` | ADR-010 §1 | side-effect-specific; references `DecisionRequest` |
| `AuthorityContribution` | ADR-010 §2 | typed `AuthoritySource` implementations contributing versioned constraints over principal, action, resource, context |
| `AuthorizationDecision` | ADR-010 §3 | bound to `ActionRequest`, accepted `DecisionRequest`, context and source versions |
| `AuthorizationEnvelope` | ADR-010 §3 | permitted action classes, candidate/model/judge/evaluation/compute providers/accounts, credential scopes, task-data destinations, hard target constraints, maximum spend, runtime/lifecycle/teardown, security/data rules, publication scope, adaptation rules |
| `OrchestrationDecision` | ADR-010 §7 | stable job id, deduplication key, policy version, state transitions, inputs, outputs |
| `ActionAttempt` | ADR-010 §1 | discriminated union with evaluation, execution and publication specializations |
| `AnomalyCase` | phase-plan §Phase 0 automation-boundaries, ADR-010 §8 | internal case for unknown fingerprint or significant prediction delta |
| `ProposedExternalAction` | ADR-010 §8 | rendered record in upstream shape, destination policy, provenance, deduplication, confidence, rate controls |

#### Layer 7 — Decision report and resources

Module: `src/apron/domain/schemas/reports.py`

| schema | source | key fields |
|---|---|---|
| `DecisionReport` | ADR-011 §9, phase-plan | every considered solution, rejection reason, evidence state, evaluation coverage, trade-off; preserves candidate set, exclusions, evidence states and trade-offs |
| `MaintainerBaselineAllocation` | ADR-010 §14 | owner-controlled, cannot be set by scheduler |
| `ContributedResourcePool` | ADR-010 §13 | provenance, may reduce project cost or expand coverage, cannot change evidence authority |

### 2.2 Golden fixtures

Three golden fixture directories under `tests/fixtures/golden/`:

#### `self-hosted/` — single self-hosted solution

Shape: Qwen3-8B BF16 on a single RTX 4090 via vLLM.

- `decision-request.json` — accepted request for a text-generation task
  with a quality floor, latency target and data policy
- `task-suite-spec.json` — small deterministic text task (3–5 cases with
  expected outputs, deterministic scorer)
- `application-spec.json` — minimal single-role application (one prompt
  template, no agent/tools)
- `serving-workload-spec.json` — low-concurrency serving (ISL=512,
  OSL=128, concurrency=4, P99 TTFT < 2s)
- `inference-solution.json` — one `direct_endpoint` binding with resolved
  `ArtifactSpec` (Qwen3-8B at a pinned HF revision), `ModelSpec` with
  dense GQA component graph, `ExecutionSpec` (vLLM engine), `HardwareSpec`
  (RTX 4090 24 GB), exposed `CapabilitySignature` (text→generated_text)
- `deployment-plan.json` — TP=1, BF16, `vllm serve` command, Docker
  Compose rendering
- `evaluation-protocol.json` — deterministic scorer, 1 seed, 1 repetition
- `task-attempt-records.json` — one passing attempt per case
- `verification-report.json` — memory profiling fields (model memory,
  KV-cache budget, peak activation, safety buffer) with synthetic but
  structurally correct values
- `decision-report.json` — one candidate considered, one qualified, no
  rejections

Also export:
- `exported-recipes.yaml` — the same plan exported as a recipes YAML entry
- `exported-aiconfigurator.json` — the same plan exported as an
  aiconfigurator estimate request
- `exported-inferencex.json` — the same plan exported as an
  InferenceX-shaped result row (config fields only; metric fields empty
  or marked `not_measured`)

#### `managed-api/` — single managed API solution

Shape: OpenAI `gpt-4o-2024-08-06` (or confirmed substitute).

- Same document set as self-hosted, but:
- `inference-solution.json` — one `direct_endpoint` binding with
  `provider_opaque` for hidden runtime fields; no `ArtifactSpec` (no
  self-hosted artifact); `ModelSpec` with `artifact_declared` capability
  signatures from the provider's documentation
- No `deployment-plan.json` (managed API has no Apron-renderable plan)
- No `verification-report.json` (no Apron-observable boot or memory)
- `decision-report.json` — one candidate, `provider_opaque` for memory
  and boot evidence, task evaluation evidence present

#### `compound/` — compound planner/executor/vision solution

Shape: three-role application — planning (managed API), execution
(self-hosted text model) and vision (self-hosted multimodal model).

- Same document set as self-hosted, plus:
- `application-spec.json` — three named logical roles (`planner`,
  `executor`, `vision_analyzer`) with invocation conditions and a routing
  policy
- `inference-solution.json` — `logical_route` over three `direct_endpoint`
  bindings: one managed, two self-hosted; concrete role bindings for each
  logical role; routing/fallback policy
- `task-suite-spec.json` — task requiring text+image input (proves the
  vision path) with a text output
- Two `deployment-plan.json` files — one per self-hosted endpoint
- `decision-report.json` — three endpoints, one compound solution, changed
  member changes fingerprint

#### Negative/rejection golden fixtures (phase-plan exit gate, line 29)

Under `tests/fixtures/golden/negative/`, five mandated scenarios:

| fixture | what it proves |
|---|---|
| `hardware-specific-measurement.json` | a measurement from one GPU SKU does not transfer to another; cross-hardware reuse produces a prediction, not a measurement |
| `deployment-substitution.json` | a changed target inside authorization proceeds; a changed target outside a hard boundary is denied |
| `high-throughput-task-failure.json` | high throughput with failed task outcomes cannot qualify a solution (INV-24) |
| `serving-slo-failure.json` | high task score with a violated serving SLO cannot qualify a solution (INV-24) |
| `incomparable-cost-boundary.json` | managed and self-hosted cost cannot be ranked when their cost boundary or workload horizon is incomparable |

#### Authority and automation fixtures

Under `tests/fixtures/authority/`:

| fixture | what it proves |
|---|---|
| `authorization-envelope.json` | deny overrides permit; indeterminate never authorizes; unknown obligation denies |
| `restart-no-duplication.json` | replaying a job cannot duplicate a paid task attempt or judge call (ADR-010, INV-21) |
| `teardown-convergence.json` | cancellation, timeout, controller crash and target loss converge on idempotent teardown |
| `dynamic-destination-denied.json` | authorized primary route with unauthorized fallback is denied before disclosure or spend (ADR-013) |
| `authority-contribution.json` | typed `AuthoritySource` contributing versioned constraints |
| `orchestration-dedup.json` | stable job id, deduplication key, state transitions; restart produces same result |

#### Evidence-state fixtures

Under `tests/fixtures/evidence-states/`:

| fixture | what it proves |
|---|---|
| `contested.json` | two records at different evidence levels for the same fingerprint; disagreement marked `contested`, both shown, nothing auto-resolves (INV-4, ADR-006 §8) |
| `lifecycle-correction.json` | record with `lifecycle: observed`, then corrected record with `lifecycle: superseded`; original stays visible (INV-12) |
| `cross-fingerprint-rejection.json` | a task score from a different solution fingerprint is a prediction, not a measurement; same task, two different solutions, score does not transfer (INV-25) |
| `promotion-gate-recommended.json` | an endpoint cannot reach `Recommended` without accepted task and serving evidence (INV-18) |
| `promotion-gate-qualified.json` | a solution cannot reach `Qualified` without exact-solution task reproduction plus every applicable SLO (INV-18) |
| `quality-evidence.json` | `QualityEvidence` references task-attempt and evaluation-protocol fingerprints; cannot exist as a free-standing assertion (ADR-006, ADR-011) |
| `compatibility-evidence.json` | scoped to a candidate and execution fingerprint |
| `evidence-release-manifest.json` | content-addressed manifest with record digests, CDLA-Permissive-2.0; reconstructable independently of any mirror (INV-35) |
| `workload-spec-envelope.json` | `WorkloadSpec` binds `TaskSuiteSpec` and `ServingWorkloadSpec`; task success not inferred from serving throughput (INV-24) |

### 2.3 Capability and component fixtures

Under `tests/fixtures/capabilities/`, one fixture per capability from
ADR-007 §required-fixtures:

| fixture | capability signature | component mechanisms | notes |
|---|---|---|---|
| `text-to-text.json` | `{text} → generated_text` | `autoregressive_decode` | dense GQA (Qwen3-8B shape) |
| `text-image-to-text.json` | `{text,image} → generated_text` | `autoregressive_decode` + `media_encoder` + `projector` | VL model shape; includes forbidden-combination counterexample (`{image} → generated_text` without required text) |
| `audio-file-transcription.json` | `{audio} → text` | `encoder_decoder_generation` | file-based ASR |
| `audio-stream-transcription.json` | `{audio_stream} → text_delta + timestamps` | `encoder_decoder_generation` | streaming ASR |
| `text-to-embedding.json` | `{text} → embedding_vector` | `single_pass_pooling` | embedding model |
| `query-document-score.json` | `{text,text} → score` | `single_pass_pooling` | reranker/scoring model |
| `text-to-image.json` | `{text} → generated_image` | `latent_denoising` + `vae_decode` + `media_encoder` (text encoder) | Diffusers-based pipeline; resolves real immutable pipeline into text-encoder, denoiser and decoder mechanisms; does NOT claim a vLLM endpoint or executable media proof |
| `audio-text-audio-compound.json` | compound: `{audio} → text → {text} → generated_text → {text} → audio` | ASR + LLM + TTS composition | final audio endpoint may remain externally opaque |
| `checkpoint-not-exposed.json` | checkpoint ability not exposed by engine | — | model has capability X but vLLM at the pinned image does not expose it |
| `engine-resolved-unknown.json` | engine-resolved ability with calculator `unknown` | — | vLLM discovers a capability but Apron's calculator has no mechanism branch for it |

### 2.4 Component/mechanism fixtures

Under `tests/fixtures/mechanisms/`, proving the typed component graph and
memory calculation dispatch (from architecture-dispatch-proof.md):

| fixture | mechanism | architecture reference |
|---|---|---|
| `dense-gqa.json` | `FullAttentionSpec` | Llama-3.1-8B shape |
| `mla.json` | `MLAAttentionSpec` | DeepSeek-V3 shape |
| `sliding-window.json` | `SlidingWindowSpec` | Mistral-7B shape |
| `hybrid-mamba.json` | `MambaSpec` + `FullAttentionSpec` per layer | Jamba shape |
| `moe.json` | `FullAttentionSpec` + MoE expert routing | Mixtral shape |
| `quantized-awq.json` | quantized dense | AWQ artifact |
| `quantized-fp8.json` | quantized MoE | FP8 artifact with scale tensors |
| `long-context.json` | RoPE/YaRN extension | extended context beyond checkpoint default |
| `multimodal-vl.json` | vision encoder + projector + decoder | VL model with media-token expansion |
| `multi-gpu.json` | TP=2 or TP=4 topology | multi-GPU parallelism |
| `unified-memory.json` | Apple Silicon / unified memory | non-discrete GPU memory model |

### 2.5 Topology fixtures

Under `tests/fixtures/topology/`, from ADR-013 required conformance tests:

| fixture | topology type | what it proves |
|---|---|---|
| `direct-endpoint.json` | `direct_endpoint` | single declared endpoint |
| `identical-replica.json` | `replica_pool` | two identical replicas; capacity improves concurrency, not per-request memory |
| `heterogeneous-alias.json` | `logical_route` | different models behind one alias; per-attempt target recorded or `routing_opaque` |
| `failover.json` | `logical_route` | primary + fallback; unauthorized fallback denied |
| `opaque-router.json` | `logical_route` | `routing_opaque`; alias-level evidence cannot promote members |
| `verified-sharded.json` | `distributed_execution_group` | joint capacity through verified engine mechanism; worker roles, model placement, interconnect |

### 2.6 Evaluation adapter fixtures

Under `tests/fixtures/evaluation/`:

| fixture | what it proves |
|---|---|
| `deterministic-scorer.json` | exact string match scorer; no judge, no external call |
| `model-judge.json` | model-graded evaluation with judge identity, prompt, sampling policy |
| `failed-retried-attempt.json` | failed attempt preserved; retry recorded; both in economics |
| `private-task-rejected.json` | task data classified private; unauthorized external destination denied |

### 2.7 Quantization candidate graph fixtures

Under `tests/fixtures/quantization/`:

| fixture | candidate class | what it proves |
|---|---|---|
| `checkpoint-native.json` | checkpoint-native BF16 | base artifact, no transform |
| `publisher-quant.json` | publisher-quantized FP8 | separate `ArtifactSpec` with `published_by_owner` relation, actual tensor/scale bytes |
| `third-party-quant.json` | third-party AWQ | `claimed_derived_from` relation, name similarity cannot promote lineage |
| `offline-conversion.json` | offline GPTQ conversion | `OfflineTransformSpec` produces new `ArtifactSpec`; `reproducibly_derived_from` |
| `runtime-transform.json` | vLLM online FP8 quantization | `RuntimeTransformSpec` with `none` checkpoint, engine-applied conversion |
| `structurally-compatible.json` | two artifacts with matching tensor schemas but different values | `structurally_compatible_with` relation; matching shapes cannot prove content equality |
| `quality-compared.json` | two artifacts compared on the same task | `quality_compared_with` relation; comparison references the task/protocol that produced it |
| `tokenizer-compatible.json` | two artifacts with shared tokenizer | `tokenizer_compatible_with` relation; tokenizer hash matches |
| `same-content-two-sources.json` | identical resolved artifact from HF and local | one `ArtifactIdentity` with two `ArtifactSourceObservation`s (ADR-002 §8, ADR-011) |

### 2.8 Implementation order for Part 2

1. **Migration infrastructure and canonicalization test.** Build the
   migration registry, version field convention and the
   `fingerprint_fields()` classmethod pattern. Prove that
   `model_dump(mode="json")` + `canonicalize()` works on a minimal
   frozen model with a `datetime` field. Prove that migration produces a
   new digest and old references remain valid.
2. **Layer 0 primitives + `ExecutionTarget` Protocol.** Test: each model
   serializes to canonical form and round-trips. `ExecutionTarget` Protocol
   has a fake-adapter test client.
3. **Layer 1 artifacts.** Test: `ArtifactIdentity` derived from resolved
   contents; equal contents from two sources share one identity with two
   observations.
4. **Layer 2 models + calculator contract.** Test: `ModelSpec` component
   graph builds from `config.json` fixtures; calculator Protocol dispatches
   on `ComponentMechanism` + `HardwareSpec` → `PlanningClaim`;
   unimplemented mechanism returns `unknown`. Quantization candidate graph
   resolves through one API.
5. **Layer 3 tasks.** Test: `TaskSuiteSpec` and `ServingWorkloadSpec` are
   independent; `WorkloadSpec` envelope binds them; serving throughput
   cannot establish task success.
6. **Layer 4 solutions + evaluation.** Test: `InferenceSolution` with one
   endpoint and with three endpoints uses the same API; changed member
   changes fingerprint. `EvaluationProtocol` carries fingerprint refs to
   `DecisionRequest` (not embedded — digest string). Topology fixtures
   preserve distinct semantics.
7. **Layer 5 records.** Test: `TaskAttemptRecord` carries exact
   fingerprints; cross-fingerprint result cannot be emitted as
   measurement. `VerificationReport` memory fields are non-overlapping and
   carry `ExecutionTarget` identity. `RemediationRecord` has separate
   `mechanism_outcome` and `request_outcome`. `DiagnosisRule` starts as
   `hypothesis`. All records carry `production_mode`, `reason`, `lifecycle`.
8. **Layer 6 authority.** Test: `AuthorizationEnvelope` deny overrides
   permit; indeterminate never authorizes. Restart/dedup fixtures prove
   `OrchestrationDecision` idempotency. Dynamic-destination authorization
   denied for unauthorized fallback.
9. **Layer 7 reports.** Test: `DecisionReport` preserves every considered
   solution, rejection reason, evidence state. Promotion-gate test: no
   endpoint reaches `Recommended` without task+serving evidence; no
   solution reaches `Qualified` without exact-solution reproduction (INV-18).
10. **llmcalc legacy import.** Import the 101 entries as
    `owner_attested_legacy_boot`. Test: entries seed candidates but do not
    satisfy current verification gates; fallback formulas are not carried.
11. **Extension-point Protocols.** Define the remaining 9 Protocols
    (framework-spec.md §1): `EngineAdapter`, `ArtifactSourceResolver`,
    `RenderTarget`, `EvidenceSource`, `PlanningSource`,
    `EvaluationAdapter`, `SignalSource`, `AuthoritySource`, `Publisher`.
    Each with a fake-adapter test client. These are `typing.Protocol`
    in the domain layer; full conformance suites in `conformance/` are
    built as a prerequisite for Phase 1a.
12. **Golden fixtures.** Build the three success golden fixture sets + the
    five negative/rejection fixtures from the schemas. Test: each
    round-trips through serialization and migration.
13. **Evidence-state, authority and evaluation fixtures.** Build the
    contested, lifecycle, restart/dedup, cross-fingerprint, promotion-gate,
    QualityEvidence, CompatibilityEvidence, EvidenceReleaseManifest and
    remaining ArtifactRelation fixtures. Test: each exercises its stated
    invariant.
14. **Export tests.** Test: each `DeploymentPlan` golden fixture exports
    losslessly to the three pinned ecosystem shapes and back, using
    explicit shared-field tables (§3.5a).

---

## Part 3 — Exit gate as tests

### 3.1 Test location

All Phase 0 tests go in `tests/unit/`. Test files: `test_primitives.py`,
`test_artifacts.py`, `test_models.py`, `test_tasks.py`,
`test_solutions.py`, `test_records.py`, `test_authority.py`,
`test_reports.py`, `test_golden.py`, `test_export.py`,
`test_evidence_states.py`, `test_promotion.py`, `test_provenance.py`.

### 3.2 Round-trip tests

For every schema model:

```python
def test_round_trip_<schema>(fixture):
    """Serialize to canonical JSON, deserialize, re-serialize. Bytes match."""
    canonical = canonicalize(fixture.model_dump(mode="json"))
    restored = Schema.model_validate_json(canonical)
    assert canonicalize(restored.model_dump(mode="json")) == canonical
```

Schemas without file-based golden fixtures (authority-layer schemas,
`QualityEvidence`, `CompatibilityEvidence`, etc.) use programmatically
constructed instances in the test body — `Schema(field=..., ...)`.

For every golden fixture:

```python
def test_golden_round_trip(golden_dir):
    """Every JSON file in the golden fixture directory round-trips."""
```

### 3.3 Migration tests

```python
def test_migration_v1_to_v2(v1_fixture):
    """V1 fixture migrates to V2 without losing meaning."""
    v2 = migrate_v1_to_v2(v1_fixture)
    assert v2.schema_version == 2
    # Assert all semantic fields are preserved
```

From the first draft, every schema starts at version 1. The migration
infrastructure (version field, migration registry, forward-only
migration functions) is built with the first schema and tested with a
synthetic v1→v2 migration that adds one optional field.

### 3.4 Determinism tests

```python
def test_deterministic_plan(fixed_clock, fixed_id_gen, fixed_rng):
    """Same inputs + same injected ports = byte-identical plan."""
    plan1 = generate_plan(request, clock=fixed_clock, id_gen=fixed_id_gen, rng=fixed_rng)
    plan2 = generate_plan(request, clock=fixed_clock, id_gen=fixed_id_gen, rng=fixed_rng)
    assert canonicalize(plan1.model_dump(mode="json")) == canonicalize(
        plan2.model_dump(mode="json")
    )
```

### 3.5 Export tests (ADR-002 §4)

For each golden self-hosted `DeploymentPlan`:

```python
def test_export_to_recipes(plan, recipes_fixture):
    """Plan exports to recipes YAML; shared fields match."""


def test_export_to_aiconfigurator(plan, aiconfigurator_fixture):
    """Plan exports to aiconfigurator estimate request; shared fields match."""


def test_export_to_inferencex(plan, inferencex_fixture):
    """Plan exports to InferenceX row shape; config fields match."""


def test_round_trip_recipes(plan):
    """Export to recipes, import back, export again. Shared fields identical."""
```

The export functions are in `src/apron/adapters/renderers/` (one per
ecosystem shape). The import functions are in
`src/apron/adapters/evidence/sources/` (one per ecosystem shape). Both
are tested against the pinned fixtures.

### 3.5a Shared-field tables

Each ecosystem shape has an explicit shared-field table. "Shared" means
the field exists in both Apron's `DeploymentPlan` and the external
format with compatible semantics. Fields outside this table are
format-specific and are dropped on export or left empty on import.

The implementer builds these tables from the pinned source files at
implementation time. Each table records:

- Apron field name and type
- External field name and type
- Direction: bidirectional / export-only / import-only
- Conversion: identity / unit change / enum mapping
- Required-but-unpopulated: external required fields Apron exports
  with a documented convention (e.g. aiconfigurator
  `provenance.source_type` → `"custom"`, `batch_size` → Apron's
  calculated engine batch size)

Round-trip test: export a `DeploymentPlan` fixture to external format,
import it back, export again. Every shared field is identical on the
second export. Non-shared fields are absent or defaulted.

### 3.6 State explicitness tests

```python
def test_unknown_mechanism_explicit():
    """A component with no implemented calculator returns unknown, not a fallback."""


def test_unsupported_adapter_explicit():
    """A quantization method with no adapter returns unsupported, not a silent skip."""


def test_opaque_managed_api():
    """A managed API solution has provider_opaque for boot/memory, not fabricated evidence."""


def test_estimated_values_labeled():
    """Every estimated value carries epistemic_status=predicted with uncertainty."""
```

### 3.7 Fingerprint tests

```python
def test_solution_fingerprint_changes_on_member_change():
    """Changing a model, routing rule or modality transition changes the solution fingerprint."""


def test_solution_fingerprint_stable_on_irrelevant_change():
    """Changing display metadata does not change the fingerprint."""


def test_topology_fingerprint_distinct():
    """direct_endpoint, replica_pool and distributed_execution_group have distinct fingerprints."""


def test_replica_not_sharded():
    """Two replicas cannot satisfy a per-request memory requirement exceeding each replica."""
```

### 3.8 Evaluation privacy tests

```python
def test_private_task_rejected_from_unauthorized_destination():
    """A task classified private cannot be sent to an unauthorized external destination."""


def test_cross_fingerprint_score_not_measurement():
    """A task score from a different solution fingerprint is a prediction, not a measurement."""


def test_failed_attempts_preserved():
    """Failed and retried attempts appear in economics aggregates."""
```

### 3.9 Capability combination tests

```python
def test_capability_combination_semantics():
    """T+I (required together) != T/I (either alone). Schema preserves the distinction."""


def test_result_shape_preserved():
    """Each capability fixture preserves its declared result shape through serialization."""


def test_capability_state_separate():
    """artifact_declared, engine_resolved, endpoint_exposed and evidence states are independent."""
```

### 3.10 Quantization graph tests

```python
def test_quantization_candidate_api():
    """Every quantization fixture resolves through the same candidate API."""


def test_artifact_identity_survives_round_trip():
    """Artifact, transform and execution identities survive serialization round-trip."""


def test_unsupported_adapter_fails_explicitly():
    """An unsupported quantization adapter raises a typed error, not a silent fallback."""


def test_name_similarity_cannot_promote_lineage():
    """Two artifacts with matching names but different manifests remain different."""


def test_actual_tensor_bytes_for_memory():
    """Memory calculation uses actual tensor headers and auxiliary-scale bytes, not nominal bit width."""
```

### 3.11 Promotion-gate tests (INV-18)

```python
def test_no_recommended_without_task_and_serving_evidence():
    """An endpoint cannot reach Recommended without accepted task and
    serving evidence for the exact endpoint/application fingerprint."""


def test_no_qualified_without_exact_reproduction():
    """A solution cannot reach Qualified without exact-solution task
    reproduction plus every applicable SLO."""


def test_single_solution_not_optimal():
    """A single measured solution cannot be labeled best or optimal."""
```

### 3.12 Restart and deduplication tests (ADR-010, INV-21)

```python
def test_replay_no_duplicate_paid_attempt():
    """Replaying a job cannot duplicate a paid task attempt or judge call."""


def test_teardown_convergence():
    """Cancellation, timeout, crash and target loss converge on idempotent teardown."""


def test_duplicate_event_no_duplicate_publication():
    """Replayed events cannot duplicate paid executions or external publications."""
```

### 3.13 Evidence-state tests

```python
def test_contested_evidence():
    """Two records at different levels that disagree are marked contested;
    nothing auto-resolves (INV-4)."""


def test_lifecycle_correction():
    """Corrected record has lifecycle superseded; original stays visible with
    lifecycle observed (INV-12)."""


def test_epistemic_proven_constraint():
    """At least one fixture uses proven_constraint status — e.g. a
    component constraint derived from the pinned vLLM source."""
```

### 3.14 Generated-image fixture test

```python
def test_generated_image_fixture():
    """The text-to-image capability fixture resolves a real immutable pipeline
    into text-encoder, denoiser and decoder mechanisms without claiming a vLLM
    endpoint or executable media proof."""
```

### 3.15 Budget feasibility test

```python
def test_phase_1a_budget_feasible():
    """The accepted Phase 1a run plan is budget-feasible with ContributedResourcePool = 0.
    This is a schema-level test: the MaintainerBaselineAllocation fixture
    and a Phase 1a run plan fixture (built as part of golden fixtures)
    together prove the budget constraint holds."""
```

### 3.16 Provenance integrity test

```python
def test_external_format_provenance():
    """Every pinned external format has valid provenance: URL, commit, date, digest, license."""
    for source_dir in fixtures_dir.iterdir():
        provenance = json.loads((source_dir / "provenance.json").read_text())
        for entry in provenance["pinned_files"]:
            path = source_dir / entry["local_path"]
            assert path.exists()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
```

---

## Exit gate summary

Phase 0 is complete when `prek run --all-files` passes and:

1. Every exit-gate external format is pinned with provenance (Part 1,
   5 sources + llmcalc legacy, §1.3 step 8 test green).
2. Every schema from `DecisionRequest` through `DecisionReport` plus
   `RemediationRecord`, `DiagnosisRule` and `ExecutionTarget` is a
   frozen Pydantic model with discriminated unions, version field and
   migration from v1 (Part 2, Layers 0–7).
3. The 10 extension-point `typing.Protocol` contracts are defined with
   fake-adapter test clients (§2.8 steps 2 and 11).
4. The calculator Protocol dispatches on mechanism + hardware + workload
   and returns `unknown` for unimplemented mechanisms (§2.8 step 4).
5. Three success golden fixtures (self-hosted, managed, compound) and
   five negative/rejection golden fixtures round-trip through
   serialization and migration (§3.2).
6. Generated endpoint plans and decision reports are deterministic under
   injected ports (§3.4).
7. Every self-hosted `DeploymentPlan` exports losslessly for shared
   fields (per explicit shared-field tables, §3.5a) to the three pinned
   ecosystem shapes (§3.5).
8. Unknown, unsupported, opaque and estimated states are explicit (§3.6).
9. Changed solution members change the fingerprint; display metadata
   does not; topology fixtures preserve distinct semantics (§3.7).
10. Evaluation adapters cannot leak private fixtures or transfer scores
    across fingerprints (§3.8).
11. Capability fixtures preserve combination semantics and result shape
    (§3.9).
12. Quantization fixtures resolve through one candidate API with actual
    tensor bytes; all six `ArtifactRelation` variants fixtured (§3.10).
13. No endpoint reaches `Recommended` without task+serving evidence; no
    solution reaches `Qualified` without exact reproduction (§3.11,
    INV-18).
14. Restart/dedup fixtures prove no duplicate paid attempts (§3.12,
    ADR-010, INV-21).
15. Contested evidence, lifecycle correction and `proven_constraint`
    states are exercised (§3.13).
16. The text-to-image fixture resolves a real pipeline without claiming
    executable proof (§3.14).
17. The Phase 1a run plan is budget-feasible at zero contributed
    resources (§3.15).
18. The llmcalc legacy entries seed candidates without satisfying
    verification gates; fallback formulas are not carried (§2.8 step 10).
