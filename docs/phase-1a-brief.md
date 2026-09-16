# Phase 1a implementation brief — first complete vLLM product loop

**Status:** draft
**Date:** 2026-09-13
**Authority:** phase-plan.md §Phase 1a, ADR-001, ADR-002, ADR-003, ADR-005,
ADR-006, ADR-007, ADR-010, ADR-011
**Preconditions met:** Phase 0 exit gate (frozen schemas, 10 Protocols,
determinism ports, canonical digest, golden fixtures, 349 tests); external
format pins with provenance
**Precondition not met:** conformance suites for extension points (built
in Part 1 of this brief, before the first real adapter)

## Goal

Phase 1a proves the complete product loop end-to-end against one
internal conformance fixture on one rented CUDA target. After this
phase, the machinery works: resolve an artifact, predict memory, render
a deployment plan, boot vLLM, measure memory, run a deterministic task,
measure serving, inject a failure, diagnose it, correct the plan, reboot,
and replay the accepted task — all producing schema-valid records with
exact fingerprints.

The tasks below produce five things in order:

1. **Conformance suites** — the acceptance tests every adapter must pass
   before it ships, built before the first real adapter.
2. **GPU-free components** — calculator, HF Hub resolver, renderers,
   planning source, the `apron plan` command.
3. **Execution components** — rented-provider ExecutionTarget, vLLM
   engine adapter, deterministic evaluation adapter, the remaining CLI
   commands.
4. **Orchestration** — qualification graph, decision report generation,
   local MCP server.
5. **The fixture run** — one accepted request, one model, one GPU, the
   whole loop including failure injection and correction.

## Implementer notes

The following choices are left to the implementer because they have no
architectural consequence:

- **Deterministic task cases:** pick 3–5 short text prompts with
  unambiguous expected outputs (e.g. arithmetic, simple factual recall).
  The task suite exists to prove the contract, not to measure model
  quality.
- **RunPod region and availability zone:** pick whatever has RTX 4090
  Secure Cloud available at implementation time.
- **Network volume naming:** any name; the brief specifies the lifecycle
  (create before GPU instance, attach at boot, destroy at teardown).
- **Docker Compose formatting:** the rendered Compose file is test data;
  whitespace and comment style are not constrained.

---

## Part 1 — Conformance suites

Conformance suites are the acceptance tests for extension-point adapters.
They must exist before the first real adapter (engineering-standards.md
§7, phase-0-brief.md §2.8 step 10 note). Each suite is a pytest module
under `tests/conformance/` that accepts a factory fixture and exercises
the Protocol contract with a fake adapter as first client.

### 1.1 Suite location and structure

Location: `tests/conformance/`

Each suite is a module named `test_<protocol>.py`. The suite imports the
Protocol from `apron.domain.protocols` and accepts a `pytest.fixture`
that returns an adapter instance. A real adapter's own test module
provides the fixture and collects the conformance suite via
`pytest_plugins`.

### 1.2 Suites to build

Seven conformance suites are Phase 1a prerequisites (the three that
Phase 1a does not exercise — SignalSource, Publisher, AuthoritySource
beyond interactive — get minimal suites; full suites come in Phase 1b/2):

| suite | Protocol | key assertions |
|---|---|---|
| `test_engine_adapter.py` | `EngineAdapter` | resolve_support returns typed result; validate rejects incompatible config; render produces non-empty plan; verify returns VerificationReport; classify returns typed failure or success; extract_schema returns engine constraints |
| `test_artifact_resolver.py` | `ArtifactSourceResolver` | locate returns ArtifactLocator; resolve returns ArtifactSourceObservation with immutable revision, manifest, digests; same content from two sources shares one ArtifactIdentity |
| `test_render_target.py` | `RenderTarget` | render accepts DeploymentPlan and produces a string; round-trip parse of rendered output recovers shared fields |
| `test_evidence_source.py` | `EvidenceSource` | source_name is non-empty; collect returns list of dicts; entries carry required fields |
| `test_planning_source.py` | `PlanningSource` | predict accepts ModelSpec + HardwareSpec + WorkloadShape and returns PlanningClaim; unknown mechanism returns claim with `unknown` status |
| `test_evaluation_adapter.py` | `EvaluationAdapter` | accepts returns bool; prepare returns protocol state; execute returns TaskAttemptRecord list; collect aggregates scores; failed attempts preserved |
| `test_execution_target.py` | `ExecutionTarget` | kind/operator/provider/hardware/execution_fingerprint are non-empty; lifecycle methods callable in order; teardown is idempotent |

### 1.3 Implementation order for Part 1

1. Create `tests/conformance/` with a shared `conftest.py` providing
   fake-adapter factories.
2. Write suites in dependency order: execution_target, evidence_source,
   planning_source, artifact_resolver, render_target, engine_adapter,
   evaluation_adapter.
3. Every suite passes with the fake adapter. A real adapter plugs in by
   providing the factory fixture in its own test module.

---

## Part 2 — GPU-free components

These components run without a GPU. They are tested locally and in CI
(pure and contract tiers).

### 2.1 HF Hub artifact resolver

Module: `src/apron/adapters/evidence/hf_hub.py`

Implements `ArtifactSourceResolver` Protocol. Given an `ArtifactLocator`
with `source_kind: "huggingface"`, resolves to an
`ArtifactSourceObservation` with:

- Immutable revision SHA (resolved from requested revision or HEAD)
- Manifest from `model.safetensors.index.json` (total_size, weight_map)
- Per-file content digests (SHA-256 of `config.json`,
  `model.safetensors.index.json`)
- Model card metadata (license, pipeline_tag, tags)
- `gated` and `safetensors.parameters` from the API response

Uses `huggingface_hub` Python API. `huggingface_hub` is a dev dependency
only (not a production dependency); the resolver is used at plan time
and in CI, not at serving time.

**Test:** passes the `test_artifact_resolver.py` conformance suite.
Additional test: resolving Qwen3-8B at the pinned revision
(`b968826d9c46dd6066d109eabc6255188de91218`) produces the same digests
as the pinned HF Hub fixtures.

### 2.1a Identity chain construction

Module: `src/apron/application/orchestration/resolution.py`

Constructs the full identity chain from resolver output to deployable
solution. This is the "candidate graph resolution" the phase plan
requires (line 55).

**Steps:**

1. `ArtifactSourceObservation` → `ArtifactIdentity` (derived from
   resolved file contents and component structure, not registry name)
2. `ArtifactIdentity` + `config.json` → `ArtifactSpec` (weight manifest,
   content digests, config digest, tokenizer identity, lineage claims)
3. `ArtifactSpec` → candidate graph: enumerate the checkpoint-native
   BF16 artifact plus any published quantized variants with
   `ArtifactRelation` links. Unsupported quantization adapters remain
   visible as `unsupported`; they do not collapse the graph.
4. `ArtifactSpec` + pinned engine constraints → `ExecutionSpec` (engine
   image digest, resolved checkpoint method, selected kernels)
5. `ArtifactSpec` + `ExecutionSpec` + `HardwareSpec` + `CapabilitySignature`
   → `InferenceSolution` with one `direct_endpoint` binding

**Legacy evidence integration:** the `CatalogueImportSource` adapter
(Phase 0) feeds its 101 entries into the candidate graph at step 3.
Legacy entries seed candidates with `import_status:
owner_attested_boot` but cannot satisfy verification gates. The fixture
run exercises this flow: the legacy Qwen3-8B entry (if present) appears
in the candidate set alongside the freshly resolved artifact.

**Test:** Qwen3-8B resolution produces a candidate graph with at least
the BF16 checkpoint-native artifact. The `InferenceSolution` fingerprint
changes when `ArtifactSpec` or `ExecutionSpec` changes. Legacy entries
appear in the candidate set but do not satisfy `deployment_feasible`.

### 2.1b Engine constraint extraction

Module: `src/apron/adapters/backends/vllm_constraints.py`

Extracts version-pinned engine constraints from the vLLM image and
persists them as an `ExecutionSpec` input to the calculator.

**Extracted constraints:**

- Supported architectures (from `supported_models.md` or runtime
  `--help` inspection)
- Task registry (`generate`, `embed`, `classify`, `score`, `reward`,
  `transcription`)
- KV cache spec classes (`FullAttentionSpec`, `MLAAttentionSpec`,
  `SlidingWindowSpec`, `MambaSpec`)
- Available quantization schemes and their compute-capability
  requirements
- Default and maximum `max_model_len`

Constraints are pinned to the image tag and digest. A different image
produces different constraints.

**Test:** extraction from the pinned image tag produces constraints
consistent with the pinned `tests/fixtures/external-formats/vllm/`
source files.

### 2.1c Tensor byte extraction

Module: `src/apron/adapters/evidence/safetensors_reader.py`

Reads safetensors file headers to extract per-tensor byte counts for
memory calculation. This is distinct from the HF Hub resolver's
`total_size` field — the calculator needs per-tensor shapes and dtypes
to compute actual weight memory, not just a single total.

**Behavior:**

- Parses safetensors index (`weight_map`) to identify shard files
- Reads the header (first 8 bytes = header length, then JSON metadata)
  of each shard to extract tensor names, shapes, and dtypes
- Computes per-tensor byte count as `product(shape) × dtype_bytes`
- Also extracts auxiliary scale tensor bytes (for quantized artifacts
  with separate scale tensors)

For Phase 1a, the reader operates on cached files from HF Hub download.
It does not download shards itself.

**Test:** reader on the pinned Qwen3-8B `model.safetensors.index.json`
fixture produces non-zero byte counts that sum to approximately
`metadata.total_size`.

### 2.2 Calculator implementation

Module: `src/apron/domain/mechanisms/calculator.py`

Implements the `calculate` function contract from Phase 0
(phase-0-brief.md §2.8 step 4). Phase 1a implements one mechanism
branch: **dense GQA autoregressive decode** (`FullAttentionSpec`).

**Inputs:** `ModelSpec` component graph (from `config.json`), exact
tensor bytes (from `model.safetensors.index.json` weight_map and
headers), `HardwareSpec` (RTX 4090: 24 GB, compute capability 8.9),
`ExecutionSpec` (vLLM engine constraints from pinned image),
`WorkloadShape` (token distribution: ISL=512, OSL=128).

**Calculation (from architecture dispatch proof):**

- Model weight memory: sum of tensor bytes from safetensors index
  (actual bytes, not nominal bit-width × parameter count)
- KV cache per token: `2 × num_layers × num_kv_heads × head_dim ×
  dtype_bytes` (for GQA: num_kv_heads < num_attention_heads)
- KV cache total: `kv_per_token × (ISL + OSL) × max_batch_size`
- Activation estimate: fingerprinted prediction with uncertainty bounds
- Non-PyTorch overhead: fingerprinted prediction
- CUDA graph memory: fingerprinted prediction (applied only when
  `enforce_eager=False`)
- Available KV cache: `total_gpu_memory × gpu_memory_utilization −
  weight_memory − activation − overhead − cuda_graph`

**Output:** `PlanningClaim` with `claim_scope: "memory"`, producer
`"apron-calculator/0.1"`, `EpistemicStatus` = `predicted` with
uncertainty bounds and calibration scope.

The prediction breakdown must be **individually inspectable**, not a
single number. The `PlanningClaim` carries separate fields for:
`weight_memory_bytes`, `kv_cache_bytes`, `activation_estimate_bytes`,
`non_pytorch_overhead_bytes`, `cuda_graph_estimate_bytes`,
`available_kv_cache_bytes`, `total_required_bytes`. Each field is
independently queryable by the CLI (`apron plan --verbose`) and stored
in the record.

All other mechanism branches return `PlanningClaim` with `unknown`
status.

**Test:** passes `test_planning_source.py` conformance suite. Additional
tests: Qwen3-8B BF16 prediction produces a non-zero memory budget with
all breakdown fields populated; unknown mechanism returns `unknown`;
same inputs produce byte-identical output (determinism); each breakdown
field is independently non-zero.

### 2.3 Renderers

Module: `src/apron/adapters/renderers/`

Two `RenderTarget` implementations:

#### 2.3.1 `vllm_serve.py` — native `vllm serve` command

Accepts a `DeploymentPlan` and renders a `vllm serve` command string
with all resolved arguments. Arguments come from the plan's engine
configuration, not from a hardcoded template.

#### 2.3.2 `docker_compose.py` — Docker Compose file

Accepts a `DeploymentPlan` and renders a `docker-compose.yaml` with the
pinned `vllm/vllm-openai:<tag>` image, volume mounts, GPU device
requests, environment variables, and port mapping.

#### 2.3.3 Ecosystem export renderers

Three export renderers (one per pinned ecosystem shape):

- `recipes_export.py` — exports `DeploymentPlan` to vllm-recipes YAML
- `aiconfigurator_export.py` — exports to aiconfigurator estimate request
- `inferencex_export.py` — exports to InferenceX result row (config
  fields only; metric fields `not_measured`)

Each renderer's shared-field table is derived from the pinned external
format fixtures (phase-0-brief.md §3.5a).

**Test:** passes `test_render_target.py` conformance suite. Export tests:
every `DeploymentPlan` golden fixture exports losslessly for shared
fields to all three ecosystem shapes and round-trips (existing
`test_export.py` assertions, now with real renderers replacing stubs).

### 2.4 CLI `plan` command

Module: `src/apron/interfaces/cli.py`

```
apron plan REQUEST_FILE [--target TARGET_SPEC]
```

Reads an accepted `DecisionRequest` from a JSON file. Resolves the
artifact (HF Hub), builds `ModelSpec` from `config.json`, dispatches
the calculator, generates `DeploymentPlan` candidates, renders them,
and writes `deployment-plan.json` to the output directory.

GPU-free. No provider credentials required. No paid actions.

**Target spec** (optional): a JSON file or inline `--gpu-sku` /
`--gpu-memory` / `--gpu-count` flags specifying the `HardwareSpec` to
plan against. Defaults to the fixture's declared target.

**Test:** `apron plan` with the self-hosted golden fixture's
`decision-request.json` produces a valid `deployment-plan.json` that
matches the golden fixture (field by field for identity fields;
timestamps and display metadata may differ).

---

## Part 3 — Execution components

These components require a GPU target or external service. They are
tested first against fakes in CI, then against the real target in the
fixture run (Part 5).

### 3.1 Fixture selection

The artifact and target are selected by the recorded engineering rule
(phase-plan.md §Phase 1a, line 51). The rule's criteria and the
selected fixture:

| criterion | evidence |
|---|---|
| immutable official revision | Qwen/Qwen3-8B at HF revision `b968826d9c46dd6066d109eabc6255188de91218` |
| clear test rights | Apache-2.0 license; no gated access |
| support in pinned vLLM image | `Qwen2ForCausalLM` in supported_models.md; `generate` task |
| implemented memory mechanism | dense GQA → `FullAttentionSpec` → calculator branch implemented in §2.2 |
| checkpoint-native fit with headroom | BF16 weights ~16 GB; RTX 4090 24 GB; headroom for KV cache at ISL=512/OSL=128/concurrency=4 |
| reproducibility | deterministic scorer, fixed seed, 1 repetition |
| isolation of product path | no eval vendor, no managed provider, no contributed resource |
| availability | RunPod Secure Cloud RTX 4090 |
| total expected workload cost | <$2 (see §3.3 budget proof) |

**Selected fixture:** Qwen3-8B BF16 on RTX 4090 via vLLM.
**Selected image:** `vllm/vllm-openai:v0.29.0` (latest stable at
brief date). The implementer must verify that the KV cache interface
at this tag is compatible with the pinned source commit's
`FullAttentionSpec`; if not, pin the latest compatible release and
record the deviation.

### 3.2 Rented-provider ExecutionTarget

Module: `src/apron/adapters/backends/runpod.py`

Implements `ExecutionTarget` Protocol with `kind: "rented-provider"`,
`provider: "runpod"`.

**Lifecycle:**

1. `prepare()` — validate RunPod API credentials (from environment
   `RUNPOD_API_KEY`), check GPU availability.
2. `provision()` — create a CPU-only instance, create network volume,
   download model weights to volume (`huggingface_hub` snapshot download
   with pinned revision), stop CPU instance. Then create GPU instance
   (RTX 4090 Secure Cloud) with network volume attached and
   `vllm/vllm-openai:v0.29.0` image.
3. `execute(command)` — run a command inside the GPU instance.
4. `observe()` — collect GPU utilization, memory usage, process status.
5. `collect()` — download logs, profiling data, benchmark results.
6. `teardown()` — stop GPU instance, optionally destroy network volume.
   Idempotent: calling teardown twice is safe. Teardown runs on
   cancellation, timeout, crash, and target loss.

**Runtime hardware detection:** at provision time, the target queries
the actual GPU to build `HardwareSpec` from detected values — not
hardcoded constants. Detection reads: `torch.cuda.get_device_name()`,
`torch.cuda.get_device_properties()` (total memory, compute
capability), `nvidia-smi` (driver version), and
`torch.version.cuda`. The detected `HardwareSpec` is compared against
the plan's declared target; a mismatch (wrong SKU, insufficient
memory) aborts before any paid execution.

**Execution fingerprint:** built from detected hardware — GPU SKU,
total memory, compute capability, vLLM image digest (resolved at
provision time), PyTorch/CUDA/driver versions, `enforce_eager`
setting, TP/PP topology (`TP=1, PP=1`). No hardcoded values.

**No-GPU fallback:** when `RUNPOD_API_KEY` is absent or the target is
unavailable, returns a typed `hardware_unavailable` outcome without
attempting execution. This is the INV-30 path.

**Test:** passes `test_execution_target.py` conformance suite with a
fake RunPod client. Integration test against real RunPod is Part 5.

### 3.3 Budget proof

The Phase 1a run must be budget-feasible at `ContributedResourcePool = 0`
(phase-plan.md §Phase 0 exit gate, already tested in golden fixture
`phase-1a-run-plan.json`).

Updated budget from current RunPod pricing (2026-09-13):

| item | estimate |
|---|---|
| CPU instance for weight staging (1 hr) | $0.10 |
| Network volume (50 GB, 1 day) | $0.01 |
| GPU boot + initial measurement (~15 min) | $0.19 |
| Failure injection boot (~5 min) | $0.06 |
| Corrected boot + measurement (~15 min) | $0.19 |
| Serving benchmark via `vllm bench` (~10 min) | $0.12 |
| Task replay after correction (~5 min) | $0.06 |
| Headroom for retries and cleanup | $0.27 |
| **Total** | **$1.00** |

`MaintainerBaselineAllocation`: $5.00 (covers Phase 1a with margin for
iteration). The implementer updates the
`tests/fixtures/golden/maintainer-baseline-allocation.json` fixture
with the real allocation and re-runs `test_phase_1a_budget_feasible`.

### 3.4 vLLM engine adapter

Module: `src/apron/adapters/backends/vllm_engine.py`

Implements `EngineAdapter` Protocol.

| method | behavior |
|---|---|
| `resolve_support(model_spec, image_tag)` | checks `architectures` against pinned supported_models.md; returns typed support result with task set (`generate`, `embed`, etc.) |
| `validate(plan, target)` | checks TP divisibility against `num_attention_heads`/`num_kv_heads`, dtype compatibility with compute capability, max-model-len against checkpoint default |
| `render(plan)` | delegates to `vllm_serve.py` and `docker_compose.py` renderers |
| `verify(plan, target)` | executes boot sequence in pinned image; collects 15 non-overlapping memory profiling fields from vLLM's memory profiler; returns `VerificationReport` |
| `classify(error)` | parses vLLM error output into typed failure classes: `oom`, `engine_init`, `max_model_len`, `dtype_incompatible`, `tp_divisibility`, `quant_compute_capability`, `unknown` |
| `extract_schema(image_tag)` | extracts engine constraints from image: supported architectures, task registry, KV cache specs, available quantization schemes |

**Memory profiling fields** (from ADR-006, phase-plan.md §Phase 0 line
31): initial_total_memory, initial_free_memory, requested_memory,
model_weight_memory, persistent_consumption, transient_peak_headroom,
non_pytorch_increase, cuda_graph_estimate, cuda_graph_applied,
cuda_graph_actual, available_kv_cache_memory, safety_buffer,
profiling_num_layers, profiling_num_kv_heads, profiling_head_dim.

**Test:** passes `test_engine_adapter.py` conformance suite with a fake
engine backend. Integration test against real vLLM is Part 5.

### 3.5 Deterministic evaluation adapter

Module: `src/apron/adapters/evaluations/deterministic_scorer.py`

Implements `EvaluationAdapter` Protocol. For Phase 1a, this is the only
evaluation adapter — no model judge, no external eval service.

**Scorer:** exact string match after whitespace normalization. A case
passes when the model output matches the expected output.

**Behavior:**

- `accepts(protocol)` — returns True when `protocol.scorer_type ==
  "deterministic_exact_match"`
- `prepare(protocol)` — validates that every case has an `expected`
  field
- `execute(protocol, endpoint)` — sends each case to the vLLM OpenAI
  endpoint (`/v1/completions`), collects responses, scores each
- `collect(attempts)` — returns `TaskAttemptRecord` list with
  per-case scores, input/output tokens, latency, cost

Every attempt is recorded, including failures. Failed attempts are not
hidden from economics.

**Test:** passes `test_evaluation_adapter.py` conformance suite.
Additional test: scoring against a fake endpoint with known outputs
produces the expected pass/fail results.

### 3.6 CLI execution commands

Module: `src/apron/interfaces/cli.py`

#### `apron verify`

```
apron verify PLAN_FILE --target TARGET_SPEC
```

Runs bounded deployment verification against the specified target.
Boots vLLM, collects memory profiling, runs the deterministic task
suite, collects serving metrics via `vllm bench`, saves local
`VerificationReport` and `TaskAttemptRecord` files, tears down.

Paid action: before confirmation, displays the applicable data
destination (where records will be saved — local directory path),
maximum estimated cost (from budget proof), and hard deadline
(maximum wall-clock time before forced teardown). Requires
confirmation (or `--yes` for standing authorization).

#### `apron deploy`

```
apron deploy PLAN_FILE --target TARGET_SPEC
```

Same as `verify` but does NOT tear down. Leaves a managed endpoint
running. Prints the endpoint URL.

#### `apron run`

```
apron run -- COMMAND [ARGS...]
```

Wraps an existing execution for observation and diagnosis. Captures
vLLM logs, detects errors, classifies failures, suggests corrections.
Does not boot or manage the endpoint — the user's command does.

#### `apron report`

```
apron report RECORD_ID
```

Read-only. Displays a `DecisionReport`, `VerificationReport`, or
`TaskAttemptRecord` from local storage. Exports to JSON, YAML, or
the three ecosystem shapes.

#### `apron submit`

```
apron submit RECORD_ID --destination DEST
```

Separate opt-in. Sanitizes the record, previews the data destination,
shows what will be published, and requires explicit confirmation.
Phase 1a exit gate does NOT require submission — this command exists
but publication is not an exit condition.

#### `apron mcp`

```
apron mcp
```

Starts the local MCP server exposing every CLI operation as an MCP
tool. Claude Code configuration: `{"command": "apron", "args": ["mcp"]}`.

**Test:** each command has a unit test with fake adapters proving the
control flow (not the GPU execution). `plan` produces a valid plan;
`verify` calls the right lifecycle methods in order; `report` renders
without error; `submit` requires confirmation.

### 3.7 Local record storage

Module: `src/apron/adapters/backends/local_store.py`

All records are saved to a local directory. The default location is
`~/.apron/records/`. Each record is a JSON file named by its
`record_digest_hex` (the SHA-256 multihash of its canonical form).

**Directory structure:**

```
~/.apron/records/
  verification-reports/
    1220<hex>.json
  task-attempt-records/
    1220<hex>.json
  remediation-records/
    1220<hex>.json
  deployment-plans/
    1220<hex>.json
  decision-reports/
    1220<hex>.json
  planning-claims/
    1220<hex>.json
  diagnosis-rules/
    1220<hex>.json
```

`apron report RECORD_ID` accepts either a full digest or a unique
prefix (minimum 8 hex characters). It searches all subdirectories
for a matching file.

`apron deploy` additionally writes a `retained-solutions/` entry
that maps a solution fingerprint to the endpoint URL, target spec,
and the `DeploymentPlan` digest. This is how a deployed solution is
"retained" — it has a local record that `apron report` can find and
that prevents teardown.

**Test:** write a record, retrieve it by full digest and by prefix.
Retained solution is findable after `deploy`.

### 3.8 Sanitization and provenance validation

Module: `src/apron/application/sanitization.py`

The exit gate requires records "pass sanitization and provenance
validation." This module provides two functions:

**`sanitize(record) → SanitizedRecord`:**

- Strips any field that could contain credentials (`RUNPOD_API_KEY`,
  auth tokens, bearer headers)
- Strips raw prompt text if the task is classified private
- Preserves all fingerprints, digests, and structural fields
- Returns a `SanitizedRecord` wrapper with `sanitized: True` flag

**`validate_provenance(record) → list[str]`:**

- Every fingerprint reference resolves to an existing record in local
  storage (or is marked `external`)
- `decision_request_digest` matches a stored `DecisionRequest`
- `execution_fingerprint` contains all required fields (GPU SKU,
  memory, compute capability, image digest)
- `corrects` field (if present) points to a stored record
- Returns a list of validation errors (empty = valid)

**Test:** a record with embedded credentials fails sanitization. A
record with a dangling fingerprint reference fails provenance
validation. A clean record passes both.

---

## Part 4 — Orchestration

### 4.1 Qualification graph

Module: `src/apron/application/orchestration/qualification.py`

Implements the deterministic obligation graph from phase-plan.md §Phase
0 line 25 and ADR-011 §9:

```
candidate_discovered
  → capability_eligible        (CapabilitySignature check)
  → policy_checked             (privacy/license/data-residency)
  → identity_resolved          (ArtifactIdentity from HF Hub)
  → deployment_feasible        (calculator prediction fits target)
  → authorized                 (AuthorizationEnvelope permits)
  → task_evaluated             (EvaluationAdapter runs task suite)
  → serving_verified           (vllm bench against ServingWorkloadSpec)
  → task_reproduced            (exact-solution replay on retained endpoint)
  → qualified
```

Phase 1a exercises this graph for one candidate. The graph is the same
for managed API and compound solutions (those skip deployment_feasible
and serving_verified, marked `provider_opaque`).

**Key constraint:** implementations may interleave measurements to
minimize cost but cannot skip an obligation. For Phase 1a with one
candidate, the order is fixed: resolve → predict → boot → measure →
evaluate → benchmark → replay.

**Evidence-bound result states (ADR-006):** the qualification graph
tracks the candidate's output tier at each node and advances it only
when evidence justifies the transition:

| after node | output tier | evidence required |
|---|---|---|
| `deployment_feasible` | `candidate` | GPU-free prediction only |
| `serving_verified` | `conservative` | boot-verified, memory measured |
| `task_evaluated` | toward `recommended` | task evidence for exact endpoint |
| `task_reproduced` | `qualified` | exact-solution reproduction + all SLOs |

A candidate cannot skip tiers. The `qualification_status` field on
`CandidateEntry` in the `DecisionReport` is set to the highest tier
the evidence justifies — never higher.

**Output:** `DecisionReport` with one candidate, its
`qualification_status` and `qualification_graph_state` (which node was
reached), evidence references, and economics.

**Test:** the qualification graph accepts the self-hosted golden fixture
and advances through every node to `qualified`. A fixture missing a
required capability is pruned at `capability_eligible`. A fixture
violating a constraint is pruned at `policy_checked`. A candidate that
passes boot but fails task evaluation cannot reach `recommended`. A
candidate that passes task evaluation but fails serving SLO cannot
reach `qualified`.

### 4.2 Decision report generation

Module: `src/apron/application/orchestration/decision.py`

Constructs a `DecisionReport` from the qualification graph output.
Preserves every considered solution (even if only one), rejection
reasons, evidence state, evaluation coverage, and economics.

For Phase 1a: single candidate, `disclosed_comparable_set: []` (cannot
be `measured_efficient`), per-candidate economics with
`market_equivalent_price`, `gross_attributable_cost`,
`subsidy_applied: 0`, `project_out_of_pocket_cost`.

**Test:** generated report matches the self-hosted golden fixture
structure. `disclosed_comparable_set` is empty for single candidate.

---

## Part 5 — The fixture run

This is the actual GPU execution. It exercises the full product loop.

### 5.1 Accepted fixture documents

Under `tests/fixtures/phase-1a-run/` (separate from golden fixtures,
which are synthetic):

| document | content |
|---|---|
| `decision-request.json` | accepted request for Qwen3-8B text generation, quality floor (3/5 cases pass), P99 TTFT < 2s, ISL=512/OSL=128/concurrency=4, Apache-2.0 only |
| `task-suite-spec.json` | 3–5 deterministic text cases with exact expected outputs, `deterministic_exact_match` scorer |
| `application-spec.json` | minimal single-role (one prompt template, no agent/tools) |
| `serving-workload-spec.json` | ISL=512, OSL=128, concurrency=4, P99 TTFT < 2s, P99 TPOT < 100ms |
| `evaluation-protocol.json` | deterministic scorer, 1 seed, 1 repetition, no judge |

These are real accepted documents, not golden fixtures. The golden
fixtures are the synthetic reference; these are the live input to the
run.

### 5.2 Execution sequence

The fixture run is a single orchestrated sequence. Each step produces
records that are inputs to the next.

1. **Resolve artifact.** `apron plan decision-request.json` resolves
   Qwen3-8B from HF Hub, builds ModelSpec, dispatches calculator,
   generates DeploymentPlan, renders `vllm serve` command and Docker
   Compose. Saves `deployment-plan.json`.

2. **Predict memory.** Calculator produces a `PlanningClaim` with
   predicted memory budget. This is the GPU-free prediction that will
   be compared against measurement.

3. **Boot and measure.** `apron verify deployment-plan.json --target
   runpod-4090` provisions the target (§3.2), boots vLLM, collects
   15 memory profiling fields. Saves `VerificationReport` with
   `EpistemicStatus: measured`.

4. **Record prediction delta.** Compare predicted memory (step 2)
   against measured memory (step 3). The delta is a first-class
   diagnostic that calibrates future predictions.

5. **Run task suite.** The evaluation adapter sends each case to the
   vLLM endpoint, scores responses, records `TaskAttemptRecord` per
   case.

6. **Measure serving.** Run `vllm bench` with the accepted
   `ServingWorkloadSpec` parameters. Record TTFT, TPOT, throughput
   percentiles.

7. **Inject failure.** One deliberate, source-backed, reversible
   failure: set `max_num_seqs=256` (or equivalent OOM-inducing batch
   parameter) to trigger an OOM at boot. The failure is source-backed:
   the OOM threshold is traceable to vLLM's memory allocation code at
   the pinned revision.

8. **Diagnose.** The engine adapter classifies the error (`oom`).
   A `DiagnosisRule` with `status: hypothesis` proposes a correction:
   reduce `max_num_seqs` to fit within the measured memory budget.

9. **Correct and reboot.** Generate a corrected `DeploymentPlan` with
   the reduced parameter. Reboot vLLM with the correction. Collect
   memory profiling again. The corrected boot succeeds.

10. **Promote diagnosis.** The correction restored engine health →
    `mechanism_outcome: verified`. The `DiagnosisRule` advances to
    `status: mechanism_verified`.

11. **Replay accepted task.** Re-run the deterministic task suite on
    the corrected endpoint. All cases pass → `request_outcome:
    satisfied`. The public result is `Fixed`.

12. **Generate records.** Save `RemediationRecord` with
    `mechanism_outcome: verified`, `request_outcome: satisfied`,
    `corrected_plan_digest`, `accepted_request_digest`.

13. **Generate decision report.** `DecisionReport` with one qualified
    candidate, evidence chain, economics.

14. **Teardown.** Destroy GPU instance and network volume.

### 5.3 Records produced

The run produces these schema-valid records in local storage:

- `VerificationReport` × 2 (initial boot, corrected boot)
- `TaskAttemptRecord` × cases × 2 (initial run, replay after correction)
- `RemediationRecord` × 1 (Fixed)
- `DiagnosisRule` × 1 (mechanism_verified)
- `DeploymentPlan` × 2 (original, corrected)
- `DecisionReport` × 1
- `PlanningClaim` × 1 (GPU-free prediction)

Every record carries exact fingerprints:
`decision_request_digest`, `task_suite_fingerprint`,
`application_fingerprint`, `evaluation_protocol_fingerprint`,
`solution_fingerprint`, `execution_fingerprint`.

### 5.4 What the run does NOT prove

- General model quality (the task suite is an engineering fixture)
- Broad candidate discovery (one preselected model)
- External evaluation services (deterministic scorer only)
- Managed API path (self-hosted only; managed-api golden fixture is
  Phase 0 schema evidence)
- Compound routing (single endpoint)
- Publication (submit exists but is not exercised)
- Contributed resources (ContributedResourcePool = 0)

---

## Part 6 — Exit gate as tests

### 6.1 Test location

Phase 1a tests go in:
- `tests/conformance/` — extension-point conformance suites
- `tests/unit/` — adapter unit tests, CLI tests, orchestration tests
- `tests/integration/` — GPU execution tests (path-triggered, not
  every PR)

### 6.2 Conformance suite tests

```python
def test_engine_adapter_conformance(vllm_engine_adapter):
    """vLLM adapter passes the EngineAdapter conformance suite."""

def test_artifact_resolver_conformance(hf_hub_resolver):
    """HF Hub resolver passes the ArtifactSourceResolver conformance suite."""

def test_execution_target_conformance(runpod_target):
    """RunPod target passes the ExecutionTarget conformance suite."""

def test_evaluation_adapter_conformance(deterministic_scorer):
    """Deterministic scorer passes the EvaluationAdapter conformance suite."""

def test_render_target_conformance(vllm_serve_renderer):
    """vllm serve renderer passes the RenderTarget conformance suite."""
```

### 6.3 Calculator tests

```python
def test_calculator_qwen3_8b_bf16_prediction():
    """Qwen3-8B BF16 produces a non-zero memory prediction with
    uncertainty bounds for RTX 4090."""

def test_calculator_unknown_mechanism():
    """An MLA mechanism returns unknown, not a fallback."""

def test_calculator_deterministic():
    """Same inputs + same ports = byte-identical PlanningClaim."""
```

### 6.4 CLI tests

```python
def test_cli_plan_produces_valid_plan():
    """apron plan with the self-hosted golden fixture produces a valid
    DeploymentPlan."""

def test_cli_verify_lifecycle():
    """apron verify calls prepare → provision → execute → observe →
    collect → teardown in order, with fake target."""

def test_cli_report_read_only():
    """apron report renders a record without side effects."""

def test_cli_submit_requires_confirmation():
    """apron submit without --yes prompts for confirmation."""

def test_cli_no_gpu_returns_typed_outcome():
    """apron verify without RUNPOD_API_KEY returns
    hardware_unavailable, not an error."""
```

### 6.5 Resolution and identity chain tests

```python
def test_identity_chain_from_hf_hub():
    """ArtifactSourceObservation → ArtifactIdentity → ArtifactSpec →
    ExecutionSpec → InferenceSolution constructed without hardcoding."""

def test_candidate_graph_includes_unsupported():
    """Unsupported quantization variants remain visible as unsupported
    in the candidate graph, not silently dropped."""

def test_legacy_evidence_seeds_candidates():
    """CatalogueImportSource entries appear in the candidate graph
    but do not satisfy deployment_feasible."""

def test_engine_constraint_extraction():
    """Constraints extracted from pinned image are consistent with
    pinned external-format vLLM source files."""

def test_tensor_byte_extraction():
    """Safetensors header reader produces per-tensor byte counts
    that sum to approximately total_size."""
```

### 6.6 Qualification graph tests

```python
def test_qualification_graph_single_candidate():
    """One candidate advances through all nodes to qualified."""

def test_qualification_graph_capability_pruning():
    """Missing capability prunes at capability_eligible."""

def test_qualification_graph_policy_pruning():
    """Violated constraint prunes at policy_checked."""

def test_qualification_graph_evidence_tiers():
    """A candidate that passes boot but fails task cannot reach
    recommended. A candidate that passes task but fails serving SLO
    cannot reach qualified."""
```

### 6.7 Storage and sanitization tests

```python
def test_record_storage_round_trip():
    """Write a record to local store, retrieve by full digest and
    by 8-character prefix."""

def test_retained_solution_findable():
    """After deploy, the retained solution is findable by
    apron report."""

def test_sanitization_strips_credentials():
    """A record containing RUNPOD_API_KEY fails sanitization and
    the key is stripped from the sanitized output."""

def test_provenance_validation_dangling_ref():
    """A record with a fingerprint pointing to a non-existent
    record fails provenance validation."""

def test_provenance_validation_clean():
    """A record with all references resolving passes provenance
    validation."""
```

### 6.8 Integration tests (GPU-dependent)

These run against the real RunPod target. They are the exit gate.

```python
def test_fixture_run_record_1():
    """Internal deployment evidence record #1 exists with exact
    fingerprints, predicted-vs-measured memory, and
    EpistemicStatus: measured."""

def test_fixture_run_remediation_record_1():
    """Remediation record #1 exists with mechanism_outcome: verified,
    request_outcome: satisfied, corrected_plan_digest, and
    accepted_request_digest."""

def test_corrected_endpoint_reproduces_task():
    """The corrected endpoint reproduces the accepted deterministic
    task result (same cases pass before and after correction)."""

def test_records_schema_valid():
    """All produced records pass schema validation and round-trip
    through canonicalize → deserialize → re-canonicalize."""

def test_records_pass_sanitization():
    """All produced records pass sanitization and provenance validation
    (no credentials, no raw prompts, fingerprints resolve)."""

def test_no_gpu_non_measurement():
    """apron verify with no GPU target returns a typed non-measurement
    outcome, not an error."""

def test_target_loss_triggers_teardown():
    """A simulated target loss triggers idempotent teardown."""

def test_failed_run_triggers_teardown():
    """A boot failure triggers teardown, not a dangling instance."""

def test_self_hosted_acceptance_inv30():
    """The self-hosted acceptance suite passes with external evaluation
    services, managed inference providers, compound routing and
    contributed resources disabled (INV-30)."""
```

### 6.9 Export tests (extended)

```python
def test_real_plan_exports_to_recipes():
    """The real DeploymentPlan (not golden fixture) exports to recipes
    YAML with shared fields matching the pinned vllm-recipes format."""

def test_real_plan_exports_to_aiconfigurator():
    """The real DeploymentPlan exports to aiconfigurator estimate
    request with shared fields matching the pinned format."""

def test_real_plan_exports_to_inferencex():
    """The real DeploymentPlan exports to InferenceX row shape."""
```

---

## Exit gate summary

Phase 1a is complete when `prek run --all-files` passes and:

1. Seven conformance suites exist under `tests/conformance/`, each
   passing with a fake adapter and with the real adapter (Part 1).
2. The calculator produces a non-zero memory prediction for Qwen3-8B
   BF16 on RTX 4090 with all breakdown fields individually populated,
   and returns `unknown` for unimplemented mechanisms (Part 2, §2.2).
3. `apron plan` produces a valid `DeploymentPlan` from an accepted
   `DecisionRequest` without GPU access, including candidate graph
   resolution with legacy evidence integration (Part 2, §2.1a, §2.4).
4. Every `DeploymentPlan` exports losslessly for shared fields to the
   three pinned ecosystem shapes (Part 2, §2.3.3).
5. The full identity chain — `ArtifactSourceObservation` →
   `ArtifactIdentity` → `ArtifactSpec` → `ExecutionSpec` →
   `InferenceSolution` — is constructed from HF Hub resolution and
   engine constraint extraction, not hardcoded (Part 2, §2.1a, §2.1b).
6. Internal deployment evidence record #1 exists with exact
   decision/task/application/evaluator/solution and execution
   fingerprints, provider, instance, digest, predicted-versus-measured
   memory, and serving observations (Part 5, §5.3).
7. Remediation record #1 exists with `mechanism_outcome: verified`
   and `request_outcome: satisfied`; the corrected endpoint
   reproduces the accepted deterministic task result (Part 5, §5.2
   steps 7–12).
8. Hardware detection is runtime, not hardcoded: `HardwareSpec` is
   built from the actual GPU at provision time (Part 3, §3.2).
9. A no-GPU target returns a typed non-measurement outcome; target
   loss and every failed run trigger teardown (Part 3, §3.2).
10. `apron verify` displays data destination, estimated cost, and hard
    deadline before confirmation (Part 3, §3.6).
11. `apron report` is read-only and `apron submit` requires explicit
    confirmation (Part 3, §3.6).
12. All records are saved to local storage by content digest and
    retrievable by `apron report` using digest or prefix (Part 3, §3.7).
13. All records pass sanitization (no credentials or private data
    leaked) and provenance validation (all fingerprint references
    resolve) (Part 3, §3.8).
14. Evidence-bound result states advance correctly through the
    qualification graph: `candidate` → `conservative` → `recommended`
    → `qualified`, never skipping a tier (Part 4, §4.1).
15. The self-hosted acceptance suite passes with external evaluation
    services, managed inference providers, compound routing and
    contributed resources disabled (INV-30).
16. Publication and provider participation are not exit conditions.
