# Failure knowledge base — diagnostic intelligence that compounds

**Date:** 2026-09-07 · **Source:** design grilling session — "how to make failure collection practically useful and meaningful"
**Authority:** ADR-003, INV-2, INV-10

## 1. Purpose

The knowledge base transforms individual failure observations into reusable diagnostic intelligence. Not a dumb lookup table ("this error → this fix") but a graph of relationships: what breaks together, what fixes what, how confidently, at what cost, and on which engine version. Every boot — successful or failed — feeds it. The agent reads it before every dispatch and applies all known fixes at once, not one at a time.

## 2. The failure landscape — 5 layers

Every vLLM deployment failure falls into one of five layers. Each layer has different properties for collection, reusability, and downstream value.

### Layer 1 — Config failures (BEFORE weight download)

The pinned vLLM config source contains multiple explicit rejection paths across `vllm/config/model.py`, `vllm/config/vllm.py`, and `vllm/config/scheduler.py`. Syntactic `raise` counts are not treated as semantic validator counts.

| failure | example | deterministic? | reusable? |
|---|---|---|---|
| dtype vs declared scheme capability | one specific quantization scheme below its version-pinned minimum | **Within the exact scheme/version scope** | **YES — scoped rule, never generalized to all FP8** |
| TP divisibility | 32 heads % 3 TP ≠ 0 | **Always** — same model = same failure | **YES — permanent rule** |
| PP not supported | architecture lacks `SupportsPP` | **Always** — same architecture | **YES — permanent rule** |
| max-model-len exceeds derived | requested 128K, model supports 32K | **Always** — same model config | **YES — permanent rule** |
| quant method incompatible | AWQ with unsupported activation dtype | **Always** | **YES — permanent rule** |

**Knowledge base value: HIGHEST.** See it once, prevent it forever. These become version-pinned prediction rules; they do not become vLLM validation without GPU execution.

### Layer 2 — Load failures (DURING weight loading)

| failure | example | deterministic? | reusable? |
|---|---|---|---|
| OOM during weight load | 64 GB weights on 24 GB GPU | **Always** — same math | **YES** — but the static prediction should have caught an artifact larger than declared VRAM |
| trust_remote_code required | custom model implementation | **Within the exact artifact/implementation scope** | **YES — resolved implementation requirement, not a family flag** |
| Weight format unsupported | GGUF in vLLM, bin format | **Always** — wrong format | **YES — config resolution gate** |
| Gated model, no token | Llama without HF token | **Always** — same auth state | **NO** — user-specific, not model-specific |
| Corrupted download | network glitch | **Never** — transient | **NO — filter out, don't record** |

**Knowledge base value: MEDIUM.** A trivially oversized artifact should have been caught by the prediction — if it was not, the knowledge base calibrates the estimate. `trust_remote_code` is resolved for the exact artifact and selected engine implementation; it is not transferred across a broad model family. Corrupted downloads are NOISE — they must be recognized and discarded.

### Layer 3 — Init failures ("Engine core initialization failed")

The `Engine core initialization failed` umbrella appears repeatedly in the preserved GitHub issue snapshot and hides many distinct causes; no total issue count is used as a product claim:

| actual cause | deterministic? | reusable? |
|---|---|---|
| KV cache won't fit after weights + activation | **Always** for this config | **YES** — calibrates the activation estimate |
| CUDA graph capture failed | **Execution-stack-specific** — model, backend, graph mode, GPU and versions matter | **YES — fingerprinted rule only after failure/correction/mechanism proof; each application separately checks the accepted request** |
| CUDA graph replay error | **Execution-stack-specific** | **YES — fingerprinted rule; `enforce_eager` is a hypothesis until proven for that incident** |
| NCCL init failed (multi-GPU) | **Topology-specific** — PCIe vs NVLink, driver version | **SOMETIMES** — depends on infrastructure |
| Kernel compilation failure | **Execution-fingerprint-specific** — artifact, scheme, selected backend/kernel, GPU, engine and compiler versions all matter | **SOMETIMES** — starts as a hypothesis; correction requires matching re-execution proof |

**Knowledge base value: HIGH.** These are failures a CPU prediction may miss because they depend on activation memory and CUDA graph behavior. Each measured failure can improve later predictions without being relabeled as validation.

### Layer 4 — Runtime failures (DURING serving, after successful boot)

| failure | example | deterministic? | reusable? |
|---|---|---|---|
| OOM during inference | concurrency spike exceeds KV budget | **At that concurrency** | **YES — max safe concurrency for this config** |
| Numerical instability | NaN in attention with FP16 on Gemma | **Always** for that dtype+model | **YES — permanent dtype rule** (use bfloat16) |
| NCCL timeout during TP | slow PCIe link under load | **Probabilistic** | **SOMETIMES** — topology-dependent |
| Generation quality degraded | wrong sampling params after engine update | **Version-specific** | **YES — per-version note** |

**Knowledge base value: MEDIUM.** These happen after a successful boot. They affect the benchmark record and max-concurrency recommendation, not the feasibility verdict.

### Layer 5 — Transient / infrastructure failures

| failure | example | deterministic? | reusable? |
|---|---|---|---|
| Provider API error | RunPod 503 | **Never** | **NO — retry, don't record** |
| GPU hardware fault | bit flip, driver crash | **Never** | **NO — different instance** |
| HuggingFace rate limit | 429 during download | **Never** | **NO — wait and retry** |
| Docker image pull timeout | network congestion | **Never** | **NO — retry** |

**Knowledge base value: ZERO.** Must be RECOGNIZED and FILTERED OUT. If the knowledge base records a provider 503 as a model failure, it poisons the data.

### Distinguishing useful failures from noise

| signal | means RECORD IT | means FILTER IT |
|---|---|---|
| Error from vLLM's own raise sites | Config/model problem | — |
| Python traceback inside vLLM code | Init/runtime failure | — |
| Same error on retry with same config | Deterministic | — |
| Error mentions NCCL/network/timeout | — | Probably transient — retry once, record only if it persists |
| Error mentions file I/O, download, permission | — | Infrastructure — retry, don't record |
| Error resolves on a different instance | — | Hardware fault — don't record |

**Practical filter:** the agent retries ONCE on transient-looking errors. If the same error persists on retry, it's deterministic — record it. If it resolves, discard it. Costs one extra boot ($0.17-$1) but prevents poisoning the knowledge base.

## 3. Knowledge base architecture — canonical records, rules and a SurrealDB diagnostic graph

Two layers, working together:

### Layer A — Deterministic rules (existing, from framework-spec.md §6a)

YAML files in `rules/`. One rule per proven failure class. Fingerprint → correction, with scoped mechanism-proving records. CI-validated. INV-2 separately enforces the accepted-request result for every application.

These handle KNOWN failures instantly. The agent checks rules first — if a fingerprint matches, the fix is applied without reasoning. Milliseconds.

### Layer B — Diagnostic graph in embedded SurrealDB

A SurrealDB graph stores accumulated failure experience as entities and relationships. It discovers co-occurrence, scoped compatibility and correction patterns from canonical evidence, remediation and orchestration records. It is the source of new rule hypotheses, not the authority that verifies them: only matching execution records can promote a diagnosis rule. The graph does not replace canonical records or deterministic rules and must be rebuildable from them.

**Selected implementation: embedded SurrealDB.** The official Python SDK supports in-process `mem://`, persistent `file://`, and persistent `surrealkv://` connections without a separate database server. That directly supports a local-first CLI while retaining structured filters, graph traversal and optional vector indexing in one query model. The diagnostic graph uses persistent SurrealKV storage; in-memory mode is limited to tests.

The product requires three query capabilities:
- "What co-occurs with this error?" → graph traversal
- "What breaks on A100 with DeepSeek?" → structured filter
- "Find the most similar past failure" → an optional similarity index whose result never proves a rule

Embedded connections currently lack the SDK's session, transaction and live-query APIs. Phase 0 therefore verifies that Apron's write/rebuild/crash-recovery path does not require those features before the storage contract is frozen. SurrealDB remains behind an internal repository interface: replacing it cannot change public schemas, record identity or evidence semantics.

**Initial SurrealQL projection schema:**

```sql
-- Entities
DEFINE TABLE failure_case SCHEMAFULL;
DEFINE FIELD timestamp ON failure_case TYPE datetime;
DEFINE FIELD error_fingerprint ON failure_case TYPE option<string>;
DEFINE FIELD error_text ON failure_case TYPE string;
DEFINE FIELD failure_layer ON failure_case TYPE int;  -- 1-5
DEFINE FIELD is_deterministic ON failure_case TYPE bool;
DEFINE FIELD model_family ON failure_case TYPE string;
DEFINE FIELD model_id ON failure_case TYPE string;
DEFINE FIELD architecture ON failure_case TYPE string;
DEFINE FIELD gpu_type ON failure_case TYPE string;
DEFINE FIELD gpu_count ON failure_case TYPE int;
DEFINE FIELD topology ON failure_case TYPE option<string>;
DEFINE FIELD quantization_candidate_id ON failure_case TYPE option<record>;
DEFINE FIELD artifact_id ON failure_case TYPE record;
DEFINE FIELD runtime_transform_id ON failure_case TYPE option<record>;
DEFINE FIELD execution_id ON failure_case TYPE record;
DEFINE FIELD quantization_fingerprint ON failure_case TYPE option<string>;
DEFINE FIELD engine_tag ON failure_case TYPE string;
DEFINE FIELD fix_applied ON failure_case TYPE option<string>;
DEFINE FIELD fix_succeeded ON failure_case TYPE bool;
DEFINE FIELD fix_source ON failure_case TYPE string;  -- rule/agent/human
DEFINE FIELD boot_attempts ON failure_case TYPE int;
DEFINE FIELD cost_usd ON failure_case TYPE float;

DEFINE TABLE hardware SCHEMAFULL;
DEFINE FIELD gpu_type ON hardware TYPE string;
DEFINE FIELD compute_capability ON hardware TYPE float;
DEFINE FIELD vram_gb ON hardware TYPE int;
DEFINE FIELD supported_dtypes ON hardware TYPE array<string>;
DEFINE FIELD interconnects ON hardware TYPE array<string>;

DEFINE TABLE model_family SCHEMAFULL;
DEFINE FIELD name ON model_family TYPE string;
DEFINE FIELD always_requires ON model_family TYPE array<object>;
DEFINE FIELD known_quirks ON model_family TYPE array<object>;

-- Relationships
DEFINE TABLE co_occurs SCHEMAFULL TYPE RELATION IN failure_case OUT failure_case;
DEFINE FIELD probability ON co_occurs TYPE float;

DEFINE TABLE proves SCHEMAFULL TYPE RELATION IN failure_case OUT record();  -- links to diagnosis rule proving records

DEFINE TABLE requires SCHEMAFULL TYPE RELATION IN record() OUT record();
DEFINE FIELD reason ON requires TYPE string;
DEFINE FIELD since_version ON requires TYPE option<string>;

DEFINE TABLE incompatible_with SCHEMAFULL TYPE RELATION IN record() OUT record();
DEFINE FIELD evidence ON incompatible_with TYPE string;

-- Indexes
DEFINE INDEX idx_model_family ON failure_case FIELDS model_family;
DEFINE INDEX idx_gpu_type ON failure_case FIELDS gpu_type;
DEFINE INDEX idx_engine_tag ON failure_case FIELDS engine_tag;
DEFINE INDEX idx_fingerprint ON failure_case FIELDS error_fingerprint;
```

**Non-KV memory profiles in the knowledge base (corrected 2026-09-07):**

The earlier design treated `peak_activation_memory` as a portable pure activation value. That is false. At the inspected vLLM revision it is `transient_peak_headroom + cudagraph_memory_estimate_applied`; separately adding the graph estimate can double-count it. Each Tier 1 boot and opted-in user deployment therefore stores the raw non-overlapping profiler and post-capture observations instead of one normalized activation number.

The measurement is valid only for its exact execution fingerprint: immutable artifact and quantization identity; normalized engine config and image digest; GPU SKU, total memory and compute capability; selected attention/linear/MoE backends; PyTorch, CUDA/runtime and driver versions; graph mode/capture sizes; allocator environment; TP/PP/DP/EP topology and rank; and profiling/workload shape. This is deliberately stricter than “same architecture” or “same GPU class.” vLLM's own startup-plan key already includes config, device identity/capability, PyTorch/CUDA, rank and world size, and also applies a free-memory gate before reuse.

```sql
-- Store raw profile observations from one execution
CREATE memory_profile SET
  execution_fingerprint = "sha256:<complete normalized fingerprint>",
  artifact_revision = "<immutable revision>",
  container_digest = "sha256:<image>",
  gpu_sku = "RTX 4090",
  compute_capability = "8.9",
  profile_shape = { max_num_batched_tokens: 2048 },
  initial_total_bytes = 25757220864,
  initial_free_bytes = 25000000000,
  requested_bytes = 23181498777,
  model_memory_bytes = 16380000000,
  persistent_consumed_bytes = 17000000000,
  transient_peak_headroom_bytes = 1500000000,
  non_torch_increase_bytes = 200000000,
  cudagraph_estimate_bytes = 314572800,
  cudagraph_estimate_applied = true,
  cudagraph_actual_bytes = 330000000,
  available_kv_cache_bytes = 4000000000,
  source_record = "record-0012";

-- Only an exact match can be returned as matching prior measurement evidence.
SELECT * FROM memory_profile
WHERE execution_fingerprint = "sha256:<complete normalized fingerprint>"
LIMIT 1;
```

Records with incomplete or different fingerprints are queried separately as calibration examples. Their values may inform a prediction with uncertainty and source lineage, but are never returned as measurements for the proposed execution.

**Example queries the agent runs:**

```sql
-- Before dispatching a boot: what's known about this combination?
SELECT *, ->co_occurs->failure_case AS related_failures
FROM failure_case
WHERE model_family = "DeepSeek"
  AND gpu_type = "A100"
  AND quantization_fingerprint = "<typed weight/activation/KV + backend fingerprint>";

-- What does this exact artifact/engine implementation require?
SELECT always_requires FROM execution_requirement
WHERE artifact_revision = "<immutable revision>"
  AND engine_image_digest = "sha256:<digest>";

-- Fix success rate for a specific correction
SELECT fix_applied, 
       count(IF fix_succeeded THEN 1 END) AS successes,
       count() AS total
FROM failure_case
WHERE error_fingerprint = "fp8e4nv_not_supported"
GROUP BY fix_applied;

-- Find failure chains: what ELSE breaks when this breaks?
SELECT ->co_occurs->failure_case.error_fingerprint AS chain
FROM failure_case
WHERE error_fingerprint = "trust_remote_code_missing";
```

### How the two layers work together

1. **New failure arrives.** Agent checks RULES first (YAML, deterministic, instant). Fingerprint match? → check the fix's success rate in the case graph for this exact model+GPU+quant context. 100% success? → apply with confidence. 0% or mixed? → do NOT apply; check what distinguished successes from failures. The agent NEVER proposes a fix it doesn't have evidence for.

2. **No rule matches but similar cases exist.** Agent queries the GRAPH: "show me failures on the same model_family + gpu_type + quantization." If a fix has 100% success across similar cases → propose it. If mixed → propose only if the current context matches the SUCCESS cases, not the failure cases. If 0% → don't propose.

3. **No cases match either.** Genuinely new failure. Agent reads vLLM source to trace the error. Proposes a hypothesis correction. Tests it ONCE. If boot succeeds → new case + new rule. If boot fails → case stored with `fix_succeeded=false`. The agent does NOT retry a failed fix — one attempt, one data point.

4. **Pattern extraction (periodic agent task).** Query cases for patterns with 100% fix success rate across N+ occurrences → promote to permanent rule in `rules/`. Cases become rules when evidence is strong enough.

5. **Provider circuit breaker (Layer 5 enforcement).** After 3 consecutive provider-level errors (HTTP 5xx, timeout, connection refused) from the same provider, the agent stops dispatching to that provider. Does NOT record these as model failures. Resumes when the provider responds. Prevents an outage from eating the cycle budget.

6. **Operational state.** Every dispatch, proposed external action and publication attempt is stored through the durable orchestration contracts from ADR-010, keyed by stable job/action id and deduplication key. At the start of each cycle, the controller reads that state to avoid duplicate paid runs or external mutations.

## 4. Scale trajectory

| stage | entities | relationships | storage contract |
|---|---|---|---|
| Initial loop | measured, failed and remediation records | rule proofs and execution lineage | canonical versioned records + embedded SurrealDB projection |
| Broader evidence | additional artifacts, fingerprints and targets | scoped co-occurrence and correction outcomes | same contracts and SurrealDB graph; indexes justified by query measurements |
| Continuous lab | release-spanning records and actions | provenance, freshness, supersession and publication lineage | same public contracts; embedded-versus-remote operation decided by measured concurrency and durability needs |

Capacity claims are established by benchmarks over representative record and query fixtures. Schema migration and export/rebuild tests remain required even though SurrealDB supports flexible entity and relationship types.

## 5. What the knowledge base gives the product

| capability | without knowledge base | with knowledge base |
|---|---|---|
| Fix a known error | lookup table, one fix | fix + co-occurring fixes applied together |
| Fix an unknown error | agent guesses from source | agent finds similar past cases, proposes highest-success-rate fix |
| Budget per failure | unknown cost per boot attempt | estimated cost from past cases with same failure class |
| Model family quirks | hardcoded flags | learned from accumulated cases, promoted to rules |
| Version-specific breaks | discovered by user crash | anticipated from Tier 0 diff + graph query |
| Transient vs deterministic | retry and hope | one retry to classify, then record or discard |

## Source references

- vLLM troubleshooting: `vLLM troubleshooting documentation at the pinned source commit`
- vLLM exception types: `vLLM exception types at the pinned source commit`
- SurrealDB embedded Python databases and limitations: [surrealdb.com/docs/reference/python/concepts/embedded-databases](https://surrealdb.com/docs/reference/python/concepts/embedded-databases)
- Official SurrealDB Python SDK: [github.com/surrealdb/surrealdb.py](https://github.com/surrealdb/surrealdb.py)
- Knowledge graph + LLM for fault diagnosis: [ScienceDirect 2026](https://www.sciencedirect.com/science/article/pii/S156849462600356X)
- Case-based reasoning in industrial diagnosis: [ResearchGate](https://www.researchgate.net/publication/360297091)
