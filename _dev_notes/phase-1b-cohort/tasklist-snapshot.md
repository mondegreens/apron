# Task dashboard snapshot v3 — 2026-09-26 (Layer F gate, second resubmission)

Status / blockedBy, one row per task (supervisor has no TaskList tool; this file mirrors it).

| # | Task | Status | Blocked by |
|---|------|--------|-----------|
| 1 | ANCHOR | open (anchor) | — |
| 2 | F1 recursive identity, vectors, migration note | completed | — |
| 3 | F2 strict schemas | completed | — |
| 4 | F3 verdicts from stored evidence | completed | — |
| 5 | F4 attempts stored; solution fp; production caller | completed | (was #16, done) |
| 6 | F5 diagnosis config | completed | — |
| 7 | F6 provenance + masking | completed | — |
| 8 | F7 token isolation (code; GPU evidence = L0-A3 in #20) | completed | — |
| 9 | F8 attention-free → unknown | completed | — |
| 10 | F9 real adapters on suites | completed | — |
| 11 | F10/F11/F12 (re-pin after rebuild = #29) | completed | — |
| 12 | F1/F2 approval issue (#41) | completed | — |
| 13 | L1 safety net, gpu_count, SKUs, rates | completed | — |
| 14 | L2 budget ledger + cost per schema | completed | — |
| 15 | L3 fingerprints from objects; placeholders removed | completed | — |
| 16 | L4 scheduler + orchestrator + composition root | completed | — |
| 17 | Serving measurement code | completed | — |
| 18 | §10 fix-proof code (real-log tests split to #32) | completed | — |
| 19 | Layer F gate | pending (this checkpoint) | 2–11 (done), 12 (done), 13–18 (done), 30, 31 (done) |
| 20 | L0 proofs (GPU) | pending | 19, 29 |
| 21 | L5 cohort + six proofs + serving (GPU) | pending | 20, 29, 32 |
| 22 | L6 exit gate test | pending | 21 |
| 23 | L7 findings article draft PR | pending | 22 |
| 24 | Gate: full regression zero red + cross-suite | pending | 23, 32 |
| 25 | Gate: blast radius proof | pending | 23, 32 |
| 26 | Gate: memory policy compliance | pending | 23, 32 |
| 27 | Gate: code review | pending | 23, 32 |
| 28 | Commit + PR | pending | 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 32 |
| 29 | Rebuild runner image, push, re-pin, pull by digest | pending (needs owner go: publishes to GHCR, moves :latest) | — |
| 30 | Scorer: template-aware chat_template_kwargs | completed | — |
| 31 | Layer F gate prerequisites (tests hook, marker, guards, all hooks) | completed | — |
| 32 | §10.3 per-class tests on REAL L0-F logs | pending | 20 (blocks 21, 28) |

## file:line evidence for DONE rows (resolved by grep, 66 anchors, 0 missing)

```
F1	src/apron/domain/fingerprints.py:69	def identity_projection
F1	src/apron/domain/fingerprints.py:90	def fingerprint_hex
F2	src/apron/domain/schemas/tasks.py:85	input_sequence_length
F2	src/apron/domain/schemas/migrations/__init__.py:55	def load_record
F2	src/apron/interfaces/cli.py:489	def _load_request_file
F3	src/apron/application/orchestration/qualification.py:83	def advance
F3	src/apron/domain/verdicts.py:47	def task_verdict
F3	src/apron/domain/verdicts.py:75	def reproduction_verdict
F3	src/apron/application/orchestration/serving.py:42	def evaluate_serving_slos
F4	src/apron/domain/schemas/solutions.py:113	class SolutionIdentity
F4	src/apron/application/orchestration/evidence.py:48	def solution_fingerprint
F4	src/apron/application/orchestration/evidence.py:172	def build_task_attempt
F4	src/apron/application/orchestration/cohort.py:743	def qualify_cohort
F4	src/apron/interfaces/cli.py:293	gpu_count=target.gpu_count
F5	src/apron/application/orchestration/diagnosis_pipeline.py:42	def diagnosis_model_config
F5	src/apron/interfaces/cli.py:761	def _resolve_diagnosis_config
F6	src/apron/application/sanitization.py:18	SECRET_VALUE_PATTERNS =
F6	src/apron/adapters/backends/llm_classifier.py:234	def classifier_input_digest
F6	src/apron/application/orchestration/remediation.py:405	classifier_model_id=diagnosis
F6	src/apron/application/orchestration/remediation.py:334	def _authorize_classification
F7	docker/start.sh:31	/run/apron/hf_token
F7	src/apron/adapters/backends/vllm_engine.py:503	def token_environ_check
F7	src/apron/adapters/backends/vllm_engine.py:405	def boot(
F8	src/apron/domain/mechanisms/model_spec_builder.py:28	def _is_attention_free
F9	tests/conformance/plugin.py:86	def valid_plan
F9	tests/unit/test_vllm_engine.py:18	from conformance.test_engine_adapter import
F9	tests/unit/test_runpod.py:18	from conformance.test_execution_target import
F9	tests/unit/test_deterministic_scorer.py:14	from conformance.test_evaluation_adapter import
F9	src/apron/adapters/backends/vllm_engine.py:601	def profiling_shape
F10	src/apron/application/orchestration/evidence.py:263	def store_validated
F11	src/apron/adapters/runner_image.py:12	RUNNER_IMAGE_DIGEST =
F11	src/apron/adapters/renderers/docker_compose.py:5	RUNNER_IMAGE
F12	src/apron/application/orchestration/plan_pipeline.py:62	self.observation = observation
L1	src/apron/adapters/backends/runpod.py:316	def _register_teardown_guard
L1	src/apron/adapters/backends/runpod.py:356	def cleanup_orphaned_pods
L1	src/apron/adapters/backends/runpod.py:39	TEARDOWN_BACKOFF_SECONDS =
L1	src/apron/adapters/backends/runpod.py:82	"NVIDIA B200"
L1	src/apron/adapters/backends/runpod.py:281	gpu_count=self._gpu_count
L1	src/apron/application/cost_estimator.py:31	def hourly_rate
L2	src/apron/application/orchestration/budget.py:131	def replay
L2	src/apron/application/orchestration/budget.py:246	def attribute_costs
L2	src/apron/adapters/backends/ledger_file.py:34	os.fsync
L3	src/apron/application/orchestration/evidence.py:127	class EvidenceContext
L3	src/apron/application/orchestration/cohort.py:167	def request_for
L4	src/apron/application/orchestration/cohort.py:280	def execute_solution
L4	src/apron/application/orchestration/cohort.py:92	def classify_harness_error
L4	src/apron/application/orchestration/cohort.py:678	def run_cohort
L4	src/apron/application/orchestration/scheduler.py:108	def rank_candidates
L4	src/apron/interfaces/cohort_root.py:140	class CohortPlanner
Serving	src/apron/adapters/backends/vllm_engine.py:441	def benchmark_serving
Serving	src/apron/adapters/backends/vllm_engine.py:485	def build_serving_report
§10	src/apron/application/orchestration/remediation.py:82	SIX_CLASSES: tuple
§10	src/apron/application/orchestration/remediation.py:226	def prove_fix
§10	src/apron/application/orchestration/remediation.py:188	def promoted_rule
§10	src/apron/application/orchestration/correction.py:362	def _retarget_memory
§10	src/apron/application/orchestration/correction.py:391	def _retarget_capability
§10	src/apron/application/orchestration/correction.py:440	STRATEGY_EXTRACTED_KEYS
§10	src/apron/adapters/backends/llm_classifier.py:47	def root_cause_window
§10	src/apron/domain/schemas/records.py:205	class DiagnosisRule
§10	src/apron/adapters/backends/rule_repository.py:27	def write_version
#30	src/apron/application/orchestration/evidence.py:222	def chat_template_kwargs
#30	src/apron/application/orchestration/plan_pipeline.py:154	def _download_chat_template
#31	.pre-commit-config.yaml:40	id: tests
#31	lychee.toml:11	exclude_path
#31	tests/conftest.py:22	def _no_test_writes_real_run_artifacts
```

## Test counts

- Base bd376ef (clean checkout): 774 collected — 723 passed, 51 skipped.
- Now: 1324 collected — 1262 passed, 62 skipped.
- Skips (exact, summed from `pytest -rs`): 45 need keys (39 test_level2_real_llm ANTHROPIC_API_KEY, 3 test_level3_gpu, 2 test_exit_gate, 1 test_fixture_run — RUNPOD_API_KEY); 12 need the pinned vLLM source (6 test_rule_strategy_alignment, 6 test_source_scanner); 5 opt-in GPU steps (test_cohort_run).
- With APRON_VLLM_SOURCE=/Users/vlad/repos/apron/.sources/vllm/vllm the 6 alignment source-line tests pass (30/30 in that module).

## Removed test IDs (declared; each replaced by a stronger assertion)

| Removed | Replacement | Why |
|---|---|---|
| test_diagnosis_six_classes[oom-kv-cache, max_model_len, dtype_incompatible, tp_divisibility, quant_compute_capability, lora_config] | [1-oom_weight_load, 2-oom_kv_cache-behind-wrapper, 3-max_model_len, 4-dtype_incompatible, 5-tp_divisibility, 6-quant_compute_capability, extra-lora_config] | ADR-005 six per PLAN §10.1 (old set lacked weight-load OOM, used a made-up dtype message); each param now asserts Gate A + Gate B + the exact corrected field; lora kept |
| test_correction::TestClampMaxModelLen::test_fallback_when_no_derived (asserted 4096) | test_falls_back_to_resolved_max_position_embeddings + test_no_evidence_is_infeasible_not_4096 | no guessed default (§10.3) |
| test_correction::TestFallbackDtype::test_low_cc_falls_to_float16 | test_low_cc_without_bf16_falls_to_float32 | float16 is the refused dtype; pre-SM80 has no bf16 |
| test_correction::TestFallbackDtype::test_quant_first_supported | test_supported_list_string_is_not_read | INV-6: free text no longer drives a fix |
| test_correction::TestReduceMemoryPressure::test_weight_oom_fallback_gpu_util (asserted 0.90) | test_no_extraction_is_infeasible_not_a_default | §10.3: 0.90 is vLLM's default; the fix boot would fail the same way |
| test_qualification::test_graph_state_records_reasons | test_graph_state_records_evidence_digests | F3 |
| test_qualification::test_task_failure_stops_at_identity_resolved | test_task_failure_below_quality_floor | F3 |

## prek run --all-files (actual tails, full tracked tree incl. new files, dependency files staged)

```
ruff check...............................................................Passed
ruff format..............................................................Passed
types....................................................................Passed
boundaries...............................................................Passed
identity.................................................................Passed
determinism..............................................................Passed
tests....................................................................Passed
supply-chain.............................................................Passed
docs.....................................................................Passed
Detect hardcoded secrets.................................................Passed
```
Root cause of the false "passed" in v2: prek lints tracked files only; the new files were untracked until `git add -N`. no-agent-branding runs at commit-msg.

## Plan deviations for the owner (stop point 1)

- a. tp_divisibility.json does not declare `gpu_count` (§10.3 lists it). config/model.py:1414-1418 never prints it; `_reduce_tensor_parallel` reads it from the plan's `resource_allocation` (the requested execution).
- b. dtype class cites config/model.py:2262 ("The model type {model_type!r} does not support float16"), the line that prints the fields, instead of the table at :2243-2260.
- c. (fixed, no longer a deviation) `_retarget_capability` returns no correction when the rule has no kernel architectures for the method.
