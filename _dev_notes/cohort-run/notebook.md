# Cohort run notebook (PLAN §12)

Timestamped, facts only: expected vs happened, surprises, availability
waits, retries, dead ends, fixes that failed.  Times are local (UTC-6)
unless marked.  Machine events are in `events.jsonl`.

## 2026-09-26 — L0-A (pod safety net)

- **15:0x — attempt 1, RTX A5000.** Expected: pod, SIGKILL, orphan cleanup.
  Happened: `create_pod` refused, "no longer any instances available". No
  pod, $0.  Surprise: the GPU price listing showed the A5000 as available;
  the listing is not a capacity signal.
- **15:15 — attempt 2, RTX 4090** (pod `ljf4phxhq229f1`, ~5 min).  SSH came up;
  hardware detection failed with CUDA error 804 "forward compatibility was
  attempted on non supported HW".  The image is CUDA 13.0 with
  `NVIDIA_DISABLE_REQUIRE`, so it started on a host whose driver is older than
  CUDA 13; forward compatibility does not work on GeForce.  The process's
  atexit guard terminated the pod (safety net worked on the error path).
  Fix: `create_pod(allowed_cuda_versions=["13.0"])` (commit 36caaba,
  "place pods only on hosts whose driver supports the image's CUDA").
- **~15:3x — attempt 3, RTX 4090** (pod `1whhoofog4ig7t`, ~7 min).  Pod came
  up; process SIGKILLed as designed; **orphan cleanup terminated nothing** and
  the pod stayed billing.  Cause: `runpod.get_pods()` returns `uptimeSeconds`
  at the top level, the cleanup read `runtime.uptimeInSeconds` (never in that
  query), so every pod looked 0 s old.  The unit test's fake used the assumed
  shape.  Pod terminated by hand; account verified at 0 pods.  Fix: read the
  real field; the fake now mirrors the SDK query; a test pins the field
  against `runpod/api/queries/pods.py`.
- **attempts 4-5.** With the CUDA 13.0 requirement, no Secure capacity for
  4090, L4, RTX 3090, A6000, A5000, L40 (all refused at creation, $0).
- **Capacity check (read-only `lowestPrice.stockStatus`, Secure, 1 GPU):**
  4090 "Low" on CUDA 13.0 and none on 12.9/12.8; H100 "Medium" on 13.0;
  B200 "Low" only with no CUDA filter, none on 13.0.  Considered rebuilding
  the image on vLLM v0.29.0 `+cu129`; rejected — 12.9 hosts show no 4090
  stock.  Open: the class 6 fixed boot on B200 cannot use the 13.0 filter as
  is; decide with evidence at class 6.
- Spend so far ≈ $0.15 (two 4090 pods, ~12 min total).
- **15:37 — attempt 6, RTX 4090 (pod `pcx46p9oe2n3cz`, ~9 min): L0-A PASSED.**
  Pod up, process SIGKILLed, next start's orphan cleanup terminated it,
  account back to 0 pods (`l0a-kill-test.json`).  Measured: pod created
  15:37:33, SSH up 15:46:25 — ~9 min of image pull on a fresh host.
- Estimate change (deviation from H1's formula): the H1 estimate had no image
  pull term, so an L0-A3 boot was held at $0.19 against a realistic ~$0.30 —
  every run would have tripped the 1.5x overrun stop on an estimate error.
  `estimate_cost` now adds the measured 9 min pull.
- Spend so far ≈ $0.26.

## 2026-09-26 — L0-A3 (measurement stability)

- **~16:00 — run 1 boot failed (4090, ~11 min, $0.133 local clock / $0.027
  RunPod-reported).**  Expected: healthy Qwen3-1.7B.  Happened: EngineCore
  died in FlashInfer's first JIT build, `FileNotFoundError: 'ninja'`, surfaced
  as "Engine core initialization failed".  Cause: the new SSH-driven boot
  (F7) runs `vllm serve` in sshd's default environment; start.sh exports the
  container's PATH/CUDA_HOME to /etc/apron_environment but only ~/.bashrc
  sources it, which a non-interactive SSH command never reads.  Phase 1a
  booted from start.sh and inherited the container env, so this was new.
  Fix: serve and bench commands source /etc/apron_environment first (then
  `env -u` the tokens).  Stored as a failed boot record — a harness defect,
  not a model failure; it is not diagnosis material.
- Note: local-clock cost and RunPod-reported cost differ ~5x on a short pod
  (RunPod appears to count from container start, after the image pull).  Both
  are recorded (M3); the ledger settles on the local clock (conservative).
- Spend so far ≈ $0.39 (local clock).
- **Incident (~17:10): prek ran the suite with keys loaded.**  I ran
  `prek run --all-files` in a shell where `.env` was sourced; the `tests`
  hook then ran the key-gated integration tests (`test_fixture_run`,
  `test_level2_real_llm`, `test_level3_gpu`) live for ~10 min before I killed
  it.  After: 0 pods on the account, `currentSpendPerHr` 0, no leaked-pod
  log, no new ledger or record entries.  Likely spend: a few Anthropic Haiku
  calls (cents); not provable from here.  Fix: the `tests` hook now runs
  `env -u RUNPOD_API_KEY -u ANTHROPIC_API_KEY -u HF_TOKEN`, verified by a
  full prek run with the keys loaded.
- RunPod balance read at the same time: $10.07 — enough for the approved L0
  (≈$7.49), not for the full cohort (plan ≈$55–70): top-up needed before L5.

## 2026-09-26 — L0-F (failure reproduction)

- **16:16 — class 6 on H100 ($0.61): failed, NOT as named.**  Expected the
  capability error at `config/vllm.py:791`.  Happened: `ValueError` at
  `config/model.py:1334`, "The quantization method fp_quant is deprecated …
  set `--allow-deprecated-quantization`".  The deprecation check runs before
  the capability check.  fp_quant is the only v0.29.0 method with
  `get_min_capability() == 100` (`fp_quant.py:64`), so no other method can
  stand in.  Replacement broken plan (§6.1): same checkpoint on H100 with
  `allow_deprecated_quantization: true` in the plan, which only warns and
  reaches the capability check.  Found on the way: every boolean in
  `engine_configuration` was rendered `--flag true`, which vLLM's
  BooleanOptionalAction parser rejects; booleans now render `--flag` /
  `--no-flag`.  The failed record stays as evidence.  The class 6 broken boot
  must be re-run with the replacement plan.
- **L0-A3 (in flight): the pod-age fix of L0-A was wrong too.**  For a pod up
  810 s, `get_pods()` reported `uptimeSeconds: 0`; only the single-pod query
  has a real age (`lastStartedAt`, `runtime.uptimeInSeconds`).  The L0-A pass
  did not catch it because its cleanup uses age 0.  The start-of-run cleanup
  (age 3600) would have terminated nothing.  Ages now come from
  `lastStartedAt` (covers a pod still pulling its image); verified live: the
  in-flight L0-A3 pod read 1644 s.
- **L0-A3 run 1, second attempt (22:27-22:56 UTC, $0.36 local / $0.20
  RunPod-rate): health deadline passed while the engine was still starting.**
  Pod up 22:40 (13 min pull on this host), weights + profile + CUDA graphs
  done 22:51:38, then silence until the 900 s deadline at 22:56 — the sampler
  warm-up JIT-compiles FlashInfer on a cold pod (the earlier ninja failure
  showed that path).  Checked FlashInfer 0.6.18: it compiles only for the
  detected GPU unless FLASHINFER_CUDA_ARCH_LIST is set, so the image's
  5-arch TORCH_CUDA_ARCH_LIST is not the cause.  Fixes: boot deadline 1800 s
  (cold boot); the stored log folds the 10-second "waiting for core engine"
  heartbeat that had filled the whole 8000-char tail; estimates use the
  measured pull (13 min) and cold-boot times (+10 min per size class).
- **Class 1 (Qwen3-14B BF16, RTX 4090, $0.11): weight-load OOM as expected,
  but not at the named call site.**  vLLM v0.29.0 loads through the v2 model
  runner (`v1/worker/gpu/model_runner.py:384`), which has no OOM wrapper; the
  error is PyTorch's own `torch.OutOfMemoryError` from the `torch.empty`
  weight allocation at `model_executor/layers/linear.py:192`.  The plan's
  "Failed to load model - not enough GPU memory" handler
  (`v1/worker/gpu_model_runner.py:5460`) is in the legacy runner, not on this
  path.  The failure class is right; the rule and the broken case now cite
  the real path (pinned source checked).  No replacement plan needed: any
  weight-load OOM takes this path.
- **Class 2 (Qwen3-8B, max_model_len 40960, RTX 4090): failed as named.**
- **Class 3 (Mistral-7B, max_model_len 999999, RTX 4090, $0.02): failed as
  named.**  This pod's host had the image cached: SSH up in 28 s.
- **Class 5 capacity (23:30 UTC).**  No 3- or 4-GPU Secure stock on CUDA 13
  hosts for any consumer/pro GPU; only A100-SXM (x3, x4 "Low") and H100.
  Owner approved A100 or H100 for class 5; the broken case now uses
  4x A100-SXM ($6.36/h).  The TP divisibility check does not depend on the
  GPU model.  The earlier batch (classes 1,2,3,5) was stopped while waiting
  for 4x4090 stock (no pod, no open hold); class 5 was re-queued last.
- **Class 4 (gemma-2-2b-it float16, real Gemma, 4090, $0.07): failed as named.**
- **Class 6 re-run (H100, with allow_deprecated_quantization, $0.31): failed as
  named** at the capability check.
- **L0-A3 run 1, third attempt (23:36-23:56 UTC): hung — my bug.**  Owner
  asked whether the pod was working; I checked inside it over SSH: vLLM was
  healthy (/health 200, 21.5 GiB in use) but had received no request.  The
  launch `. env && cd && env … nohup vllm … &` backgrounds the whole list; the
  subshell (not exec'd since the env-file change) kept the SSH channel's
  output open, so the launch command never returned and the harness never
  polled health.  paramiko's recv_exit_status ignores the channel timeout, so
  nothing bounded the wait.  Stopped with SIGINT: `finally` tore the pod
  down; the next step's ledger replay settled the open hold at the estimate
  ($0.48, flagged `replayed:estimate`).  Fixes: the launch redirects the
  whole group and execs vLLM (a pipe test reproduces the hang: old 8.0 s vs
  new 0.0 s); `execute()` enforces its own deadline and raises
  RemoteCommandTimeout (never retried); downloads get a longer one.
- **Class 5 (Qwen3-8B TP 3 on 4x A100-SXM, $0.52): failed as named.**
- **Verification pass over all six L0-F logs (owner asked for evidence, not
  assumptions):** each failure checked against where vLLM actually raised it
  in the pinned source, not only by text match — class 1
  `linear.py:192` under `gpu/model_runner.py:384` (frames in the log);
  class 2 `kv_cache_utils.py:879` (innermost engine frame); class 3 message
  template `config/model.py:2501-2504`; class 4 `config/model.py:2262`;
  class 5 `raise` at `config/model.py:1414`; class 6 `raise` at
  `config/vllm.py:791`.
- **L0-A3 PASSED (00:02-00:18 UTC, pods `jaw52t8omxeh6w` and
  `fv58mbh1ieedqv`, $0.10 each).**  Watched from inside the pod every 2 min.
  Both runs: healthy, 3/3 task cases accepted, serving 50/50 completed,
  p99 TTFT ~51 ms / TPOT 5.4 ms (SLO pass), F7 check token-free.  The two
  memory reports are distinct records from different hosts (different
  execution fingerprints) with byte-identical weights, KV cache, activation
  and persistent memory: 0% spread, no uncertainty note needed.  Limit: two
  boots on one GPU type.  First real prediction deltas: weights +0.56 GiB,
  total +1.54 GiB (calculator over-predicts Qwen3-1.7B).
- L0-A age-path proof, attempt 1: RunPod API "Something went wrong" at
  creation, no pod, $0; retried.
- **L0-A age path PASSED (00:18-00:27 UTC, pod `bdjkw5nbsj625z`, ~$0.11).**
  Process SIGKILLed; the pod was 521 s old; `cleanup_orphaned_pods(max_age
  300)` terminated it (age from `lastStartedAt`); account at 0 pods.  L0 done.
- **Owner feedback: RunPod showed mostly pods being re-created.**  True: a
  fresh pod per run spent 5-13 min pulling the image for seconds to minutes of
  work.  Built pod reuse (`TargetPool`): one live pod per requested execution,
  hygiene between boots, other models' weights evicted, a pod dropped after
  any harness failure, a parked pod holds budget (`pod-idle:` hold, replayed
  after a crash).  First synthetic run exposed a parked pod billing while
  other GPU types ran; fixed: one live pod at a time and the chosen solutions
  run grouped by execution.  Synthetic cohort + fix proofs: 18 executions on
  10 pods.  Not yet verified on RunPod.

## 2026-09-27 — before L5

- Owner topped up RunPod: balance $37.37.  Llama access now granted.
- Capacity waiting built into `run_cohort` (stock poll before a new pod,
  refusals retried, never counted as failures).  Checked live (read-only):
  1x4090 stock yes, 4x4090 no, 4xA100-SXM yes.
- **#32 done.**  The production classifier (Haiku 4.5) run on each real
  L0-F log tail exactly as the fix proof reads it: all six diagnosed as their
  class; typed extractions match the log (class 1's byte counts within 0.2%,
  evidence-only fields).  $0.064, recorded in the ledger.  Replayed in CI
  through the real pipeline: 1 → RTX A6000 (predicted 31.9 GiB), 2 →
  max_model_len 32640, 3 → 32768, 4 → bfloat16, 5 → TP 2, 6 → B200.
- **Stock map by host CUDA version (read-only, 2026-09-27).**  RunPod's
  `allowedCudaVersions` matches the host's version exactly, and the API
  accepts values past its documented list: B200 hosts run CUDA 13.2, so the
  "13.0" filter had excluded every B200 (and some 4090) host.  Filter now
  13.0-13.4.  RTX A6000 exists only on CUDA 12.8 hosts (cannot run the CUDA
  13 image); L4, A5000, L40: no Secure stock; 4x4090 none; 4x A100-SXM/H100
  and B200 x1/x4 yes.
- Class 1's retarget now chooses from GPUs with image-compatible stock at
  diagnosis time (availability removes an option): with the A6000 out, the
  next cheapest fitting GPU.  Live catalog x1 now: 4090, A100-SXM, H100, B200.

## 2026-09-27 — L5 cohort

- **4090 pod `17xicayf2q4xfm`, reused for 4 solutions:** Qwen3-8B-GPTQ
  ($0.118, paid the image pull), gemma-2-2b ($0.025), Llama-3.1-8B ($0.030),
  Mistral-7B ($0.031) — all healthy, 3 task attempts and serving each.
  Qwen3-32B@4090 stored as predicted infeasible; Qwen3-1.7B@4090 skipped
  (measured in L0-A3).  Verified inside the pod: 3 chat requests + 55
  benchmark requests per model, all 200.
- **A100 PCIe pod `tbwmztvyaiaej2`:** DeepSeek-V2-Lite (MLA/MoE) healthy
  ($0.25).  Qwen3-32B failed in the tokenizer constructor ("vocab and merges
  must be both from memory or both filenames").
- **L4 pod `uctmsoz02bawey`** (L4 had stock this time): Qwen3-1.7B ($0.085)
  and Mistral-7B ($0.036) healthy.
- **H100 pod `fy7vvlu4f5l5l5`:** Qwen3-32B, same tokenizer error, 64 s
  after SSH on a fresh pod.
- **My wrong turns, recorded:** I first blamed the reused A100 pod, then
  misread the L4 pod as the H100 and claimed the "experiment" had answered
  it.  Both wrong; corrected to the owner from the events.
- **Root cause (looked, not guessed):** Qwen3-32B's tokenizer files are
  byte-identical to Qwen3-1.7B's.  Inside the class 1 fix pod: `/workspace`
  is the 50 GB container disk — the SDK mounts the requested 100 GB volume at
  `/runpod-volume` by default — and the download command ended in `| tail
  -20`, so its exit status was tail's: a failed download always looked
  successful.  The 65.5 GB model filled the disk, the partial directory had
  no vocab/merges, vLLM failed in the tokenizer.  Fixes: exit status kept +
  every repo file verified at its size (DOWNLOAD_INCOMPLETE is a harness
  failure); weights on the volume (`/runpod-volume/models`), mount explicit.
  Both Qwen3-32B records are harness failures mislabelled as model failures;
  the two rows are re-run after the fix proofs.
- Fix proofs started 01:52 UTC.  Class 1 retargeted to RTX A6000 (had CUDA
  13.x stock at diagnosis time); download verified (28 GB).
- **Class 1 PROVEN (02:05 UTC):** broken Qwen3-14B BF16 on RTX 4090 (L0-F
  log) → real classifier (Haiku 4.5, $0.011) → oom_weight_load →
  retarget_memory → RTX A6000 → fixed boot healthy, 3/3 tasks, serving SLO
  pass → mechanism verified + request satisfied = **Fixed**; rule
  `oom_weight_load` promoted to v2 mechanism_verified citing the boot
  (previous version in `rules/vllm-v0.29/history/`).  Pod $0.106.
- The fix proofs then stopped: class 2's fixed pod was refused ("no longer
  any instances") and `prove_fix` had no capacity waiting — a gap in my prep
  (I had added it to the cohort loop only).  Fixed: one helper
  (`execute_when_available`) for both; no capacity within the wait records the
  proof as not evaluated, never failed, and promotes nothing.
- **Class 2 PROVEN (02:20 UTC):** Qwen3-8B, max_model_len 40960 → 32640 (the
  value vLLM printed), RTX 4090 → fixed boot healthy, tasks + SLO pass →
  Fixed; rule `oom_kv_cache` promoted.  $0.15.
- **Class 3: fixed boot never happened.**  My new download check flagged
  Mistral's `tokenizer.model` (listed 130 bytes, a git link, written as the
  587 KB target) twice → harness:download_incomplete.  Worse, the proof then
  recorded the mechanism as "failed" — false: a harness failure is no
  evidence about the correction.  Fixes: the check flags only missing or
  truncated files; a fixed boot that never reached the engine is recorded
  "not_evaluated".  The false class 3 record stays in the evidence; class 3
  is re-run.
- **Class 4 PROVEN (02:23 UTC):** gemma-2-2b-it float16 → bfloat16 on the
  RTX 4090 → Fixed; rule `dtype_incompatible` promoted.  $0.025.
- **Class 5 PROVEN (02:31 UTC):** Qwen3-8B TP 3 on 4× A100-SXM → TP 2 → Fixed;
  rule `tp_divisibility` promoted.  $0.85 (4 GPUs).
- **Class 6: the retarget removed the capability error, then weights failed
  to load (02:43 UTC).**  fp_quant model on a B200 (the only catalog GPU the
  QuTLASS kernels are built for): "no module or parameter named
  'layers.0.mlp.down_proj.backward_hadamard_matrix' in Qwen3Model".  Pinned
  v0.29.0 `fp_quant.py:198-209` registers `forward_hadamard_matrix` only.
  Afterwards I read the safetensors headers of all 32 FPQuant checkpoints on
  the Hub: every one stores `backward_hadamard_matrix`, so no GPU makes them
  load in v0.29.0.  No public report of this found (web search, 2026-09-27).
  $1.39 — avoidable: the header check is free and should have preceded the
  B200 boot (the L0-F failure stopped at the capability check, before
  weights, so it hid this).
- **Qwen3-32B re-runs, after the volume + download fixes:** A100 PCIe healthy,
  served the benchmark (02:55 UTC, $0.30); H100 healthy (03:11 UTC, $0.96).
- **Class 3 PROVEN on re-run (03:24 UTC):** Mistral-7B max_model_len 999999 →
  32768 on the RTX 4090 → Fixed; rule `max_model_len` promoted.  $0.14.
  `fix-proofs.json` now merges re-runs by class instead of overwriting.
- Exit gate over the real records: 8 of 9 items pass; item 2 is short class 6.
- **Class 6 decision (owner, 2026-09-27):** the §10.1 fallback ("artifact
  substitution to the declared base model") had a gap: the checkpoint's card
  declares no base model (empty template; `_name_or_path` empty — true of all
  32 FPQuant repos).  Owner: the agent proposes the base, code confirms it;
  prefer a quantized sibling the GPU runs over the full-size original.
  Implemented as rule v2 strategy `substitute_artifact` (v1 in history):
  classifier proposes the base (Haiku 4.5, $0.0017) → Hub confirms it exists,
  is unquantized, same architecture → the Hub's model tree lists 455
  checkpoints marked *quantized* from `Qwen/Qwen3-0.6B` (fine-tunes are a
  separate relation) → 432 in formats v0.29.0 cannot load (GGUF, MLX, and
  bitsandbytes, which v0.29.0 dropped although the latest docs list it) → 20
  candidates with identical architecture → objective: base publisher first,
  closest weight precision, downloads → `Qwen/Qwen3-0.6B-FP8` on the same
  H100.  Caveat: the Hub's "quantized" relation is publisher-declared and
  sometimes wrong (`buttercoconut/Qwen3-ko-alpaca-0.6B-Q4` is a fine-tune);
  the publisher preference keeps those out here, the architecture check
  cannot.  Label per §10.2: Fixed if the request is satisfied (the request
  does not pin the artifact); the record shows the model change.
- **Class 6 re-run with substitute_artifact (04:00-04:12 UTC):** real
  classifier on the stored H100 log → quant_compute_capability → live lineage
  search → `Qwen/Qwen3-0.6B-FP8` on the same H100.  Weights loaded (2.1 GB, as
  predicted), healthy, served the benchmark.  Tasks 1/3: "2 + 2" → `2`,
  "10 * 5" → `10 * 5 = 5`, capital of France → `Paris`; below the 0.60 floor.
  Checked the raw outputs: real answers, no thinking text, no parse fault.
  Mechanism verified + request violated = **Alternative with trade-offs**;
  rule `quant_compute_capability` promoted to v3 mechanism_verified.  $0.72.
  Not isolated: whether FP8 or the 0.6B size (or raw completions without the
  chat template, used for every model) causes the misses; the requested
  checkpoint never ran, so there is no like-for-like baseline.
- **Exit gate over the real records: 9 of 9 items pass** (item 6 waits on
  Part 3, as planned).  Ledger total $9.01 (runs $8.82, classifier $0.16,
  idle pods $0.03), plus ≈$0.37 of unledgered L0-A kill tests.

## 2026-09-27 — review of the run: gaps found and fixed (owner: "we fix them")

- **Correction to my own report:** I said tasks are sent as plain completions
  without the chat template.  Wrong: `/v1/chat/completions` with the template
  (`deterministic_scorer.py:148`); the `/v1/completions` lines in pod logs are
  the serving benchmark.  DeepSeek-V2-Lite's 0/3 (" 4\n\nUser: What is") is
  a **base** model in a chat task (the seed picked the base, not `-Chat`);
  Llama's "Paris." failed only on the full stop (protocol normalizes
  whitespace only).  Qwen3-0.6B-FP8's misses are wrong answers.
- **Calculator (fixed, aeae42e):** tied `lm_head` counted twice (Qwen3-1.7B
  +17%); without an index only the first dtype counted (Qwen3-0.6B-FP8
  -22%: its BF16 embeddings, not its FP8 layers); whole-model bytes against
  vLLM's per-rank figure (TP 2: +100%); the TP choice never saw the heads
  (always TP 1).  After: all 14 measured points within 1.5%, FP8 5.4%.
- **Protocol fields (fixed, 553981d):** `sampling_top_p` and `stopping_rules`
  were fingerprinted but never sent; `repetitions`/`concurrency` > 1 ran once
  silently.  Now sent, or refused.  The chat template digest is recorded.
- **Substitution safety (fixed, e772fbc):** the Hub's "quantized" relation is
  publisher-declared: `buttercoconut/Qwen3-ko-alpaca-0.6B-Q4` is a fine-tune.
  Final-norm weights are unquantized in every format: 19 of 20 candidates
  match the base bit for bit, the fine-tune does not and is dropped.  The
  requested FPQuant checkpoint matches too, confirming the proposed base.
  (`GaborMadarasz/…gptq_hungarian_news` matches: its name is the GPTQ
  calibration set, not a fine-tune — my earlier guess was wrong.)
- **Spend (fixed, 900c34e):** RunPod's bill for the window shows five pods the
  ledger never tracked, $0.65: the four L0-A kill tests and
  `oxlwqogyhgfc2m` (09-26, ~9.5 min at A100-PCIe pricing, 150 GB disk) which
  no run artifact, transcript or local session names — unattributed, now in
  the ledger.  The L0-A3 hung attempt was settled at a 2x estimate ($0.48 vs
  $0.24 billed) because a crash replay could not read a terminated pod's
  cost; terminated pods are now costed from the billing API.  Before this
  cohort (09-21..23) the account shows $17.15 of earlier work, not ours.
- **fp_quant root cause:** vllm-project/vllm#44122 ("[Refactor] Remove dead
  code fp quant", 2026-06-03, first in v0.23.0) removed
  `backward_hadamard_matrix`; v0.11.1 registered it; `main` (2026-09-27)
  still lacks it.  Draft issue for the owner:
  `_dev_notes/cohort-run/upstream-vllm-fp-quant-issue.md`.

## 2026-09-27 — pass 2 and the Phase 2 items (owner: "do it and the rest")

- **DeepSeek-V2-Lite-Chat (05:12-05:24 UTC, A100 PCIe, $0.33):** healthy,
  3/3 (" Paris", " 4", " 50").  The base model's 0/3 was a base model on a
  chat task.  A6000 would be cheaper but its hosts run CUDA 12.8 only.
- **Re-scoring, no GPU:** the Phase 1b rule (a trailing full stop ignored)
  applied to all 51 recorded answers of 16 solutions; each re-score is a new
  attempt, reason task_rescore, linked to its source.  Llama 2/3 → 3/3.
  Found on the way: resume and the findings indexed attempts by solution
  only (a new rule would have been skipped or pooled), and the gate checked
  attempts against a solution's first protocol.  Both fixed.
- **Load check (item 11):** tensor names from the safetensors headers against
  each quantized method's registered parameters (create_weights, pinned).
  Flags the FPQuant checkpoint (196 tensors), passes every FP8/GPTQ/AWQ/
  compressed-tensors checkpoint that loaded.  Would have saved the $1.39.
- **Activation (item 9):** measured peak ≠ 10% of weights.  It tracks the
  output projection (vocab × hidden × dtype / TP) + 2 × profiled tokens ×
  hidden × dtype (tokens = vLLM's max_num_batched_tokens default: 2048, 8192
  on ≥70 GiB non-A100).  14 single-GPU points within 0.02 GiB.  Mechanism
  not traced; the TP 2 point (0.21 measured, 0.61 predicted) does not fit.
- **Mamba (item 10):** ssm_decode from vLLM's state shapes (conv
  I×(k−1) + SSM I×state per layer and sequence).  Found on the way: vLLM
  serves a float32 checkpoint in 16 bits, so weights were predicted 2× high
  for fp32 checkpoints (Mamba-2.8B: 11.07 GB stored → 5.16 GiB predicted).
  Not yet checked against a boot.
- **Pending the owner's go (no GPU without the owner, 2026-09-27):** one
  RTX 4090 pod for Qwen3-0.6B + Qwen3-0.6B-FP8 (class 6: FP8 or size?) and
  Mamba-2.8B (validates ssm_decode); est. ~$0.5-0.8 pooled.
- Draft article PR: mondegreens/apron#43 (stacked on #42).

## 2026-09-27 — pass 3: one RTX 4090 for Qwen3-0.6B, its FP8 copy, Mamba-2.8B

- Owner's go.  Pod `idy00eidbjrj64`, three solutions on one pod, $0.17
  total (FP8 $0.12, Mamba $0.03, base $0.02; the image pull is on the first).
- **Qwen3-0.6B-FP8 and Qwen3-0.6B answer identically** on the 4090: "2",
  "10 * 5 = 5", "Paris" — also identical to the FP8 run on the H100.  FP8 did
  not cause class 6's misses.
- **Independent reference, no GPU:** Qwen3-0.6B in transformers on CPU,
  greedy, the same rendered prompt (chat template checked by rendering it:
  non-thinking format, `enable_thinking=False` applied).  Same answers at 8
  tokens.  At 64 tokens: "10 * 5 = 50" (10 tokens — Qwen writes one digit per
  token) and "2" followed by end-of-turn.  So "10 * 5 = 5" was **our harness
  cutting a correct answer** at max_tokens 8, recorded as a plain wrong
  answer; "2 + 2 = 2" is the model's own non-thinking greedy answer.
- **Mamba-2.8B:** healthy, served the benchmark.  Weights measured 10.31 GiB
  against 5.16 predicted: our plan named float32 (from the config) while the
  calculator assumed vLLM's 16-bit default — the planner and the calculator
  disagreed.  Activation 0.52 GiB measured, 0.52 with the float32 plan.
  The per-sequence SSM state is not observable in what we record: not
  validated.  All chat task requests: HTTP 400 — a base model without a chat
  template; the record kept only "400".
- Fixed: the planner now names vLLM's served dtype (float32 checkpoints in 16
  bits); a model without a chat template is not sent the chat suite (reason
  recorded per case); an engine refusal keeps its reason; a cut answer
  (finish_reason "length") is marked "truncated at max_tokens".  Not changed:
  the suite's max_tokens (a task-suite change re-runs every model's tasks —
  owner's call).

## 2026-09-27 — attribution of pod `oxlwqogyhgfc2m`, and a correction

- RunPod's hourly billing (grouped by GPU type) puts it at 2026-09-26
  22:00-23:00 UTC, NVIDIA A100 80GB PCIe, 568 s, $0.2569.  No cohort event
  names an A100 in that hour.  At that time I ran the local hook with the
  API keys loaded; the key-gated integration tests ran live and I stopped
  them (fixed in ec0e0d8, 22:16 UTC: the hook now strips the keys).  One of
  them, `tests/integration/test_fixture_run.py` (Phase 1a: "spends real money
  (~$0.50)"), rents the cheapest available GPU that fits; with no 4090 stock
  on CUDA 13 it would take an A100 PCIe.  **Correction:** I reported that
  incident as "0 pods".  It was one pod, $0.26 — already in the ledger as
  `reconcile:unledgered:oxlwqogyhgfc2m`.
- Ledger corrected to RunPod's bill: 26 signed `correct` entries, -$0.10
  net (the L0-A3 crash replay settled at a $0.48 estimate, billed $0.24;
  the rest are sub-cent clock differences).  Ledger pods = bill = $9.9041.

## 2026-09-27 — pass 4: Mamba-2.8B in 16 bits (owner: "confirm it with a run")

- Pod on an RTX 4090, $0.10.  The planner now names bfloat16 for the float32
  checkpoint (vLLM's own default).  Predicted vs measured: weights 5.16 vs
  5.23 GiB (-1.3%); activation 0.259 vs 0.260; state per sequence 11.875 vs
  11.881 MiB (+0.05%), measured as available KV memory 14.84 GiB / maximum
  concurrency 1279 (the report now records vLLM's pool line).  The Mamba-1
  state model holds; a test keeps it against the record.
- Tasks: not sent — the base model has no chat template; recorded per case.
- Also traced this session (read-only agents): the activation peak is
  TorchInductor's combo-kernel benchmarking on a cold compile (an embedding-
  sized random tensor); with TP > 1 the embedding is a custom op and the peak
  is the sampler's logits.  All 16 points within 0.02 GiB.  And the FPQuant
  failure: vLLM regression #44122; vLLM's own test used our checkpoint but
  never ran (no test_ prefix); our invocation does not affect loading.

## 2026-09-27 — class 6 on Qwen3-8B (owner's go), and a load-check error

- **Why re-run class 6.**  The exit gate now follows PLAN §10.2 to the letter
  (a class ending `violated` does not satisfy item 4).  On Qwen3-0.6B no
  substitute could meet the 0.60 task floor: every 0.6B variant answers
  "2 + 2" with "2", in vLLM and in transformers on CPU (pass 3).  Owner's
  decision: the broken plan uses `ISTA-DASLab/Qwen3-8B-FPQuant-RTN-MXFP4` on
  the same H100 with `allow_deprecated_quantization`.  The 0.6B record stays
  in the evidence.
- **Broken boot (H100, pod `mp8toichl9aavu`, $0.76 with the image pull):**
  failed as named, capability check `config/vllm.py:791`.
- **Correction:** classifier (Haiku 4.5, $0.0017) → quant_compute_capability;
  proposed base `Qwen/Qwen3-8B`, confirmed on the Hub.  89 listed checkpoints
  had a format v0.29.0 loads; dropped: two speculative-decoding draft models
  listed as "quantized from" Qwen3-8B (another architecture), fine-tunes by
  final-norm weights.  Chosen `Qwen/Qwen3-8B-AWQ` (base publisher, 4 bits like
  the request) on the same H100.
- **Fixed boot:** healthy; weights 5.68 predicted / 5.82 GiB measured;
  tasks 3/3 ("4", "50", "Paris"); serving SLO pass → mechanism verified +
  request satisfied = **Fixed**; rule `quant_compute_capability` v4 promoted
  citing the boot.  $0.19, plus $0.20 of pod idle between the two boots.
- **Engine facts generated from source** (owner agreed): the minimum
  capabilities and registered parameter names now come from the pinned vLLM
  by `scripts/generate_vllm_facts.py`; a test ties them to the image's vLLM
  pin.  Generating replaced one hand-written error (modelopt FP8 params).
- **Found while writing the article: the load check dropped a loadable
  candidate.**  `nvidia/Qwen3-8B-NVFP4` was dropped as "would not load:
  k_scale x36, v_scale x36" — wrong twice: (1) a `modelopt` checkpoint with
  quant_algo NVFP4 is served by ModelOptNvFp4Config (min capability 75,
  its own linear method), not the FP8 config (`modelopt.py:1063-1070`);
  (2) k/v_scale belong to the attention layer's KV-cache method
  (`kv_cache.py:57`), mapped from checkpoint names by
  `base_config.py:195`.  Fixed from source (generated), with tests; the
  lineage search re-recorded (Haiku $0.0017): 64 candidates, NVFP4 now one of
  them, the choice unchanged (AWQ first by publisher).  Still not recognised
  (reported as such, never as "will not load"): compressed-tensors NVFP4,
  ModelOpt MXFP8 / mixed precision / W4A16.
- **Billing at 18:40 UTC:** RunPod has not billed pods `7l0uecp1u30fhb`
  (pass 4) and `mp8toichl9aavu` (class 6) yet; ledger pods $11.16 vs billed
  $9.90 — the difference is these two pods.  Re-run the reconciliation once
  they post.

## 2026-09-27 — scope grows: modern models, $500 cap, staged weights

- **Owner decisions:** Phase 1b continues with modern models in four groups
  (A single GPU: gpt-oss-20b/120b, GLM-4.7-Flash, Gemma-4-31B; B hybrid
  linear attention: Qwen3.6/3.8, Nemotron-3; C 2-4 GPUs: MiniMax-M2.7,
  DeepSeek-V4-Flash; D 8 GPUs: GLM-5.3, DeepSeek-V3.2, Kimi K2).  The cap
  goes from $100 to **$500**, with a stop and report after groups A-C.
  First: staged weights, checked on one model.
- **Why staged weights:** the phase plan's GPU dollar protection rule 1
  ("never download on GPU-billed time") was dropped for 1b (the SDK had no
  volume call; volumes lock to a datacenter).  Fine for 1-30 GB models;
  700 GB-1 TB on 8 GPUs at $37-54/h is not.
- **Found (docs and live read-only API, 2026-09-27):** network volumes are
  created over REST (`/v1/networkvolumes`); CPU pods accept a network volume;
  `dataCenters.storageSupport` and `lowestPrice(dataCenterId)` tell where.
  Volume reads are 200-400 MB/s (10 GB/s peak): loading 700 GB lazily would
  keep 8 GPUs waiting ~40 min, so vLLM reads staged files ahead
  (`--safetensors-load-strategy prefetch`, pinned `weight_utils.py:871`) when
  the checkpoint fits host memory.  CUDA-13 stock in storage datacenters is
  thin today (H100 x1 in EU-FR-1, EUR-NO-2, US-NE-1; B200 x1 in EU-RO-1,
  US-CA-2; no x8 anywhere).
- **Built:** `runpod_storage.py` (volumes, datacenter choice, CPU stager pod),
  GPU pods attach the volume at the same path, staged weights never evicted,
  `staging.py` (provider-neutral, paid through the budget), records say
  where weights came from and split pod time into `weights` and
  `engine_start`, storage accrued to the ledger, `run_cohort(repeat=True)`.

## 2026-09-27 — first staged runs (one model, owner's go), paused by the owner

- **Try 1 (19:27 UTC):** volume created in AP-JP-1 (the first storage
  datacenter alphabetically with H100 stock); the CPU stager was refused
  ("no longer any instances") although the datacenter listed CPU stock.
  No GPU pod.  A one-minute probe showed the refusal was the 8-vCPU size:
  4 vCPUs were given ($0.002, terminated).  Fixed: the stager steps down
  through the datacenter's stocked flavors at 8/4/2 vCPUs, and a weights
  site needs GPU and CPU stock.  The empty JP volume was deleted ($0.0007
  of storage accrued).
- **Try 2:** no storage datacenter had H100 x1 (CUDA 13) together with CPU
  stock.  Where both existed: EUR-IS-1 (A100-SXM4-80GB), US-CA-2 (H200,
  B200), EU-RO-1 (RTX PRO 6000).
- **Try 3 (19:33 UTC):** Qwen3-32B on A100-SXM4-80GB in EUR-IS-1 (cheapest
  GPU that fits 61 GB with KV room and had a CPU stager beside it; a new
  point: Qwen3-32B was measured on A100 PCIe, not SXM).  Volume
  `f6arz2r1s4` (85 GB) created; CPU stager `f4v1gvrczhj9mi` ($0.24/h) still
  pulling the 9 GB runner image after 5 minutes.
- **Owner feedback during the run:** RunPod's US regions are fast, Europe
  much slower, Asia slower still — never pick a region at random.  Now:
  US, then CA, then EU, Asia last; each staging event records its
  datacenter so speed can be measured per datacenter.
- **Owner stepped away (no pods without the owner):** I interrupted the run
  before any GPU pod.  My SIGINT went to every process in the chain, so
  Python got it twice: the second one interrupted the stager's cleanup and
  the atexit guard terminated the pod instead.  Zero pods after.  The hold
  had no pod id (the stager never finished provisioning), so replay settled
  it at the $0.15 estimate; withdrawn by a signed correction — the pod's
  real bill enters at the next reconciliation as unledgered.  Fixed: the
  stager is torn down before any cost query, and a stager stopped
  mid-provision is still annotated.
- **Kept:** volume `f6arz2r1s4` in EUR-IS-1 (85 GB, ~$0.20/day, empty or
  near empty).  Given the region feedback, the next staged run should go to
  a US datacenter; this volume can then be deleted (owner's call).

## 2026-09-27 — GPU-free predictions for groups A-D (owner away: no pods)

`scripts/modern_predictions.py` → `modern-predictions.json`: the production
plan pipeline on each approved model and proposed GPU, no pod, no paid call.
It found three errors of ours before any money was spent:

- **Multi-GPU seeds were planned whole on one GPU.** `plan_seed` never passed
  the GPU count as the split, so MiniMax-M2.7 on 2x H200 read 214 GiB "per
  GPU" → infeasible.  Now TP = count, predicted per GPU, and a per-GPU claim
  must fit one GPU (it was compared against all of them).
- **Hybrid models got a confident plain-attention estimate.** Nemotron-H
  (Mamba2 + attention) was "planned" with KV for all 52 layers.  vLLM marks
  25 model classes `IsHybrid`; that list is now generated from the pinned
  source and those architectures are an explicit unknown until the
  calculator models their state.  Qwen3.5/3.6/3.8 are among them.
- **The load check called every large MoE unloadable.** MiniMax-M2.7,
  GLM-5.3, DeepSeek-V3.2, Kimi K2 and GLM-4.7-Flash store the MoE router's
  `e_score_correction_bias` — the model's own parameter, not the fp8
  method's.  Rule now: a tensor is refused when no file of the pinned vLLM
  names it at all (a generated dictionary of the source's names, 52,513
  names, checked by digest); `backward_hadamard_matrix` (class 6) is in no
  v0.29.0 file, so that refusal stands.  (vLLM 0.29 keeps some models under
  `vllm/models/`, e.g. DeepSeek V4: the dictionary covers the whole package.)

Result (per GPU): gpt-oss-20b 12.8 GiB on a 4090; gpt-oss-120b 60.8 on an
H100; GLM-4.7-Flash 58.2 (MLA) on an H100; MiniMax-M2.7 107 on 2x H200;
DeepSeek-V4-Flash 38.9 on 4x H200; GLM-5.3 88.0, DeepSeek-V3.2 80.3 on 8x
H200; Kimi K2 119.8 on 8x B200.  Unknown: Gemma 4 (multimodal config, the
calculator reads no nested `text_config`), Qwen3.8/3.6 and Nemotron-3
(hybrid).  To check before trusting: DeepSeek-V4 is planned as plain GQA —
its attention is new (compressed/sparse) and may need its own mechanism.
H200 memory in GPU_SPECS is not yet confirmed on a pod.

## 2026-09-27 — the calculator learns hybrid and mixed-attention models (no spend)

- Two read-only traces of the pinned vLLM (`hybrid-memory-trace.md`,
  `gemma4-memory-trace.md`) gave how v0.29.0 pages per-layer caches:
  state layers set the attention block size, pages are unified, layers are
  grouped, and one request reserves a counted number of blocks per group.
- New `layered_decode` mechanism (`domain/mechanisms/layered.py`) for the
  Qwen3.5 family (gated delta net + attention), NemotronH (Mamba2 +
  attention) and Gemma 4 (sliding + global attention, global K copied into
  v_proj: +210 MiB).  Its reservation per request equals the traced figures
  byte for byte for all four models (tests).  Other hybrids (Jamba,
  Falcon-H1, ...) stay unknown.
- Found on the way: the source-name dictionary skipped names starting with
  a capital, so Qwen3.6's `A_log` read as unloadable; newer configs name the
  dtype `dtype`, not `torch_dtype`; multimodal configs keep the language
  model in `text_config` (vLLM sizes from it) — all fixed.
- Now every one of the 12 models has a GPU-free verdict: all "planned".
  Gemma 4-31B on an H100 64.7 GiB, Qwen3.8-27B 56.8, Qwen3.6-35B-A3B 37.7,
  Nemotron 3 20.4 (per GPU, at the plan's 640-token context).
- Worth knowing for the article: at vLLM's default context (262,144 tokens)
  Gemma 4-31B needs >= 33.3 GiB of KV beside ~58 GiB of weights on an H100 —
  it would not start there; Apron's plan sets the context to the workload.
- Open: Nemotron's checkpoint asks for an FP8 KV cache (hf_quant_config);
  the prediction uses bf16 (conservative) until a boot shows which vLLM
  picks.  Qwen3.5 and Gemma 4 load their vision towers by default.
