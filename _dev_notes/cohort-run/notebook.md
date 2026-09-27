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
