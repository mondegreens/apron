# Cohort run notebook (PLAN §12)

Timestamped, facts only: expected vs happened, surprises, availability
waits, retries, dead ends, fixes that failed.  Times are local (UTC-5)
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
