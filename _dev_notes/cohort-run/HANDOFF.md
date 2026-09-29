# Phase 1b cohort — handoff (2026-09-29)

Read this before acting. It is the state of the cohort work, what is left, how
to run it, and the traps that cost money or time. Older history:
`_dev_notes/cohort-run/notebook.md`. Process: `docs/development-process.md`.

## Where things are

- Worktree `.claude/worktrees/interesting-shtern-147258`, branch
  `claude/interesting-shtern-147258`. Pushed to `phase-1b/evidence-cohort`
  (PR mondegreens/apron#42) up to `75ba44f`; **later commits are local only**
  (`35f0422` … `6d14069`): push when the owner says so.
- Article: branch `phase-1b/findings-article`, draft PR #43. Never open new PRs
  for the article; never open PRs or issues without the owner.
- RunPod: **no pods, no network volumes** (the 3358 GB volume was deleted on
  2026-09-29, owner's go). Balance ~$3.66 — top-up needed before any GPU run.
- Ledger: $112.76 spent of the $200 cap (`AUTHORIZED_USD`, owner 2026-09-28).
  No open holds. Reconciled to the RunPod bill up to 2026-09-29 01:12 UTC.

## Models: what is measured, what is left

| Group | Models | State |
|---|---|---|
| Base cohort | 15 small/medium (Qwen3, Mistral, Llama 3.1, Gemma 2, DeepSeek-V2-Lite, Mamba, quantized variants) | done, committed, six fix proofs |
| A | gpt-oss-120b, GLM-4.7-Flash, Gemma 4 31B (H100) | done, healthy |
| B | Qwen3.8-27B, Qwen3.6-35B-A3B-FP8, Muse-Glimmer-30B, Nemotron-3.5 (H100) | booted healthy; Muse scored 0/3 (task #69/F8) |
| B, never run | gpt-oss-20b (RTX 4090), Nemotron-3-Nano-30B-A3B-NVFP4 (H100) | cheap; P1 and #33 |
| C | GLM-5.3-Flash | healthy on 4xH200 with `enforce_eager` (4/5, reasoning miss was our check — fixed); TPOT 111 ms fails the 100 ms SLO |
| C, never booted | DeepSeek-V4.1-Flash, DeepSeek-V4-Flash-0731, Qwen3.8-Flash-Next | P2; ~$15 each on 4xH200 |
| D | GLM-5.3, DeepSeek-V4-Pro-0813, MiniMax-M3 (8xH200) | not run: no 8-GPU machine was in stock all of 2026-09-28/29 |

## Open work (task ids from the session's task list)

GPU-free (do these first):

1. **F4 — calculator vs the GLM 4xB200 record.** `pending-records/README.md`.
   CUDA-graph memory for Glm5Next does not fit `layers x graphs x constant`
   (`calculator.cuda_graph_estimate_bytes`): 4.28 GiB measured vs 2.30
   predicted on 4xB200 (~83 graphs, 1.2 MiB/layer-graph) while 2xB200
   (max_num_seqs 13, 4 graphs) implies ~9 MiB. Torch peak 3.39 vs 2.82 GiB at
   16384 tokens. Read vLLM v0.30 `v1/worker/gpu/cudagraph_utils.py` 847-960
   (the estimate: two largest sizes captured, rest extrapolated) and what
   Glm5Next allocates on first capture (`layered.first_capture_bytes`). Fix
   term by term, then move the record back into `records/` and regenerate
   (`scripts/calculator_recheck.py`, `scripts/cohort_findings.py`).
   **Never widen a tolerance.**
2. **F3 — rule:** GLM-5.3-Flash on vLLM v0.30 with CUDA graphs replays moved
   inputs ("Input tensor addresses changed between capture and replay",
   `compilation/breakable_cudagraph.py:419-424`) on the tool-call request, on
   B200 and H200 → `enforce_eager`. Hypothesis rule in `rules/vllm-v0.30/`
   with that source observation. See "DEBUG" below for why this is real.
3. **F6 — estimates.** `CohortPlanner.plan_for` holds a flat 25 minutes; a run
   that downloads 330 GB and loads slowly cost $13.62 vs $7.65 and tripped the
   ">50% overrun" stop. Add download (size / ~0.9 GB/s measured) and load time.
4. **F8 — Muse-Glimmer 0/3.** Re-read its task records with what GLM taught
   (effort vs switch, parser). Decide the fix before any paid re-run.
5. **F7 — reconcile + findings + article.** `scripts/reconcile_billing.py`
   after today's pods post; regenerate findings/README; put the GLM lessons
   into the article (#43).
6. Older in-progress items: #48 model list, #49/#52/#56-#58 engine versions,
   #62/#74 startup peaks + encoder plumbing, #65/#66 calculator layouts for
   DeepSeek V4 / Qwen4Exp / Glm5Next, #68 Qwen3.6 peak miss.

Paid (each needs the owner's explicit "да" with a stated price):

- **P1** Nemotron-3-Nano NVFP4 on H100 and **#33** gpt-oss-20b on RTX 4090 —
  a few dollars.
- **P2** group C rest on 4xH200 with download on the pod. Qwen3.8-Flash-Next:
  the plan's max_num_seqs exceeded its KV blocks (714 > 626) — fix the plan
  (calculator #66) before paying.
- **Re-run GLM-5.3-Flash eager** once, to record the fixed reasoning check.
- **#37 group D** — needs 8xH200/B200 in stock and ~$40-60 per model.

## How to run (no volume: weights download on the pod)

All GPU steps are pytest steps in `tests/integration/test_cohort_run.py`,
selected by `APRON_COHORT_STEP`. Keys come from `/Users/vlad/repos/apron/.env`
(never print them). Template:

```
set -a; . /Users/vlad/repos/apron/.env; set +a
export APRON_COHORT_STEP=variant APRON_TASK_SUITE=v3 PYTHONUNBUFFERED=1 \
  APRON_BASE_SOLUTION=<recorded solution fp from solutions.jsonl> \
  APRON_ENGINE_SET='{"enforce_eager": "true"}' \
  APRON_GPU_SKU="NVIDIA H200" APRON_WEIGHTS=download APRON_REPEAT=1 \
  APRON_RUN_TAG=<tag>
env -u ANTHROPIC_API_KEY uv run pytest tests/integration/test_cohort_run.py \
  -m cohort -k plan_variant -s -q -p no:cacheprovider 2>&1 \
  | sed -E 's/rpa_[A-Za-z0-9]+/rpa_REDACTED/g; s/hf_[A-Za-z0-9]{20,}/hf_REDACTED/g' > <log>
```

- `variant`: a recorded plan with engine settings changed, optionally another
  GPU, weights downloaded on the pod. `APRON_REPEAT=1` is required when the
  same solution was measured before, otherwise it is skipped as "already
  measured" (costs nothing, does nothing).
- `cohort`: seeds from an approved list (`APRON_APPROVED=approved-*.json`),
  downloads on the pod wherever stock is (pod volume sized from the model).
- `prestage` **no longer works** — it needs the deleted volume.
- The run waits for stock silently (polls every 2 min, up to
  `APRON_CAPACITY_WAIT`); an empty RunPod console during that wait is normal.
  Check stock yourself before telling the owner anything:
  `RunPodTarget(gpu_type=..., gpu_count=n).stock_status()`.

Watching a run: events land in `_dev_notes/cohort-run/events.jsonl`
(`hold` → `provisioned` with `start_attempts` → `executed`); pod system and
container logs in `_dev_notes/cohort-run/pod-logs/<pod>.log`; vLLM's own log
is `/var/log/vllm.log` on the pod (SSH). Poll events with a loop and check
that the thing you wait for actually advances — a watcher that only checks
"alive" once burned 11 idle minutes of a 4xH200 pod.

Stopping a run after the current model: SIGINT the **python** process
(`pgrep -f "python.*pytest tests/integration/test_cohort_run.py"`), not uv —
it tears the pod down and settles the hold. Then confirm no pods remain.

## Tips and traps (each one cost money or time)

**Owner rules.** Short, plain Russian answers. No GPU spend without an explicit
"да" after you state the price. Web-search and confirm before claiming
anything; say "I don't know yet" rather than guess. Never contact RunPod
support. Own mistakes plainly. RunPod Secure only. Keys only from env; mask
logs in any scratch script (`install_log_masking()` first). `prek run
--all-files` before every commit, gate on its exit code. No tool branding or
Co-Authored-By in commits.

**Never blind yourself.** Pods run vLLM with `VLLM_LOGGING_LEVEL=DEBUG` on
purpose: DEBUG turns on vLLM's CUDA-graph input-address check
(`is_debugging_mode = VLLM_LOGGING_LEVEL == "DEBUG"`, breakable_cudagraph.py:288,
cuda_graph.py:191), which exists to prevent silent wrong outputs. It caught
GLM-5.3-Flash. Do not switch logging, checks or guards off to make a failure
go away; search why the guard exists and ask the owner.

**RunPod.**
- Network volumes attach only to pods in their own datacenter; weights cannot
  move with the GPU. Downloading on the pod ran ~0.9 GB/s (330 GB in 5.5 min).
- Pods' own volumes are network-backed too: loading 62 shards took ~20 min
  (18 s/shard). Budget for it.
- Some hosts never start a pod. The start is now watched through the pod
  log API (`GET https://api.runpod.io/v2/pods/{id}/logs?source=system|container`,
  SSE; logs vanish when the pod is terminated): no system line in 3 min or
  10 min of silence → the pod is abandoned and another rented (3 tries),
  each abandoned pod billed on its own ledger line. Machine
  `u4mmovdxujq8` (US-CA-2, 2xB200) never started a pod three times. You
  cannot exclude a machine in the create API.
- A 4xH200 host (`1pbo55fc2rpr`, EUR-IS) had a broken NVSwitch: NCCL "unhandled
  cuda error" at init; NCCL WARN "Failed to bind NVLink SHARP (NVLS)";
  `NCCL_NVLS_ENABLE=0` works. Apron now classifies and retries this.
- Stock is volatile: 4xB200/8xGPU come and go; 4xH200 appeared in EUR-IS and
  CA-MTL when US had none. Offer the alternative instead of waiting silently.

**vLLM / models.**
- GLM-5.x templates always open `<think>`; effort (`reasoning_effort` low/high,
  default max) is the only knob. On trivial prompts it thinks only at max.
- If you ever launch vLLM by hand on a pod, launch it exactly like Apron:
  `. /etc/apron_environment && exec env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN
  HF_HUB_OFFLINE=1 NCCL_DEBUG=WARN /opt/venv/bin/vllm serve ...` — without the
  container env, FlashInfer's JIT misses ninja/CUDA_HOME and writes broken
  cubins (`!cubin.empty()` assert).
- `pkill -f "vllm serve"` inside a command that contains that text kills your
  own shell; match `bin/[v]llm serve`.
- Kept-pod debugging was a temporary option and is removed; to debug live,
  add it again deliberately and remove it after.

**Records and money.**
- Records are evidence: never edit them; a lost cause goes in as a
  `boot_evidence` event (findings reads it); a measurement the calculator
  cannot yet explain waits in `pending-records/`.
- The ledger is append-only (deleting lines is refused); correct with
  `correct` entries, reconcile with `scripts/reconcile_billing.py`.
- Local shell is macOS zsh: no `timeout`; `pytest -s | sed` buffers — set
  `PYTHONUNBUFFERED=1`.
