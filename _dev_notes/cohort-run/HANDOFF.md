# Phase 1b cohort — handoff (2026-09-29)

Read this before acting. It is the state of the cohort work, what is left, how
to run it, and the traps that cost money or time. The previous session's task
list does not carry over; the list of open work below is complete. Its labels
(H1, H2, … and G1, G2, …) belong to this document only — they are not the
plan's §5 items F1-F12.

Other sources: run history `_dev_notes/cohort-run/notebook.md` (its last entry
is 2026-09-28, group A; the later history is in the commits and here);
process `docs/development-process.md`; repo rules `AGENTS.md`.

## The plan

- **Binding text:** `/Users/vlad/repos/apron/_thoughts/phase-1b-cohort/PLAN.md`.
  `_thoughts/` is git-ignored and lives in the main checkout, not in this
  worktree: read it by that absolute path. Its header's "code base
  `bd376ef`" line is historical; the work lives on this branch.
- **Rev 3 (§0-§17, 2026-09-25)** — the original Evidence Cohort: foundations,
  L0 proofs, budget ledger, fingerprints, scheduler/orchestrator, six
  failure-fix proofs, exit gate test (`tests/test_exit_gate_1b_cohort.py`,
  run by CI with `APRON_COHORT_GATE=1`), findings article 1 (draft PR #43).
  Rev 3 was complete when its records were the only ones; see G-gate below
  for what the Rev 4 records broke.
- **Rev 4 amendment (§18, 2026-09-27):** current open models in groups A-D
  (list: `cohort/modern-models.json`), engine version per plan (vLLM v0.30.0
  added), task suites v2/v3, calculator coverage for hybrid attention,
  article 2. "Done" for Rev 4 is §18.2. Its bullet "every boot of a staged
  model reads weights from the volume" is obsolete (staging dropped, §18.4.2):
  every boot records `weights_source` instead.
- **Changes after Rev 4 (§18.4, 2026-09-28/29):** cap $200 (§18.1.2's "$100,
  stop before group D" is superseded; group D stays an owner checkpoint);
  staged weights dropped, weights download on the pod; pod start watched
  through RunPod's log API; no fixed boot limit; vLLM keeps logging at DEBUG;
  host faults are not model failures; unexplained measurements wait in
  `pending-records/`.
- **Status against the plan:** §18.3 (updated 2026-09-29) and the table below.
  Rev 4 is **not done**: three group C models never booted, group D not run,
  article 2 lacks the group C results, the exit gate fails on the Rev 4 records.

## Where things are

- Worktree `/Users/vlad/repos/apron/.claude/worktrees/interesting-shtern-147258`,
  branch `claude/interesting-shtern-147258`. The remote
  `phase-1b/evidence-cohort` (PR mondegreens/apron#42) is at `75ba44f`;
  **every commit after `75ba44f` is local**. Push only when the owner says so.
- Article: branch `phase-1b/findings-article`, draft PR #43. Never open new PRs
  for the article; never open PRs or issues without the owner.
- RunPod: no pods, no network volumes (the 3358 GB volume was deleted on
  2026-09-29, owner's go). Balance ~$3.66: the owner tops up before any GPU run.
- Ledger: **$136.94 spent of the $200 cap, $63.06 left** after the
  reconcile of 2026-09-29 21:05 UTC (`c29717f`): +$17.34 of it is the 4xH200
  pod `yduvqoaynvo8ep`, kept after the GLM variant for debugging by hand and
  never ledgered. 13 pods, G1's among them, not billed yet: reconcile again.
  RunPod balance $53.66 before G1.
- vLLM sources, outside the worktree: v0.30.0 at
  `/Users/vlad/repos/apron/.sources/vllm-v0.30.0/`, v0.29.0 at
  `/Users/vlad/repos/apron/.sources/vllm/`.

## Models: what is measured, what is left

Group membership follows `cohort/modern-models.json`.

| Group | Model | State |
|---|---|---|
| Base cohort | 15 small/medium (Qwen3, Mistral, Llama 3.1, Gemma 2, DeepSeek-V2-Lite, Mamba, quantized variants) | done, six fix proofs |
| A | gpt-oss-120b, Gemma 4 31B, GLM-4.7-Flash (measured once, then replaced in the list) | done, healthy (H100) |
| A | Muse-Glimmer-30B | booted healthy (H100, ran with group B), suite **0/3** — H4 |
| A | gpt-oss-20b (RTX 4090) | done 2026-09-29 (G1): 3/3, serving SLO passes, $0.13 |
| B | Qwen3.8-27B (v0.30), Qwen3.6-35B-A3B-FP8, Nemotron-3.5-Lightning | done, healthy (H100) |
| B (dropped) | Nemotron-3-Nano NVFP4 | dropped from the list ("replaced by Nemotron-3.5"); a stale seed row remains — H6 |
| C | GLM-5.3-Flash | healthy on 4xH200 with `enforce_eager` (solution `1220020ae7d2d05c…`): suite v3 4/5 — the miss was the reasoning check's own effort setting, fixed in `e0857da`; TPOT 111 ms fails the 100 ms SLO |
| C | DeepSeek-V4.1-Flash, DeepSeek-V4-Flash-0731, Qwen3.8-Flash-Next | **never booted** — G2 (Qwen needs H2 first) |
| D | GLM-5.3, DeepSeek-V4-Pro-0813, MiniMax-M3 (8xH200) | **not run** — G4 |

What happened to GLM-5.3-Flash, in order:
- 2xB200: booted; with CUDA graphs it died mid-suite, and the log was lost.
- 4xB200 (pod `uxt0bh4pm5rhx4`): booted with CUDA graphs. It passed the 32k
  needle, then died on the tool call with "Input tensor addresses changed
  between capture and replay".
- 4xH200 eager (pod `0u6rww3x7ya7yt`, solution `1220020ae7d2d05c…`): healthy.
- 4xH200 again (pod `yduvqoaynvo8ep`): the host had a broken NVSwitch (NCCL at
  init). Recorded as a failed boot and kept as a `boot_evidence` event.
- By hand on that pod, with `NCCL_NVLS_ENABLE=0`:
  - eager: served;
  - GLM reasons on "17 + 25" only at the template's default effort;
  - CUDA graphs: the same tool-call assert as on 4xB200.

## Open work

GPU-free — do these first, in this order:

- **H1 — calculator vs the GLM 4xB200 record — deferred (owner, 2026-09-29).**
  Not a phase blocker: the target GPU for GLM-5.3-Flash is 4xH200
  (`cohort/modern-models.json`), where it is measured; §18.2 lets a miss stay
  visible. The record waits in `pending-records/`; its README has what was
  traced (2026-09-29) and what needs a probe. Never widen a tolerance.
- **H2 — Qwen3.8-Flash-Next plan — checked 2026-09-29, no code change.**
  - Target 4xH200: vLLM's default 1024 sequences against ~6,645 predicted
    state blocks (63.35 GiB KV, 0.0095 GiB per block): the refusal cannot
    happen there short of a tens-of-GiB error.
  - 2xB200 refusal (record `1220a2044547c634…`): 841 blocks predicted, 626 in
    vLLM, about 4 GiB high at TP 2 on SM100. One number cannot tell the KV
    budget from the block size; stays visible.
  - The 4xH100 failure (record `122069c0258a5541…`, max_num_seqs 692) was not
    the model: NCCL "unhandled cuda error" at `ncclCommInitRank`, 4 s after
    start, the host fault now classified and retried with
    `NCCL_NVLS_ENABLE=0`. The record still says `boot:model_failure`: correct
    it in findings with a `boot_evidence` event (G5), never by editing it.
- **H3 — diagnosis rule.** vLLM v0.30 with CUDA graphs replays moved inputs
  for GLM-5.3-Flash (`breakable_cudagraph.py:419-424`). This showed on the
  tool-call request, on B200 and on H200. The fix is `enforce_eager`.
  - Write it as a hypothesis rule in `rules/vllm-v0.30/`, in the schema of the
    existing rules there, with this source observation.
  - The check runs only at DEBUG (line 288, `cuda_graph.py:191`). It exists to
    prevent silent wrong outputs: see "Never blind yourself".
- **H4 — Muse-Glimmer 0/3 — resolved (2026-09-29).** The 0/3 is solution
  `1220155171f73cac…`, served without a reasoning parser: the reasoning
  landed in the answer ("to=self … assistant to=user50"). Solution
  `1220cf5cc8099fd5…` with `reasoning_parser: muse_glimmer` scored 3/3. The
  article's model card showed the first solution by digest; it now shows the
  latest one (`2f32050`). No paid re-run needed.
- **H5 — cost estimates — done (`657e13e`).** `scheduler.run_cost` is the
  slowest measured case per term (image pull 13 min, download 28.3 GB/min,
  engine start max(size class, checkpoint / 11.5 GB/min), evaluation 5 min);
  `plan_for` reuses it through the seed row; `CohortPlanner.download_weights`
  drops the download for a staged volume. `tests/unit/test_run_cost_vs_cohort.py`:
  no recorded run cost more than its estimate (worst 60%; the stop is 150%).
  These are the budget's holds, not expected spend: holds for everything left
  (G1, G2, group D) are ~$290 downloading on the pod, ~$225 staged.
- **H6 — seed cleanup.** `cohort/phase-1b-seed.json` still has a row for
  Nemotron-3-Nano, which `cohort/modern-models.json` dropped. Keep the seed
  consistent with the list; ask the owner before removing a model.
- **G-gate — exit gate — 9/9 on the real records (`bfc5dd0`).** Item 7a
  accepts an execution in an owner-approved list whose model is on
  `cohort/modern-models.json` (§18.1.1). Item 9 attributes `pod-abandoned:`
  spends to the next settle, leaves out settles flagged `interrupted*` and
  `probe:` settles (no records), and counts `pending-records/`. Rankings now
  record their cost model; older ones re-derive with H1's rates.
- **Older, partly done:**
  - Engine versions (`engine-versions-inventory.md`):
    - readable engine version (+ driver/CUDA/PyTorch) on PlanningClaim and
      VerificationReport, per ADR-003 §5;
    - facts per version everywhere, no silent default;
    - the engine version per row in findings and the article.
  - Startup peaks:
    - encoder-phase plumbing: processor configs feed the activation estimate,
      and untraced vision towers are flagged, not a silent 0;
    - Qwen3.6 peak 1.92 vs 1.01 GiB predicted;
    - Muse 1.83 vs 2.71.
  - KV layouts: packed (DeepSeek V4/V4.1, Qwen4Exp) and glm5_next are
    implemented in `layered.py`; what is left from the traces
    (`*-kv-trace.md`) is the weight side (mtp.* and the host-RAM tables such
    as DeepSeek's Engram). Check against the traces before writing code.
- **G5 — bill, findings, article.**
  1. Run `scripts/reconcile_billing.py` once today's pods post.
  2. Regenerate findings/README.
  3. Put the GLM and RunPod lessons into article 2. The prose is edited on
     branch `phase-1b/findings-article` (PR #43; check it out in its own
     worktree); this branch holds only the generated tables.
  4. Bring `notebook.md` up to date.

Paid — each needs the owner's explicit "да" after you state the price:

| Id | What | Estimate |
|---|---|---|
| G1 | gpt-oss-20b on RTX 4090 — **done** (`65f867b`) | $0.13 |
| G2 | DeepSeek-V4.1-Flash, DeepSeek-V4-Flash-0731, Qwen3.8-Flash-Next on 4xH200 (`approved-groupC-h200x4-rest.json`, suite v3), after H2 and H5 | ~$15 each at $18.36/h |
| G3 | GLM-5.3-Flash eager once more, to record the fixed reasoning check, after H5 | ~$14 |
| G4 | Group D on 8xH200 or 8xB200, when stock exists | ~$40-60 each |

Money:
- G1 + G2 + G3 ≈ $60 leaves ~$27 of the $200 cap.
- Group D (~$120-180) does not fit, and needs an owner decision on the cap first.

## Groups C and D: GPU-free readiness (2026-09-29)

Planned with today's code and live rates (suite v3 plans), all on vLLM v0.30.0:

| Model | GPUs | Weights / total / GPU (GiB per GPU) | KV GiB | Hold (download) | Notes |
|---|---|---|---|---|---|
| DeepSeek-V4.1-Flash | 4xH200 | 77.98 / 81.67 / 125.82 | 44.60 | $24.80 | host RAM 188.83 GiB (Engram tables) |
| DeepSeek-V4-Flash-0731 | 4xH200 | 37.25 / 46.55 / 125.82 | 85.63 | $21.08 | — |
| Qwen3.8-Flash-Next | 4xH200 | 58.76 / 62.95 / 125.82 | 63.35 | $23.17 | host RAM 95.37 GiB; 1024 seqs vs ~6,645 blocks |
| GLM-5.3 | 8xH200 | 88.20 / 96.75 / 125.82 | 31.89 | $67.57 | GlmMoeDsa: breakable graphs; the DEBUG input-address check caught GLM-5.3-Flash (Glm5Next) |
| DeepSeek-V4-Pro-0813 | 8xH200 | 101.94 / 115.12 / 125.82 | 19.81 | $77.84 | — |
| MiniMax-M3 | 8xH200 | 99.79 / 104.99 / 125.82 | 22.21 | $74.95 | vision tower peak not traced |

- Host RAM: pods are created without `min_memory_in_gb` (SDK default 1).
  RunPod lists 188 GB per H200 (`lowestPrice.minMemory`, 1 GPU), so 4xH200
  holds ~750 GB against V4.1's ~203 GB. Not requested explicitly.
- Pod disk: download-on-pod sizes the pod volume at largest x 1.15 + 10 GB,
  ~1,037 GB for DeepSeek-V4-Pro. RunPod's docs name no maximum for a pod
  volume (it is the host's local disk): unknown. A network volume holds
  up to 4,000 GB (`runpod_storage.MAX_VOLUME_GB`); group D's three
  checkpoints are 2,502.7 GB. Group D: stage on a network volume.
- Stock: `stock-watch.jsonl` (read-only poll every 10 min, 4x/8x H200/B200).

## How to run (no volume: weights download on the pod)

The GPU steps are pytest steps in `tests/integration/test_cohort_run.py`,
chosen by `APRON_COHORT_STEP`. Every other step skips itself. Keys come from
`/Users/vlad/repos/apron/.env`. Never print them.

A recorded plan with changed settings (G3):

```
set -a; . /Users/vlad/repos/apron/.env; set +a
export APRON_COHORT_STEP=variant APRON_TASK_SUITE=v3 PYTHONUNBUFFERED=1 \
  APRON_BASE_SOLUTION=122042ea2f378f3800be85605c652337274ae3dd50da9aedec28d83e37f70b6384db \
  APRON_ENGINE_SET='{"enforce_eager": "true"}' \
  APRON_GPU_SKU="NVIDIA H200" APRON_WEIGHTS=download APRON_REPEAT=1 \
  APRON_RUN_TAG=glm53flash-h200x4-eager-rerun
env -u ANTHROPIC_API_KEY uv run pytest tests/integration/test_cohort_run.py \
  -m cohort -k plan_variant -s -q -p no:cacheprovider 2>&1 \
  | sed -E 's/rpa_[A-Za-z0-9]+/rpa_REDACTED/g; s/hf_[A-Za-z0-9]{20,}/hf_REDACTED/g' > <log>
```

`APRON_BASE_SOLUTION` is the recorded B200x4 GLM plan. It is in
`solutions.jsonl`, and the variant file `variant-glm53flash-h200x4-eager.json`
names it as `base_solution`.

Seeds from an approved list (G1, G2):

```
export APRON_COHORT_STEP=cohort APRON_APPROVED=approved-groupC-h200x4-rest.json \
  APRON_RUN_TAG=groupC-h200x4-rest APRON_TASK_SUITE=v3 PYTHONUNBUFFERED=1
env -u ANTHROPIC_API_KEY uv run pytest tests/integration/test_cohort_run.py -m cohort -s -q \
  -p no:cacheprovider 2>&1 | sed -E '...same mask...' > <log>
```

- `cohort` downloads on the pod wherever stock is. The pod's volume is sized
  from the largest model.
- `variant` needs `APRON_REPEAT=1` for a solution measured before. Without
  it the run is skipped as "already measured": it costs nothing and does
  nothing.
- `prestage` **no longer works**: it needs the deleted volume.
- Before starting, check stock per datacenter:
  `RunPodStorage(key).stock_in(dc, "NVIDIA H200", 4)` for each `dc` from the
  `dataCenters` GraphQL query. `RunPodTarget.stock_status()` gives only
  High/Medium/Low, not where.
- A run waits for stock silently. It polls every 2 min, for up to
  `APRON_CAPACITY_WAIT` seconds (6 h in the test's own wait, 2 h in
  `build_ports`). An empty RunPod console during that wait is normal. Tell
  the owner which GPU you are waiting for, and offer a region or GPU that is
  in stock instead of waiting in silence.

Watching a run:
- Events go to `_dev_notes/cohort-run/events.jsonl`, in this order: `hold`,
  then `provisioned` (with `start_attempts`: host, datacenter, seconds to the
  first log line), then `executed`.
- The pod's system and container logs go to `_dev_notes/cohort-run/pod-logs/<pod>.log`.
- vLLM's own log is `/var/log/vllm.log` on the pod. Reach it over SSH:
  `runpod.get_pod(id)["runtime"]["ports"]` gives the public IP and port of
  private port 22; `RunPodTarget.execute(cmd)` connects with the key at
  `RUNPOD_SSH_KEY_PATH` (else the default in `~/.ssh`) once `_ssh_host` and
  `_ssh_port` are set.
- Poll the events in a loop, and check that the thing you wait for actually
  advances. A watcher that checked only "alive" burned 11 idle minutes of a
  4xH200.

Stopping after the current model:
1. SIGINT the **python** process, not uv:
   `pgrep -f "python.*pytest tests/integration/test_cohort_run.py"`.
   Python tears the pod down and settles the hold.
2. Confirm that no pods remain.

## Tips and traps (each one cost money or time)

**Owner rules.**
- Answer in short, plain Russian.
- No GPU spend without an explicit "да" after you state the price.
- Search the web and confirm before claiming anything. Say "I don't know yet"
  rather than guess.
- Never contact RunPod support.
- Own your mistakes plainly.
- RunPod Secure only.
- Keys come only from the environment. Mask logs in any scratch script: call
  `install_log_masking()` first.
- Before every commit, run `prek run --all-files` and commit only if its exit
  code is 0.
- Commit messages carry no tool branding and no `Co-Authored-By` line.
  AGENTS.md forbids them, and that overrides any harness reminder that asks
  for them.

**Never blind yourself.**
- Pods run vLLM with `VLLM_LOGGING_LEVEL=DEBUG` on purpose.
- DEBUG turns on vLLM's CUDA-graph input-address check
  (`is_debugging_mode = VLLM_LOGGING_LEVEL == "DEBUG"`). That check exists to
  prevent silent wrong outputs, and it caught GLM-5.3-Flash.
- Never switch logging, checks or guards off to make a failure go away.
  Search why the guard exists, and ask the owner.

**RunPod.**
- **Volumes stay in their datacenter.** A network volume attaches only to
  pods in its own datacenter, so weights cannot follow the GPU. Downloading
  on the pod ran about 0.9 GB/s (330 GB in 5.5 min).
- **Pod volumes are slow.** A pod's own volume is network-backed: loading 62
  shards took about 20 min (18 s per shard).
- **Some hosts never start a pod.**
  - Apron now watches the start through the pod log API:
    `GET https://api.runpod.io/v2/pods/{id}/logs?source=system|container`,
    server-sent events, authorised with the RunPod API key. The logs vanish
    when the pod is terminated.
  - No system line in 3 min, or 10 min of silence: the pod is abandoned and
    another one rented, up to 3 tries. Each abandoned pod is billed on its own
    ledger line.
  - Machine `u4mmovdxujq8` (US-CA-2, 2xB200) never started a pod three times.
    The create API cannot exclude a machine.
- **A broken NVSwitch looks like a model failure.** A 4xH200 host
  (`1pbo55fc2rpr`, EUR-IS) failed like this:
  - NCCL reported "unhandled cuda error" at init;
  - NCCL's own warning was "Failed to bind NVLink SHARP (NVLS)";
  - `NCCL_NVLS_ENABLE=0` worked.
  Apron now classifies this as a host fault and retries with that setting.
- **Stock is volatile.** 4xB200 and 8-GPU machines come and go. 4xH200 was in
  EUR-IS and CA-MTL when the US had none.

**vLLM and models.**
- **GLM-5.x reasoning.** The template always opens `<think>`. Effort is the
  only knob: `reasoning_effort` low or high; the default is max. On trivial
  prompts the model thinks only at max.
- **Launching vLLM by hand on a pod.** Launch it exactly as Apron does:
  `. /etc/apron_environment && exec env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN
  HF_HUB_OFFLINE=1 NCCL_DEBUG=WARN /opt/venv/bin/vllm serve ...`. Without the
  container environment, FlashInfer's JIT misses ninja and CUDA_HOME and
  writes broken cubins (the `!cubin.empty()` assert).
- **`pkill` can kill your own shell.** `pkill -f "vllm serve"` inside a
  command that contains that text matches the command itself. Match
  `bin/[v]llm serve` instead.
- **No kept pods.** Keeping a pod after a run was a temporary option and is
  removed. To debug a live model, add it again deliberately, then remove it.

**Records and money.**
- Records are evidence, and they are never edited.
  - A cause that a stored log lost goes in as a `boot_evidence` event;
    findings read it.
  - A measurement the calculator cannot yet explain waits in
    `pending-records/`.
- The ledger is append-only: deleting lines is refused. Correct it with
  `correct` entries, and reconcile with `scripts/reconcile_billing.py`.
- The local shell is macOS zsh:
  - there is no `timeout`;
  - `pytest -s | sed` buffers, so set `PYTHONUNBUFFERED=1`.
