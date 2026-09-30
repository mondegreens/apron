# Pending records

Records the cohort measured that the calculator cannot yet explain. They are
real measurements, kept verbatim; they sit here, outside `records/`, only
because `tests/unit/test_calculator_vs_cohort.py` fails on them until the
calculator is fixed. Move each back into `_dev_notes/cohort-run/records/<kind>/` in the commit that
makes the calculator match it. Never widen a tolerance to admit one.

| Record | What | Why it waits |
|---|---|---|
| `verification-reports/1220b7a182caf10d…` | GLM-5.3-Flash, 4x B200, CUDA graphs on, memory report (pod uxt0bh4pm5rhx4, 2026-09-29) | CUDA-graph estimate measured 4.28 GiB vs 2.30 predicted; torch peak 3.39 vs 2.82 GiB at 16,384 tokens; KV budget 2.61 GiB over (buffer 2.19). The per-layer-graph constant does not fit Glm5Next: ~1.2 MiB here, ~9 MiB on 2x B200 (record `12208ef6f21ef008…`). Handoff item H1. |
| `verification-reports/12204fbbf9605e62…` | Qwen3.8-Flash-Next, 4x H200, vLLM v0.30.0, memory report (pod md3kso5eretn9o, 2026-09-29): healthy, 5/5 deployment checks | Startup (torch) peak 2.01 GiB measured vs 1.10 predicted at 8,192 tokens, TP 4. See below. |
| `verification-reports/1220a687eae36995…` | DeepSeek-V4-Flash-0731, 4x H200, vLLM v0.30.0, memory report (pod md3kso5eretn9o, 2026-09-29): healthy, 4/4 deployment checks (image: text-only) | Startup peak 3.10 GiB vs 1.00 predicted; CUDA-graph estimate 2.43 vs 1.12; weights 37.80 vs 37.25 (-1.5%). KV budget +3.74 GiB. The breakable-graph class, below. |

## `1220b7a182caf10d…`: what was traced (2026-09-29, GPU-free)

The KV budget is 2.61 GiB high from three terms (predicted / measured GiB):
torch peak 2.82 / 3.39, non-torch 1.05 / 0.87, CUDA-graph estimate 2.30 / 4.28.

**Torch peak.** vLLM v0.30 runs Glm5Next uncompiled: the architecture is in
`DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES` and breakable graphs set
`CompilationMode.NONE` (`config/vllm.py:77-89, 756-770`).  The forward rule's
"graph" term is GLM-4.7-Flash's compiled (inductor) buffer count, whose head
term divides by TP.  Take out the terms the rule already has (MLA prefill
dummy, workspace, the multimodal T x H) and every GLM-5.3-Flash boot leaves the
same TP-independent remainder, about 16.5 [T, hidden] bf16 tensors:
B200 TP 2 at 16,384 tokens 16.3, B200 TP 4 at 16,384 16.8, H200 TP 4 at 8,192
16.6.  Part of it is mHC's four residual streams, [T, 4, H] bf16
(`layers/mhc.py:969-972`; `kernels/mhc/tilelang.py:812-930` holds the old and
the new residual at once).  Which tensors are live together is not readable
from source: it needs a memory-history probe (`peak-probe/`), as for the
other models.  Fitting 16.5 x T x H from these three boots would be a fit,
not a trace.

**CUDA-graph estimate.**
- Breakable graphs make the mixed mode PIECEWISE: vLLM captures a PIECEWISE
  graph for every capture size (to 1024 on SM100) and measures them all in
  full, besides the FULL decode graphs it extrapolates from two samples with a
  1 MiB per-graph floor (`v1/worker/gpu/cudagraph_utils.py:318-346, 398-484,
  929-942`).  The calculator counts FULL decode graphs only.
- The estimate is measured before `kernel_warmup` and the real capture after
  it (`v1/worker/gpu_worker.py:574-590, 800-830`), so first-call allocations
  land in the estimate only: estimate minus actual is 1.59 GiB on 2x B200 and
  1.42 GiB on 4x B200; `layered.first_capture_bytes` traces 0.39 GiB (the
  FlashInfer sparse-MLA workspace, the KDA conv weights).  The rest is in
  FlashInfer, DeepGEMM and TileLang, whose sources are not in `.sources/`.
- The pod's vLLM log was not kept (only the record's numbers), so the 4.28 GiB
  cannot be split further from this boot.

What closes it: a boot with graphs on B200 that keeps vLLM's full log, and a
peak probe of the profile run.  Not planned (owner, 2026-09-29).

## `12204fbbf9605e62…`: what was traced (2026-09-29, GPU-free)

KV budget, predicted minus measured, per GPU (GiB) after the fixes below:
weights 59.82 / 60.87, torch peak 1.10 / 2.01, non-torch 1.35 / 1.21, CUDA-graph
estimate 1.25 / 1.49; available KV 62.30 / 60.24 (+2.06, inside the 2.19
buffer).  The record waits only on the startup peak.

Fixed while tracing (the weight check had read 82.60 GiB against 60.87):
- the offline check did not subtract the n-gram table vLLM keeps in pinned
  host memory (95.37 GiB): `host_bytes` in the fixture row,
  `recorded_loaded_bytes` subtracts it, as the planner always did;
- Qwen4Exp's replicated weights, 1.06 GiB per GPU at TP 4:
  `plan_pipeline._QWEN4_EXP_REPLICATED` (hyper-connections, router, QSA
  indexer projection, PLE key/value projections, each cited to vLLM);
- its vision tower is Qwen3-VL's (`qwen4_exp/nvidia/model.py:922-926`):
  `calculator._VISION_FAMILIES`.

Not explained:
- weights: 1.05 GiB per GPU over the plan after the replicated term.  Not the
  MTP layer's 1.21 GiB unless vLLM loads it (not found), not the vision tower
  (sharded, `mm_encoder_tp_mode = "weights"`), not the n-gram prefetch buffer
  (0.04 GiB, `ngram_embedding.py:421-427`);
- torch peak: 2.01 GiB.  The Qwen3-VL encoder moment gives 1.03 GiB at TP 4
  (its attention is head-sharded: `qwen2_5_vl.py:411-464`), the language
  forward 0.88.  Which tensors are live at the peak needs a memory-history
  probe (`peak-probe/`); fitting a constant would not be a trace.

## One class behind all three: vLLM v0.30's breakable CUDA graphs

GLM-5.3-Flash (Glm5NextForConditionalGeneration), Qwen3.8-Flash-Next
(Qwen4ExpForConditionalGeneration) and DeepSeek-V4-Flash-0731
(DeepseekV4ForCausalLM) are all in `DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES`
(`config/vllm.py:77-104`): vLLM runs them uncompiled (`CompilationMode.NONE`)
with PIECEWISE graphs for every capture size.  The calculator's startup-peak
forward rule counts compiled (inductor) buffers, fitted on GLM-4.7-Flash, and
its CUDA-graph rule counts FULL decode graphs per layer.  Both under-predict
for this class: startup peak 3.39/2.82 (GLM-5.3-Flash B200), 2.01/1.10 (Qwen),
3.10/1.00 (DeepSeek); graph estimate 4.28/2.30, 1.49/1.25, 2.43/1.12.  Closing
it takes the peak probe (`scripts/peak_probe.py`) on one of them; the same fix
then applies to group D (GLM-5.3 and MiniMax-M3 are in the list too).
