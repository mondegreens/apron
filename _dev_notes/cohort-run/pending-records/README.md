# Pending records

Records the cohort measured that the calculator cannot yet explain. They are
real measurements, kept verbatim; they sit here, outside `records/`, only
because `tests/unit/test_calculator_vs_cohort.py` fails on them until the
calculator is fixed. Move each back into `_dev_notes/cohort-run/records/<kind>/` in the commit that
makes the calculator match it. Never widen a tolerance to admit one.

| Record | What | Why it waits |
|---|---|---|
| `verification-reports/1220b7a182caf10d…` | GLM-5.3-Flash, 4x B200, CUDA graphs on, memory report (pod uxt0bh4pm5rhx4, 2026-09-29) | CUDA-graph estimate measured 4.28 GiB vs 2.30 predicted; torch peak 3.39 vs 2.82 GiB at 16,384 tokens; KV budget 2.61 GiB over (buffer 2.19). The per-layer-graph constant does not fit Glm5Next: ~1.2 MiB here, ~9 MiB on 2x B200 (record `12208ef6f21ef008…`). Handoff item H1. |

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
