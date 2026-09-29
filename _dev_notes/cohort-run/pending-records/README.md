# Pending records

Records the cohort measured that the calculator cannot yet explain. They are
real measurements, kept verbatim; they sit here, outside `records/`, only
because `tests/unit/test_calculator_vs_cohort.py` fails on them until the
calculator is fixed. Move each back into `_dev_notes/cohort-run/records/<kind>/` in the commit that
makes the calculator match it. Never widen a tolerance to admit one.

| Record | What | Why it waits |
|---|---|---|
| `verification-reports/1220b7a182caf10d…` | GLM-5.3-Flash, 4x B200, CUDA graphs on, memory report (pod uxt0bh4pm5rhx4, 2026-09-29) | CUDA-graph estimate measured 4.28 GiB vs 2.30 predicted; torch peak 3.39 vs 2.82 GiB at 16,384 tokens; KV budget 2.61 GiB over (buffer 2.19). The per-layer-graph constant does not fit Glm5Next: ~1.2 MiB here, ~9 MiB on 2x B200 (record `12208ef6f21ef008…`). Handoff item H1. |
