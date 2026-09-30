"""The cost estimate against what every recorded run cost.

A run that costs more than 1.5 x its estimate stops the cohort after it
(``budget.OVERRUN_FACTOR``).  The estimate is the slowest measured case
(``scheduler.run_cost``), so no recorded run costs more than its estimate:
the pod's rate from the report, the model's download size and size class from
its seed row, the download counted unless a network volume held the weights.

The rates in ``scheduler`` (measured 2026-09-26/29):
- download 12.5 GB/min: DeepSeek-V4.1-Flash, 510.3 GB in 2,438.9 s on its
  4x H200 pod, the slowest; DeepSeek-V4-Flash-0731 166.9 GB in 706.6 s;
  gemma-4-31B-it 62.6 GB in 132.6 s on a CPU stager; GLM-5.3 755.7 GB in
  1,357.7 s; GLM-5.3-Flash 328.4 GB in 368.9 s; Qwen3.8-Flash-Next 360.0 GB
  in 338.2 s (events.jsonl "staged", the report's ``phase_seconds.weights``);
- load 11.5 GB/min: GLM-5.3-Flash's engine start, 28.4 min from the 4x H200
  pod's own volume; 24.8 and 21.3 min from a network volume on 2x and 4x B200;
- evaluation 5 min: suite v3 and the serving measurement took 4.2 min on the
  4x H200 run;
- image pull 13 min: L0-A3 (notebook, 2026-09-26); 3.8-6.5 min to the first
  container line since (events.jsonl "provisioned").
"""

from __future__ import annotations

from pathlib import Path

import pytest

from apron.application.orchestration.scheduler import run_cost
from apron.interfaces.cohort_root import load_cohort_run, load_seed

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "_dev_notes" / "cohort-run"


def _runs() -> list[str]:
    if not (RUN / "records").is_dir():
        return []
    records = load_cohort_run(RUN).records
    seeded = {s.model_id for s in load_seed()}
    return sorted(
        digest
        for digest, report in records.reports.items()
        if report.claim_scope == "memory"
        and report.gross_attributable_cost
        and report.hourly_rate
        and (entry := records.solutions.get(report.solution_fingerprint or "")) is not None
        and entry.model_id in seeded
    )


@pytest.mark.skipif(not (RUN / "records").is_dir(), reason="cohort records not present")
@pytest.mark.parametrize("digest", _runs(), ids=lambda d: d[4:16])
def test_no_recorded_run_costs_more_than_its_estimate(digest: str) -> None:
    records = load_cohort_run(RUN).records
    report = records.reports[digest]
    entry = records.solutions[report.solution_fingerprint or ""]
    seed = next(s for s in load_seed() if s.model_id == entry.model_id)
    count = entry.requested_execution.gpu_count
    estimate = run_cost(
        seed.weight_gb,
        seed.size_class,
        count,
        float(report.hourly_rate or 0) / count,  # the report's rate is the pod's
        download="network volume" not in (report.weights_source or ""),
    )
    actual = float(report.gross_attributable_cost or 0)
    assert actual <= estimate, (
        f"{entry.model_id} on {count}x {entry.requested_execution.gpu_sku}: "
        f"${actual:.2f} against an estimate of ${estimate:.2f}"
    )
