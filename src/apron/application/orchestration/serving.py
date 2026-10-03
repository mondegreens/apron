"""Serving measurement verdict against the declared SLO (D2, §9.1 step 5).

Apron measures against the SLO the accepted ``ServingWorkloadSpec`` declares
and never tunes for speed.  The verdict is computed from the stored
``serving_performance`` VerificationReport; it is independent of the task
verdict (INV-24).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from apron.domain.schemas.records import VerificationReport
    from apron.domain.schemas.tasks import ServingWorkloadSpec

MAX_FAILURE_RATE = 0.20

# (SLO field on ServingWorkloadSpec, measured key in serving_latency_ms)
_SLO_METRICS: tuple[tuple[str, str], ...] = (
    ("p99_ttft_ms", "p99_ttft_ms"),
    ("p99_tpot_ms", "p99_tpot_ms"),
)


@dataclass(frozen=True)
class ServingSloResult:
    verdict: Literal["pass", "fail", "none_declared"]
    reasons: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.verdict != "fail"


def latency_key(percentile: int, metric: str) -> str:
    """Key used in ``VerificationReport.serving_latency_ms``: ``p99_ttft_ms``."""
    return f"p{percentile}_{metric}_ms"


def evaluate_serving_slos(
    report: VerificationReport,
    workload: ServingWorkloadSpec,
) -> ServingSloResult:
    """Compare measured p99 TTFT/TPOT with the declared SLO.

    Fails when the report is not a serving measurement, when zero requests
    completed, when more than 20% of requests failed, when a declared SLO's
    metric was not measured, or when a measured p99 exceeds its SLO.  With no
    SLO declared the verdict is ``none_declared``: a measurement, not a pass.
    """
    if report.claim_scope != "serving_performance":
        return ServingSloResult("fail", (f"not a serving report: {report.claim_scope}",))

    completed = report.serving_completed or 0
    failed = report.serving_failed or 0
    if completed == 0:
        return ServingSloResult("fail", ("zero completed requests",))
    total = completed + failed
    failure_rate = failed / total
    if failure_rate > MAX_FAILURE_RATE:
        return ServingSloResult(
            "fail", (f"failure rate {failure_rate:.0%} exceeds {MAX_FAILURE_RATE:.0%}",)
        )

    declared = [
        (slo_field, key, getattr(workload, slo_field))
        for slo_field, key in _SLO_METRICS
        if getattr(workload, slo_field) is not None
    ]
    if not declared:
        return ServingSloResult("none_declared", ("no serving SLO declared",))

    measured = report.serving_latency_ms or {}
    reasons: list[str] = []
    for slo_field, key, limit in declared:
        value = measured.get(key)
        if value is None:
            reasons.append(f"{key} not measured but {slo_field} is declared")
        elif value > limit:
            reasons.append(f"{key} {value:.1f} exceeds SLO {limit:.1f}")
    if reasons:
        return ServingSloResult("fail", tuple(reasons))
    return ServingSloResult(
        "pass",
        tuple(f"{key} {measured[key]:.1f} within SLO {limit:.1f}" for _, key, limit in declared),
    )
