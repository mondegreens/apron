"""Generic diagnosis pipeline — classify, extract, match, correct.

Composes classification (adapters), rule matching (domain), and
correction (application) into a single pipeline call. Produces a
corrected plan; deployment is the caller's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from apron.application.orchestration.correction import compute_correction
from apron.domain.diagnosis import match_rule

if TYPE_CHECKING:
    from apron.domain.schemas.primitives import HardwareSpec
    from apron.domain.schemas.solutions import DeploymentPlan


@dataclass(frozen=True)
class DiagnosisPipelineResult:
    failure_class: str
    extracted: dict[str, int | float | str]
    rule_matched: bool
    correction_strategy: str | None
    corrected_plan: DeploymentPlan | None
    result_label: str
    error_trace: str


def run_diagnosis_pipeline(
    error: str,
    engine: Any,
    plan: DeploymentPlan,
    model_config: dict[str, Any],
    hardware: HardwareSpec,
    rules: list[dict[str, Any]],
    verification_report: dict[str, Any] | None = None,
) -> DiagnosisPipelineResult:
    """Run the full diagnosis pipeline: classify → extract → match → correct.

    The pipeline does NOT deploy the corrected plan. Deployment,
    mechanism proof, and request replay are the caller's responsibility.
    """
    classification = engine.classify(error)
    failure_class = classification["failure_class"]

    if failure_class == "unknown":
        return DiagnosisPipelineResult(
            failure_class="unknown",
            extracted={},
            rule_matched=False,
            correction_strategy=None,
            corrected_plan=None,
            result_label="Unrecognized failure",
            error_trace=error[:2000],
        )

    extracted = engine.extract(error, failure_class)

    rule = match_rule(failure_class, rules)
    if rule is None:
        return DiagnosisPipelineResult(
            failure_class=failure_class,
            extracted=extracted,
            rule_matched=False,
            correction_strategy=None,
            corrected_plan=None,
            result_label="No rule matched",
            error_trace=error[:2000],
        )

    strategy = rule["correction_strategy"]
    corrected = compute_correction(
        strategy, extracted, plan, model_config, hardware, verification_report
    )

    if corrected is None:
        return DiagnosisPipelineResult(
            failure_class=failure_class,
            extracted=extracted,
            rule_matched=True,
            correction_strategy=strategy,
            corrected_plan=None,
            result_label="Correction infeasible",
            error_trace=error[:2000],
        )

    label = "Corrected"
    if corrected == plan:
        label = "No effective correction"

    return DiagnosisPipelineResult(
        failure_class=failure_class,
        extracted=extracted,
        rule_matched=True,
        correction_strategy=strategy,
        corrected_plan=corrected,
        result_label=label,
        error_trace=error[:2000],
    )
