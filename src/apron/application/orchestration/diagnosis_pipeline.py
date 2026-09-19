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
from apron.domain.fingerprints import fingerprint_hex

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
    corrects: str | None = None
    extraction_confidence: float = 1.0
    version_match: bool | None = None


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
    import logging

    _log = logging.getLogger(__name__)

    classification = engine.classify(error)
    failure_class = classification["failure_class"]

    version_match: bool | None = None
    if hasattr(engine, "detect_engine_version"):
        detected = engine.detect_engine_version(error)
        if detected is not None:
            expected = engine.engine_version.lstrip("v")
            version_match = detected == expected
            if not version_match:
                _log.warning(
                    "Engine version mismatch: rules expect %s, error output shows %s",
                    engine.engine_version,
                    detected,
                )

    if failure_class == "unknown":
        return DiagnosisPipelineResult(
            failure_class="unknown",
            extracted={},
            rule_matched=False,
            correction_strategy=None,
            corrected_plan=None,
            result_label="Unrecognized failure",
            error_trace=error[:2000],
            version_match=version_match,
        )

    extracted = engine.extract(error, failure_class)

    confidence = 1.0
    if hasattr(engine, "extraction_confidence"):
        confidence = engine.extraction_confidence(failure_class, extracted)
        if confidence < 0.5:
            _log.warning(
                "Low extraction confidence (%.0f%%) for %s — error format may have changed",
                confidence * 100,
                failure_class,
            )

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
            extraction_confidence=confidence,
            version_match=version_match,
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
            extraction_confidence=confidence,
            version_match=version_match,
        )

    label = "Corrected"
    if corrected == plan:
        label = "No effective correction"
    elif _serving_degraded(corrected, plan, verification_report):
        label = "Alternative with trade-offs"

    original_digest = fingerprint_hex(plan)

    return DiagnosisPipelineResult(
        failure_class=failure_class,
        extracted=extracted,
        rule_matched=True,
        correction_strategy=strategy,
        corrected_plan=corrected,
        result_label=label,
        error_trace=error[:2000],
        corrects=original_digest,
        extraction_confidence=confidence,
        version_match=version_match,
    )


def _serving_degraded(
    corrected: DeploymentPlan,
    original: DeploymentPlan,
    vr: dict[str, Any] | None,
) -> bool:
    """Check if the correction degrades serving capacity.

    Returns True when the corrected max_num_seqs is below the original,
    or when the corrected max_model_len is below the original and a
    verification report shows limited KV cache.
    """
    corrected_seqs = int(corrected.engine_configuration.get("max_num_seqs", "256"))
    original_seqs = int(original.engine_configuration.get("max_num_seqs", "256"))
    if corrected_seqs < original_seqs:
        return True

    corrected_len = int(corrected.engine_configuration.get("max_model_len", "0"))
    original_len = int(original.engine_configuration.get("max_model_len", "0"))
    return 0 < corrected_len < original_len and original_len > 0


_MAX_CORRECTION_ATTEMPTS = 3


def iterate_diagnosis(
    errors: list[str],
    engine: Any,
    plan: DeploymentPlan,
    model_config: dict[str, Any],
    hardware: HardwareSpec,
    rules: list[dict[str, Any]],
    verification_report: dict[str, Any] | None = None,
) -> DiagnosisPipelineResult:
    """Run diagnosis up to _MAX_CORRECTION_ATTEMPTS times with cycle detection.

    Each error in the list is diagnosed against the corrected plan from
    the previous iteration. Aborts if the same failure_class recurs.
    """
    current_plan = plan
    seen_classes: set[str] = set()
    last_result: DiagnosisPipelineResult | None = None

    for error in errors[:_MAX_CORRECTION_ATTEMPTS]:
        result = run_diagnosis_pipeline(
            error, engine, current_plan, model_config, hardware, rules, verification_report
        )
        last_result = result

        if result.failure_class in seen_classes:
            return DiagnosisPipelineResult(
                failure_class=result.failure_class,
                extracted=result.extracted,
                rule_matched=result.rule_matched,
                correction_strategy=result.correction_strategy,
                corrected_plan=None,
                result_label="Cycle detected",
                error_trace=result.error_trace,
            )

        seen_classes.add(result.failure_class)

        if result.corrected_plan is None:
            return result

        current_plan = result.corrected_plan

    assert last_result is not None
    return last_result
