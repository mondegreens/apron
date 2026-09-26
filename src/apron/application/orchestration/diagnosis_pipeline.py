"""Generic diagnosis pipeline — classify, extract, match, correct.

Composes classification (adapters), rule matching (domain), and
correction (application) into a single pipeline call. Produces a
corrected plan; deployment is the caller's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from apron.application.orchestration.correction import (
    CorrectionContext,
    apply_correction_spec,
    compute_correction,
)
from apron.domain.diagnosis import (
    build_extraction_schemas,
    match_rule,
)
from apron.domain.diagnosis import (
    extraction_confidence as compute_confidence,
)
from apron.domain.fingerprints import fingerprint_hex

if TYPE_CHECKING:
    from apron.domain.schemas.primitives import HardwareSpec
    from apron.domain.schemas.solutions import DeploymentPlan


DIAGNOSIS_CONFIG_FIELDS: tuple[str, ...] = (
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
    "model_type",
    "architectures",
    "quantization_config",
)


def diagnosis_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """The resolved ``config.json`` fields diagnosis and correction read (F5).

    Multimodal checkpoints nest the language model under ``text_config``;
    its fields fill any that the top level lacks.  Absent fields stay absent.
    """
    text_config = config.get("text_config")
    nested = text_config if isinstance(text_config, dict) else {}
    selected: dict[str, Any] = {}
    for key in DIAGNOSIS_CONFIG_FIELDS:
        if key in config:
            selected[key] = config[key]
        elif key in nested:
            selected[key] = nested[key]
    return selected


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
    classification_confidence: float = 1.0
    requires_gpu_verification: bool = True
    # F6 provenance: which classifier saw which input, and what it returned.
    classifier_model_id: str | None = None
    classifier_input_digest: str | None = None
    evidence_span: str = ""


def run_diagnosis_pipeline(
    error: str,
    engine: Any,
    plan: DeploymentPlan,
    model_config: dict[str, Any],
    hardware: HardwareSpec,
    rules: list[dict[str, Any]],
    verification_report: dict[str, Any] | None = None,
    correction_context: CorrectionContext | None = None,
) -> DiagnosisPipelineResult:
    """Run the full diagnosis pipeline: classify → extract → match → correct.

    The pipeline does NOT deploy the corrected plan. Deployment,
    mechanism proof, and request replay are the caller's responsibility.
    """
    import logging

    _log = logging.getLogger(__name__)

    classification = engine.classify(error)
    failure_class = classification["failure_class"]
    provenance: dict[str, Any] = {
        "classifier_model_id": classification.get("classifier_model_id"),
        "classifier_input_digest": classification.get("classifier_input_digest"),
        "evidence_span": str(classification.get("evidence_span") or ""),
    }

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
            **provenance,
        )

    extracted = engine.extract(error, failure_class)
    schemas = build_extraction_schemas(rules)
    confidence = compute_confidence(failure_class, extracted, schemas)

    classification_confidence = classification.get("confidence", 1.0)
    if classification_confidence < 0.5:
        _log.warning(
            "Low classification confidence (%.0f%%) for %s",
            classification_confidence * 100,
            failure_class,
        )
        return DiagnosisPipelineResult(
            failure_class=failure_class,
            extracted=extracted,
            rule_matched=False,
            correction_strategy=None,
            corrected_plan=None,
            result_label="Low classification confidence",
            error_trace=error[:2000],
            extraction_confidence=confidence,
            version_match=version_match,
            **provenance,
        )

    if confidence == 0.0 and failure_class != "unknown":
        _log.info(
            "No fields extracted for %s — correction will use fallback strategy",
            failure_class,
        )

    if version_match is False:
        _log.warning("Version mismatch — refusing to apply version-pinned correction")
        return DiagnosisPipelineResult(
            failure_class=failure_class,
            extracted=extracted,
            rule_matched=False,
            correction_strategy=None,
            corrected_plan=None,
            result_label="Engine version mismatch",
            error_trace=error[:2000],
            extraction_confidence=confidence,
            version_match=version_match,
            **provenance,
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
            **provenance,
        )

    strategy_name = rule.get("correction_strategy")
    correction_spec = rule.get("correction_spec")
    strategy: str = strategy_name or (
        correction_spec.get("action", "spec") if correction_spec else "unknown"
    )
    if strategy_name:
        corrected = compute_correction(
            strategy_name,
            extracted,
            plan,
            model_config,
            hardware,
            verification_report,
            rule=rule,
            context=correction_context,
        )
    elif correction_spec:
        corrected = apply_correction_spec(correction_spec, extracted, plan)
    else:
        corrected = None

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
            **provenance,
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
        **provenance,
    )


def _serving_degraded(
    corrected: DeploymentPlan,
    original: DeploymentPlan,
    vr: dict[str, Any] | None,
) -> bool:
    """Check if the correction degrades serving capacity.

    Checks three conditions:
    1. max_num_seqs decreased
    2. max_model_len decreased
    3. KV cache budget can't serve original concurrency (when vr available)
    """
    corrected_seqs = int(corrected.engine_configuration.get("max_num_seqs", "256"))
    original_seqs = int(original.engine_configuration.get("max_num_seqs", "256"))
    if corrected_seqs < original_seqs:
        return True

    corrected_len = int(corrected.engine_configuration.get("max_model_len", "0"))
    original_len = int(original.engine_configuration.get("max_model_len", "0"))
    if 0 < corrected_len < original_len and original_len > 0:
        return True

    if vr is not None:
        available_kv = vr.get("available_kv_cache_memory", 0)
        if available_kv > 0 and corrected_len > 0:
            kv_per_token_estimate = available_kv / max(original_len, 1)
            corrected_capacity = available_kv / max(corrected_len * kv_per_token_estimate, 1)
            if corrected_capacity < original_seqs:
                return True

    return False


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
