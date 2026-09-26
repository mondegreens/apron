"""Generic correction function — dispatches on strategy name.

Given a correction strategy (from a diagnosis rule), extracted values
(from error classification), and the current plan/hardware context,
compute a corrected DeploymentPlan. Pure computation, no I/O, no GPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from apron.domain.schemas.primitives import HardwareSpec
    from apron.domain.schemas.solutions import DeploymentPlan


@dataclass(frozen=True)
class CatalogEntry:
    """A GPU the solution may be retargeted to, with its per-GPU hourly rate."""

    hardware: HardwareSpec
    hourly_rate: float


@dataclass(frozen=True)
class CorrectionContext:
    """Facts a retarget strategy needs beyond the error and the plan.

    ``catalog`` is the provider's GPU list (the composition root builds it
    from the adapter's GPU_SPECS and rates); ``predicted_total_bytes`` is the
    plan pipeline's calculator prediction for this plan.
    """

    catalog: tuple[CatalogEntry, ...] = field(default=())
    predicted_total_bytes: int | None = None


logger = logging.getLogger(__name__)

CORRECTION_BOUNDS: dict[str, tuple[int | float, int | float]] = {
    "max_model_len": (1, 1_000_000),
    "max_num_seqs": (1, 4096),
    "gpu_memory_utilization": (0.1, 0.95),
}

TOP_LEVEL_FIELDS = frozenset(
    {
        "tensor_parallel",
        "pipeline_parallel",
        "expert_parallel",
        "data_parallel",
        "dtype",
        "batch_size",
    }
)

# Requested-execution fields: a retarget changes the GPU, not engine flags.
RESOURCE_FIELDS = frozenset({"gpu_sku"})

# Strategies that choose new hardware by its capacity or capability; the
# old hardware's feasibility check does not apply to them.
RETARGET_STRATEGIES = frozenset({"retarget_memory", "retarget_capability"})


def compute_correction(
    strategy: str,
    extracted: Mapping[str, int | float | str],
    plan: DeploymentPlan,
    model_config: dict[str, Any],
    hardware: HardwareSpec,
    verification_report: dict[str, Any] | None,
    planning_source: Any = None,
    rule: dict[str, Any] | None = None,
    context: CorrectionContext | None = None,
) -> DeploymentPlan | None:
    """Compute a corrected plan by dispatching on strategy name.

    Returns None if the correction is infeasible (model doesn't fit).
    When a planning_source is provided, re-runs the GPU-free calculator
    prediction to verify the corrected plan fits the target hardware.
    """
    dispatch = _STRATEGIES.get(strategy)
    if dispatch is None:
        msg = f"Unknown correction strategy: {strategy!r}"
        raise ValueError(msg)

    overrides = dispatch(
        extracted,
        plan,
        model_config,
        hardware,
        verification_report,
        rule=rule,
        context=context or CorrectionContext(),
    )
    if overrides is None:
        return None
    overrides = _validate_bounds(overrides)
    corrected = _apply_overrides(plan, overrides)
    if strategy in RETARGET_STRATEGIES:
        return corrected
    if not _check_feasibility(
        corrected, hardware, verification_report, planning_source, model_config
    ):
        return None
    return corrected


def _validate_bounds(overrides: dict[str, Any]) -> dict[str, Any]:
    """Clamp corrected values to known-safe ranges."""
    result = dict(overrides)
    for key, (lo, hi) in CORRECTION_BOUNDS.items():
        if key in result:
            val = result[key]
            if isinstance(val, str):
                try:
                    val = type(lo)(val)
                except (ValueError, TypeError):
                    continue
            if val < lo:
                logger.warning("Clamping %s from %s to %s", key, val, lo)
                result[key] = str(lo) if isinstance(overrides[key], str) else lo
            elif val > hi:
                logger.warning("Clamping %s from %s to %s", key, val, hi)
                result[key] = str(hi) if isinstance(overrides[key], str) else hi
    return result


def _check_feasibility(
    corrected: DeploymentPlan,
    hardware: HardwareSpec,
    vr: dict[str, Any] | None,
    planning_source: Any = None,
    model_config: dict[str, Any] | None = None,
) -> bool:
    """Check whether the corrected plan can physically fit on the hardware.

    When a planning_source is available, re-runs the GPU-free calculator
    prediction. Otherwise falls back to comparing model weight memory
    from the verification report against GPU capacity.
    """
    if hardware.total_memory_bytes <= 0:
        return True

    if planning_source is not None and model_config:
        try:
            claim = planning_source.predict(model_config, hardware, {})
            proposed = claim.proposed_configuration
            total_required = proposed.get("total_required_bytes", 0)
            if total_required > 0:
                usable = int(hardware.total_memory_bytes * 0.95)
                if total_required > usable:
                    logger.warning(
                        "Correction infeasible: predicted %d bytes exceeds "
                        "95%% of GPU memory (%d bytes)",
                        total_required,
                        hardware.total_memory_bytes,
                    )
                    return False
                return True
        except Exception:
            logger.debug("Calculator re-prediction failed", exc_info=True)

    if vr is None:
        return True
    weight_memory = vr.get("model_weight_memory", 0)
    if weight_memory <= 0:
        return True
    usable = int(hardware.total_memory_bytes * 0.95)
    if weight_memory > usable:
        logger.warning(
            "Correction infeasible: model weights (%d bytes) exceed 95%% of GPU memory (%d bytes)",
            weight_memory,
            hardware.total_memory_bytes,
        )
        return False
    return True


def _apply_overrides(plan: DeploymentPlan, overrides: dict[str, Any]) -> DeploymentPlan:
    plan_updates: dict[str, Any] = {}
    config_updates: dict[str, Any] = {}
    resource_updates: dict[str, str] = {}
    removals: list[str] = []

    for key, value in overrides.items():
        if value is None:
            removals.append(key)
        elif key in TOP_LEVEL_FIELDS:
            plan_updates[key] = value
        elif key in RESOURCE_FIELDS:
            resource_updates[key] = str(value)
        else:
            config_updates[key] = str(value)

    if resource_updates:
        plan_updates["resource_allocation"] = {**plan.resource_allocation, **resource_updates}

    if config_updates or removals:
        new_config = {**plan.engine_configuration, **config_updates}
        for r in removals:
            new_config.pop(r, None)
        plan_updates["engine_configuration"] = new_config

    return plan.model_copy(update=plan_updates)


# -------------------------------------------------------------------
# Strategy implementations
# -------------------------------------------------------------------


def _reduce_memory_pressure(
    extracted: Mapping[str, int | float | str],
    plan: DeploymentPlan,
    model_config: dict[str, Any],
    hardware: HardwareSpec,
    vr: dict[str, Any] | None,
    **_kw: Any,
) -> dict[str, Any] | None:
    """Memory pressure after KV-cache sizing: reduce what the engine must hold.

    - KV cache cannot hold one full sequence: clamp ``max_model_len`` to
      ``estimated_max_model_len`` (v1/core/kv_cache_utils.py:876).
    - Sampler warm-up ran out of memory: halve ``max_num_seqs`` from
      ``max_num_seqs_attempted`` (v1/worker/gpu_model_runner.py:6373).
    Without either value there is no evidence-based correction: resetting
    gpu_memory_utilization to vLLM's own default would reproduce the failure.
    """
    if "estimated_max_model_len" in extracted:
        return {"max_model_len": str(int(extracted["estimated_max_model_len"]))}
    if "max_num_seqs_attempted" in extracted:
        halved = max(1, int(extracted["max_num_seqs_attempted"]) // 2)
        return {"max_num_seqs": str(halved)}
    return None


def _clamp_max_model_len(
    extracted: Mapping[str, int | float | str],
    plan: DeploymentPlan,
    model_config: dict[str, Any],
    hardware: HardwareSpec,
    vr: dict[str, Any] | None,
    **_kw: Any,
) -> dict[str, Any] | None:
    """Clamp to the derived maximum vLLM printed (config/model.py:2502), else
    to the resolved config's ``max_position_embeddings`` (F5).  No evidence,
    no correction — never a guessed default."""
    derived_max = extracted.get("derived_max")
    if derived_max is not None:
        return {"max_model_len": str(int(derived_max))}
    model_max = model_config.get("max_position_embeddings")
    if model_max is None:
        return None
    return {"max_model_len": str(int(model_max))}


def _fallback_dtype(
    extracted: Mapping[str, int | float | str],
    plan: DeploymentPlan,
    model_config: dict[str, Any],
    hardware: HardwareSpec,
    vr: dict[str, Any] | None,
    rule: dict[str, Any] | None = None,
    **_kw: Any,
) -> dict[str, Any] | None:
    """float16 refused for the model type: switch to bfloat16 where the GPU has it.

    Evidence only: the extracted ``unsupported_dtype`` and ``model_type``
    (config/model.py:2262) or the resolved config's ``model_type`` against
    the rule's float16 blocklist.  No evidence, no correction.
    """
    try:
        cc = float(hardware.compute_capability)
    except (ValueError, TypeError):
        cc = 0.0
    unsupported = str(extracted.get("unsupported_dtype", ""))
    model_type = str(extracted.get("model_type") or model_config.get("model_type") or "")
    blocklist = frozenset(rule.get("float16_blocklist", [])) if rule else frozenset()
    if unsupported != "float16" and model_type not in blocklist:
        return None
    if cc and cc < 8.0:
        return {"dtype": "float32"}  # no bfloat16 before SM80
    return {"dtype": "bfloat16"}


def _reduce_tensor_parallel(
    extracted: Mapping[str, int | float | str],
    plan: DeploymentPlan,
    model_config: dict[str, Any],
    hardware: HardwareSpec,
    vr: dict[str, Any] | None,
    **_kw: Any,
) -> dict[str, Any] | None:
    num_heads = int(extracted.get("num_heads", 0))
    if num_heads == 0:
        num_heads = int(model_config.get("num_attention_heads", 0))
    if num_heads == 0:
        return None
    num_kv_heads = int(
        model_config.get("num_kv_heads", model_config.get("num_key_value_heads", num_heads))
    )
    current_tp = plan.tensor_parallel
    # gpu_count from extraction or plan metadata. When unknown (0),
    # don't constrain — use head divisibility only. Level 3 tests
    # extract gpu_count from real error output; Level 2 tests verify
    # head-divisibility math without hardware constraints.
    # gpu_count is not printed by the divisibility error; it comes from the
    # plan's requested execution (resource_allocation).
    gpu_count = int(plan.resource_allocation.get("gpu_count", 0) or 0)

    for tp in range(current_tp - 1, 0, -1):
        if gpu_count > 0 and tp > gpu_count:
            continue
        if num_heads % tp == 0 and num_kv_heads % tp == 0:
            return {"tensor_parallel": tp}

    return {"tensor_parallel": 1}


def _remove_quantization(
    extracted: Mapping[str, int | float | str],
    plan: DeploymentPlan,
    model_config: dict[str, Any],
    hardware: HardwareSpec,
    vr: dict[str, Any] | None,
    **_kw: Any,
) -> dict[str, Any]:
    return {"quantization": None}


def _fallback_engine_config(
    extracted: Mapping[str, int | float | str],
    plan: DeploymentPlan,
    model_config: dict[str, Any],
    hardware: HardwareSpec,
    vr: dict[str, Any] | None,
    **_kw: Any,
) -> dict[str, Any] | None:
    for value in extracted.values():
        s = str(value).lower()
        if "enable-lora" in s or "enable_lora" in s:
            return {"enable_lora": "true"}
    # Remove any extracted field that matches an engine_configuration key
    for key in ("field_path", "flag", "get_args"):
        field = str(extracted.get(key, "")).replace("-", "_").lstrip("-")
        if field:
            if field in plan.engine_configuration:
                return {field: None}
            return {}
    return None


def _cheapest(entries: list[CatalogEntry]) -> CatalogEntry | None:
    return min(entries, key=lambda e: (e.hourly_rate, e.hardware.gpu_sku), default=None)


def _retarget_memory(
    extracted: Mapping[str, int | float | str],
    plan: DeploymentPlan,
    model_config: dict[str, Any],
    hardware: HardwareSpec,
    vr: dict[str, Any] | None,
    context: CorrectionContext | None = None,
    **_kw: Any,
) -> dict[str, Any] | None:
    """Weights do not fit: move to the cheapest GPU whose usable memory holds
    the calculator's predicted total for this plan (not the error text)."""
    if context is None or not context.predicted_total_bytes or not context.catalog:
        return None
    utilization = float(plan.engine_configuration.get("gpu_memory_utilization", "0.90"))
    fits = [
        e
        for e in context.catalog
        if e.hardware.gpu_sku != hardware.gpu_sku
        and int(e.hardware.total_memory_bytes * utilization) >= context.predicted_total_bytes
    ]
    choice = _cheapest(fits)
    return None if choice is None else {"gpu_sku": choice.hardware.gpu_sku}


def _capability_int(cc: str) -> int:
    major, _, minor = cc.partition(".")
    return int(major) * 10 + int(minor or 0)


def _retarget_capability(
    extracted: Mapping[str, int | float | str],
    plan: DeploymentPlan,
    model_config: dict[str, Any],
    hardware: HardwareSpec,
    vr: dict[str, Any] | None,
    rule: dict[str, Any] | None = None,
    context: CorrectionContext | None = None,
    **_kw: Any,
) -> dict[str, Any] | None:
    """Quantization needs a newer GPU: move to the cheapest GPU whose
    architecture the method's kernels were *built for*.

    ``cc >= min`` is not enough: an RTX 5090 (12.0) passes vLLM's check, but
    fp_quant's QuTLASS kernels are compiled for SM100 only, so it would crash.
    The rule's ``kernel_architectures`` names the built architectures per
    method (PLAN §10.1).  Without an entry for the method there is no
    evidence of where it runs, so there is no correction.
    """
    if context is None or not context.catalog:
        return None
    minimum = int(extracted.get("min_capability", 0) or 0)
    if minimum <= 0:
        return None
    quant = model_config.get("quantization_config") or {}
    method = str(quant.get("quant_method", "")) if isinstance(quant, dict) else ""
    built = tuple((rule or {}).get("kernel_architectures", {}).get(method, ()))
    if not built:
        return None
    candidates = [
        e
        for e in context.catalog
        if e.hardware.gpu_sku != hardware.gpu_sku
        and _capability_int(e.hardware.compute_capability) >= minimum
        and e.hardware.compute_capability in built
    ]
    choice = _cheapest(candidates)
    return None if choice is None else {"gpu_sku": choice.hardware.gpu_sku}


_STRATEGIES: dict[str, Any] = {
    "reduce_memory_pressure": _reduce_memory_pressure,
    "clamp_max_model_len": _clamp_max_model_len,
    "fallback_dtype": _fallback_dtype,
    "reduce_tensor_parallel": _reduce_tensor_parallel,
    "remove_quantization": _remove_quantization,
    "fallback_engine_config": _fallback_engine_config,
    "retarget_memory": _retarget_memory,
    "retarget_capability": _retarget_capability,
}

# Keys each strategy reads from the extraction (§10.3 rule<->strategy alignment).
STRATEGY_EXTRACTED_KEYS: dict[str, frozenset[str]] = {
    "reduce_memory_pressure": frozenset({"estimated_max_model_len", "max_num_seqs_attempted"}),
    "clamp_max_model_len": frozenset({"derived_max"}),
    "fallback_dtype": frozenset({"model_type", "unsupported_dtype"}),
    "reduce_tensor_parallel": frozenset({"num_heads"}),
    "retarget_memory": frozenset(),
    "retarget_capability": frozenset({"min_capability"}),
}
# Keys a strategy reads from the resolved config.json (F5) instead.
STRATEGY_CONFIG_KEYS: dict[str, frozenset[str]] = {
    "clamp_max_model_len": frozenset({"max_position_embeddings"}),
    "fallback_dtype": frozenset({"model_type"}),
    "reduce_tensor_parallel": frozenset({"num_attention_heads", "num_key_value_heads"}),
    "retarget_capability": frozenset({"quantization_config"}),
}


def apply_correction_spec(
    spec: dict[str, Any],
    extracted: Mapping[str, int | float | str],
    plan: DeploymentPlan,
) -> DeploymentPlan | None:
    """Apply a data-driven correction spec from a scanner-generated rule.

    Supports generic operations: set_field, remove_field, scale_field,
    use_alternative, fallback. Returns None if no action can be taken.
    """
    action = spec.get("action", "")
    field = spec.get("field", "")

    if action == "set_field" and field:
        source = spec.get("source", "")
        if source and source.startswith("extracted."):
            key = source.split(".", 1)[1]
            value = extracted.get(key)
            if value is not None:
                return _apply_overrides(plan, {field: str(value)})
        static_value = spec.get("value")
        if static_value is not None:
            return _apply_overrides(plan, {field: str(static_value)})

    if action == "remove_field" and field:
        return _apply_overrides(plan, {field: None})

    if action == "scale_field" and field:
        factor = float(spec.get("factor", 0.5))
        current = plan.engine_configuration.get(field)
        if current is not None:
            try:
                scaled = int(float(current) * factor)
                return _apply_overrides(plan, {field: str(max(1, scaled))})
            except (ValueError, TypeError):
                pass
        for _key, value in extracted.items():
            try:
                scaled = int(float(value) * factor)
                return _apply_overrides(plan, {field: str(max(1, scaled))})
            except (ValueError, TypeError):
                continue

    if action == "use_alternative":
        return None

    return None
