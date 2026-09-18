"""Generic correction function — dispatches on strategy name.

Given a correction strategy (from a diagnosis rule), extracted values
(from error classification), and the current plan/hardware context,
compute a corrected DeploymentPlan. Pure computation, no I/O, no GPU.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from apron.domain.schemas.primitives import HardwareSpec
from apron.domain.schemas.solutions import DeploymentPlan

logger = logging.getLogger(__name__)

CORRECTION_BOUNDS: dict[str, tuple[int | float, int | float]] = {
    "max_model_len": (1, 1_000_000),
    "max_num_seqs": (1, 4096),
    "gpu_memory_utilization": (0.1, 0.95),
    "tensor_parallel": (1, 8),
}

TOP_LEVEL_FIELDS = frozenset({
    "tensor_parallel",
    "pipeline_parallel",
    "expert_parallel",
    "data_parallel",
    "dtype",
    "batch_size",
})


def compute_correction(
    strategy: str,
    extracted: Mapping[str, int | float | str],
    plan: DeploymentPlan,
    model_config: dict[str, Any],
    hardware: HardwareSpec,
    verification_report: dict[str, Any] | None,
) -> DeploymentPlan | None:
    """Compute a corrected plan by dispatching on strategy name.

    Returns None if the correction is infeasible (model doesn't fit).
    """
    dispatch = _STRATEGIES.get(strategy)
    if dispatch is None:
        msg = f"Unknown correction strategy: {strategy!r}"
        raise ValueError(msg)

    overrides = dispatch(extracted, plan, model_config, hardware, verification_report)
    if overrides is None:
        return None
    overrides = _validate_bounds(overrides)
    corrected = _apply_overrides(plan, overrides)
    if not _check_feasibility(corrected, hardware, verification_report):
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
) -> bool:
    """Check whether the corrected plan can physically fit on the hardware.

    Uses the verification report (when available) to compare model weight
    memory against GPU capacity. Returns False if infeasible.
    """
    if hardware.total_memory_bytes <= 0 or vr is None:
        return True
    weight_memory = vr.get("model_weight_memory", 0)
    if weight_memory <= 0:
        return True
    usable = int(hardware.total_memory_bytes * 0.95)
    if weight_memory > usable:
        logger.warning(
            "Correction infeasible: model weights (%d bytes) exceed "
            "95%% of GPU memory (%d bytes)",
            weight_memory,
            hardware.total_memory_bytes,
        )
        return False
    return True


def _apply_overrides(plan: DeploymentPlan, overrides: dict[str, Any]) -> DeploymentPlan:
    plan_updates: dict[str, Any] = {}
    config_updates: dict[str, Any] = {}
    removals: list[str] = []

    for key, value in overrides.items():
        if value is None:
            removals.append(key)
        elif key in TOP_LEVEL_FIELDS:
            plan_updates[key] = value
        else:
            config_updates[key] = str(value)

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
) -> dict[str, Any] | None:
    if "estimated_max_model_len" in extracted:
        return {"max_model_len": str(int(extracted["estimated_max_model_len"]))}
    if "max_num_seqs_attempted" in extracted:
        halved = max(1, int(extracted["max_num_seqs_attempted"]) // 2)
        return {"max_num_seqs": str(halved)}
    if vr is not None and hardware.total_memory_bytes > 0:
        weight_mem = vr.get("model_weight_memory", 0)
        if weight_mem > int(hardware.total_memory_bytes * 0.95):
            return None
    return {"gpu_memory_utilization": "0.90"}


def _clamp_max_model_len(
    extracted: Mapping[str, int | float | str],
    plan: DeploymentPlan,
    model_config: dict[str, Any],
    hardware: HardwareSpec,
    vr: dict[str, Any] | None,
) -> dict[str, Any]:
    derived_max = extracted.get("derived_max")
    if derived_max is not None:
        return {"max_model_len": str(int(derived_max))}
    return {"max_model_len": "4096"}


def _fallback_dtype(
    extracted: Mapping[str, int | float | str],
    plan: DeploymentPlan,
    model_config: dict[str, Any],
    hardware: HardwareSpec,
    vr: dict[str, Any] | None,
) -> dict[str, Any]:
    cc = hardware.compute_capability
    if cc < "8.0":
        return {"dtype": "float16"}
    unsupported = str(extracted.get("unsupported_dtype", ""))
    if unsupported == "float16":
        return {"dtype": "bfloat16"}
    return {"dtype": "bfloat16"}


def _reduce_tensor_parallel(
    extracted: Mapping[str, int | float | str],
    plan: DeploymentPlan,
    model_config: dict[str, Any],
    hardware: HardwareSpec,
    vr: dict[str, Any] | None,
) -> dict[str, Any]:
    num_heads = int(extracted.get("num_heads", 0))
    num_kv_heads = int(
        model_config.get("num_kv_heads", model_config.get("num_key_value_heads", num_heads))
    )
    current_tp = plan.tensor_parallel

    for tp in range(current_tp - 1, 0, -1):
        if num_heads % tp == 0 and num_kv_heads % tp == 0:
            return {"tensor_parallel": tp}

    return {"tensor_parallel": 1}


def _remove_quantization(
    extracted: Mapping[str, int | float | str],
    plan: DeploymentPlan,
    model_config: dict[str, Any],
    hardware: HardwareSpec,
    vr: dict[str, Any] | None,
) -> dict[str, Any]:
    return {"quantization": None}


def _fallback_engine_config(
    extracted: Mapping[str, int | float | str],
    plan: DeploymentPlan,
    model_config: dict[str, Any],
    hardware: HardwareSpec,
    vr: dict[str, Any] | None,
) -> dict[str, Any]:
    fix = str(extracted.get("suggested_fix", ""))
    if "enable-lora" in fix.lower():
        return {"enable_lora": "true"}
    return {}


_STRATEGIES: dict[str, Any] = {
    "reduce_memory_pressure": _reduce_memory_pressure,
    "clamp_max_model_len": _clamp_max_model_len,
    "fallback_dtype": _fallback_dtype,
    "reduce_tensor_parallel": _reduce_tensor_parallel,
    "remove_quantization": _remove_quantization,
    "fallback_engine_config": _fallback_engine_config,
}
