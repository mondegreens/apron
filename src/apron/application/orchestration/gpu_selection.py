"""GPU selection — pick the cheapest available GPU where the model fits.

Connects the calculator's prediction to the provider's available inventory.
The decision request's budget and optimization objective drive the choice.
"""

from __future__ import annotations

from typing import Any


def select_gpu(
    available_gpus: list[dict[str, Any]],
    model_metadata: dict[str, Any],
    planning_source: Any,
    budget_max_usd: float,
    estimated_duration_hours: float = 0.25,
) -> dict[str, Any] | None:
    """Select the cheapest GPU where the model fits within budget.

    Returns the selected GPU dict (with gpu_type_id, hourly_rate_usd,
    hardware_spec, and the calculator's prediction), or None if nothing fits.
    """
    candidates: list[dict[str, Any]] = []

    for gpu in available_gpus:
        hw = gpu["hardware_spec"]
        hourly = gpu["hourly_rate_usd"]
        estimated_cost = hourly * estimated_duration_hours

        if estimated_cost > budget_max_usd:
            continue

        workload_shape = model_metadata.get(
            "workload_shape", {"isl": 512, "osl": 128, "max_batch_size": 4}
        )
        claim = planning_source.predict(model_metadata, hw, workload_shape)

        if claim.proposed_configuration.get("status") == "unknown":
            continue

        total_required = claim.proposed_configuration.get("total_required_bytes", 0)
        gpu_usable = int(hw.total_memory_bytes * 0.90)

        if total_required > gpu_usable:
            continue

        candidates.append(
            {
                **gpu,
                "estimated_cost_usd": round(estimated_cost, 2),
                "prediction": claim.proposed_configuration,
                "headroom_bytes": gpu_usable - total_required,
            }
        )

    if not candidates:
        return None

    return min(candidates, key=lambda c: c["hourly_rate_usd"])
