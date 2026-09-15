"""GPU selection — pick the cheapest available GPU where the model fits.

Connects the calculator's prediction to the provider's available inventory.
The decision request's budget and optimization objective drive the choice.
"""

from __future__ import annotations

from typing import Any


def select_gpu(
    available_gpus: list[dict[str, Any]],
    prediction: dict[str, Any],
    budget_max_usd: float,
    estimated_duration_hours: float = 0.25,
) -> dict[str, Any] | None:
    """Select the cheapest GPU where the model fits within budget.

    ``prediction`` is the calculator's proposed_configuration dict
    containing ``total_required_bytes``. The total is hardware-independent
    for TP=1 — weights, KV cache, activation and overhead don't change
    across GPUs. Only the "does it fit?" threshold changes.

    Returns the selected GPU dict or None if nothing fits.
    """
    total_required = prediction.get("total_required_bytes", 0)
    if total_required <= 0:
        return None

    candidates: list[dict[str, Any]] = []

    for gpu in available_gpus:
        hw = gpu["hardware_spec"]
        hourly = gpu["hourly_rate_usd"]
        estimated_cost = hourly * estimated_duration_hours

        if estimated_cost > budget_max_usd:
            continue

        gpu_usable = int(hw.total_memory_bytes * 0.90)
        if total_required > gpu_usable:
            continue

        candidates.append(
            {
                **gpu,
                "estimated_cost_usd": round(estimated_cost, 2),
                "headroom_bytes": gpu_usable - total_required,
            }
        )

    if not candidates:
        return None

    return min(candidates, key=lambda c: c["hourly_rate_usd"])
