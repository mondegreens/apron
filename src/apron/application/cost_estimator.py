"""Cost estimation from plan + target provider."""

from typing import Any

PROVIDER_RATES: dict[str, dict[str, float]] = {
    "runpod": {"RTX 4090": 0.74, "A100 SXM": 1.59, "H100 SXM": 3.49},
}


def estimate_cost(
    plan_data: dict[str, Any],
    provider: str,
    estimated_duration_minutes: float | None = None,
) -> dict[str, Any]:
    gpu_sku = plan_data.get("resource_allocation", {}).get("gpu_sku", "")
    hourly = PROVIDER_RATES.get(provider, {}).get(gpu_sku, 0.0)

    if estimated_duration_minutes is None:
        weight_str = plan_data.get("resource_allocation", {}).get("weight_bytes", "0")
        weight_gb = int(weight_str) / 1e9 if weight_str else 0
        if weight_gb < 20:
            estimated_duration_minutes = 15.0
        elif weight_gb < 150:
            estimated_duration_minutes = 30.0
        else:
            estimated_duration_minutes = 60.0

    return {
        "estimated_cost_usd": round(hourly * estimated_duration_minutes / 60, 2),
        "hourly_rate_usd": hourly,
        "estimated_duration_minutes": estimated_duration_minutes,
        "provider": provider,
    }
