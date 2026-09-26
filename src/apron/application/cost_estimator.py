"""Cost estimation from plan + target provider.

Rates are read from the provider API at dispatch time (phase-plan GPU-dollar
rule); ``PROVIDER_RATES`` is only the offline fallback.  It is keyed by the
same GPU type IDs as the RunPod adapter's ``GPU_SPECS`` and covers every SKU
there (``tests/unit/test_cost_rates_cover_specs.py``).  The estimate is stored
beside the actual cost.
"""

from typing import Any

# RunPod Secure on-demand $/GPU-hour, from https://www.runpod.io/pricing,
# retrieved 2026-09-26.  Offline fallback only.
PROVIDER_RATES: dict[str, dict[str, float]] = {
    "runpod": {
        "NVIDIA GeForce RTX 4090": 0.74,
        "NVIDIA GeForce RTX 3090": 0.50,
        "NVIDIA RTX A5000": 0.27,
        "NVIDIA L4": 0.49,
        "NVIDIA RTX A6000": 0.53,
        "NVIDIA L40": 0.82,
        "NVIDIA A100 80GB PCIe": 1.59,
        "NVIDIA A100-SXM4-80GB": 1.59,
        "NVIDIA H100 80GB HBM3": 3.49,
        "NVIDIA B200": 6.79,
    },
}
RATES_RETRIEVED = "2026-09-26"


def hourly_rate(
    provider: str,
    gpu_sku: str,
    *,
    live_rates: dict[str, float] | None = None,
    gpu_count: int = 1,
) -> tuple[float, str]:
    """Pod $/hour for *gpu_count* GPUs, and where the per-GPU rate came from.

    The live API rate wins; the offline table is the fallback.  Unknown SKUs
    raise instead of silently costing zero.
    """
    if live_rates and gpu_sku in live_rates:
        return live_rates[gpu_sku] * gpu_count, "provider_api"
    table = PROVIDER_RATES.get(provider, {})
    if gpu_sku not in table:
        raise KeyError(f"no rate for {provider}/{gpu_sku}")
    return table[gpu_sku] * gpu_count, f"offline_table_{RATES_RETRIEVED}"


def estimate_cost(
    plan_data: dict[str, Any],
    provider: str,
    estimated_duration_minutes: float | None = None,
) -> dict[str, Any]:
    allocation = plan_data.get("resource_allocation", {})
    gpu_sku = allocation.get("gpu_sku", "")
    gpu_count = int(allocation.get("gpu_count", "1") or 1)
    per_gpu = PROVIDER_RATES.get(provider, {}).get(gpu_sku, 0.0)
    hourly = per_gpu * gpu_count

    if estimated_duration_minutes is None:
        weight_str = allocation.get("weight_bytes", "0")
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
        "gpu_count": gpu_count,
        "estimated_duration_minutes": estimated_duration_minutes,
        "provider": provider,
    }
