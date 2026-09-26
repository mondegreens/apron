"""§6.2: the offline rate table is keyed by GPU_SPECS names and covers every SKU."""

from __future__ import annotations

import pytest

from apron.adapters.backends.runpod import GPU_SPECS
from apron.application.cost_estimator import PROVIDER_RATES, estimate_cost, hourly_rate


def test_every_gpu_spec_has_an_offline_rate() -> None:
    assert set(GPU_SPECS) <= set(PROVIDER_RATES["runpod"])


def test_no_rate_for_an_unknown_sku() -> None:
    assert set(PROVIDER_RATES["runpod"]) <= set(GPU_SPECS)


def test_live_rate_wins_and_cost_scales_with_gpu_count() -> None:
    rate, source = hourly_rate(
        "runpod",
        "NVIDIA GeForce RTX 4090",
        live_rates={"NVIDIA GeForce RTX 4090": 0.69},
        gpu_count=4,
    )
    assert rate == pytest.approx(2.76)
    assert source == "provider_api"
    rate, source = hourly_rate("runpod", "NVIDIA GeForce RTX 4090", gpu_count=4)
    assert rate == pytest.approx(2.96)
    assert source.startswith("offline_table_")


def test_unknown_sku_raises_instead_of_costing_zero() -> None:
    with pytest.raises(KeyError):
        hourly_rate("runpod", "NVIDIA Imaginary 9000")


def test_estimate_cost_multiplies_by_gpu_count() -> None:
    plan = {
        "resource_allocation": {
            "gpu_sku": "NVIDIA GeForce RTX 4090",
            "gpu_count": "4",
            "weight_bytes": "16000000000",
        }
    }
    result = estimate_cost(plan, "runpod")
    assert result["hourly_rate_usd"] == pytest.approx(2.96)
    assert result["gpu_count"] == 4
