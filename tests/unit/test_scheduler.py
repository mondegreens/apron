"""§9.2 evidence-gap scheduler: skip measured, estimate cost, budget filter, rank by value."""

from __future__ import annotations

import pytest

from apron.application.orchestration.scheduler import (
    CandidateSeed,
    Coverage,
    estimate_cost,
    obligations_met,
    rank_candidates,
)

RATES = {"NVIDIA GeForce RTX 4090": 0.74, "NVIDIA L4": 0.49, "NVIDIA A100 80GB PCIe": 1.59}


def _seed(model: str, gpu: str = "NVIDIA GeForce RTX 4090", **kw) -> CandidateSeed:  # type: ignore[no-untyped-def]
    defaults = {
        "size_class": "small",
        "hardware_class": "consumer",
        "mechanism": "autoregressive_decode",
        "weight_gb": 4.0,
    }
    defaults.update(kw)
    return CandidateSeed(model_id=model, gpu_sku=gpu, **defaults)


def test_cost_estimate_is_weight_proportional_and_size_scaled() -> None:
    small = _seed("a", weight_gb=6.0)
    large = _seed("b", weight_gb=66.0, size_class="large")
    assert estimate_cost(small, 0.74) == pytest.approx((1 + 15) / 60 * 0.74, abs=1e-3)
    assert estimate_cost(large, 0.74) == pytest.approx((11 + 35) / 60 * 0.74, abs=1e-3)


def test_multi_gpu_cost_scales_with_count() -> None:
    one = _seed("a")
    four = _seed("a", gpu_count=4)
    assert estimate_cost(four, 0.74) == pytest.approx(4 * estimate_cost(one, 0.74), rel=1e-3)


def test_prediction_error_candidates_cost_nothing() -> None:
    assert estimate_cost(_seed("m", prediction_error=True, mechanism="unknown"), 0.74) == 0.0


def test_measured_candidates_are_skipped() -> None:
    seed = _seed("a")
    ranking = rank_candidates(
        [seed], Coverage(), measured=[seed.key], remaining_budget=10, rates=RATES
    )
    assert ranking.ranked == []
    assert ranking.skipped == [(seed, "already measured")]


def test_unaffordable_candidates_are_skipped_with_a_reason() -> None:
    big = _seed("big", weight_gb=400.0, size_class="large")
    ranking = rank_candidates([big], Coverage(), measured=[], remaining_budget=0.5, rates=RATES)
    assert ranking.ranked == []
    assert "exceeds remaining" in ranking.skipped[0][1]


def test_missing_obligations_rank_first_then_features_then_cost() -> None:
    covered = Coverage(
        sizes=frozenset({"small"}),
        hardware=frozenset({"consumer"}),
        mechanisms=frozenset({"autoregressive_decode"}),
        models=frozenset({"x"}),
        gpus=frozenset({"NVIDIA GeForce RTX 4090"}),
    )
    redundant = _seed("x")
    mla = _seed(
        "y", mechanism="mla_decode", size_class="large", weight_gb=31.0, features=("mla", "moe")
    )
    cheap_new_gpu = _seed("x", gpu="NVIDIA L4", hardware_class="datacenter")
    ranking = rank_candidates(
        [redundant, cheap_new_gpu, mla], covered, measured=[], remaining_budget=50, rates=RATES
    )
    order = [r.seed.key for r in ranking.ranked]
    assert order[0] == mla.key  # size + mechanism + new model + new gpu... most obligations
    assert order[-1] == redundant.key  # nothing new
    assert "mechanism:mla_decode" in ranking.ranked[0].obligations


def test_ranking_is_greedy_over_updated_coverage() -> None:
    a = _seed("a", gpu="NVIDIA L4", hardware_class="datacenter")
    b = _seed("b", gpu="NVIDIA L4", hardware_class="datacenter")
    ranking = rank_candidates([a, b], Coverage(), measured=[], remaining_budget=50, rates=RATES)
    second = ranking.ranked[1]
    assert "hardware:datacenter" not in second.obligations  # the first already covered it


def test_no_name_based_preference() -> None:
    """Swapping names leaves the ranking shape unchanged (exit gate 7)."""
    s1 = [_seed("zeta"), _seed("alpha", gpu="NVIDIA L4", hardware_class="datacenter")]
    s2 = [_seed("alpha"), _seed("zeta", gpu="NVIDIA L4", hardware_class="datacenter")]
    r1 = rank_candidates(s1, Coverage(), measured=[], remaining_budget=50, rates=RATES)
    r2 = rank_candidates(s2, Coverage(), measured=[], remaining_budget=50, rates=RATES)
    assert [r.seed.gpu_sku for r in r1.ranked] == [r.seed.gpu_sku for r in r2.ranked]


def test_obligations_include_quantization_and_prediction_error() -> None:
    q = _seed("q", quantized=True)
    p = _seed("p", prediction_error=True, mechanism="unknown")
    assert "quantization" in obligations_met(q, Coverage())
    assert "prediction_error" in obligations_met(p, Coverage())
    assert "new_gpu" not in obligations_met(p, Coverage())


def test_gpu_without_rate_is_skipped() -> None:
    seed = _seed("a", gpu="NVIDIA Imaginary")
    ranking = rank_candidates([seed], Coverage(), measured=[], remaining_budget=50, rates=RATES)
    assert ranking.skipped[0][1].startswith("no rate")
