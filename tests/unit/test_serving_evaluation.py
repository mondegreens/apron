"""evaluate_serving_slos: pass, TTFT fail, TPOT fail, none declared, zero
completions, high failure rate, missing metric (§13)."""

from __future__ import annotations

from typing import Any

from apron.application.orchestration.serving import (
    MAX_FAILURE_RATE,
    evaluate_serving_slos,
    latency_key,
)
from apron.domain.schemas.records import VerificationReport
from apron.domain.schemas.tasks import ServingWorkloadSpec

_FP = "1220" + "ab" * 32
_SLO = ServingWorkloadSpec(
    input_sequence_length=512,
    output_sequence_length=128,
    concurrency=4,
    p99_ttft_ms=2000,
    p99_tpot_ms=100,
)


def _report(**overrides: Any) -> VerificationReport:
    data: dict[str, Any] = {
        "target_kind": "rented-provider",
        "operator": "apron",
        "execution_fingerprint": _FP,
        "serving_completed": 50,
        "serving_failed": 0,
        "serving_latency_ms": {"p99_ttft_ms": 800.0, "p99_tpot_ms": 30.0},
        "claim_scope": "serving_performance",
        "production_mode": False,
        "reason": "serving_measurement",
        "lifecycle": "observed",
    }
    data.update(overrides)
    return VerificationReport(**data)


def test_pass_within_slo() -> None:
    result = evaluate_serving_slos(_report(), _SLO)
    assert result.verdict == "pass"
    assert result.passed


def test_ttft_fail() -> None:
    report = _report(serving_latency_ms={"p99_ttft_ms": 2400.0, "p99_tpot_ms": 30.0})
    result = evaluate_serving_slos(report, _SLO)
    assert result.verdict == "fail"
    assert result.reasons == ("p99_ttft_ms 2400.0 exceeds SLO 2000.0",)


def test_tpot_fail() -> None:
    report = _report(serving_latency_ms={"p99_ttft_ms": 800.0, "p99_tpot_ms": 150.0})
    result = evaluate_serving_slos(report, _SLO)
    assert result.verdict == "fail"
    assert result.reasons == ("p99_tpot_ms 150.0 exceeds SLO 100.0",)


def test_none_declared_is_a_measurement_not_a_pass() -> None:
    result = evaluate_serving_slos(_report(), ServingWorkloadSpec(concurrency=4))
    assert result.verdict == "none_declared"
    assert result.passed


def test_zero_completions_fail() -> None:
    result = evaluate_serving_slos(_report(serving_completed=0, serving_failed=50), _SLO)
    assert result.verdict == "fail"
    assert result.reasons == ("zero completed requests",)


def test_zero_completions_fail_even_without_slo() -> None:
    result = evaluate_serving_slos(_report(serving_completed=0), ServingWorkloadSpec())
    assert result.verdict == "fail"


def test_failure_rate_over_threshold_fails() -> None:
    result = evaluate_serving_slos(_report(serving_completed=39, serving_failed=11), _SLO)
    assert result.verdict == "fail"
    assert "exceeds 20%" in result.reasons[0]


def test_failure_rate_at_threshold_is_allowed() -> None:
    completed, failed = 40, 10
    assert failed / (completed + failed) == MAX_FAILURE_RATE
    result = evaluate_serving_slos(
        _report(serving_completed=completed, serving_failed=failed), _SLO
    )
    assert result.verdict == "pass"


def test_missing_metric_fails() -> None:
    report = _report(serving_latency_ms={"p99_ttft_ms": 800.0})
    result = evaluate_serving_slos(report, _SLO)
    assert result.verdict == "fail"
    assert result.reasons == ("p99_tpot_ms not measured but p99_tpot_ms is declared",)


def test_non_serving_report_fails() -> None:
    result = evaluate_serving_slos(_report(claim_scope="memory"), _SLO)
    assert result.verdict == "fail"


def test_latency_key_format() -> None:
    assert latency_key(99, "ttft") == "p99_ttft_ms"
    assert latency_key(50, "e2el") == "p50_e2el_ms"
