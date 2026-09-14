"""Conformance suite: EvaluationAdapter Protocol."""

from apron.domain.protocols import EvaluationAdapter


def test_satisfies_protocol(evaluation_adapter):
    assert isinstance(evaluation_adapter, EvaluationAdapter)


def test_accepts_returns_bool(evaluation_adapter):
    result = evaluation_adapter.accepts({"harness": "inspect_ai"})
    assert isinstance(result, bool)


def test_accepts_rejects_unknown_harness(evaluation_adapter):
    result = evaluation_adapter.accepts({"harness": "nonexistent_harness"})
    assert result is False


def test_prepare_returns_nonempty_dict(evaluation_adapter):
    result = evaluation_adapter.prepare({"harness": "inspect_ai", "scorer": "exact_match"})
    assert isinstance(result, dict)
    assert len(result) > 0


def test_execute_returns_list_of_dicts_with_score(evaluation_adapter):
    results = evaluation_adapter.execute(
        {"harness": "inspect_ai"},
        "http://localhost:8000",
    )
    assert isinstance(results, list)
    assert len(results) > 0
    for r in results:
        assert "score" in r


def test_failed_attempt_is_preserved(evaluation_adapter):
    results = evaluation_adapter.execute(
        {"harness": "inspect_ai"},
        "http://localhost:8000",
    )
    failed = [r for r in results if not r.get("accepted", True)]
    assert len(failed) > 0


def test_collect_aggregates_scores(evaluation_adapter):
    attempts = [
        {"case_id": "c1", "score": 1.0},
        {"case_id": "c2", "score": 0.5},
        {"case_id": "c3", "score": 0.0},
    ]
    collected = evaluation_adapter.collect(attempts)
    assert isinstance(collected, dict)
    assert "aggregate_score" in collected
    assert "total_attempts" in collected
    assert collected["total_attempts"] == 3
