"""Unit tests for DeterministicScorer EvaluationAdapter.

Collects the EvaluationAdapter conformance suite via pytest_plugins.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from apron.adapters.evaluations.deterministic_scorer import DeterministicScorer
from apron.domain.protocols import EvaluationAdapter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def scorer() -> DeterministicScorer:
    return DeterministicScorer()


def _make_fake_response(text: str, prompt_tokens: int = 10, completion_tokens: int = 5) -> Any:
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {
        "choices": [{"text": text}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }
    mock.raise_for_status = MagicMock()
    return mock


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_evaluation_adapter_protocol(scorer: DeterministicScorer) -> None:
    assert isinstance(scorer, EvaluationAdapter)


def test_harness_name(scorer: DeterministicScorer) -> None:
    assert scorer.harness_name == "deterministic_exact_match"


def test_harness_version(scorer: DeterministicScorer) -> None:
    assert scorer.harness_version == "0.1"


# ---------------------------------------------------------------------------
# accepts
# ---------------------------------------------------------------------------


def test_accepts_deterministic(scorer: DeterministicScorer) -> None:
    assert scorer.accepts({"scorer_type": "deterministic_exact_match"}) is True


def test_rejects_other_scorer(scorer: DeterministicScorer) -> None:
    assert scorer.accepts({"scorer_type": "llm_judge"}) is False


def test_rejects_missing_scorer_type(scorer: DeterministicScorer) -> None:
    assert scorer.accepts({}) is False


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def test_prepare_valid_cases(scorer: DeterministicScorer) -> None:
    protocol = {
        "cases": [{"id": "c1", "prompt": "p", "expected": "e"}],
        "model_id": "m",
    }
    result = scorer.prepare(protocol)
    assert result["cases"] == protocol["cases"]


def test_prepare_rejects_missing_expected(scorer: DeterministicScorer) -> None:
    protocol = {"cases": [{"id": "c1", "prompt": "p"}]}
    with pytest.raises(ValueError, match="missing 'expected'"):
        scorer.prepare(protocol)


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


def test_execute_correct_answer(scorer: DeterministicScorer) -> None:
    protocol = {
        "cases": [{"id": "c1", "prompt": "2+2?", "expected": "4", "max_tokens": 8}],
        "model_id": "test-model",
    }
    with patch("apron.adapters.evaluations.deterministic_scorer.httpx.post") as mock_post:
        mock_post.return_value = _make_fake_response("4")
        results = scorer.execute(protocol, "http://localhost:8000")

    assert len(results) == 1
    assert results[0]["score"] == 1
    assert results[0]["accepted"] is True
    assert results[0]["status"] == "completed"


def test_execute_wrong_answer(scorer: DeterministicScorer) -> None:
    protocol = {
        "cases": [{"id": "c1", "prompt": "2+2?", "expected": "4", "max_tokens": 8}],
        "model_id": "test-model",
    }
    with patch("apron.adapters.evaluations.deterministic_scorer.httpx.post") as mock_post:
        mock_post.return_value = _make_fake_response("5")
        results = scorer.execute(protocol, "http://localhost:8000")

    assert results[0]["score"] == 0
    assert results[0]["accepted"] is False


def test_execute_whitespace_normalization(scorer: DeterministicScorer) -> None:
    protocol = {
        "cases": [{"id": "c1", "prompt": "q", "expected": "hello world", "max_tokens": 8}],
        "model_id": "test-model",
    }
    with patch("apron.adapters.evaluations.deterministic_scorer.httpx.post") as mock_post:
        mock_post.return_value = _make_fake_response("  hello   world  ")
        results = scorer.execute(protocol, "http://localhost:8000")

    assert results[0]["score"] == 1


def test_execute_failed_request_preserved(scorer: DeterministicScorer) -> None:
    protocol = {
        "cases": [{"id": "c1", "prompt": "q", "expected": "e", "max_tokens": 8}],
        "model_id": "test-model",
    }
    with patch("apron.adapters.evaluations.deterministic_scorer.httpx.post") as mock_post:
        mock_post.side_effect = ConnectionError("refused")
        results = scorer.execute(protocol, "http://localhost:8000")

    assert len(results) == 1
    assert results[0]["score"] == 0
    assert results[0]["status"] == "failed"
    assert "refused" in results[0]["error"]


def test_execute_multiple_cases(scorer: DeterministicScorer) -> None:
    protocol = {
        "cases": [
            {"id": "c1", "prompt": "2+2?", "expected": "4", "max_tokens": 8},
            {"id": "c2", "prompt": "10*5?", "expected": "50", "max_tokens": 8},
            {"id": "c3", "prompt": "capital?", "expected": "Paris", "max_tokens": 8},
        ],
        "model_id": "test-model",
    }
    responses = [
        _make_fake_response("4"),
        _make_fake_response("50"),
        _make_fake_response("London"),
    ]
    with patch("apron.adapters.evaluations.deterministic_scorer.httpx.post") as mock_post:
        mock_post.side_effect = responses
        results = scorer.execute(protocol, "http://localhost:8000")

    assert len(results) == 3
    assert results[0]["score"] == 1
    assert results[1]["score"] == 1
    assert results[2]["score"] == 0


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------


def test_collect_aggregates(scorer: DeterministicScorer) -> None:
    attempts = [
        {"case_id": "c1", "score": 1},
        {"case_id": "c2", "score": 1},
        {"case_id": "c3", "score": 0},
    ]
    result = scorer.collect(attempts)
    assert result["total_attempts"] == 3
    assert result["passed"] == 2
    assert result["failed"] == 1
    assert abs(result["pass_rate"] - 2 / 3) < 0.01


def test_collect_empty(scorer: DeterministicScorer) -> None:
    result = scorer.collect([])
    assert result["total_attempts"] == 0
    assert result["pass_rate"] == 0.0


def test_collect_all_pass(scorer: DeterministicScorer) -> None:
    attempts = [{"case_id": f"c{i}", "score": 1} for i in range(5)]
    result = scorer.collect(attempts)
    assert result["pass_rate"] == 1.0


# ---------------------------------------------------------------------------
# Determinism: temperature and seed
# ---------------------------------------------------------------------------


def test_execute_sends_temperature_zero_and_seed(scorer: DeterministicScorer) -> None:
    protocol = {
        "cases": [{"id": "c1", "prompt": "q", "expected": "e", "max_tokens": 8}],
        "model_id": "test-model",
    }
    with patch("apron.adapters.evaluations.deterministic_scorer.httpx.post") as mock_post:
        mock_post.return_value = _make_fake_response("e")
        scorer.execute(protocol, "http://localhost:8000")

    call_args = mock_post.call_args
    body = call_args.kwargs.get("json") or call_args[1].get("json")
    assert body["temperature"] == 0
    assert body["seed"] == 42
