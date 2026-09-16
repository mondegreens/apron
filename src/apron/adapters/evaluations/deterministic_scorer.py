"""Deterministic exact-match evaluation adapter.

Implements EvaluationAdapter Protocol (domain/protocols.py).
Sends cases sequentially with temperature=0 and seed=42
for deterministic output.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class DeterministicScorer:
    """EvaluationAdapter for deterministic exact-match scoring."""

    _harness_name = "deterministic_exact_match"
    _harness_version = "0.1"

    def __init__(self, timeout: int = 60) -> None:
        self._timeout = timeout

    @property
    def harness_name(self) -> str:
        return self._harness_name

    @property
    def harness_version(self) -> str:
        return self._harness_version

    def accepts(self, protocol: dict[str, Any]) -> bool:
        return protocol.get("scorer_type") == "deterministic_exact_match"

    def prepare(self, protocol: dict[str, Any]) -> dict[str, Any]:
        cases = protocol.get("cases", [])
        for case in cases:
            if "expected" not in case:
                raise ValueError(f"Case {case.get('id', '<unknown>')} missing 'expected' field")
        return {
            "cases": cases,
            "model_id": protocol.get("model_id", ""),
            "endpoint": protocol.get("endpoint", ""),
        }

    def execute(self, protocol: dict[str, Any], endpoint: str) -> list[dict[str, Any]]:
        cases = protocol.get("cases", [])
        model_id = protocol.get("model_id", "")
        attempts: list[dict[str, Any]] = []

        for case in cases:
            attempt = self._execute_case(case, model_id, endpoint)
            attempts.append(attempt)

        return attempts

    def collect(self, attempts: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(attempts)
        passed = sum(1 for a in attempts if a.get("score") == 1)
        return {
            "aggregate_score": passed / total if total > 0 else 0.0,
            "total_attempts": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "attempts": attempts,
        }

    def _execute_case(self, case: dict[str, Any], model_id: str, endpoint: str) -> dict[str, Any]:
        case_id = case.get("id", "")
        prompt = case.get("prompt", "")
        expected = case.get("expected", "")
        max_tokens = case.get("max_tokens", 128)

        start = time.monotonic()
        try:
            response = httpx.post(
                f"{endpoint}/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "seed": 42,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            elapsed = time.monotonic() - start

            data = response.json()
            output = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            normalized_output = " ".join(output.split())
            normalized_expected = " ".join(expected.split())
            score = 1 if normalized_output == normalized_expected else 0

            return {
                "case_id": case_id,
                "output": output,
                "expected": expected,
                "score": score,
                "accepted": score == 1,
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "time_seconds": round(elapsed, 3),
                "status": "completed",
            }
        except Exception as exc:
            elapsed = time.monotonic() - start
            return {
                "case_id": case_id,
                "score": 0,
                "accepted": False,
                "status": "failed",
                "error": str(exc),
                "time_seconds": round(elapsed, 3),
            }
