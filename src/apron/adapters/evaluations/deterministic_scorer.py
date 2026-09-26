"""Deterministic exact-match evaluation adapter.

Implements EvaluationAdapter Protocol (domain/protocols.py).
Sends cases sequentially with the protocol's temperature and seed
(defaults 0 and 42) for deterministic output.

Normalization is declared by the protocol's ``deterministic_checks`` (M9):
each named check is applied to both output and expected value before exact
comparison, and an unknown check name is an error, not a silent no-op.
``chat_template_kwargs`` is sent only when the caller passes it — i.e. when
the model's chat template accepts it (§9.1 step 4).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_TERMINAL_PUNCTUATION = ".!?。"


def _whitespace(text: str) -> str:
    return " ".join(text.split())


def _strip_terminal_punctuation(text: str) -> str:
    return text.rstrip(_TERMINAL_PUNCTUATION)


def _casefold(text: str) -> str:
    return text.casefold()


NORMALIZATIONS = {
    "whitespace_normalized_exact_match": _whitespace,
    "strip_terminal_punctuation": _strip_terminal_punctuation,
    "casefold": _casefold,
}
DEFAULT_CHECKS = ("whitespace_normalized_exact_match",)


def normalize(text: str, checks: tuple[str, ...]) -> str:
    for check in checks:
        text = NORMALIZATIONS[check](text)
    return text


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
        checks = tuple(protocol.get("deterministic_checks") or DEFAULT_CHECKS)
        unknown = [c for c in checks if c not in NORMALIZATIONS]
        if unknown:
            raise ValueError(f"Unknown deterministic checks: {unknown}")
        seeds = protocol.get("seeds") or [42]
        prepared: dict[str, Any] = {
            "cases": cases,
            "model_id": protocol.get("model_id", ""),
            "endpoint": protocol.get("endpoint", ""),
            "deterministic_checks": list(checks),
            "temperature": protocol.get("sampling_temperature") or 0,
            "seed": seeds[0],
        }
        if protocol.get("chat_template_kwargs"):
            prepared["chat_template_kwargs"] = dict(protocol["chat_template_kwargs"])
        return prepared

    def execute(self, protocol: dict[str, Any], endpoint: str) -> list[dict[str, Any]]:
        cases = protocol.get("cases", [])
        model_id = protocol.get("model_id", "")
        attempts: list[dict[str, Any]] = []

        for case in cases:
            attempt = self._execute_case(case, model_id, endpoint, protocol)
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

    def _execute_case(
        self,
        case: dict[str, Any],
        model_id: str,
        endpoint: str,
        protocol: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        protocol = protocol or {}
        case_id = case.get("id", "")
        prompt = case.get("prompt", "")
        expected = case.get("expected", "")
        # Strict task suites carry string values; the API wants an integer.
        max_tokens = int(case.get("max_tokens", 128))
        checks = tuple(protocol.get("deterministic_checks") or DEFAULT_CHECKS)

        body: dict[str, Any] = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": protocol.get("temperature", 0),
            "seed": protocol.get("seed", 42),
        }
        if protocol.get("chat_template_kwargs"):
            body["chat_template_kwargs"] = protocol["chat_template_kwargs"]

        start = time.monotonic()
        try:
            response = httpx.post(
                f"{endpoint}/v1/chat/completions",
                json=body,
                timeout=self._timeout,
            )
            response.raise_for_status()
            elapsed = time.monotonic() - start

            data = response.json()
            output = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            normalized_output = normalize(output or "", checks)
            normalized_expected = normalize(expected, checks)
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
