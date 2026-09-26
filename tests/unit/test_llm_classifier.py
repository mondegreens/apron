"""Tests for the LLM classifier — no API calls required.

Tests tool schema construction, evidence verification, system prompt
building, and caching. The actual LLM call is tested separately via
an ANTHROPIC_API_KEY-gated integration test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apron.adapters.backends.llm_classifier import (
    _build_extraction_tool,
    _build_system_prompt,
    _classify_tool,
    _verify_evidence,
)
from apron.adapters.backends.rule_loader import load_rules
from apron.domain.diagnosis import build_extraction_schemas, build_failure_classes


@pytest.fixture()
def rules() -> list[dict]:
    rules_dir = Path(__file__).parents[2] / "rules"
    return load_rules(rules_dir, "vllm", "v0.29")


@pytest.fixture()
def classes(rules: list[dict]) -> list[str]:
    return build_failure_classes(rules)


@pytest.fixture()
def schemas(rules: list[dict]) -> dict:
    return build_extraction_schemas(rules)


class TestToolSchemaConstruction:
    def test_classify_tool_has_required_fields(self, classes: list[str]) -> None:
        tool = _classify_tool(classes)
        schema = tool["input_schema"]
        assert "failure_class" in schema["properties"]
        assert "confidence" in schema["properties"]
        assert "evidence_span" in schema["properties"]
        assert schema["properties"]["failure_class"]["enum"] == classes

    def test_classify_tool_enum_includes_unknown(self, classes: list[str]) -> None:
        assert "unknown" in classes

    def test_classify_tool_enum_includes_all_rule_families(
        self, rules: list[dict], classes: list[str]
    ) -> None:
        families = {r["error_family"] for r in rules}
        for f in families:
            assert f in classes

    def test_extraction_tool_has_class_specific_fields(
        self, classes: list[str], schemas: dict
    ) -> None:
        tool = _build_extraction_tool("tp_divisibility", classes, schemas)
        props = tool["input_schema"]["properties"]
        tp_fields = schemas.get("tp_divisibility", [])
        for field_name, _ in tp_fields:
            assert field_name in props
            assert f"{field_name}_evidence" in props

    def test_extraction_tool_is_valid_json_schema(self, classes: list[str], schemas: dict) -> None:
        for cls in schemas:
            tool = _build_extraction_tool(cls, classes, schemas)
            dumped = json.dumps(tool)
            loaded = json.loads(dumped)
            assert loaded["name"] == "classify_and_extract"


class TestSystemPrompt:
    def test_prompt_lists_all_classes(self, rules: list[dict]) -> None:
        prompt = _build_system_prompt(rules)
        families = {r["error_family"] for r in rules}
        for f in families:
            assert f in prompt

    def test_prompt_includes_unknown(self, rules: list[dict]) -> None:
        prompt = _build_system_prompt(rules)
        assert "unknown" in prompt

    def test_prompt_includes_examples(self, rules: list[dict]) -> None:
        prompt = _build_system_prompt(rules)
        assert len(prompt) > 200


class TestEvidenceVerification:
    def test_verified_value_kept(self, schemas: dict) -> None:
        extracted = {
            "failure_class": "tp_divisibility",
            "num_heads": 28,
            "num_heads_evidence": "attention heads (28)",
            "tp_size": 4,
            "tp_size_evidence": "tensor parallel size (4)",
        }
        error = (
            "Total number of attention heads (28) must be divisible by tensor parallel size (4)."
        )
        result = _verify_evidence(extracted, error, schemas)
        assert result["num_heads"] == 28
        assert result["tp_size"] == 4

    def test_unverified_value_removed(self, schemas: dict) -> None:
        extracted = {
            "failure_class": "tp_divisibility",
            "num_heads": 28,
            "num_heads_evidence": "THIS STRING IS NOT IN THE ERROR",
        }
        error = "some error text"
        result = _verify_evidence(extracted, error, schemas)
        assert result["num_heads"] is None

    def test_null_evidence_removes_value(self, schemas: dict) -> None:
        extracted = {
            "failure_class": "tp_divisibility",
            "num_heads": 28,
            "num_heads_evidence": None,
        }
        error = "some error"
        result = _verify_evidence(extracted, error, schemas)
        assert result["num_heads"] is None

    def test_null_value_kept_as_null(self, schemas: dict) -> None:
        extracted = {
            "failure_class": "tp_divisibility",
            "num_heads": None,
        }
        error = "some error"
        result = _verify_evidence(extracted, error, schemas)
        assert result["num_heads"] is None


class TestClassificationCache:
    def test_cache_returns_same_result(self) -> None:
        from apron.adapters.backends.llm_classifier import _classification_cache

        _classification_cache["test_key"] = {
            "failure_class": "oom",
            "confidence": 0.9,
        }
        assert _classification_cache["test_key"]["failure_class"] == "oom"
        del _classification_cache["test_key"]


class TestClassifierProvenance:
    """F6: every result names the classifier model and the digest of its input."""

    def _fake_anthropic(self, responses: list[dict]):  # type: ignore[no-untyped-def]
        from types import SimpleNamespace

        calls: list[dict] = []

        class _Messages:
            def create(self, **kwargs):  # type: ignore[no-untyped-def]
                calls.append(kwargs)
                block = SimpleNamespace(type="tool_use", input=responses[len(calls) - 1])
                return SimpleNamespace(content=[block])

        class _Client:
            messages = _Messages()

        return SimpleNamespace(Anthropic=lambda: _Client()), calls

    def test_result_carries_model_and_input_digest(self, rules: list[dict]) -> None:
        import sys
        from unittest.mock import patch

        from apron.adapters.backends import llm_classifier as lc

        error = "ValueError: The model's max seq len (40960) is larger than KV cache"
        fake, calls = self._fake_anthropic(
            [
                {"failure_class": "unknown", "confidence": 0.2, "evidence_span": ""},
            ]
        )
        lc._classification_cache.clear()
        with patch.dict(sys.modules, {"anthropic": fake}):
            result = lc.classify_and_extract(error, rules=rules)
        system = calls[0]["system"]
        expected = lc.classifier_input_digest(lc.DEFAULT_CLASSIFIER_MODEL, system, error)
        assert calls[0]["model"] == lc.DEFAULT_CLASSIFIER_MODEL
        assert result["classifier_model_id"] == lc.DEFAULT_CLASSIFIER_MODEL
        assert result["classifier_input_digest"] == expected
        assert expected.startswith("1220")
        lc._classification_cache.clear()


class TestRootCauseAboveWrapper:
    """§10.1 class 2: the classifier sees the root cause, never only the wrapper."""

    LOG = (
        "INFO loading weights\n" * 3000
        + "ERROR [core.py:900] EngineCore failed to start.\n"
        + "Traceback (most recent call last):\n"
        + '  File "vllm/v1/core/kv_cache_utils.py", line 879, in _check_enough_kv_cache_memory\n'
        + "ValueError: To serve at least one request with the model's max seq len (40960), "
        + "(5.62 GiB KV cache is needed, which is larger than the available KV cache memory "
        + "(2.40 GiB). Based on the available memory, the estimated maximum model length is "
        + "17472.\n"
        + "INFO shutting down\n" * 50
        + "RuntimeError: Engine core initialization failed. See root cause above. "
        + "Failed core proc(s): {}\n"
    )

    def test_window_contains_root_cause(self) -> None:
        from apron.adapters.backends.llm_classifier import root_cause_window

        window = root_cause_window(self.LOG)
        assert window is not None
        assert "estimated maximum model length is 17472" in window
        assert "See root cause above" not in window

    def test_truncated_input_keeps_the_root_cause(self) -> None:
        from apron.adapters.backends.llm_classifier import _extract_relevant

        assert len(self.LOG) > 8000
        relevant = _extract_relevant(self.LOG, limit=8000)
        assert len(relevant) <= 8000
        assert "estimated maximum model length is 17472" in relevant

    def test_no_wrapper_no_window(self) -> None:
        from apron.adapters.backends.llm_classifier import root_cause_window

        assert root_cause_window("ValueError: boom") is None

    def test_prompt_tells_the_model_to_skip_the_wrapper(self, rules: list[dict]) -> None:
        prompt = _build_system_prompt(rules)
        assert "never the wrapper itself" in prompt


class TestEnumExtraction:
    def test_enum_fields_are_constrained_in_the_tool(
        self, rules: list[dict], classes: list[str], schemas: dict
    ) -> None:
        from apron.domain.diagnosis import build_extraction_enums

        tool = _build_extraction_tool(
            "dtype_incompatible", classes, schemas, build_extraction_enums(rules)
        )
        props = tool["input_schema"]["properties"]
        assert props["model_type"]["enum"] == ["gemma2", "gemma3", "gemma3_text", "glm4", None]
        assert props["unsupported_dtype"]["enum"] == ["float16", None]

    def test_engine_rejects_values_outside_the_enum(self, rules: list[dict], monkeypatch) -> None:  # type: ignore[no-untyped-def]
        from apron.adapters.backends import llm_classifier
        from apron.adapters.backends.vllm_engine import VllmEngineAdapter

        monkeypatch.setattr(
            llm_classifier,
            "classify_and_extract",
            lambda error, rules=None: {"model_type": "llama", "unsupported_dtype": "float16"},
        )
        extracted = VllmEngineAdapter(rules=rules).extract("x", "dtype_incompatible")
        assert extracted == {"unsupported_dtype": "float16"}
