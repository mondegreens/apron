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
