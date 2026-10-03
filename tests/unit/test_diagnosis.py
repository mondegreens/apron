"""Tests for diagnosis rule loading and matching."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apron.adapters.backends.rule_loader import load_rules
from apron.domain.diagnosis import match_rule


@pytest.fixture()
def rules_dir(tmp_path: Path) -> Path:
    vllm_dir = tmp_path / "vllm-v0.29"
    vllm_dir.mkdir()
    return tmp_path


@pytest.fixture()
def sample_rule() -> dict:
    return {
        "schema_version": 4,
        "engine": "vllm",
        "engine_version": "v0.29.0",
        "error_family": "oom",
        "correction_strategy": "reduce_memory_pressure",
        "source_sites": [{"file": "v1/worker/gpu_model_runner.py", "line": 5460}],
        "status": "hypothesis",
    }


# -------------------------------------------------------------------
# rule_loader
# -------------------------------------------------------------------


class TestLoadRules:
    def test_loads_valid_json(self, rules_dir: Path, sample_rule: dict) -> None:
        rule_file = rules_dir / "vllm-v0.29" / "oom.json"
        rule_file.write_text(json.dumps(sample_rule))
        result = load_rules(rules_dir, "vllm", "v0.29")
        assert len(result) == 1
        assert result[0]["error_family"] == "oom"

    def test_loads_multiple_files_sorted(self, rules_dir: Path, sample_rule: dict) -> None:
        vllm_dir = rules_dir / "vllm-v0.29"
        for name in ("oom", "dtype_incompatible", "tp_divisibility"):
            rule = {**sample_rule, "error_family": name}
            (vllm_dir / f"{name}.json").write_text(json.dumps(rule))
        result = load_rules(rules_dir, "vllm", "v0.29")
        assert len(result) == 3
        assert result[0]["error_family"] == "dtype_incompatible"

    def test_empty_dir_returns_empty(self, rules_dir: Path) -> None:
        result = load_rules(rules_dir, "vllm", "v0.29")
        assert result == []

    def test_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        result = load_rules(tmp_path, "vllm", "v0.99")
        assert result == []

    def test_malformed_json_raises(self, rules_dir: Path) -> None:
        (rules_dir / "vllm-v0.29" / "bad.json").write_text("{invalid")
        with pytest.raises(ValueError, match="Malformed JSON"):
            load_rules(rules_dir, "vllm", "v0.29")

    def test_missing_required_fields_raises(self, rules_dir: Path) -> None:
        incomplete = {"schema_version": 1, "engine": "vllm"}
        (rules_dir / "vllm-v0.29" / "incomplete.json").write_text(json.dumps(incomplete))
        with pytest.raises(ValueError, match="missing required fields"):
            load_rules(rules_dir, "vllm", "v0.29")

    def test_v3_rule_file_is_migrated(self, rules_dir: Path, sample_rule: dict) -> None:
        v3 = {**sample_rule, "schema_version": 3, "extraction_schema": {"x": "string"}}
        (rules_dir / "vllm-v0.29" / "oom.json").write_text(json.dumps(v3))
        result = load_rules(rules_dir, "vllm", "v0.29")
        assert result[0]["schema_version"] == 4
        assert result[0]["extraction_schema"]["x"]["type"] == "string"

    def test_unknown_rule_key_fails_loudly(self, rules_dir: Path, sample_rule: dict) -> None:
        """F2: a rule file with a key the schema does not know is an error."""
        bad = {**sample_rule, "corection_strategy": "typo"}
        (rules_dir / "vllm-v0.29" / "oom.json").write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="corection_strategy"):
            load_rules(rules_dir, "vllm", "v0.29")

    def test_version_strips_v_prefix(self, rules_dir: Path, sample_rule: dict) -> None:
        (rules_dir / "vllm-v0.29" / "oom.json").write_text(json.dumps(sample_rule))
        result = load_rules(rules_dir, "vllm", "v0.29")
        assert len(result) == 1
        result2 = load_rules(rules_dir, "vllm", "0.29")
        assert len(result2) == 1


# -------------------------------------------------------------------
# Actual rules from rules/vllm-v0.29/
# -------------------------------------------------------------------


class TestActualRules:
    def test_load_shipped_rules(self) -> None:
        rules_dir = Path(__file__).parents[2] / "rules"
        rules = load_rules(rules_dir, "vllm", "v0.29")
        assert len(rules) >= 6
        families = {r["error_family"] for r in rules}
        assert {
            "oom_weight_load",
            "oom_kv_cache",
            "max_model_len",
            "dtype_incompatible",
            "tp_divisibility",
            "quant_compute_capability",
        }.issubset(families)
        assert "oom" not in families  # split into weight-load and KV-cache (§10.1)


# -------------------------------------------------------------------
# match_rule
# -------------------------------------------------------------------


class TestMatchRule:
    def test_matches_by_family(self) -> None:
        rules = [
            {"error_family": "oom", "correction_strategy": "reduce_memory_pressure"},
            {"error_family": "dtype_incompatible", "correction_strategy": "fallback_dtype"},
        ]
        result = match_rule("oom", rules)
        assert result is not None
        assert result["correction_strategy"] == "reduce_memory_pressure"

    def test_unknown_returns_none(self) -> None:
        rules = [{"error_family": "oom", "correction_strategy": "reduce_memory_pressure"}]
        assert match_rule("unknown", rules) is None

    def test_empty_rules_returns_none(self) -> None:
        assert match_rule("oom", []) is None

    def test_first_match_wins(self) -> None:
        rules = [
            {"error_family": "oom", "correction_strategy": "first"},
            {"error_family": "oom", "correction_strategy": "second"},
        ]
        result = match_rule("oom", rules)
        assert result is not None
        assert result["correction_strategy"] == "first"
