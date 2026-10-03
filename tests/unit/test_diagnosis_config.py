"""F5: diagnosis receives the resolved config.json fields, never a hard-coded dict."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from apron.application.orchestration.diagnosis_pipeline import (
    DIAGNOSIS_CONFIG_FIELDS,
    diagnosis_model_config,
)

if TYPE_CHECKING:
    import pytest

HF = Path(__file__).parents[1] / "fixtures" / "external-formats" / "huggingface-hub"


def test_selects_exactly_the_diagnosis_fields() -> None:
    config = json.loads((HF / "config.json").read_text())
    selected = diagnosis_model_config(config)
    assert set(selected) <= set(DIAGNOSIS_CONFIG_FIELDS)
    assert selected["num_attention_heads"] == config["num_attention_heads"]
    assert selected["num_key_value_heads"] == config["num_key_value_heads"]
    assert selected["max_position_embeddings"] == config["max_position_embeddings"]
    assert selected["model_type"] == config["model_type"]
    assert selected["architectures"] == config["architectures"]


def test_quantization_config_is_passed_through() -> None:
    config = {"model_type": "qwen3", "quantization_config": {"quant_method": "fp_quant"}}
    assert diagnosis_model_config(config)["quantization_config"] == {"quant_method": "fp_quant"}


def test_text_config_fills_missing_fields() -> None:
    config = {
        "model_type": "gemma3",
        "architectures": ["Gemma3ForConditionalGeneration"],
        "text_config": {"num_attention_heads": 8, "max_position_embeddings": 131072},
    }
    selected = diagnosis_model_config(config)
    assert selected["model_type"] == "gemma3"
    assert selected["num_attention_heads"] == 8
    assert selected["max_position_embeddings"] == 131072


def test_cli_diagnose_resolves_real_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from apron.adapters.evidence.hf_hub import FixtureHFHubResolver
    from apron.interfaces import cli

    monkeypatch.setattr(cli, "_build_resolver", lambda: FixtureHFHubResolver(HF))
    config = cli._resolve_diagnosis_config("Qwen/Qwen3-8B")
    assert config["num_attention_heads"] == 32
    assert config["num_key_value_heads"] == 8


def test_cli_diagnose_passes_resolved_config_to_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    from apron.adapters.evidence.hf_hub import FixtureHFHubResolver
    from apron.application.orchestration import diagnosis_pipeline
    from apron.interfaces import cli

    seen: dict[str, Any] = {}

    def fake_pipeline(error, engine, plan, model_config, hardware, rules):  # type: ignore[no-untyped-def]
        seen["model_config"] = model_config
        return diagnosis_pipeline.DiagnosisPipelineResult(
            failure_class="unknown",
            extracted={},
            rule_matched=False,
            correction_strategy=None,
            corrected_plan=None,
            result_label="Unrecognized failure",
            error_trace=error,
        )

    monkeypatch.setattr(cli, "_build_resolver", lambda: FixtureHFHubResolver(HF))
    monkeypatch.setattr(diagnosis_pipeline, "run_diagnosis_pipeline", fake_pipeline)
    cli._diagnose("boom", "Qwen/Qwen3-8B")
    assert seen["model_config"]["num_attention_heads"] == 32


def test_cli_diagnose_rejects_stored_plan_with_unknown_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The stored plan used to be filtered to known keys — a silent drop (F2)."""
    from apron.adapters.backends.local_store import LocalRecordStore
    from apron.interfaces import cli

    store = LocalRecordStore(tmp_path)
    digest = store.store({"schema_version": 1, "tensor_parallel": 2, "tp_typo": 4})
    monkeypatch.setattr(cli, "_default_store", lambda: store)
    result = cli._diagnose("boom", "Qwen/Qwen3-8B", plan_digest=digest)
    assert "not a valid DeploymentPlan" in result["error"]
    assert "tp_typo" in result["error"]


def test_served_model_parsing() -> None:
    from apron.interfaces.cli import _served_model

    assert _served_model(["vllm", "serve", "Qwen/Qwen3-8B", "--dtype", "bf16"]) == "Qwen/Qwen3-8B"
    assert _served_model(["python", "-m", "x"]) is None
