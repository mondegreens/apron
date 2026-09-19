"""Integration test: six failure classes through one diagnosis pipeline.

Each test constructs a synthetic error using the actual pinned vLLM
v0.29.0 error format, runs the full pipeline (classify → extract →
match rule → compute correction), and verifies the corrected plan.

Exit gate for Phase 1b diagnosis engine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from apron.adapters.backends.rule_loader import load_rules
from apron.adapters.backends.vllm_engine import VllmEngineAdapter
from apron.application.orchestration.diagnosis_pipeline import run_diagnosis_pipeline
from apron.domain.schemas.primitives import HardwareSpec
from apron.domain.schemas.solutions import DeploymentPlan


@pytest.fixture()
def engine() -> VllmEngineAdapter:
    return VllmEngineAdapter()


@pytest.fixture()
def rules() -> list[dict]:
    rules_dir = Path(__file__).parents[2] / "rules"
    return load_rules(rules_dir, "vllm", "v0.29.0")


@pytest.fixture()
def hardware() -> HardwareSpec:
    return HardwareSpec(
        gpu_sku="RTX 4090",
        total_memory_bytes=25_769_803_776,
        compute_capability="8.9",
    )


@pytest.fixture()
def model_config() -> dict:
    return {"num_attention_heads": 32, "num_kv_heads": 8}


# The six failure classes with actual vLLM v0.29.0 error formats
_SIX_CLASSES = [
    pytest.param(
        "oom",
        (
            "To serve at least one request with the model's max seq len "
            "(32768), (14.50 GiB KV "
            "cache is needed, which is larger than the available KV cache "
            "memory (6.20 GiB). "
            "Based on the available memory, "
            "the estimated maximum model length is 8192."
        ),
        {"max_model_len": "8192"},
        id="oom-kv-cache",
    ),
    pytest.param(
        "max_model_len",
        (
            "User-specified max_model_len (131072) is greater "
            "than the derived max_model_len (max_position_embeddings="
            "32768 or model_max_length=None in model's config.json)."
        ),
        {"max_model_len": "32768"},
        id="max_model_len",
    ),
    pytest.param(
        "dtype_incompatible",
        (
            "The model type 'Qwen3MoeForCausalLM' does not support float16. "
            "Reason: quantized models require bfloat16"
        ),
        None,
        id="dtype_incompatible",
    ),
    pytest.param(
        "tp_divisibility",
        ("Total number of attention heads (28) must be divisible by tensor parallel size (4)."),
        None,
        id="tp_divisibility",
    ),
    pytest.param(
        "quant_compute_capability",
        (
            "The quantization method gptq "
            "is not supported for the current GPU. Minimum "
            "capability: 80. Current capability: 75."
        ),
        None,
        id="quant_compute_capability",
    ),
    pytest.param(
        "engine_init",
        "LoRA is not enabled. Use --enable-lora to enable LoRA.",
        None,
        id="engine_init",
    ),
]


@pytest.mark.parametrize(("expected_class", "error", "expected_config"), _SIX_CLASSES)
def test_six_classes_through_pipeline(
    engine: VllmEngineAdapter,
    rules: list[dict],
    hardware: HardwareSpec,
    model_config: dict,
    expected_class: str,
    error: str,
    expected_config: dict | None,
) -> None:
    """Each failure class produces a corrected plan through one generic pipeline."""
    plan = DeploymentPlan(
        tensor_parallel=4,
        dtype="float16",
        engine_configuration={
            "gpu_memory_utilization": "0.90",
            "max_model_len": "131072",
            "max_num_seqs": "256",
            "quantization": "gptq",
        },
    )

    result = run_diagnosis_pipeline(error, engine, plan, model_config, hardware, rules)

    assert result.failure_class == expected_class
    assert result.rule_matched, f"No rule matched for {expected_class}"
    assert result.corrected_plan is not None, f"No correction for {expected_class}"
    assert result.result_label in ("Corrected", "Alternative with trade-offs")
    assert isinstance(result.corrected_plan, DeploymentPlan)

    if expected_config:
        for key, value in expected_config.items():
            assert result.corrected_plan.engine_configuration[key] == value


def test_unknown_gets_trace_only(
    engine: VllmEngineAdapter,
    rules: list[dict],
    hardware: HardwareSpec,
    model_config: dict,
) -> None:
    """Unknown failure class gets trace capture, no correction."""
    result = run_diagnosis_pipeline(
        "RuntimeError: NCCL timeout on rank 0",
        engine,
        DeploymentPlan(),
        model_config,
        hardware,
        rules,
    )
    assert result.failure_class == "unknown"
    assert result.corrected_plan is None
    assert result.error_trace
