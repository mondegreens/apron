"""Integration test: six failure classes through one diagnosis pipeline.

Each test constructs a synthetic error using the actual pinned vLLM
v0.29.0 error format, runs the full pipeline (classify → extract →
match rule → compute correction), and verifies the corrected plan.

Exit gate for Phase 1b diagnosis engine.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import FakeDiagnosisEngine

from apron.adapters.backends.rule_loader import load_rules
from apron.application.orchestration.correction import CatalogEntry, CorrectionContext
from apron.application.orchestration.diagnosis_pipeline import run_diagnosis_pipeline
from apron.domain.schemas.primitives import HardwareSpec
from apron.domain.schemas.solutions import DeploymentPlan


@pytest.fixture()
def engine() -> FakeDiagnosisEngine:
    return FakeDiagnosisEngine()


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


_4090 = HardwareSpec(
    gpu_sku="NVIDIA GeForce RTX 4090", total_memory_bytes=24 << 30, compute_capability="8.9"
)
_A6000 = HardwareSpec(
    gpu_sku="NVIDIA RTX A6000", total_memory_bytes=48 << 30, compute_capability="8.6"
)
_H100 = HardwareSpec(
    gpu_sku="NVIDIA H100 80GB HBM3", total_memory_bytes=80 << 30, compute_capability="9.0"
)
_B200 = HardwareSpec(
    gpu_sku="NVIDIA B200", total_memory_bytes=179 << 30, compute_capability="10.0"
)
_CONTEXT = CorrectionContext(
    catalog=(
        CatalogEntry(_4090, 0.74),
        CatalogEntry(_A6000, 0.53),
        CatalogEntry(_H100, 3.49),
        CatalogEntry(_B200, 6.79),
    ),
    predicted_total_bytes=33_000_000_000,
)

_ENGINE_WRAPPER = (
    "\nERROR [core.py] EngineCore failed to start.\n"
    "RuntimeError: Engine core initialization failed. See root cause above. "
    "Failed core proc(s): {}\n"
)

# The six ADR-005 classes as defined in PLAN §10.1.  Error text follows the
# pinned vLLM v0.29.0 format strings (file:line in each rule's extraction
# schema); the real captured logs from L0-F replace these in
# tests/unit/test_fix_proof_real_logs.py.  Each row: expected family (Gate A),
# the broken plan, hardware, and the field the correction must change (Gate B).
_SIX_CLASSES = [
    pytest.param(
        "oom_weight_load",
        "ERROR Failed to load model - not enough GPU memory. Try lowering "
        "--gpu-memory-utilization to free memory for weights, increasing "
        "--tensor-parallel-size, or using --quantization. See "
        "https://docs.vllm.ai/en/latest/configuration/conserving_memory/ for more tips. "
        "(original error: CUDA out of memory. Tried to allocate 1.50 GiB. GPU 0 has a total "
        "capacity of 23.52 GiB of which 1.12 GiB is free.)",
        DeploymentPlan(
            dtype="bfloat16",
            engine_configuration={"gpu_memory_utilization": "0.90"},
            resource_allocation={"gpu_sku": _4090.gpu_sku, "gpu_count": "1"},
        ),
        _4090,
        ("resource_allocation", "gpu_sku", "NVIDIA RTX A6000"),
        id="1-oom_weight_load",
    ),
    pytest.param(
        "oom_kv_cache",
        "ValueError: To serve at least one request with the model's max seq len "
        "(40960), (5.62 GiB KV cache is needed, which is larger than the available KV cache "
        "memory (2.40 GiB). Based on the available memory, the estimated maximum model "
        "length is 17472. Try increasing `gpu_memory_utilization` (which also controls CPU "
        "memory on the CPU backend) or decreasing `max_model_len` when initializing the engine."
        + _ENGINE_WRAPPER,
        DeploymentPlan(
            dtype="bfloat16",
            engine_configuration={"max_model_len": "40960"},
            resource_allocation={"gpu_sku": _4090.gpu_sku, "gpu_count": "1"},
        ),
        _4090,
        ("engine_configuration", "max_model_len", "17472"),
        id="2-oom_kv_cache-behind-wrapper",
    ),
    pytest.param(
        "max_model_len",
        "ValueError: User-specified max_model_len (999999) is greater than the derived "
        "max_model_len (max_position_embeddings=32768 or model_max_length=None in model's "
        "config.json). To allow overriding this maximum, set the env var "
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1.",
        DeploymentPlan(
            dtype="bfloat16",
            engine_configuration={"max_model_len": "999999"},
            resource_allocation={"gpu_sku": _4090.gpu_sku, "gpu_count": "1"},
        ),
        _4090,
        ("engine_configuration", "max_model_len", "32768"),
        id="3-max_model_len",
    ),
    pytest.param(
        "dtype_incompatible",
        "ValueError: The model type 'gemma2' does not support float16. Reason: Numerical "
        "instability. Please use bfloat16 or float32 instead.",
        DeploymentPlan(
            dtype="float16", resource_allocation={"gpu_sku": _4090.gpu_sku, "gpu_count": "1"}
        ),
        _4090,
        ("plan", "dtype", "bfloat16"),
        id="4-dtype_incompatible",
    ),
    pytest.param(
        "tp_divisibility",
        "ValueError: Total number of attention heads (32) must be divisible by tensor "
        "parallel size (3).",
        DeploymentPlan(
            tensor_parallel=3,
            dtype="bfloat16",
            resource_allocation={"gpu_sku": _4090.gpu_sku, "gpu_count": "4"},
        ),
        _4090,
        ("plan", "tensor_parallel", 2),
        id="5-tp_divisibility",
    ),
    pytest.param(
        "quant_compute_capability",
        "ValueError: The quantization method fp_quant is not supported for the current GPU. "
        "Minimum capability: 100. Current capability: 90.",
        DeploymentPlan(
            dtype="bfloat16", resource_allocation={"gpu_sku": _H100.gpu_sku, "gpu_count": "1"}
        ),
        _H100,
        ("resource_allocation", "gpu_sku", "NVIDIA B200"),
        id="6-quant_compute_capability",
    ),
    pytest.param(
        "lora_config",
        "LoRA is not enabled. Use --enable-lora to enable LoRA.",
        DeploymentPlan(engine_configuration={"max_model_len": "4096"}),
        _4090,
        ("engine_configuration", "enable_lora", "true"),
        id="extra-lora_config",
    ),
]


@pytest.mark.parametrize(
    ("expected_class", "error", "broken_plan", "hardware", "expected_change"), _SIX_CLASSES
)
def test_six_classes_through_pipeline(
    engine: FakeDiagnosisEngine,
    rules: list[dict],
    expected_class: str,
    error: str,
    broken_plan: DeploymentPlan,
    hardware: HardwareSpec,
    expected_change: tuple,
) -> None:
    """Each class: one generic pipeline call, Gate A (family) and Gate B (a real change)."""
    model_config = {
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 32768,
        "model_type": "gemma2" if expected_class == "dtype_incompatible" else "qwen3",
        "quantization_config": {"quant_method": "fp_quant"},
    }
    result = run_diagnosis_pipeline(
        error,
        engine,
        broken_plan,
        model_config,
        hardware,
        rules,
        correction_context=_CONTEXT,
    )

    assert result.failure_class == expected_class  # Gate A
    assert result.rule_matched, f"No rule matched for {expected_class}"
    assert result.corrected_plan is not None, f"No correction for {expected_class}"
    assert result.corrected_plan != broken_plan  # Gate B
    where, key, value = expected_change
    if where == "plan":
        assert getattr(result.corrected_plan, key) == value
    else:
        assert getattr(result.corrected_plan, where)[key] == value


def test_unknown_gets_trace_only(
    engine: FakeDiagnosisEngine,
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
