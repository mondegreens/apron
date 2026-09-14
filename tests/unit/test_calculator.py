"""Tests for the mechanism-aware calculator and ModelSpec builder."""

import json
from pathlib import Path

import pytest
from apron.domain.canonical import canonicalize
from apron.domain.mechanisms import CalculatorInput, ComponentMechanism, TextWorkload, calculate
from apron.domain.mechanisms.calculator import (
    DTYPE_BYTES,
    _extract_gqa_params,
    _extract_mla_params,
    calculate_autoregressive_decode,
    calculate_mla_decode,
)
from apron.domain.mechanisms.model_spec_builder import build_model_spec
from apron.domain.schemas.primitives import HardwareSpec

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "external-formats" / "huggingface-hub"


@pytest.fixture(autouse=True)
def _ensure_calculator_registered():
    from apron.domain.mechanisms import _CALCULATOR_REGISTRY

    _CALCULATOR_REGISTRY["autoregressive_decode"] = calculate_autoregressive_decode
    _CALCULATOR_REGISTRY["mla_decode"] = calculate_mla_decode
    yield


RTX_4090 = HardwareSpec(
    gpu_sku="RTX 4090",
    total_memory_bytes=25_769_803_776,
    compute_capability="8.9",
)


def _load_qwen3_config() -> dict:
    return json.loads((FIXTURE_DIR / "config.json").read_bytes())


def _load_safetensors_metadata() -> dict:
    index = json.loads((FIXTURE_DIR / "model.safetensors.index.json").read_bytes())
    return index.get("metadata", {})


def _qwen3_input(
    *,
    isl: int = 512,
    osl: int = 128,
    max_batch_size: int = 4,
    hardware: HardwareSpec | None = None,
) -> CalculatorInput:
    config = _load_qwen3_config()
    metadata = _load_safetensors_metadata()
    config["total_weight_bytes"] = metadata["total_size"]
    return CalculatorInput(
        mechanism=ComponentMechanism(mechanism="autoregressive_decode", role="decoder"),
        workload=TextWorkload(kind="text", input_length=isl, output_length=osl),
        artifact_metadata=config,
        hardware=hardware or RTX_4090,
        execution_spec_data={"max_batch_size": max_batch_size},
    )


# ---------------------------------------------------------------------------
# GQA parameter extraction
# ---------------------------------------------------------------------------


def test_extract_gqa_params_qwen3():
    config = _load_qwen3_config()
    params = _extract_gqa_params(config)
    assert params is not None
    assert params["num_layers"] == 36
    assert params["num_kv_heads"] == 8
    assert params["head_dim"] == 128
    assert params["num_attention_heads"] == 32


def test_extract_gqa_params_missing_fields():
    assert _extract_gqa_params({}) is None
    assert _extract_gqa_params({"num_hidden_layers": 32}) is None


def test_extract_gqa_params_zero_attention_heads():
    config = {
        "num_hidden_layers": 1,
        "num_key_value_heads": 1,
        "hidden_size": 64,
        "num_attention_heads": 0,
    }
    assert _extract_gqa_params(config) is None


# ---------------------------------------------------------------------------
# Calculator: Qwen3-8B on RTX 4090
# ---------------------------------------------------------------------------


def test_qwen3_8b_all_breakdown_fields_positive():
    result = calculate(_qwen3_input())
    assert result is not None
    for key in (
        "weight_memory_bytes",
        "kv_cache_bytes",
        "activation_estimate_bytes",
        "non_pytorch_overhead_bytes",
        "cuda_graph_estimate_bytes",
        "total_required_bytes",
        "available_kv_cache_bytes",
    ):
        assert result[key] > 0, f"{key} should be > 0"


def test_qwen3_8b_total_fits_rtx_4090():
    result = calculate(_qwen3_input())
    assert result is not None
    assert result["total_required_bytes"] < 25_769_803_776


def test_qwen3_8b_weight_bytes():
    result = calculate(_qwen3_input())
    assert result is not None
    assert result["weight_memory_bytes"] == 16_381_470_720


def test_qwen3_8b_kv_per_token():
    result = calculate(_qwen3_input())
    assert result is not None
    # 2 * 36 * 8 * 128 * 2 = 147,456
    assert result["kv_per_token_bytes"] == 147_456


def test_qwen3_8b_kv_cache():
    result = calculate(_qwen3_input())
    assert result is not None
    # 147,456 * (512 + 128) * 4 = 377,487,360
    assert result["kv_cache_bytes"] == 147_456 * (512 + 128) * 4


# ---------------------------------------------------------------------------
# Calculator: unknown mechanism
# ---------------------------------------------------------------------------


def test_unknown_mechanism_returns_none():
    inputs = CalculatorInput(
        mechanism=ComponentMechanism(mechanism="multi_latent_attention", role="decoder"),
        workload=TextWorkload(kind="text", input_length=512, output_length=128),
        artifact_metadata=_load_qwen3_config(),
        hardware=RTX_4090,
        execution_spec_data={},
    )
    result = calculate(inputs)
    assert result is None


# ---------------------------------------------------------------------------
# Calculator: determinism
# ---------------------------------------------------------------------------


def test_determinism():
    inp = _qwen3_input()
    r1 = calculate(inp)
    r2 = calculate(inp)
    assert r1 is not None
    assert r2 is not None
    b1 = canonicalize(r1)
    b2 = canonicalize(r2)
    assert b1 == b2


# ---------------------------------------------------------------------------
# Calculator: edge cases
# ---------------------------------------------------------------------------


def test_enforce_eager_no_cuda_graphs():
    config = _load_qwen3_config()
    metadata = _load_safetensors_metadata()
    config["total_weight_bytes"] = metadata["total_size"]
    inputs = CalculatorInput(
        mechanism=ComponentMechanism(mechanism="autoregressive_decode", role="decoder"),
        workload=TextWorkload(kind="text", input_length=512, output_length=128),
        artifact_metadata=config,
        hardware=RTX_4090,
        execution_spec_data={"enforce_eager": True, "max_batch_size": 4},
    )
    result = calculate(inputs)
    assert result is not None
    assert result["cuda_graph_estimate_bytes"] == 0


def test_zero_weight_bytes_returns_none():
    config = _load_qwen3_config()
    config["total_weight_bytes"] = 0
    inputs = CalculatorInput(
        mechanism=ComponentMechanism(mechanism="autoregressive_decode", role="decoder"),
        workload=TextWorkload(kind="text", input_length=512, output_length=128),
        artifact_metadata=config,
        hardware=RTX_4090,
        execution_spec_data={},
    )
    result = calculate(inputs)
    assert result is None


# ---------------------------------------------------------------------------
# ModelSpec builder
# ---------------------------------------------------------------------------


def test_build_model_spec_qwen3():
    config = _load_qwen3_config()
    spec = build_model_spec(
        config,
        repository="Qwen/Qwen3-8B",
        revision="abc123",
        license_id="apache-2.0",
    )
    assert spec.repository == "Qwen/Qwen3-8B"
    assert len(spec.components) == 1
    assert spec.components[0].mechanism == "autoregressive_decode"
    assert spec.components[0].role == "decoder"
    assert spec.component_bytes_dtype == {"decoder": "bfloat16"}


def test_build_model_spec_unknown_arch():
    config = {"architectures": ["NovelArchForCausalLM"]}
    spec = build_model_spec(config)
    assert spec.components[0].mechanism == "autoregressive_decode"


def test_build_model_spec_no_architectures():
    spec = build_model_spec({})
    assert spec.components[0].mechanism == "UnknownArchitecture"


def test_dtype_bytes_map():
    assert DTYPE_BYTES["bfloat16"] == 2
    assert DTYPE_BYTES["float16"] == 2
    assert DTYPE_BYTES["float32"] == 4


# ---------------------------------------------------------------------------
# MLA (Multi-Latent Attention) — DeepSeek-V3
# ---------------------------------------------------------------------------

DEEPSEEK_FIXTURE = FIXTURE_DIR / "DeepSeek-V3-config.json"


def _load_deepseek_config() -> dict:
    return json.loads(DEEPSEEK_FIXTURE.read_bytes())


def _deepseek_input(
    *,
    total_weight_bytes: int = 671_000_000_000,
) -> CalculatorInput:
    config = _load_deepseek_config()
    config["total_weight_bytes"] = total_weight_bytes
    return CalculatorInput(
        mechanism=ComponentMechanism(mechanism="mla_decode", role="decoder"),
        workload=TextWorkload(kind="text", input_length=512, output_length=128),
        artifact_metadata=config,
        hardware=HardwareSpec(
            gpu_sku="H100 SXM",
            total_memory_bytes=85_899_345_920,
            compute_capability="9.0",
        ),
        execution_spec_data={"max_batch_size": 4},
    )


def test_mla_extract_params():
    config = _load_deepseek_config()
    params = _extract_mla_params(config)
    assert params is not None
    assert params["kv_lora_rank"] == 512
    assert params["qk_rope_head_dim"] == 64
    assert params["num_layers"] == 61


def test_mla_kv_per_token():
    result = calculate(_deepseek_input())
    assert result is not None
    # (512 + 64) x 2 bytes x 61 layers = 70,272 bytes/token
    assert result["kv_per_token_bytes"] == (512 + 64) * 2 * 61
    assert result["kv_per_token_bytes"] == 70_272


def test_mla_cache_per_layer():
    result = calculate(_deepseek_input())
    assert result is not None
    # (512 + 64) x 2 bytes = 1,152 bytes per layer
    assert result["cache_per_layer_bytes"] == 1152


def test_mla_vs_gqa_kv_difference():
    """MLA KV cache should be ~25x smaller than naive GQA interpretation."""
    config = _load_deepseek_config()
    config["total_weight_bytes"] = 671_000_000_000
    mla_result = calculate(_deepseek_input())
    assert mla_result is not None

    gqa_params = _extract_gqa_params(config)
    assert gqa_params is not None
    gqa_kv = 2 * gqa_params["num_layers"] * gqa_params["num_kv_heads"] * gqa_params["head_dim"] * 2
    ratio = gqa_kv / mla_result["kv_per_token_bytes"]
    assert ratio > 20


def test_mla_mechanism_detail():
    result = calculate(_deepseek_input())
    assert result is not None
    assert result["mechanism_detail"] == "mla"


def test_mla_all_breakdown_fields_positive():
    result = calculate(_deepseek_input())
    assert result is not None
    for key in (
        "weight_memory_bytes",
        "kv_cache_bytes",
        "activation_estimate_bytes",
        "non_pytorch_overhead_bytes",
        "cuda_graph_estimate_bytes",
        "total_required_bytes",
    ):
        assert result[key] > 0, f"{key} should be > 0"


def test_mla_missing_kv_lora_rank_returns_none():
    config = _load_deepseek_config()
    del config["kv_lora_rank"]
    assert _extract_mla_params(config) is None


# ---------------------------------------------------------------------------
# MoE activation adjustment
# ---------------------------------------------------------------------------


def test_moe_activation_adjustment():
    """DeepSeek-V3: 256 experts, 8 active → activation scaled by 8/256."""
    result = calculate(_deepseek_input())
    assert result is not None
    weight_bytes = 671_000_000_000
    full_activation = int(weight_bytes * 0.10)
    expected = int(full_activation * 8 / 256)
    assert result["activation_estimate_bytes"] == expected


def test_non_moe_activation_unchanged():
    """Qwen3-8B has no MoE fields — activation stays at 10% of weights."""
    result = calculate(_qwen3_input())
    assert result is not None
    expected = int(16_381_470_720 * 0.10)
    assert result["activation_estimate_bytes"] == expected


# ---------------------------------------------------------------------------
# Sliding window KV cap
# ---------------------------------------------------------------------------


def test_sliding_window_caps_kv_cache():
    """When sliding_window < seq_len, KV cache is bounded."""
    config = _load_qwen3_config()
    metadata = _load_safetensors_metadata()
    config["total_weight_bytes"] = metadata["total_size"]
    config["sliding_window"] = 256
    inputs = CalculatorInput(
        mechanism=ComponentMechanism(mechanism="autoregressive_decode", role="decoder"),
        workload=TextWorkload(kind="text", input_length=512, output_length=128),
        artifact_metadata=config,
        hardware=RTX_4090,
        execution_spec_data={"max_batch_size": 4},
    )
    result = calculate(inputs)
    assert result is not None
    # Window=256 < seq_len=640, so KV cache uses 256
    assert result["effective_seq_len"] == 256
    expected_kv = result["kv_per_token_bytes"] * 256 * 4
    assert result["kv_cache_bytes"] == expected_kv


def test_no_sliding_window_uses_full_seq():
    """Without sliding_window, full ISL+OSL is used."""
    result = calculate(_qwen3_input())
    assert result is not None
    assert result["effective_seq_len"] == 640
    expected_kv = result["kv_per_token_bytes"] * 640 * 4
    assert result["kv_cache_bytes"] == expected_kv


def test_sliding_window_larger_than_seq_no_effect():
    config = _load_qwen3_config()
    metadata = _load_safetensors_metadata()
    config["total_weight_bytes"] = metadata["total_size"]
    config["sliding_window"] = 4096
    inputs = CalculatorInput(
        mechanism=ComponentMechanism(mechanism="autoregressive_decode", role="decoder"),
        workload=TextWorkload(kind="text", input_length=512, output_length=128),
        artifact_metadata=config,
        hardware=RTX_4090,
        execution_spec_data={"max_batch_size": 4},
    )
    result = calculate(inputs)
    assert result is not None
    assert result["effective_seq_len"] == 640


# ---------------------------------------------------------------------------
# ModelSpec builder — MLA detection
# ---------------------------------------------------------------------------


def test_build_model_spec_deepseek_v3_detects_mla():
    config = _load_deepseek_config()
    spec = build_model_spec(config, repository="deepseek-ai/DeepSeek-V3")
    assert spec.components[0].mechanism == "mla_decode"


def test_build_model_spec_qwen3_stays_gqa():
    config = _load_qwen3_config()
    spec = build_model_spec(config, repository="Qwen/Qwen3-8B")
    assert spec.components[0].mechanism == "autoregressive_decode"
