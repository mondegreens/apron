"""Per-layer caches counted as vLLM v0.29.0 pages them (domain/mechanisms/layered.py).

Expected bytes are the ones traced from the pinned source
(_dev_notes/cohort-run/hybrid-memory-trace.md, gemma4-memory-trace.md) with the
configs of Qwen/Qwen3.8-27B, Qwen/Qwen3.6-35B-A3B-FP8,
nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 and google/gemma-4-31B-it (read
2026-09-27; only the fields the formulas use are copied here).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from apron.domain.mechanisms.layered import (
    Blocks,
    block_accounting,
    bytes_per_sequence,
    duplicated_weight_bytes,
    family,
    kv_layout,
    layer_kinds,
    padded_weight_bytes,
)
from apron.domain.mechanisms.model_spec_builder import build_model_spec

L = 262_144

QWEN38 = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "text_config": {
        "model_type": "qwen3_5_text",
        "num_hidden_layers": 64,
        "full_attention_interval": 4,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "hidden_size": 5120,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "mamba_ssm_dtype": "float32",
    },
}
QWEN36 = {
    **QWEN38,
    "text_config": {
        **QWEN38["text_config"],
        "model_type": "qwen3_5_moe_text",
        "num_hidden_layers": 40,
        "num_key_value_heads": 2,
        "linear_num_value_heads": 32,
    },
}
NEMOTRON = {
    "architectures": ["NemotronHForCausalLM"],
    "model_type": "nemotron_h",
    "hybrid_override_pattern": "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "hidden_size": 2688,
    "mamba_num_heads": 64,
    "mamba_head_dim": 64,
    "ssm_state_size": 128,
    "n_groups": 8,
    "conv_kernel": 4,
    "mamba_ssm_cache_dtype": "float32",
}
# Nemotron-3.5 (nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) names its
# layers in a list; transformers turns it into the same pattern for vLLM.
_BLOCKS = {"M": "mamba", "E": "moe", "*": "attention"}
NEMOTRON_LIST = {k: v for k, v in NEMOTRON.items() if k != "hybrid_override_pattern"} | {
    "layers_block_type": [_BLOCKS[c] for c in str(NEMOTRON["hybrid_override_pattern"])],
    "mtp_layers_block_type": ["attention", "moe"],
}
GEMMA4 = {
    "architectures": ["Gemma4ForConditionalGeneration"],
    "model_type": "gemma4",
    "text_config": {
        "model_type": "gemma4_text",
        "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 10,
        "head_dim": 256,
        "global_head_dim": 512,
        "num_key_value_heads": 16,
        "num_global_key_value_heads": 4,
        "attention_k_eq_v": True,
        "sliding_window": 1024,
        "hidden_size": 5376,
        "num_kv_shared_layers": 0,
    },
}


def _per_sequence(config: dict, in_flight: int = 16_384) -> int | None:
    kinds = layer_kinds(config, tp=1, kv_dtype_bytes=2, model_dtype_bytes=2)
    assert kinds is not None
    return bytes_per_sequence(kinds, L, in_flight_tokens=in_flight)


def test_hybrid_state_models_match_the_traced_reservation() -> None:
    assert _per_sequence(QWEN38) == 17_520_656_384  # 16.32 GiB
    assert _per_sequence(QWEN36) == 5_514_854_400
    assert _per_sequence(NEMOTRON) == 1_725_628_416
    assert _per_sequence(NEMOTRON_LIST) == 1_725_628_416


def test_gemma4_sliding_and_global_layers_match_the_traced_reservation() -> None:
    # H100: max_num_batched_tokens 8192, two batches in flight
    assert _per_sequence(GEMMA4, in_flight=16_384) == 13_637 * 2_621_440
    # A100: 2048 per batch
    assert _per_sequence(GEMMA4, in_flight=4_096) == 9_797 * 2_621_440
    # vLLM copies k_proj into v_proj for the 10 global layers: 210 MiB of bf16
    assert duplicated_weight_bytes(GEMMA4, tp=1, dtype_bytes=2) == 210 * 2**20


def test_these_models_get_the_layered_mechanism_other_hybrids_stay_unknown() -> None:
    for config in (QWEN38, QWEN36, NEMOTRON, NEMOTRON_LIST, GEMMA4):
        assert family(config) is not None
        assert build_model_spec(config).components[0].mechanism == "layered_decode"
    jamba = {"architectures": ["JambaForCausalLM"], "num_attention_heads": 32}
    spec = build_model_spec(jamba, unmodelled_architectures=frozenset({"JambaForCausalLM"}))
    assert spec.components[0].mechanism == "unknown"
    # a block type transformers does not know: no pattern, no estimate
    odd = NEMOTRON_LIST | {"layers_block_type": ["mamba", "retention"]}
    assert family(odd) is None


def test_prefix_caching_off_reserves_one_state_page() -> None:
    kinds = layer_kinds(QWEN38, tp=1, kv_dtype_bytes=2, model_dtype_bytes=2)
    assert kinds is not None
    assert bytes_per_sequence(kinds, L, in_flight_tokens=0, prefix_caching=False) == 17_366_515_712


def test_attention_the_calculator_does_not_count_is_an_explicit_unknown() -> None:
    """DeepSeek V4 (compressed KV + indexer) was planned as plain attention
    with a confident KV figure (GPU-free check, 2026-09-28).  Any attention
    layout layered.py does not read is unknown, whatever the architecture name."""
    from apron.domain.mechanisms.model_spec_builder import unmodelled_attention

    deepseek_v4 = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "num_attention_heads": 64,
        "compress_ratios": [0, 0, 4, 128],
        "index_topk": 512,
    }
    glm5_next = {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": {
            "num_attention_heads": 64,
            "kv_lora_rank": 512,
            "layer_types": ["linear_attention", "deepseek_sparse_attention"],
        },
    }
    for config in (deepseek_v4, glm5_next):
        assert unmodelled_attention(config) is not None
        assert build_model_spec(config).components[0].mechanism == "unknown"
    # Plain and sliding attention stay modelled (gpt-oss, Mistral).
    gpt_oss = {
        "architectures": ["GptOssForCausalLM"],
        "num_attention_heads": 64,
        "layer_types": ["sliding_attention", "full_attention"],
        "sliding_window": 128,
    }
    assert unmodelled_attention(gpt_oss) is None
    assert build_model_spec(gpt_oss).components[0].mechanism == "autoregressive_decode"


# ---------------------------------------------------------------------------
# vLLM v0.30.0 layouts: DeepSeek V4 / V4.1, Qwen4Exp, GLM5Next.  Expected
# numbers from _dev_notes/cohort-run/deepseek-v4-kv-trace.md and
# qwen4exp-glm5next-kv-trace.md, where vLLM's own grouping functions
# (kv_cache_utils.py) were run on the specs the model code builds.  Configs
# as recorded with the safetensors headers (scripts/record_tensor_headers.py).
# ---------------------------------------------------------------------------

HEADERS = Path(__file__).parents[1] / "fixtures" / "cohort" / "headers"


def _recorded(name: str) -> dict[str, Any]:
    config: dict[str, Any] = json.loads((HEADERS / f"{name}.json").read_text())["config"]
    return config


V4 = _recorded("deepseek-ai__DeepSeek-V4-Flash-0731")
V41 = _recorded("deepseek-ai__DeepSeek-V4.1-Flash")
QWEN4 = _recorded("Qwen__Qwen3.8-Flash-Next")
GLM5 = _recorded("zai-org__GLM-5.3-Flash")
MIB_1 = 1_048_576
K_256 = 262_144


def _blocks(
    config: dict[str, Any],
    length: int,
    *,
    tp: int = 4,
    in_flight: int = 16_384,
    prefix_caching: bool = True,
    sm: int | None = 90,
) -> Blocks | None:
    kinds = layer_kinds(config, tp=tp, kv_dtype_bytes=2, model_dtype_bytes=2, sm=sm)
    assert kinds is not None
    return block_accounting(
        kinds,
        length,
        in_flight_tokens=in_flight,
        prefix_caching=prefix_caching,
        layout=kv_layout(config),
    )


def _bytes(config: dict[str, Any], length: int, **kw: Any) -> int | None:
    tp, sm = kw.pop("tp", 4), kw.pop("sm", 90)
    kinds = layer_kinds(config, tp=tp, kv_dtype_bytes=2, model_dtype_bytes=2, sm=sm)
    assert kinds is not None
    return bytes_per_sequence(kinds, length, layout=kv_layout(config), **kw)


def test_deepseek_v4_pages_as_flashmla_packs_them_on_sm90() -> None:
    """fp8_ds_mla, 256-token blocks, one KV head on every rank: TP does not matter."""
    for tp in (1, 4, 8):
        assert _blocks(V4, MIB_1, tp=tp) == Blocks(1_002_240, 10_778, 5)
    assert _bytes(V4, MIB_1, in_flight_tokens=16_384) == 10_802_142_720
    assert _bytes(V4, K_256, in_flight_tokens=16_384) == 7_723_261_440
    assert _bytes(V4, MIB_1, in_flight_tokens=4_096) == 5_798_960_640
    assert _bytes(V4, K_256, in_flight_tokens=4_096) == 2_720_079_360
    # SM 10.x takes the same FlashMLA path (nvidia/model.py:1085-1117).
    assert _blocks(V4, MIB_1, sm=100) == Blocks(1_002_240, 10_778, 5)


def test_deepseek_v41_pages_as_flashmla_packs_them_on_sm90() -> None:
    for tp in (1, 4, 8):
        assert _blocks(V41, MIB_1, tp=tp) == Blocks(116_928, 20_004, 9)
    assert _bytes(V41, MIB_1, in_flight_tokens=16_384) == 2_339_027_712
    assert _bytes(V41, K_256, in_flight_tokens=16_384) == 902_216_448
    assert _bytes(V41, MIB_1, in_flight_tokens=4_096) == 2_024_725_248
    assert _bytes(V41, K_256, in_flight_tokens=4_096) == 587_913_984


def test_qwen4_exp_packs_qsa_side_caches_beside_attention() -> None:
    """Boot-log check: "attention block size to 784" at TP 4."""
    assert _blocks(QWEN4, K_256) == Blocks(10_235_904, 344, 6)
    assert _blocks(QWEN4, K_256, tp=1) == Blocks(39_739_392, 177, 6)
    assert _bytes(QWEN4, K_256, in_flight_tokens=16_384) == 3_521_150_976
    assert _bytes(QWEN4, K_256, in_flight_tokens=16_384, prefix_caching=False) == 3_480_207_360
    assert _bytes(QWEN4, K_256, tp=1, in_flight_tokens=16_384) == 7_033_872_384
    assert (
        _bytes(QWEN4, K_256, tp=1, in_flight_tokens=16_384, prefix_caching=False) == 6_874_914_816
    )
    kinds = layer_kinds(QWEN4, tp=4, kv_dtype_bytes=2, model_dtype_bytes=2, sm=90)
    assert kinds is not None and {k.block_size for k in kinds if k.kind == "full"} == {784}


def test_glm5_next_groups_kda_with_the_mla_block() -> None:
    """Boot-log check: block 1152 at TP 4."""
    assert _blocks(GLM5, K_256) == Blocks(13_394_304, 237, 6)
    assert _blocks(GLM5, K_256, tp=1) == Blocks(50_600_704, 70, 6)
    assert _bytes(GLM5, K_256, in_flight_tokens=16_384) == 3_174_450_048
    assert _bytes(GLM5, K_256, in_flight_tokens=16_384, prefix_caching=False) == 3_120_872_832
    assert _bytes(GLM5, K_256, tp=1, in_flight_tokens=16_384) == 3_542_049_280
    assert (
        _bytes(GLM5, K_256, tp=1, in_flight_tokens=16_384, prefix_caching=False) == 3_339_646_464
    )
    kinds = layer_kinds(GLM5, tp=4, kv_dtype_bytes=2, model_dtype_bytes=2, sm=90)
    assert kinds is not None and {k.block_size for k in kinds if k.kind == "full"} == {1152}


@pytest.mark.parametrize(
    ("config", "sm", "kv_cache_dtype"),
    [
        (V4, 120, "auto"),  # SM 12.x: FlashInfer pages
        (V4, None, "auto"),  # GPU unknown
        (V4, 90, "bfloat16"),  # vLLM refuses a non-fp8 cache for fp8_ds_mla
        (V41, 100, "auto"),  # SM 10.x: 528-byte record, 128-token blocks
        (QWEN4, 90, "fp8"),  # QSA refuses a non-bf16 KV cache
        (GLM5, 120, "auto"),  # SM120 switches to fp8_ds_mla
        (GLM5, 90, "fp8"),  # fp8_ds_mla or plain fp8 by backend
    ],
)
def test_v30_layouts_are_unknown_where_they_were_not_traced(
    config: dict[str, Any], sm: int | None, kv_cache_dtype: str
) -> None:
    kinds = layer_kinds(
        config, tp=4, kv_dtype_bytes=2, model_dtype_bytes=2, sm=sm, kv_cache_dtype=kv_cache_dtype
    )
    assert kinds is None


def test_v30_models_get_layered_decode_and_the_guard_still_catches_the_rest() -> None:
    for config in (V4, V41, QWEN4, GLM5):
        assert family(config) is not None
        assert build_model_spec(config).components[0].mechanism == "layered_decode"
    # Missing a field layered.py reads: not modelled, and the guard catches it.
    no_indexer = {k: v for k, v in V4.items() if k != "index_head_dim"}
    no_ple = {**QWEN4, "text_config": {**QWEN4["text_config"], "ple_layer_ids": []}}
    for config in (no_indexer, no_ple):
        assert family(config) is None
        assert build_model_spec(config).components[0].mechanism == "unknown"
    # The v0.29 families keep their accounting.
    assert kv_layout(QWEN38) == kv_layout(GEMMA4) == "grouped"


def test_deepseek_v41_mxfp4_experts_are_padded_per_rank() -> None:
    """2304 / 4 = 576 rounds to 640 per rank: 40 layers x 384 experts x 64 rows."""
    assert padded_weight_bytes(V41, tp=4) == 8_021_606_400  # 7.47 GiB per GPU
    assert padded_weight_bytes(V41, tp=1) == 0  # 2304 is a multiple of 128
    assert padded_weight_bytes(V4, tp=4) == 0  # 2048 / 4 = 512
    assert padded_weight_bytes(QWEN4, tp=4) == 0
