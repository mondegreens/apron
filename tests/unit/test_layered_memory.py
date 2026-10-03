"""Per-layer caches counted as vLLM v0.29.0 pages them (domain/mechanisms/layered.py).

Expected bytes are the ones traced from the pinned source
(docs/traces/hybrid-memory-trace.md, gemma4-memory-trace.md) with the
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
    dsa_indexer_layers,
    duplicated_weight_bytes,
    family,
    kv_layout,
    layer_kinds,
    padded_weight_bytes,
    required_block_size,
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
# numbers from docs/traces/deepseek-v4-kv-trace.md and
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


def _pages(config: dict[str, Any], *, sm: int, tp: int = 2) -> list[tuple[str, int, int, int]]:
    kinds = layer_kinds(config, tp=tp, kv_dtype_bytes=2, model_dtype_bytes=2, sm=sm)
    assert kinds is not None
    return [(k.kind, k.count, k.block_size, k.page_bytes) for k in kinds]


def test_deepseek_v41_pages_the_mxfp8_record_on_128_token_blocks_on_sm100() -> None:
    """SM 10.x keeps FlashMLA (nvidia/model.py:117-152) but writes the V4.1
    record, 512 fp8 + 16 scale bytes = 528 in 512-byte pages (attention.py:
    113-120, 459-461, 1023-1027), and the block is the FlashMLA / indexer
    backends' 128 (sparse_mla.py:88-90, indexer.py:261-263).  Bytes from vLLM's
    own _get_packed_kv_cache_groups and accounting run on these specs."""
    assert _pages(V41, sm=100) == [
        ("sliding", 40, 32, 16_896),  # 32 x 528
        ("full", 3, 128, 33_792),  # ratio 2: 64 x 528
        ("compressed", 3, 128, 8_704),  # 64 x 132 = 8448 -> 17 x 512
        ("ring", 3, 8, 32_768),  # 8 x 2 x 512 x fp32, as on SM 9.x
        ("full", 1, 128, 67_584),  # ratio 1: 128 x 528
        ("compressed", 1, 128, 16_896),  # 128 x 132
    ]
    for tp in (1, 2, 8):
        assert _blocks(V41, MIB_1, tp=tp, sm=100) == Blocks(211_968, 10_261, 6)
    assert _bytes(V41, MIB_1, sm=100, in_flight_tokens=16_384) == 2_175_003_648
    assert _bytes(V41, K_256, sm=100, in_flight_tokens=16_384) == 872_672_256
    assert _bytes(V41, MIB_1, sm=100, in_flight_tokens=4_096) == 1_849_420_800
    assert _bytes(V41, K_256, sm=100, in_flight_tokens=4_096) == 547_089_408
    assert _bytes(V41, 32_768, tp=2, sm=100, in_flight_tokens=16_384) == 492_825_600
    assert _bytes(V41, 32_768, tp=2, sm=100, in_flight_tokens=4_096) == 167_242_752
    # SM 10.3 (B300) is the same family (platforms/interface.py:481-493).
    assert _pages(V41, sm=103) == _pages(V41, sm=100)


def test_deepseek_v41_sm90_pages_are_unchanged() -> None:
    assert _pages(V41, sm=90) == [
        ("sliding", 40, 32, 19_008),
        ("full", 3, 64, 19_008),
        ("compressed", 3, 64, 4_608),
        ("ring", 3, 8, 32_768),
        ("full", 1, 64, 37_440),
        ("compressed", 1, 64, 8_640),
    ]
    assert _bytes(V41, 32_768, tp=2, sm=90, in_flight_tokens=16_384) == 483_146_496


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
        (V41, 120, "auto"),  # SM 12.x: FlashInfer's SM120 class
        (V41, 100, "bfloat16"),  # fp8_ds_mla refuses a non-fp8 cache on SM 10.x too
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
    assert padded_weight_bytes(V41, tp=4, sm=100) == 8_021_606_400  # every arch


def test_qwen4exp_bf16_experts_are_padded_to_128_on_trtllm() -> None:
    """oracle/unquantized.py:347-355, flashinfer_utils.py:309-351: the rank's
    intermediate size rounds up to 128; w13 grows by 2 x extra rows, w2 by
    extra columns, per expert and MoE layer, in bfloat16."""
    per_row = 48 * 512 * 3 * 2560 * 2  # MoE layers x experts x (w13 2 + w2 1) x H x bytes
    assert padded_weight_bytes(QWEN4, tp=2, sm=100) == per_row * (384 - 320)  # 22.5 GiB
    assert padded_weight_bytes(QWEN4, tp=2, sm=100) == 24_159_191_040
    assert padded_weight_bytes(QWEN4, tp=4, sm=100) == per_row * (256 - 160)
    assert padded_weight_bytes(QWEN4, tp=8, sm=100) == per_row * (128 - 80)
    assert padded_weight_bytes(QWEN4, tp=1, sm=100) == 0  # 640 = 5 x 128
    assert padded_weight_bytes(QWEN4, tp=2, sm=103) == per_row * 64  # SM 10.x family


@pytest.mark.parametrize("sm", [None, 80, 89, 90, 120])
def test_qwen4exp_experts_are_not_padded_off_sm100(sm: int | None) -> None:
    """TRTLLM BF16 MoE runs on the SM 10.x family only (trtllm_bf16_moe.py:
    87-93); SM 9.x takes Triton (oracle/unquantized.py:77-79), no padding."""
    assert padded_weight_bytes(QWEN4, tp=2, sm=sm) == 0


def test_trtllm_padding_follows_the_backend_gates() -> None:
    text = QWEN4["text_config"]

    def pad(backend: str = "auto", **changes: Any) -> int:
        config = {**QWEN4, "text_config": {**text, **changes}}
        return padded_weight_bytes(config, tp=2, sm=100, moe_backend=backend)

    assert pad() == 24_159_191_040
    assert pad("flashinfer_trtllm") == 24_159_191_040  # requested explicitly
    assert pad("humming") == 24_159_191_040  # falls through to auto
    assert pad("triton") == pad("flashinfer_cutlass") == 0  # no padding there
    # A quantization config takes a quantized MoE method (routed_experts.py:196-199).
    assert pad(quantization_config={"quant_method": "fp8"}) == 0
    # Softmax without renormalization is Default routing, which TRTLLM refuses.
    assert pad(norm_topk_prob=False) == 0
    assert pad(norm_topk_prob=True) == 24_159_191_040
    # The TRTLLM layout takes bfloat16 only (flashinfer_utils.py:118-121).
    assert pad(dtype="float16") == 0
    # More than 2048 experts: TRTLLM routing refuses (trtllm_bf16_moe.py:55-60).
    assert pad(num_experts=4096) == 0
    # Dense layers carry no experts (qwen4_exp/nvidia/model.py:239-244).
    assert pad(decoder_sparse_step=2) == 24_159_191_040 // 2
    assert pad(mlp_only_layers=[0, 1, 2]) == 24_159_191_040 // 48 * 45
    # Other families are not traced.
    assert padded_weight_bytes(M3, tp=2, sm=100) == 0
    assert padded_weight_bytes(GLM53, tp=4, sm=100) == 0


# ---------------------------------------------------------------------------
# GLM-5.3 (glm_moe_dsa) and MiniMax-M3 on vLLM v0.30.0: every layer a
# full-attention spec on one block size, one group of all of them
# (UniformTypeKVCacheSpecs).  Expected numbers from vLLM's own get_kv_cache_groups
# and accounting run on the specs the model code builds
# (docs/traces/glm53-minimax-m3-kv-trace.md).
# ---------------------------------------------------------------------------


GLM53 = _recorded("zai-org__GLM-5.3")
M3 = _recorded("MiniMaxAI__MiniMax-M3")


def test_glm53_pages_bf16_mla_latent_and_fp8_indexer_keys_on_64_token_blocks() -> None:
    """78 MLA pages of 64 x 1152 B + 21 indexer pages of 64 x 132 B per block;
    one KV head, so every rank keeps the whole cache whatever the TP."""
    for tp in (1, 8):
        assert _blocks(GLM53, MIB_1, tp=tp) == Blocks(5_928_192, 16_384, 1)
    assert _bytes(GLM53, MIB_1, tp=8, in_flight_tokens=16_384) == 97_127_497_728  # 90.46 GiB
    assert _bytes(GLM53, K_256, tp=8, in_flight_tokens=16_384) == 24_281_874_432  # 22.61 GiB
    # Full attention only: the in-flight tokens do not matter.
    assert _bytes(GLM53, K_256, tp=8, in_flight_tokens=4_096) == 24_281_874_432
    assert _bytes(GLM53, 640, tp=8, in_flight_tokens=16_384) == 59_281_920


def test_glm53_indexer_layers_follow_vllms_rule_not_the_config_list() -> None:
    """vLLM reads index_topk_freq / index_skip_topk_offset; the checkpoint's own
    indexer_types says the same 21 layers."""
    layers = dsa_indexer_layers(GLM53)
    assert layers == [0, 1, 2, *range(6, 78, 4)]
    assert layers == [i for i, t in enumerate(GLM53["indexer_types"]) if t == "full"]


def test_minimax_m3_pages_kv_heads_and_index_keys_on_128_token_blocks() -> None:
    """60 GQA pages (dense and sparse layers alike) + 57 index-key pages.  At
    TP 8 the 4 KV heads leave one per rank, as at TP 4."""
    for tp in (4, 8):
        assert _blocks(M3, MIB_1, tp=tp) == Blocks(5_799_936, 8_192, 1)
    assert _bytes(M3, MIB_1, tp=8, in_flight_tokens=16_384) == 47_513_075_712  # 44.25 GiB
    assert _bytes(M3, K_256, tp=8, in_flight_tokens=16_384) == 11_878_268_928  # 11.06 GiB
    assert _bytes(M3, 640, tp=8, in_flight_tokens=16_384) == 28_999_680
    assert _blocks(M3, MIB_1, tp=1) == Blocks(17_596_416, 8_192, 1)
    assert _bytes(M3, K_256, tp=1, in_flight_tokens=16_384) == 36_037_459_968


def test_minimax_m3_needs_128_token_blocks_to_boot() -> None:
    """vLLM's own block size (16, from the dense first layer) has no common
    kernel block with the sparse backends: the plan must set 128."""
    assert required_block_size(M3) == 128
    assert required_block_size(GLM53) is None
    assert required_block_size(GLM5) is None


def test_glm53_pages_the_same_on_sm100() -> None:
    """SM 10.x runs FLASHINFER_MLA_SPARSE at up to 16 heads per rank (TP 4, 8)
    and FLASHMLA_SPARSE above (TP 2) (platforms/cuda.py:96-130); neither
    rewrites a bf16 cache (mla_attention.py:359-376), and the block is still
    64, from layer 0's indexer cache (indexer.py:202-204)."""
    for tp in (2, 4, 8):
        assert _pages(GLM53, sm=100, tp=tp) == _pages(GLM53, sm=90, tp=tp)
        assert _blocks(GLM53, MIB_1, tp=tp, sm=100) == Blocks(5_928_192, 16_384, 1)
        assert _bytes(GLM53, 32_768, tp=tp, sm=100, in_flight_tokens=16_384) == 3_035_234_304
    assert _pages(GLM53, sm=90, tp=8) == [("full", 78, 64, 73_728), ("compressed", 21, 64, 8_448)]


def test_glm53_on_sm100_needs_an_indexer_on_layer_0() -> None:
    """Without one the MLA backend sets the block (FlashInfer: 32), which the
    indexer's 64-token kernel cannot serve: not modelled.  SM 9.x is 64 either way."""
    skip_first = {**GLM53, "index_topk_pattern": "S" + "F" * 77}
    assert 0 not in dsa_indexer_layers(skip_first)
    assert layer_kinds(skip_first, tp=8, kv_dtype_bytes=2, model_dtype_bytes=2, sm=100) is None
    assert layer_kinds(skip_first, tp=8, kv_dtype_bytes=2, model_dtype_bytes=2, sm=90) is not None


def test_minimax_m3_pages_the_same_on_sm100() -> None:
    """The MSA backends only swap metadata builders (sparse_attention_msa.py:
    46-51, indexer_msa.py:65-70); the pages are the layers' own specs."""
    for tp in (2, 4, 8):
        assert _pages(M3, sm=100, tp=tp) == _pages(M3, sm=90, tp=tp)
    for tp in (4, 8):
        assert _blocks(M3, MIB_1, tp=tp, sm=100) == Blocks(5_799_936, 8_192, 1)
        assert _bytes(M3, 32_768, tp=tp, sm=100, in_flight_tokens=16_384) == 1_484_783_616
    assert _pages(M3, sm=90, tp=8) == [("full", 60, 128, 65_536), ("compressed", 57, 128, 32_768)]
    assert required_block_size(M3) == 128


@pytest.mark.parametrize(
    ("config", "sm", "kv_cache_dtype"),
    [
        (GLM53, 120, "auto"),  # SM 12.x: fp8_ds_mla
        (GLM53, None, "auto"),  # GPU unknown
        (GLM53, 90, "fp8"),  # fp8_ds_mla, 656 B per token
        (GLM53, 100, "fp8"),  # FlashInfer's plain fp8 or FlashMLA's fp8_ds_mla
        (M3, 120, "auto"),
        (M3, 90, "fp8"),
        (M3, 100, "fp8"),
    ],
)
def test_glm53_and_minimax_m3_are_unknown_where_they_were_not_traced(
    config: dict[str, Any], sm: int | None, kv_cache_dtype: str
) -> None:
    kinds = layer_kinds(
        config, tp=8, kv_dtype_bytes=2, model_dtype_bytes=2, sm=sm, kv_cache_dtype=kv_cache_dtype
    )
    assert kinds is None


def test_glm53_and_minimax_m3_get_layered_decode() -> None:
    """GLM-5.3 was unknown (index_topk); MiniMax-M3 was read as an
    encoder-decoder from its ForConditionalGeneration suffix."""
    for config in (GLM53, M3):
        assert build_model_spec(config).components[0].mechanism == "layered_decode"
        assert kv_layout(config) == "uniform"
    # Without the fields layered.py reads they stay explicit unknowns.
    text = {**M3["text_config"]}
    text["sparse_attention_config"] = {
        k: v for k, v in text["sparse_attention_config"].items() if k != "sparse_attention_freq"
    }
    no_freq = {**M3, "text_config": text}
    no_index_dim = {k: v for k, v in GLM53.items() if k != "index_head_dim"}
    for config in (no_freq, no_index_dim):
        assert family(config) is None
        assert build_model_spec(config).components[0].mechanism == "unknown"


def test_gpt_oss_experts_are_padded_on_marlin_only() -> None:
    """gpt-oss's MXFP4 experts run on Marlin below SM 9.0, which rounds the
    hidden size to 256 and the rank's intermediate size to 128
    (fused_moe/oracle/mxfp4.py:745-749): 2880 -> 3072 x 2944 on an RTX 4090,
    0.855 GiB for gpt-oss-20b; OpenAI's Triton kernels (SM 9.x) keep 2880."""
    from apron.domain.mechanisms.layered import gpt_oss_marlin_padding

    gpt_oss_20b = {
        "model_type": "gpt_oss",
        "num_hidden_layers": 24,
        "num_local_experts": 32,
        "hidden_size": 2880,
        "intermediate_size": 2880,
        "quantization_config": {"quant_method": "mxfp4"},
    }
    assert gpt_oss_marlin_padding(gpt_oss_20b, tp=1, sm=89) == 917_962_752
    assert padded_weight_bytes(gpt_oss_20b, tp=1, sm=89) == 917_962_752
    assert gpt_oss_marlin_padding(gpt_oss_20b, tp=1, sm=90) == 0  # Triton
    assert gpt_oss_marlin_padding(gpt_oss_20b, tp=1, sm=None) == 0
    assert gpt_oss_marlin_padding(gpt_oss_20b, tp=1, sm=89, moe_backend="triton") == 0
    assert gpt_oss_marlin_padding({**gpt_oss_20b, "model_type": "qwen3"}, tp=1, sm=89) == 0
