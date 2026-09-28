"""The startup-peak (activation) estimate, term by term.

The expected byte counts are the allocations the GPU memory-history probe
recorded at the profile run's peak (``_dev_notes/cohort-run/peak-probe``,
vLLM v0.29.0, H100, 2026-09-28): each term below is one named tensor there.
"""

from __future__ import annotations

from typing import Any

from apron.domain.mechanisms import CalculatorInput, ComponentMechanism, TextWorkload
from apron.domain.mechanisms.calculator import (
    _activation_estimate,
    _encoder_batch,
    _encoder_peak_bytes,
    _forward_live_bytes,
    _multimodal_wrapper,
    activation_config,
)
from apron.domain.schemas.primitives import HardwareSpec

H100 = HardwareSpec(
    gpu_sku="NVIDIA H100 80GB HBM3", total_memory_bytes=85_899_345_920, compute_capability="9.0"
)
T = 8192  # vLLM's max_num_batched_tokens on an H100 (engine/arg_utils.py:2708-2727)

# Config fields as the Hub checkpoints declare them (text_config for Gemma 4).
GLM_47_FLASH: dict[str, Any] = {
    "hidden_size": 2048,
    "vocab_size": 154880,
    "intermediate_size": 10240,
    "num_attention_heads": 20,
    "n_routed_experts": 64,
    "num_experts_per_tok": 4,
    "moe_intermediate_size": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 192,
    "qk_rope_head_dim": 64,
    "v_head_dim": 256,
}
GEMMA_4_31B: dict[str, Any] = {
    "architectures": ["Gemma4ForConditionalGeneration"],
    "vision_config": {"hidden_size": 1152},
    "audio_config": None,
    "text_config": {
        "hidden_size": 5376,
        "vocab_size": 262144,
        "intermediate_size": 21504,
        "num_attention_heads": 32,
        "head_dim": 256,
        "global_head_dim": 512,
        "final_logit_softcapping": 30.0,
        "num_experts": None,
    },
}


def _inputs(tokens: int = T, seqs: int = 1024, **extra: Any) -> CalculatorInput:
    return CalculatorInput(
        mechanism=ComponentMechanism(mechanism="autoregressive_decode", role="decoder"),
        workload=TextWorkload(kind="text", input_length=512, output_length=128),
        artifact_metadata={},
        hardware=H100,
        execution_spec_data={"max_num_batched_tokens": tokens, "max_num_seqs": seqs, **extra},
    )


def _metadata(config: dict[str, Any]) -> dict[str, Any]:
    return {**activation_config(config), "torch_dtype": "bfloat16"}


def test_activation_config_merges_the_text_config_and_flags_towers() -> None:
    fields = activation_config(GEMMA_4_31B)
    assert fields["hidden_size"] == 5376
    assert fields["global_head_dim"] == 512
    assert fields["vision_config"] is True
    assert "audio_config" not in fields  # null in the checkpoint
    assert "num_experts" not in fields
    assert "architectures" not in fields


def test_glm_mla_moe_forward_is_the_probed_peak() -> None:
    """GLM-4.7-Flash: the MLA prefill-context dummy (mla_attention.py:783) and
    the fused-MoE workspace (workspace.py:207) are exactly the probe's two
    largest blocks; the graph buffers are the probe's observed counts."""
    prefill_context = 65536 * 20 * (192 + 256) * 2
    assert prefill_context == 1_174_405_120  # the probe's block
    workspace = T * 4 * (max(1536, 2048) + max(2 * 1536, 2048)) * 2
    assert workspace == 335_544_320  # the probe's block
    graph = 5 * T * 20 * 256 * 2 + 7 * T * 2048 * 2 + 2 * T * 512 * 2
    forward = _forward_live_bytes(_metadata(GLM_47_FLASH), _inputs(), 1, T, 2)
    assert forward == prefill_context + workspace + graph
    predicted = _activation_estimate(_metadata(GLM_47_FLASH), _inputs(), 1)
    assert predicted is not None
    assert predicted == forward  # above the compile-time embedding transient
    logged = 2_190_433_320
    assert abs(predicted - logged) / logged < 0.01


def test_mla_prefill_context_follows_the_engine_defaults() -> None:
    """W = round_up(min(max(8 x max_model_len, 4 x max_num_seqs x 16), 64k), 16):
    an A100's 256 sequences give 16384 tokens, an H100's 1024 reach the cap."""
    config = {**GLM_47_FLASH, "n_routed_experts": None}  # MLA alone
    graph = 5 * 2048 * 20 * 256 * 2 + 7 * 2048 * 2048 * 2 + 2 * 2048 * 512 * 2
    a100 = _forward_live_bytes(_metadata(config), _inputs(2048, 256), 1, 2048, 2)
    assert a100 - graph == 16384 * 20 * 448 * 2
    long_context = _inputs(2048, 256, max_model_len=4096)  # 8 x 4096 = 32768
    assert _forward_live_bytes(_metadata(config), long_context, 1, 2048, 2) - graph == (
        32768 * 20 * 448 * 2
    )


def test_gemma4_multimodal_drops_the_embedding_and_keeps_the_encoder_outputs() -> None:
    """Gemma 4 31B with its towers: the probe's peak in the forward --
    attention output of a global layer [T, 32 x 512], two [T, H], gate_up
    [T, 2I], activation [T, I] -- plus the dummy encoder outputs."""
    metadata = _metadata(GEMMA_4_31B)
    assert _multimodal_wrapper(metadata, _inputs())
    live = {
        "attention_out": T * 32 * 512 * 2,
        "o_proj_and_residual": 2 * T * 5376 * 2,
        "gate_up": T * 2 * 21504 * 2,
        "activation": T * 21504 * 2,
    }
    assert live == {  # the probe's blocks
        "attention_out": 268_435_456,
        "o_proj_and_residual": 2 * 88_080_384,
        "gate_up": 704_643_072,
        "activation": 352_321_536,
    }
    encoder_outputs = T * 5376 * 2  # budget bound; the probe's block is 66,060,288
    predicted = _activation_estimate(metadata, _inputs(), 1)
    assert predicted is not None
    assert predicted == sum(live.values()) + encoder_outputs
    logged = 1_599_875_317
    assert abs(predicted - logged) / logged < 0.01


def test_gemma4_language_model_only_is_the_compile_transient() -> None:
    """--language-model-only: the embedding is compiled and benchmarked; the
    probe's peak is the random embedding tensor plus two [T, H] outputs."""
    inputs = _inputs(language_model_only=True)
    metadata = _metadata(GEMMA_4_31B)
    assert not _multimodal_wrapper(metadata, inputs)
    predicted = _activation_estimate(metadata, inputs, 1)
    assert predicted is not None
    assert predicted == 262144 * 5376 * 2 + 2 * T * 5376 * 2
    assert abs(predicted - 2_995_739_688) < 2 * 2**20


def test_dense_text_model_keeps_the_compile_rule() -> None:
    """Qwen3-32B on an H100 (1.61 GiB logged): the embedding transient is
    above the forward's live set, so the compile rule still decides."""
    config = {
        "hidden_size": 5120,
        "vocab_size": 151936,
        "intermediate_size": 25600,
        "num_attention_heads": 64,
        "head_dim": 128,
    }
    compile_peak = 151936 * 5120 * 2 + 2 * T * 5120 * 2
    forward = _forward_live_bytes(_metadata(config), _inputs(), 1, T, 2)
    assert forward == T * 64 * 128 * 2 + 2 * T * 5120 * 2 + 3 * T * 25600 * 2
    assert forward < compile_peak
    assert _activation_estimate(_metadata(config), _inputs(), 1) == compile_peak


# NemotronH (Nemotron-3.5-Lightning-30B-A3B), config.json fields.
NEMOTRON_35: dict[str, Any] = {
    "model_type": "nemotron_h",
    "hidden_size": 2688,
    "vocab_size": 131072,
    "intermediate_size": 1856,
    "moe_intermediate_size": 1856,
    "n_routed_experts": 128,
    "num_experts_per_tok": 6,
    "num_attention_heads": 32,
    "head_dim": 128,
    "mamba_num_heads": 64,
    "mamba_head_dim": 64,
    "n_groups": 8,
    "ssm_state_size": 128,
    "mlp_hidden_act": "relu2",
}


def test_nemotron_h_peak_is_the_piece_between_two_mixers() -> None:
    """vLLM v0.30.0, one H100: the ungated experts' workspace, the previous
    mixer's in_proj output, SSM output and residual held through the piece,
    and the next mixer's residual, in_proj output and SSM output.  Above the
    compile transient (0.74 GiB); the log says 1.09 GiB -- 0.076 GiB (7%) is
    not accounted for from source."""
    workspace = T * 6 * (max(1856, 2688) + max(1856, 2688)) * 2  # no gate: w13 is [E, I, H]
    in_proj = T * (2 * 4096 + 2 * 8 * 128 + 64) * 2
    ssm_out = T * 4096 * 2
    residual = T * 2688 * 2
    expected = workspace + 2 * (in_proj + ssm_out + residual)
    assert expected == 1_088_421_888
    metadata = _metadata(NEMOTRON_35)
    assert _forward_live_bytes(metadata, _inputs(), 1, T, 2) == expected
    compile_peak = 131072 * 2688 * 2 + 2 * residual
    assert compile_peak < expected
    assert _activation_estimate(metadata, _inputs(seqs=556), 1) == expected
    logged = 1_170_378_588  # 1.09 GiB
    assert abs(expected - logged) / logged < 0.10


def test_gated_moe_workspace_is_unchanged_without_the_nemotron_fields() -> None:
    config = {
        key: value
        for key, value in NEMOTRON_35.items()
        if not key.startswith(("mamba", "n_g", "ssm", "mlp"))
    }
    workspace = T * 6 * (max(1856, 2688) + max(2 * 1856, 2688)) * 2
    forward = _forward_live_bytes(_metadata(config), _inputs(), 1, T, 2)
    assert forward == T * 32 * 128 * 2 + 2 * T * 2688 * 2 + workspace


def test_sampler_keeps_a_second_logits_copy_under_a_soft_cap() -> None:
    """A text-free wrapper with a tiny forward: the sampler decides.  The soft
    cap (logits / cap, tanh, * cap) holds two logits tensors at once."""
    config = {"hidden_size": 1024, "vocab_size": 262144, "vision_config": {"x": 1}}
    rows = 1024
    plain = _activation_estimate(_metadata(config), _inputs(), 1)
    assert plain == rows * 262144 * 2 + T * 1024 * 2  # logits and hidden states
    capped_config = {**config, "final_logit_softcapping": 30.0}
    capped = _activation_estimate(_metadata(capped_config), _inputs(), 1)
    assert capped == 2 * rows * 262144 * 2 + T * 1024 * 2


def test_tensor_parallel_keeps_the_gathered_logits_rule() -> None:
    config = {"hidden_size": 4096, "vocab_size": 151936}
    logits = 256 * 151936 * 2
    predicted = _activation_estimate(_metadata(config), _inputs(2048, 256), 2)
    assert predicted == 2 * logits + logits // 2 + 2048 * 4096 * 2


# ---------------------------------------------------------------------------
# Encoder phase (vLLM v0.30.0 probes of Qwen3.6-35B-A3B-FP8 and Muse-Glimmer-30B)
# ---------------------------------------------------------------------------

QWEN36_VISION: dict[str, Any] = {
    "model_type": "qwen3_5_moe",
    "text_config": {"hidden_size": 2048, "vocab_size": 248320, "num_attention_heads": 16},
    "vision_config": {
        "hidden_size": 1152,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
}
QWEN36_PROCESSOR: dict[str, Any] = {
    "image_processor": {"size": {"longest_edge": 16777216, "shortest_edge": 65536}},
    "video_processor": {"size": {"longest_edge": 25165824, "shortest_edge": 4096}},
}
MUSE_VISION: dict[str, Any] = {
    "model_type": "muse_glimmer",
    "text_config": {"hidden_size": 6656, "vocab_size": 202048, "num_attention_heads": 32},
    "vision_config": {"hidden_size": 1536, "patch_size": 14, "merge_size": 2, "patch_temporal": 2},
}
MUSE_PROCESSOR: dict[str, Any] = {
    "image_processor": {"max_image_tokens": 4096},
    "video_processor": {"max_video_frame_tokens": 144, "num_frames": 96},
}


def test_encoder_batch_is_one_qwen_image_of_16384_tokens() -> None:
    """longest_edge 4096^2 pixels / 32^2 = 16384 merged tokens per image; the
    video (two frames under 25165824 / 2) has 12288, so the image is profiled.
    The budget is max(8192, 16384): one item, 65536 patches -- the probe's
    qkv output is 65536 x 3456 x 2 = 452,984,832 B."""
    metadata = _metadata_with(QWEN36_VISION, QWEN36_PROCESSOR)
    assert _encoder_batch(metadata, _inputs(), T) == (1, 16384 * 4)
    assert 65536 * 3 * 1152 * 2 == 452_984_832


def test_encoder_batch_is_two_muse_images() -> None:
    """max_image_tokens 4096 < the 8192-token budget: two images of 16384
    patches (the video's 8 frames under max_model_len 640 give 576 tokens)."""
    metadata = _metadata_with(MUSE_VISION, MUSE_PROCESSOR)
    assert _encoder_batch(metadata, _inputs(), T) == (2, 16384)
    assert 32768 * 3 * 1536 * 2 == 301_989_888  # the probe's qkv output


def test_qwen_vl_encoder_peak_is_the_probed_block_attention() -> None:
    unit = 65536 * 1152 * 2  # the probe's 150,994,944 B blocks
    metadata = _metadata_with(QWEN36_VISION, QWEN36_PROCESSOR)
    expected = (
        2 * unit  # position embeddings, embedded patches
        + unit  # norm1
        + 3 * unit  # qkv
        + 2 * unit  # q, k contiguous
        + max(2 * unit, 2 * unit + unit + unit)  # rotary out; + attention, projection
        + 65536 * 3 * 2 * 16 * 16 * 2  # bf16 patches on the GPU: 201,326,592 B
    )
    assert _encoder_peak_bytes(metadata, _inputs(), 1, T, 2) == expected
    predicted = _activation_estimate(metadata, _inputs(), 1)
    assert predicted == expected  # above the text forward and the sampler
    assert predicted is not None
    assert abs(predicted - 2_061_584_302) / 2_061_584_302 < 0.03


def test_muse_encoder_peak_holds_the_fp32_rotary() -> None:
    unit = 32768 * 1536 * 2  # the probe's 100,663,296 B blocks
    metadata = _metadata_with(MUSE_VISION, MUSE_PROCESSOR)
    expected = (
        2 * unit  # per-image states, their concatenation
        + unit  # ln_1
        + 3 * unit  # qkv
        + 2 * unit  # stacked q, k
        + (4 * unit + 4 * unit + 2 * unit)  # FP32 copy, FP32 rotary out, bf16 back
        + 32768 * 3 * 14 * 14 * 4  # FP32 images on the GPU: 77,070,336 B
        + 16384 * 3 * 14 * 14 * 2  # bf16 copy of the last image: 19,267,584 B
    )
    assert _encoder_peak_bytes(metadata, _inputs(), 1, T, 2) == expected
    predicted = _activation_estimate(metadata, _inputs(), 1)
    assert predicted is not None
    assert abs(predicted - 1_964_947_537) / 1_964_947_537 < 0.03


def test_encoder_phase_needs_a_traced_tower_and_the_processor() -> None:
    no_processor = _metadata_with(QWEN36_VISION, None)
    assert _encoder_peak_bytes(no_processor, _inputs(), 1, T, 2) == 0
    gemma = _metadata_with({**GEMMA_4_31B, "model_type": "gemma4"}, {"image_processor": {}})
    assert _encoder_peak_bytes(gemma, _inputs(), 1, T, 2) == 0
    # --language-model-only: no encoder profiling, the embedding is compiled.
    metadata = _metadata_with(QWEN36_VISION, QWEN36_PROCESSOR)
    lm_only = _inputs(language_model_only=True)
    assert _activation_estimate(metadata, lm_only, 1) == 248320 * 2048 * 2 + 2 * T * 2048 * 2


def _metadata_with(config: dict[str, Any], processor: dict[str, Any] | None) -> dict[str, Any]:
    return {**activation_config(config, processor), "torch_dtype": "bfloat16"}
