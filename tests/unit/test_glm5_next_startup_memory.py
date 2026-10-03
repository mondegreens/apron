"""GLM-5.3-Flash (GLM5Next) on SM 10.x: the startup memory terms traced from
vLLM v0.30.0 against the 2x B200 boot (2026-09-28, TP 2, max_model_len 32768,
16,384 batched tokens, 13 sequences): weights 152.10 GiB, torch peak
4,702,989,189 B, CUDA-graph estimate 1.68 GiB (0.09 GiB actual).

Planned from the recorded config.json and safetensors headers
(tests/fixtures/cohort/headers), no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from apron.application.orchestration.plan_pipeline import (
    calculator_metadata,
    loaded_tensor_bytes,
    replicated_tensor_bytes,
    widened_tensor_bytes,
)
from apron.domain.mechanisms import CalculatorInput, ComponentMechanism, TextWorkload
from apron.domain.mechanisms.calculator import (
    _activation_estimate,
    _kpool_indexer_workspace_bytes,
    _trtllm_fp8_experts,
    activation_config,
    calculate_layered_decode,
    cuda_graph_estimate_bytes,
)
from apron.domain.mechanisms.layered import (
    FLASHINFER_MLA_SPARSE_WORKSPACE_BYTES,
    first_capture_bytes,
    startup_buffer_bytes,
)
from apron.domain.mechanisms.model_spec_builder import build_model_spec
from apron.domain.schemas.primitives import HardwareSpec

HEADERS = Path(__file__).parents[1] / "fixtures" / "cohort" / "headers"
B200 = HardwareSpec(
    gpu_sku="NVIDIA B200", total_memory_bytes=191_495_471_104, compute_capability="10.0"
)
H200 = HardwareSpec(
    gpu_sku="NVIDIA H200", total_memory_bytes=150_754_820_096, compute_capability="9.0"
)
LAYER = "model.language_model.layers"


def _record(name: str) -> dict[str, Any]:
    record: dict[str, Any] = json.loads((HEADERS / f"{name}.json").read_text())
    return record


GLM5 = _record("zai-org__GLM-5.3-Flash")
CONFIG = GLM5["config"]
META = {name: (dtype, size) for name, (dtype, size, _) in GLM5["tensors"].items()}


def _inputs(hardware: HardwareSpec, **spec: Any) -> CalculatorInput:
    return CalculatorInput(
        mechanism=ComponentMechanism(mechanism="layered_decode", role="decoder"),
        workload=TextWorkload(kind="text", input_length=32640, output_length=128),
        artifact_metadata={},
        hardware=hardware,
        execution_spec_data={
            "tensor_parallel": 2,
            "max_num_batched_tokens": 16384,
            "max_num_seqs": 13,
            "max_model_len": 32768,
            "gpu_memory_utilization": 0.9,
            "max_batch_size": 1,
            **spec,
        },
    )


def test_mla_projections_load_in_bf16_and_fp32_parameters_in_fp32() -> None:
    """The MLA layers are built without the fp8 config: their fp8 q_a / kv_a /
    q_b / o projections load as bf16, scales dropped; the hyper-connection
    mixes, the kpool APE and the KDA convolutions are float32 parameters."""
    loaded = loaded_tensor_bytes(META, CONFIG)
    q_b = f"{LAYER}.3.self_attn.q_b_proj.weight"
    assert META[q_b] == ("F8_E4M3", 25_165_824) and loaded[q_b] == 50_331_648
    assert loaded[q_b + "_scale_inv"] == 0
    kda_o = f"{LAYER}.4.self_attn.o_proj.weight"  # a KDA layer: stored bf16
    assert META[kda_o] == ("BF16", 67_108_864) and loaded[kda_o] == 67_108_864
    for name, stored in (
        (f"{LAYER}.0.hc_attn_fn", 786_432),
        (f"{LAYER}.0.self_attn.k_conv1d.weight", 65_536),
        (f"{LAYER}.3.self_attn.indexer.index_kpool_compress_ape", 1_024),
    ):
        assert META[name] == ("BF16", stored) and loaded[name] == 2 * stored
    expert = f"{LAYER}.3.mlp.experts.*.gate_proj.weight"
    assert loaded[expert] == META[expert][1]  # the experts stay fp8
    # MTP layer 45 left out: the recorded row's figure.
    assert widened_tensor_bytes(META, CONFIG) == 1_184_500_736


@pytest.mark.parametrize(
    ("name", "whole"),
    [
        (f"{LAYER}.3.self_attn.q_a_proj.weight", True),
        (f"{LAYER}.3.self_attn.kv_a_proj_with_mqa.weight", True),
        (f"{LAYER}.3.self_attn.indexer.wq_b.weight", True),
        (f"{LAYER}.0.self_attn.f_a_proj.weight", True),
        (f"{LAYER}.0.self_attn.o_norm.weight", True),
        (f"{LAYER}.3.mlp.gate.weight", True),
        (f"{LAYER}.0.hc_ffn_fn", True),
        ("model.language_model.norm.weight", True),
        ("model.visual.patch_embed.proj.weight", True),
        ("model.visual.blocks.0.attn.proj.bias", True),
        (f"{LAYER}.3.self_attn.q_b_proj.weight", False),
        (f"{LAYER}.3.self_attn.kv_b_proj.weight", False),
        (f"{LAYER}.0.self_attn.q_proj.weight", False),
        (f"{LAYER}.3.mlp.experts.*.gate_proj.weight", False),
        ("model.visual.blocks.0.attn.qkv.weight", False),
        (f"{LAYER}.45.self_attn.q_a_proj.weight", False),  # the MTP layer
    ],
)
def test_replicated_weights(name: str, whole: bool) -> None:
    assert replicated_tensor_bytes({name: 1}, CONFIG) == int(whole)


def test_replicated_total() -> None:
    loaded = loaded_tensor_bytes(META, CONFIG)
    assert replicated_tensor_bytes(loaded, CONFIG) == 697_944_824


def test_startup_buffers_are_the_top_k_index_buffers() -> None:
    """One [T, 2176] int32 buffer shared by the model, and on each of the 11
    MLA layers [T + 1, 2176] physical indices, [T + 1] counts, T request ids."""
    per_layer = 16_385 * 2_176 * 4 + 16_385 * 4 + 16_384 * 4
    assert startup_buffer_bytes(CONFIG, max_num_batched_tokens=16384) == (
        16_384 * 2_176 * 4 + 11 * per_layer
    )
    assert startup_buffer_bytes(CONFIG, max_num_batched_tokens=8192) == 856_454_700
    assert (
        startup_buffer_bytes(_record("zai-org__GLM-5.3")["config"], max_num_batched_tokens=16384)
        == 0
    )
    dense = {**CONFIG, "text_config": {**CONFIG["text_config"], "index_topk": None}}
    assert startup_buffer_bytes(dense, max_num_batched_tokens=16384) == 0


def test_first_capture_allocations() -> None:
    """FlashInfer's sparse MLA workspace on SM 10.x (FlashMLA takes only head
    576; GLM5Next's is 512) and the KDA layers' merged float32 convolutions."""
    conv = 34 * 3 * (64 * 128 // 2) * 4 * 4
    assert conv == 6_684_672
    assert (
        first_capture_bytes(CONFIG, tp=2, sm=100) == FLASHINFER_MLA_SPARSE_WORKSPACE_BYTES + conv
    )
    assert FLASHINFER_MLA_SPARSE_WORKSPACE_BYTES == 413_138_944
    assert first_capture_bytes(CONFIG, tp=2, sm=90) == conv  # SM 9.x backend: not traced
    assert first_capture_bytes(_record("zai-org__GLM-5.3")["config"], tp=8, sm=100) == 0


def test_trtllm_fp8_experts_take_no_fused_moe_workspace() -> None:
    metadata = {**activation_config(CONFIG), "torch_dtype": "bfloat16"}
    assert metadata["fp8_block_experts"] is True
    assert _trtllm_fp8_experts(metadata, _inputs(B200), 2)
    assert not _trtllm_fp8_experts(metadata, _inputs(H200), 2)  # SM 9.0: Triton
    assert not _trtllm_fp8_experts(metadata, _inputs(B200, moe_backend="triton"), 2)
    # 2048 / 32 = 64 per rank: the scales are refined and only Triton takes them.
    assert not _trtllm_fp8_experts(metadata, _inputs(B200), 32)
    unquantized = {k: v for k, v in metadata.items() if k != "fp8_block_experts"}
    assert not _trtllm_fp8_experts(unquantized, _inputs(B200), 2)
    # The Triton-shape workspace, 16384 x 8 x (4096 + 4096) x 2 B, is gone.
    triton = _activation_estimate(unquantized, _inputs(B200), 2)
    trtllm = _activation_estimate(metadata, _inputs(B200), 2)
    assert triton is not None and trtllm is not None
    assert triton - trtllm == 16_384 * 8 * (4_096 + 4_096) * 2 - 174_063_616


def test_kpool_indexer_workspace_and_mla_block() -> None:
    metadata = {**activation_config(CONFIG), "torch_dtype": "bfloat16"}
    # 40 x 32768 rows of 128 fp8 bytes, 4-byte scales, a 1 MiB radix workspace.
    assert _kpool_indexer_workspace_bytes(metadata, 32768) == 174_063_616
    aligned = _activation_estimate(metadata, _inputs(B200), 2, mla_block=2176)
    plain = _activation_estimate(metadata, _inputs(B200), 2)
    assert aligned is not None and plain is not None
    # The prefill dummy rounds 65,536 up to 31 blocks of 2,176.
    assert aligned - plain == (67_456 - 65_536) * 32 * (256 + 256) * 2


def test_plan_on_2x_b200_against_the_boot() -> None:
    """Predicted per GPU for the boot (measured: weights 163,316,131,430 B,
    peak 4,702,989,189 B, CUDA-graph estimate 1,803,886,264 B)."""
    loaded = loaded_tensor_bytes(META, CONFIG)
    total = sum(loaded.values()) - sum(
        size for name, size in loaded.items() if ".layers.45." in name
    )
    metadata = calculator_metadata(
        CONFIG,
        build_model_spec(CONFIG, repository="zai-org/GLM-5.3-Flash"),
        total_weight_bytes=total,
        replicated_weight_bytes=replicated_tensor_bytes(loaded, CONFIG),
    )
    inputs = _inputs(B200).model_copy(update={"artifact_metadata": metadata})
    predicted = calculate_layered_decode(inputs)
    assert predicted is not None
    buffers = startup_buffer_bytes(CONFIG, max_num_batched_tokens=16384)
    assert predicted["weight_memory_bytes"] == 161_357_908_984 + buffers  # 151.87 GiB
    assert predicted["activation_estimate_bytes"] == 4_833_935_360
    graphs = cuda_graph_estimate_bytes(metadata, inputs)
    assert predicted["cuda_graph_estimate_bytes"] == graphs + 419_823_616
    eager = _inputs(B200, enforce_eager=True).model_copy(update={"artifact_metadata": metadata})
    eager_predicted = calculate_layered_decode(eager)
    assert eager_predicted is not None and eager_predicted["cuda_graph_estimate_bytes"] == 0
