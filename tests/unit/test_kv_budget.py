"""The KV budget's terms: what vLLM subtracts from the requested memory.

Each term against its source (vLLM v0.29.0 / v0.30.0) or measurement; the
whole budget against every healthy cohort record is in
``test_calculator_vs_cohort.py``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, ClassVar

import pytest

import apron.domain.mechanisms.calculator  # noqa: F401 — registers calculators
from apron.adapters.backends.runpod import GPU_SPECS
from apron.adapters.backends.vllm_quantization import VLLM_DEFAULT_GPU_MEMORY_UTILIZATION
from apron.adapters.evidence.hf_hub import HFHubResolver
from apron.application.orchestration.plan_builder import PLAN_GPU_MEMORY_UTILIZATION
from apron.application.orchestration.plan_pipeline import mtp_tensor_bytes
from apron.domain.mechanisms import CalculatorInput, ComponentMechanism, TextWorkload
from apron.domain.mechanisms.calculator import (
    compile_segment_bytes,
    cuda_graph_estimate_bytes,
    non_torch_bytes,
    requested_memory_bytes,
    sampler_state_bytes,
)
from apron.domain.mechanisms.layered import KV_BUDGET_SAFETY_BUFFER_BYTES, state_block_capacity
from apron.domain.schemas.primitives import HardwareSpec
from apron.domain.schemas.solutions import DeploymentPlan

H100 = HardwareSpec(
    gpu_sku="NVIDIA H100 80GB HBM3",
    total_memory_bytes=GPU_SPECS["NVIDIA H100 80GB HBM3"]["total_memory_bytes"],
    compute_capability="9.0",
)
RTX_4090 = HardwareSpec(
    gpu_sku="NVIDIA GeForce RTX 4090",
    total_memory_bytes=GPU_SPECS["NVIDIA GeForce RTX 4090"]["total_memory_bytes"],
    compute_capability="8.9",
)
MIB = 1 << 20


def _inputs(hardware: HardwareSpec = H100, **execution: Any) -> CalculatorInput:
    return CalculatorInput(
        mechanism=ComponentMechanism(mechanism="autoregressive_decode", role="decoder"),
        workload=TextWorkload(kind="text", input_length=512, output_length=128),
        artifact_metadata={},
        hardware=hardware,
        execution_spec_data=execution,
    )


# ---------------------------------------------------------------------------
# Requested memory: the detected GPU total x the plan's utilization, rounded up
# ---------------------------------------------------------------------------


def test_detected_totals_replace_the_nominal_sizes() -> None:
    # The byte counts the cohort's pods detected (test_calculator_vs_cohort
    # ties each to a recorded fingerprint); the logs printed 79.18 / 23.52 GiB.
    assert GPU_SPECS["NVIDIA H100 80GB HBM3"]["total_memory_bytes"] == 81_079 * MIB
    assert GPU_SPECS["NVIDIA GeForce RTX 4090"]["total_memory_bytes"] == 25_250_627_584
    assert GPU_SPECS["NVIDIA L4"]["total_memory_bytes"] == 23_659_151_360
    assert GPU_SPECS["NVIDIA RTX A6000"]["total_memory_bytes"] == 50_899_648_512
    assert GPU_SPECS["NVIDIA A100 80GB PCIe"]["total_memory_bytes"] == 85_093_777_408
    assert GPU_SPECS["NVIDIA A100-SXM4-80GB"]["total_memory_bytes"] == 85_093_777_408


def test_requested_memory_is_vllms_ceiling_of_total_times_utilization() -> None:
    # v1/worker/utils.py:521-523: math.ceil(total_memory * gpu_memory_utilization)
    total = H100.total_memory_bytes
    assert requested_memory_bytes(_inputs(gpu_memory_utilization=0.9)) == -(-total * 9 // 10)
    assert requested_memory_bytes(_inputs()) == requested_memory_bytes(
        _inputs(gpu_memory_utilization=PLAN_GPU_MEMORY_UTILIZATION)
    )
    # Fix-proof plans that set none boot at vLLM's default.
    assert VLLM_DEFAULT_GPU_MEMORY_UTILIZATION == 0.92
    assert requested_memory_bytes(_inputs(gpu_memory_utilization=0.92)) == 78_216_094_024


# ---------------------------------------------------------------------------
# Non-torch: the sampler state, the encoder's embeddings buffer, a class base
# ---------------------------------------------------------------------------


def test_sampler_state_is_the_probed_bytes() -> None:
    # Peak probe 2026-09-28, vLLM v0.29.0, 1024 request slots.
    assert sampler_state_bytes(262_144, 1024) == 1_140_850_688  # Gemma 4 31B
    assert sampler_state_bytes(154_880, 1024) == 674_037_760  # GLM-4.7-Flash
    # [seqs, V] + 2 x [seqs, ceil(V / 32)], int32
    assert sampler_state_bytes(33, 2) == 2 * (33 + 2 * 2) * 4


def test_non_torch_counts_the_encoder_buffer_only_for_a_vision_wrapper() -> None:
    text = {"vocab_size": 262_144, "hidden_size": 5376, "torch_dtype": "bfloat16"}
    wrapped = {**text, "vision_config": True}
    shape: dict[str, Any] = {"max_num_seqs": 1024, "max_num_batched_tokens": 8192}
    base = non_torch_bytes(text, _inputs(**shape))
    assert base == 1_140_850_688 + 307 * MIB
    # [max_num_batched_tokens, H] in the model dtype (mm/encoder_runner.py:58-60)
    assert non_torch_bytes(wrapped, _inputs(**shape)) == base + 8192 * 5376 * 2
    # --language-model-only compiles it as text: no encoder buffer.
    assert non_torch_bytes(wrapped, _inputs(language_model_only=True, **shape)) == base


def test_non_torch_base_is_the_gpu_class_measurement() -> None:
    metadata = {"vocab_size": 32_768, "hidden_size": 4096}
    shape: dict[str, Any] = {"max_num_seqs": 256}
    sampler = sampler_state_bytes(32_768, 256)
    assert non_torch_bytes(metadata, _inputs(RTX_4090, **shape)) == sampler + 112 * MIB
    assert non_torch_bytes(metadata, _inputs(H100, **shape)) == sampler + 307 * MIB
    unknown = HardwareSpec(gpu_sku="x", total_memory_bytes=1 << 34, compute_capability="0.0")
    assert non_torch_bytes(metadata, _inputs(unknown, **shape)) == sampler + 307 * MIB


# ---------------------------------------------------------------------------
# The compile segment vLLM may hold (uncertainty, not in the prediction)
# ---------------------------------------------------------------------------


def test_compile_segment_is_the_padded_embedding_on_text_only_tp1() -> None:
    qwen3_32b = {"vocab_size": 151_936, "hidden_size": 5120, "torch_dtype": "bfloat16"}
    assert compile_segment_bytes(qwen3_32b, _inputs(), 1) == 151_936 * 5120 * 2  # 1.45 GiB
    mamba = {"vocab_size": 50_280, "hidden_size": 2560, "torch_dtype": "float32"}
    assert compile_segment_bytes(mamba, _inputs(), 1) == 50_304 * 2560 * 4  # padded to 64
    assert compile_segment_bytes(qwen3_32b, _inputs(), 2) == 0  # vLLM's own embedding op
    assert compile_segment_bytes({**qwen3_32b, "vision_config": True}, _inputs(), 1) == 0


def test_compile_segment_is_not_subtracted_from_the_prediction() -> None:
    from apron.domain.mechanisms import calculate

    config = {
        "num_hidden_layers": 64,
        "num_key_value_heads": 8,
        "num_attention_heads": 64,
        "hidden_size": 5120,
        "head_dim": 128,
        "intermediate_size": 25_600,
        "vocab_size": 151_936,
        "torch_dtype": "bfloat16",
        "total_weight_bytes": 65_524_246_528 - 1_555_824_640,
    }
    inputs = CalculatorInput(
        mechanism=ComponentMechanism(mechanism="autoregressive_decode", role="decoder"),
        workload=TextWorkload(kind="text", input_length=512, output_length=128),
        artifact_metadata=config,
        hardware=H100,
        execution_spec_data={"max_num_seqs": 1024, "max_num_batched_tokens": 8192},
    )
    result = calculate(inputs)
    assert result is not None
    assert result["compile_segment_bytes"] == 151_936 * 5120 * 2
    assert result["available_kv_cache_bytes"] == (
        result["gpu_available_bytes"]
        - result["weight_memory_bytes"]
        - result["activation_estimate_bytes"]
        - result["non_pytorch_overhead_bytes"]
        - result["cuda_graph_estimate_bytes"]
    )


# ---------------------------------------------------------------------------
# The CUDA-graph estimate vLLM subtracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seqs", "graphs"),
    [
        (256, 35),  # 1, 2, 4, 8..248 by 8, 256
        (1024, 51),  # ... and 272..512 by 16: the ceiling min(2 x 1024, 512)
        (4, 3),  # 1, 2, 4
    ],
)
def test_cuda_graph_estimate_is_layers_x_full_decode_graphs(seqs: int, graphs: int) -> None:
    metadata = {"num_hidden_layers": 36}
    per_graph = {"sm8x": 374 << 10, "sm90": 537 << 10}
    assert cuda_graph_estimate_bytes(metadata, _inputs(RTX_4090, max_num_seqs=seqs)) == (
        36 * graphs * per_graph["sm8x"]
    )
    assert cuda_graph_estimate_bytes(metadata, _inputs(H100, max_num_seqs=seqs)) == (
        36 * graphs * per_graph["sm90"]
    )


def test_cuda_graph_ceiling_is_1024_on_sm100() -> None:
    # config/vllm.py:2006-2008: default_max_graph_size 1024 on the SM100 family.
    b200 = HardwareSpec(
        gpu_sku="NVIDIA B200", total_memory_bytes=1 << 37, compute_capability="10.0"
    )
    metadata = {"num_hidden_layers": 1}
    assert cuda_graph_estimate_bytes(metadata, _inputs(b200, max_num_seqs=1024)) == (
        (51 + 32) * (537 << 10)  # 528..1024 by 16 on top of the 51 up to 512
    )


def test_no_cuda_graphs_with_enforce_eager() -> None:
    metadata = {"num_hidden_layers": 36}
    assert cuda_graph_estimate_bytes(metadata, _inputs(enforce_eager=True)) == 0


# ---------------------------------------------------------------------------
# Weights: multi-token-prediction layers vLLM's mapper drops
# ---------------------------------------------------------------------------


def test_nemotron_h_mtp_layers_are_not_loaded() -> None:
    # models/nemotron_h.py:711 (v0.29.0) / 728 (v0.30.0): "mtp" -> None
    tensors = {"mtp.layers.0.mixer.w": 10, "backbone.layers.0.mixer.w": 5, "lm_head.weight": 3}
    assert mtp_tensor_bytes(tensors, {"model_type": "nemotron_h", "num_hidden_layers": 1}) == 10
    assert mtp_tensor_bytes(tensors, {"model_type": "qwen3_next", "num_hidden_layers": 1}) == 10
    assert mtp_tensor_bytes(tensors, {"model_type": "llama", "num_hidden_layers": 1}) == 0


# ---------------------------------------------------------------------------
# The guard's safety buffer
# ---------------------------------------------------------------------------


def test_safety_buffer_is_named_and_measured() -> None:
    # 2.18 GiB measured (GLM-4.7-Flash on an H100) + 0.005 GiB log rounding,
    # rounded up to 0.01 GiB; test_calculator_vs_cohort re-derives it.
    assert round(2.19 * (1 << 30)) == KV_BUDGET_SAFETY_BUFFER_BYTES
    assert state_block_capacity(KV_BUDGET_SAFETY_BUFFER_BYTES + 10, 5) == 2


def test_guard_subtracts_the_compile_segment_on_top_of_the_buffer() -> None:
    from apron.application.orchestration.plan_pipeline import StateBlockFacts, state_block_limit

    facts = StateBlockFacts(engine_version="v0.30.0", check="config/compilation.py:1535")
    per_block = 12_877_824
    predicted = {"kv_bytes_per_block": per_block, "available_kv_cache_bytes": 10_216_498_970}
    plain = state_block_limit(predicted, 1024, facts)
    held = state_block_limit({**predicted, "compile_segment_bytes": 704_643_072}, 1024, facts)
    assert plain is not None and held is not None
    assert plain[0] == (10_216_498_970 - KV_BUDGET_SAFETY_BUFFER_BYTES) // per_block == 610
    assert held[0] == (10_216_498_970 - KV_BUDGET_SAFETY_BUFFER_BYTES - 704_643_072) // per_block
    assert "plus a 0.66 GiB compile segment vLLM may hold" in held[1]
    assert "compile segment" not in plain[1]


# ---------------------------------------------------------------------------
# A given plan (fix proof) is predicted as it boots
# ---------------------------------------------------------------------------


class _Clock:
    def now(self) -> datetime:
        return datetime(2026, 9, 28, tzinfo=UTC)


class _Resolver(HFHubResolver):
    """One float32 Mamba-1 checkpoint (state-spaces/mamba-2.8b-hf's shape)."""

    CONFIG: ClassVar[dict[str, Any]] = {
        "architectures": ["MambaForCausalLM"],
        "model_type": "mamba",
        "torch_dtype": "float32",
        "vocab_size": 50_280,
        "hidden_size": 2560,
        "intermediate_size": 5120,
        "num_hidden_layers": 64,
        "state_size": 16,
        "conv_kernel": 4,
    }

    def __init__(self) -> None:
        pass

    def _get_model_info(self, model_id: str, revision: str | None) -> dict[str, Any]:
        return {"sha": "0" * 40, "gated": False, "tags": [], "safetensors": {}}

    def _download_file(self, model_id: str, filename: str, revision: str) -> bytes | None:
        return json.dumps(self.CONFIG).encode() if filename == "config.json" else None

    def _tensor_meta(self, model_id: str, revision: str) -> dict[str, tuple[str, int]] | None:
        return {"backbone.weights": ("F32", 11_073_382_400)}


def _given(plan: DeploymentPlan) -> dict[str, Any]:
    from apron.interfaces.cohort_root import CohortPlanner

    planner = CohortPlanner(rates={}, clock=_Clock(), resolver=_Resolver())
    return dict(planner.plan_for(plan, "given").claim.proposed_configuration)


def test_a_given_plan_is_predicted_at_its_dtype_and_utilization() -> None:
    """The float32 Mamba fix proof was predicted at 16 bits and 0.90: its KV
    budget came out 5.75 GiB high (report 12200ac02bfd...)."""
    alloc = {"model_id": "state-spaces/mamba-2.8b-hf", "gpu_sku": RTX_4090.gpu_sku}
    fp32 = _given(DeploymentPlan(dtype="float32", resource_allocation=alloc))
    assert fp32["weight_memory_bytes"] == 11_073_382_400  # stored size: no downcast
    assert fp32["gpu_available_bytes"] == -(-RTX_4090.total_memory_bytes * 92 // 100)
    bf16 = _given(
        DeploymentPlan(
            dtype="bfloat16",
            resource_allocation=alloc,
            engine_configuration={"gpu_memory_utilization": "0.9"},
        )
    )
    assert bf16["weight_memory_bytes"] == 11_073_382_400 // 2
    assert bf16["gpu_available_bytes"] == -(-RTX_4090.total_memory_bytes * 9 // 10)


def test_a_given_plan_runs_on_the_vllm_its_architecture_needs() -> None:
    """A fix proof of a v0.30-only model was rebuilt on v0.29's image: another
    solution than the one that failed, so its stored failure was not found
    (Qwen3.8-Flash-Next on 2xB200, 2026-09-28).  The given plan's engine is
    chosen by architecture, as for a seed."""
    from unittest.mock import patch

    from apron.adapters.runner_image import runner_image
    from apron.interfaces.cohort_root import CohortPlanner

    planner = CohortPlanner(rates={}, clock=_Clock(), resolver=_Resolver())
    alloc = {"model_id": "state-spaces/mamba-2.8b-hf", "gpu_sku": RTX_4090.gpu_sku}
    plan = DeploymentPlan(dtype="bfloat16", resource_allocation=alloc)
    with patch.object(CohortPlanner, "engine", return_value="v0.30.0"):
        given = planner.plan_for(plan, "given")
    assert given.engine_version == "v0.30.0"
    assert given.requested.image_digest == runner_image("v0.30.0").digest
