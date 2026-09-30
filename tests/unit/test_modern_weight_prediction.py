"""GPU weights of the group C and D models as the planner predicts them for vLLM v0.30.0.

Planned from recorded config.json and safetensors headers
(tests/fixtures/cohort/headers, scripts/record_tensor_headers.py), no network.
Per GPU on an H200, group C at TP 4 from _dev_notes/cohort-run/deepseek-v4-kv-trace.md
and qwen4exp-glm5next-kv-trace.md: the ``mtp.*`` layers the loaders drop are
not weights, the n-gram tables vLLM keeps in pinned host memory are host RAM,
DeepSeek's replicated projections sit whole on every rank, and V4.1's MXFP4
experts are padded per rank.  Group D (GLM-5.3, MiniMax-M3) at TP 8 from
glm53-minimax-m3-kv-trace.md.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import apron.domain.mechanisms.calculator  # noqa: F401 — registers calculators
from apron.adapters.evidence.hf_hub import HFHubResolver
from apron.adapters.planning.calculator_source import CalculatorPlanningSource
from apron.application.orchestration.cohort import predicted_feasible
from apron.application.orchestration.plan_pipeline import (
    PlanPipelineResult,
    host_tensor_bytes,
    kv_head_tensor_bytes,
    loaded_tensor_bytes,
    mtp_tensor_bytes,
    replicated_tensor_bytes,
    run_plan_pipeline,
)
from apron.domain.schemas.primitives import HardwareSpec

HEADERS = Path(__file__).parents[1] / "fixtures" / "cohort" / "headers"
H200 = HardwareSpec(
    gpu_sku="NVIDIA H200", total_memory_bytes=150_754_820_096, compute_capability="9.0"
)
B200 = HardwareSpec(gpu_sku="NVIDIA B200", total_memory_bytes=179 << 30, compute_capability="10.0")
GIB = 1 << 30


class _Clock:
    def now(self) -> datetime:
        return datetime(2026, 9, 28, tzinfo=UTC)


class _Ids:
    def generate(self) -> str:
        return "fixed"


class _RecordedResolver(HFHubResolver):
    """The Hub as recorded: one revision, its config and tensor headers."""

    def __init__(self, record: dict[str, Any]) -> None:
        self.record = record

    def _get_model_info(self, model_id: str, revision: str | None) -> dict[str, Any]:
        return {"sha": self.record["revision"], "gated": False, "tags": [], "safetensors": {}}

    def _download_file(self, model_id: str, filename: str, revision: str) -> bytes | None:
        if filename == "config.json":
            return json.dumps(self.record["config"]).encode()
        return None

    def _tensor_meta(self, model_id: str, revision: str) -> dict[str, tuple[str, int]] | None:
        return {name: (dtype, size) for name, (dtype, size, _) in self.record["tensors"].items()}


def _record(model_id: str) -> dict[str, Any]:
    record: dict[str, Any] = json.loads(
        (HEADERS / (model_id.replace("/", "__") + ".json")).read_text()
    )
    return record


def _plan(model_id: str, tp: int = 4, hardware: HardwareSpec = H200) -> PlanPipelineResult:
    return run_plan_pipeline(
        _RecordedResolver(_record(model_id)),
        CalculatorPlanningSource(clock=_Clock()),
        model_id,
        hardware,
        clock=_Clock(),
        id_gen=_Ids(),
        tensor_parallel=tp,
        max_num_batched_tokens=8192,
        max_num_seqs=256,
    )


# model -> (GPU weights per GPU at TP 4, mtp.* bytes, host bytes, replicated bytes)
EXPECTED = {
    # (total - mtp - replicated) / 4 + replicated = 37.25 GiB
    "deepseek-ai/DeepSeek-V4-Flash-0731": (39_991_556_380, 10_862_838_300, 0, 1_316_842_460),
    # 70.51 GiB + 7.47 GiB of expert padding (576 -> 640 per rank); Engram in host RAM
    "deepseek-ai/DeepSeek-V4.1-Flash": (
        83_730_322_368,
        7_932_874_632,
        202_758_032_400,
        1_079_915_968,
    ),
    # (total - PLE n-gram table - mtp - replicated) / 4 + replicated = 59.82 GiB;
    # replicated: hyper-connections, router, QSA indexer and PLE projections
    # (plan_pipeline._QWEN4_EXP_REPLICATED).  Measured on 4x H200: 60.87 GiB.
    "Qwen/Qwen3.8-Flash-Next": (
        64_230_244_478,
        5_214_301_696,
        102_400_491_520,
        1_511_936_000,
    ),
    # (loaded - MTP layer 45 - replicated) / 4 + replicated + the top-k index
    # buffers vLLM builds at load (8192 tokens: 0.80 GiB) = 76.26 GiB.  Loaded:
    # the MLA layers' fp8 projections in bf16, the hyper-connection mixes, the
    # kpool APE and the KDA convolutions in float32 (the MTP layer's included,
    # 7,493,399,168 B stored).  Replicated: fused q_a + kv_a, the indexer, KDA
    # f_a / g_a, router gates, hyper-connections, norms, the vision convs.
    # Measured on 2x B200 at TP 2: 152.10 GiB (the plan: 151.87).
    "zai-org/GLM-5.3-Flash": (81_884_381_604, 7_594_038_912, 0, 697_944_824),
}


@pytest.mark.parametrize("model_id", sorted(EXPECTED))
def test_group_c_weights_per_gpu(model_id: str) -> None:
    weights, mtp, host, replicated = EXPECTED[model_id]
    record = _record(model_id)
    config = record["config"]
    tensors = loaded_tensor_bytes(
        {name: (dtype, size) for name, (dtype, size, _) in record["tensors"].items()}, config
    )
    assert mtp_tensor_bytes(tensors, config) == mtp
    assert host_tensor_bytes(tensors, config) == host
    assert replicated_tensor_bytes(tensors, config) == replicated

    result = _plan(model_id)
    assert result.ok, result.error
    assert result.model_spec is not None
    assert result.model_spec.components[0].mechanism == "layered_decode"
    claim = result.claim.proposed_configuration
    assert claim["weight_memory_bytes"] == weights
    assert claim["state_per_sequence_bytes"] > 0
    assert result.host_memory_bytes == host


def test_host_tables_are_named_in_the_plan_notes() -> None:
    """The n-gram tables are not GPU weights, but the pod still needs the RAM."""
    note, encoder = _plan("deepseek-ai/DeepSeek-V4.1-Flash").notes
    assert note.startswith("host RAM: 188.83 GiB") and "x 4" in note
    # Its vision tower is not traced: the plan says the peak omits it.
    assert encoder == _untraced_tower_note("deepseek_v41")
    # Qwen4Exp runs Qwen3-VL's tower, which is traced; this fixture carries no
    # processor files, so the note names those instead of an untraced tower.
    note, encoder = _plan("Qwen/Qwen3.8-Flash-Next").notes
    assert note.startswith("host RAM: 95.37 GiB") and "x 4" in note
    assert "(no processor_config.json" in encoder and "not traced" not in encoder
    assert _plan("zai-org/GLM-5.3-Flash").notes == (_untraced_tower_note("glm5_next"),)
    assert _plan("deepseek-ai/DeepSeek-V4-Flash-0731").notes == ()


def _untraced_tower_note(model_type: str) -> str:
    return (
        f"encoder startup peak not modelled for the {model_type} vision tower (not traced); "
        "activation may be under-predicted"
    )


def test_without_headers_a_model_with_host_tables_has_no_weight_figure() -> None:
    """The index or parameter counts include the host tables: unknown, not a guess."""
    from apron.application.orchestration.plan_pipeline import _resolve_weight_bytes

    class _NoHeaders:
        resolved_revision = "r"

        def __init__(self) -> None:
            self.publisher_metadata = {"parameters_BF16": "100"}

    config = _record("Qwen/Qwen3.8-Flash-Next")["config"]
    assert _resolve_weight_bytes(object(), "m", _NoHeaders(), config) == 0


def test_replicated_weights_are_whole_on_every_rank() -> None:
    """TP 8 halves the shard, not the replicated projections."""
    tp4 = _plan("deepseek-ai/DeepSeek-V4-Flash-0731").claim.proposed_configuration
    tp8 = _plan("deepseek-ai/DeepSeek-V4-Flash-0731", tp=8).claim.proposed_configuration
    replicated = EXPECTED["deepseek-ai/DeepSeek-V4-Flash-0731"][3]
    shard4 = tp4["weight_memory_bytes"] - replicated
    shard8 = tp8["weight_memory_bytes"] - replicated
    assert abs(shard4 - 2 * shard8) <= 1
    assert tp8["weight_memory_bytes"] / GIB == pytest.approx(19.24, abs=0.01)


# Group D at TP 8 on an H200 (_dev_notes/cohort-run/glm53-minimax-m3-kv-trace.md):
# model -> (GPU weights per GPU, mtp bytes, replicated bytes, KV-head bytes)
EXPECTED_D = {
    # (loaded - MTP layer 78 - replicated) / 8 + replicated = 88.20 GiB.
    # Replicated: fused q_a + kv_a, the indexer (its fp8 wk loads as bf16),
    # router gates, norms.
    "zai-org/GLM-5.3": (94_699_576_704, 10_033_419_200, 1_713_656_448, 0),
    # (loaded - replicated - KV-head) / 8 + replicated + KV-head / 4 = 99.79 GiB.
    # No mtp.* tensors stored; the float32 router gates stay float32.
    "MiniMaxAI/MiniMax-M3": (107_150_251_008, 0, 274_067_456, 1_113_587_712),
}


@pytest.mark.parametrize("model_id", sorted(EXPECTED_D))
def test_group_d_weights_per_gpu(model_id: str) -> None:
    weights, mtp, replicated, by_kv_head = EXPECTED_D[model_id]
    record = _record(model_id)
    config = record["config"]
    tensors = loaded_tensor_bytes(
        {name: (dtype, size) for name, (dtype, size, _) in record["tensors"].items()}, config
    )
    assert mtp_tensor_bytes(tensors, config) == mtp
    assert host_tensor_bytes(tensors, config) == 0
    assert replicated_tensor_bytes(tensors, config) == replicated
    assert kv_head_tensor_bytes(tensors, config) == by_kv_head

    result = _plan(model_id, tp=8)
    assert result.ok, result.error
    assert result.model_spec is not None
    assert result.model_spec.components[0].mechanism == "layered_decode"
    claim = result.claim.proposed_configuration
    assert claim["weight_memory_bytes"] == weights
    assert result.host_memory_bytes == 0
    # Both fit one H200 each at TP 8 (the planner's 0.90 of the GPU).
    assert predicted_feasible(result.claim, H200.total_memory_bytes, 8)


def test_group_d_tensors_loaded_at_vllms_width() -> None:
    """GLM-5.3's fp8 indexer wk is dequantized to bf16 and its scale dropped;
    MiniMax-M3's float32 router gate is not downcast (dtype=auto halves the
    rest of an unquantized checkpoint's F32 tensors)."""
    glm = _record("zai-org/GLM-5.3")
    wk = "model.layers.0.self_attn.indexer.wk.weight"
    loaded = loaded_tensor_bytes(
        {name: (dtype, size) for name, (dtype, size, _) in glm["tensors"].items()}, glm["config"]
    )
    assert glm["tensors"][wk][:2] == ["F8_E4M3", 786_432]
    assert loaded[wk] == 1_572_864
    assert loaded[wk + "_scale_inv"] == 0
    m3 = _record("MiniMaxAI/MiniMax-M3")
    meta = {name: (dtype, size) for name, (dtype, size, _) in m3["tensors"].items()}
    loaded = loaded_tensor_bytes(meta, m3["config"])
    gate = "language_model.model.layers.3.block_sparse_moe.gate.weight"
    patch = "vision_tower.vision_model.embeddings.patch_embedding.weight"
    assert meta[gate] == ("F32", 3_145_728) and loaded[gate] == 3_145_728
    assert meta[patch] == ("F32", 6_021_120) and loaded[patch] == 3_010_560


def test_minimax_m3_plan_sets_the_block_size_vllm_needs() -> None:
    """With vLLM's own block size (16) MiniMax-M3 does not boot."""
    plans = {
        model_id: _plan(model_id, tp=tp).plan
        for model_id, tp in (
            ("MiniMaxAI/MiniMax-M3", 8),
            ("zai-org/GLM-5.3", 8),
            ("zai-org/GLM-5.3-Flash", 4),
        )
    }
    engines = {model_id: plan.engine_configuration for model_id, plan in plans.items() if plan}
    assert len(engines) == 3
    assert engines["MiniMaxAI/MiniMax-M3"]["block_size"] == "128"
    assert "block_size" not in engines["zai-org/GLM-5.3"]
    assert "block_size" not in engines["zai-org/GLM-5.3-Flash"]


def test_qwen4exp_bf16_experts_are_padded_on_b200() -> None:
    """2x B200, TP 2 (vLLM v0.30.0 log, 2026-09-28): "Padding intermediate size
    from 320 to 384", TrtLlmBf16ExpertsMonolithic, "Model loading took 142.6
    GiB".  The H200 plan (SM 9.x, Triton) keeps the checkpoint's shard."""
    h200 = _plan("Qwen/Qwen3.8-Flash-Next", tp=2).claim.proposed_configuration
    b200 = _plan("Qwen/Qwen3.8-Flash-Next", tp=2, hardware=B200).claim.proposed_configuration
    # 117.53 GiB + half the replicated 1.41 GiB (whole on each of 2 ranks)
    assert h200["weight_memory_bytes"] == 126_948_552_956  # 118.23 GiB
    # + 48 layers x 512 experts x 3 x 2560 x 64 rows x 2 bytes = 22.5 GiB
    assert b200["weight_memory_bytes"] == 126_948_552_956 + 24_159_191_040  # 140.73 GiB


@pytest.mark.parametrize(
    ("model_id", "tp"),
    [
        ("deepseek-ai/DeepSeek-V4-Flash-0731", 2),
        ("deepseek-ai/DeepSeek-V4.1-Flash", 2),
        ("deepseek-ai/DeepSeek-V4.1-Flash", 4),
        ("zai-org/GLM-5.3-Flash", 2),
        ("zai-org/GLM-5.3", 4),
        ("zai-org/GLM-5.3", 8),
        ("MiniMaxAI/MiniMax-M3", 4),
        ("MiniMaxAI/MiniMax-M3", 8),
    ],
)
def test_other_group_c_d_models_weigh_the_same_on_b200(model_id: str, tp: int) -> None:
    """FP8 (GLM-5.3, GLM-5.3-Flash) and MXFP4 (DeepSeek) experts take other
    backends; MiniMax-M3's 16-bit experts are not a Qwen3Next MoE block and
    3072 / TP is a multiple of 128 anyway."""
    h200 = _plan(model_id, tp=tp).claim.proposed_configuration
    b200 = _plan(model_id, tp=tp, hardware=B200).claim.proposed_configuration
    assert b200["weight_memory_bytes"] == h200["weight_memory_bytes"]
