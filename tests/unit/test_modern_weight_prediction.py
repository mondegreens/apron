"""GPU weights of the group C models as the planner predicts them for vLLM v0.30.0.

Planned from recorded config.json and safetensors headers
(tests/fixtures/cohort/headers, scripts/record_tensor_headers.py), no network.
Per GPU at TP 4 on an H200, from _dev_notes/cohort-run/deepseek-v4-kv-trace.md
and qwen4exp-glm5next-kv-trace.md: the ``mtp.*`` layers the loaders drop are
not weights, the n-gram tables vLLM keeps in pinned host memory are host RAM,
DeepSeek's replicated projections sit whole on every rank, and V4.1's MXFP4
experts are padded per rank.
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
from apron.application.orchestration.plan_pipeline import (
    PlanPipelineResult,
    host_tensor_bytes,
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


def _plan(model_id: str, tp: int = 4) -> PlanPipelineResult:
    return run_plan_pipeline(
        _RecordedResolver(_record(model_id)),
        CalculatorPlanningSource(clock=_Clock()),
        model_id,
        H200,
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
    # (total - PLE n-gram table - mtp) / 4 = 58.76 GiB
    "Qwen/Qwen3.8-Flash-Next": (63_096_292_478, 5_214_301_696, 102_400_491_520, 0),
    # (total - MTP layer 45) / 4 = 74.70 GiB
    "zai-org/GLM-5.3-Flash": (80_208_343_102, 7_493_399_168, 0, 0),
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
    for model_id, gib, model_type in (
        ("deepseek-ai/DeepSeek-V4.1-Flash", "188.83", "deepseek_v41"),
        ("Qwen/Qwen3.8-Flash-Next", "95.37", "qwen4_exp"),
    ):
        note, encoder = _plan(model_id).notes
        assert note.startswith(f"host RAM: {gib} GiB")
        assert "x 4" in note
        # Their vision towers are not traced: the plan says the peak omits them.
        assert encoder == _untraced_tower_note(model_type)
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
