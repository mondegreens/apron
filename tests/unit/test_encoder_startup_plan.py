"""The vision tower's startup peak in real plans: the planner reads the processor files.

Planned offline through ``run_plan_pipeline`` from what the Hub served at each
recorded revision (tests/fixtures/cohort: configs.json, processors.json, the
stored weight bytes; scripts/kv_budget_residuals.py --configs).  The KV budget
is checked against the boots vLLM logged, with the tolerance of
test_calculator_vs_cohort's KV-budget test.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import apron.domain.mechanisms.calculator  # noqa: F401 — registers calculators
from apron.adapters.backends.runpod import GPU_SPECS
from apron.adapters.backends.vllm_quantization import engine_facts
from apron.adapters.evidence.hf_hub import HFHubResolver
from apron.adapters.planning.calculator_source import CalculatorPlanningSource
from apron.application.orchestration.plan_pipeline import (
    PROCESSOR_FILES,
    PlanPipelineResult,
    download_processor_config,
    run_plan_pipeline,
)
from apron.domain.mechanisms.calculator import (
    ENCODER_FIELDS,
    activation_config,
    compile_segment_bytes,
    encoder_peak_note,
)
from apron.domain.mechanisms.layered import KV_BUDGET_SAFETY_BUFFER_BYTES
from apron.domain.schemas.primitives import HardwareSpec

FIXTURES = Path(__file__).parents[1] / "fixtures" / "cohort"
CONFIGS = json.loads((FIXTURES / "configs.json").read_text())
PROCESSORS = json.loads((FIXTURES / "processors.json").read_text())
WEIGHTS = json.loads((FIXTURES / "weight-bytes.json").read_text())
H100_SKU = "NVIDIA H100 80GB HBM3"
H100 = HardwareSpec(
    gpu_sku=H100_SKU,
    total_memory_bytes=GPU_SPECS[H100_SKU]["total_memory_bytes"],
    compute_capability="9.0",
)
GIB = 1 << 30
# test_calculator_vs_cohort's KV-budget tolerance: over by at most the safety
# buffer (plus a compile segment, none for a multimodal wrapper), under by at
# most the largest measured under-prediction plus the log's rounding.
LOG_ROUNDING = int(0.005 * GIB)
KV_UNDER_PREDICTION = int(0.257 * GIB) + LOG_ROUNDING

# What vLLM v0.30.0 logged on an H100 (TP 1, bf16, utilization 0.9, max_model_len
# 640, the engine's default batch): available KV memory and the profiled torch
# peak, from the healthy memory reports in records/phase-1b-cohort/records.
MEASURED = {
    # report 12208654b66bdeae...
    "Qwen/Qwen3.6-35B-A3B-FP8": (34_821_447_352, 2_061_584_302),
    # report 1220ab41dd7bf8bd...
    "meta-models/Muse-Glimmer-30B": (11_800_422_645, 1_964_947_537),
}


class _Clock:
    def now(self) -> datetime:
        return datetime(2026, 9, 28, tzinfo=UTC)


class _Ids:
    def generate(self) -> str:
        return "fixed"


class _RecordedResolver(HFHubResolver):
    """The Hub as recorded for one model at its revision; logs every file asked for."""

    def __init__(self, model_id: str, *, processors: bool = True) -> None:
        self.model_id = model_id
        self.row = WEIGHTS[model_id]
        self.files = {"config.json": CONFIGS[model_id]}
        if processors:
            self.files.update(PROCESSORS.get(model_id, {}))
        self.requested: list[str] = []

    def _get_model_info(self, model_id: str, revision: str | None) -> dict[str, Any]:
        return {"sha": self.row["revision"], "gated": False, "tags": [], "safetensors": {}}

    def _download_file(self, model_id: str, filename: str, revision: str) -> bytes | None:
        assert revision == self.row["revision"]
        self.requested.append(filename)
        data = self.files.get(filename)
        return None if data is None else json.dumps(data).encode()

    def _tensor_meta(self, model_id: str, revision: str) -> dict[str, tuple[str, int]] | None:
        """The stored bytes weight-bytes.json records, under names the loader
        rules read (the lm_head, the F32 tensors, the dropped ``mtp.*``)."""
        row = self.row
        lm_head, f32, mtp = row["lm_head_bytes"], row["f32_bytes"], row["mtp_bytes"]
        meta = {"lm_head.weight": ("BF16", lm_head)} if lm_head else {}
        meta |= {"mtp.layers.0.weight": ("BF16", mtp)} if mtp else {}
        meta |= {"f32": ("F32", f32), "rest": ("BF16", row["total_bytes"] - lm_head - f32 - mtp)}
        return meta


def _plan(resolver: _RecordedResolver) -> PlanPipelineResult:
    """The plan as the v0.30.0 boots were planned: the engine's batch defaults."""
    facts = engine_facts("v0.30.0")
    memory = H100.total_memory_bytes
    return run_plan_pipeline(
        resolver,
        CalculatorPlanningSource(clock=_Clock()),
        resolver.model_id,
        H100,
        clock=_Clock(),
        id_gen=_Ids(),
        max_num_batched_tokens=facts.default_max_num_batched_tokens(memory, H100_SKU),
        max_num_seqs=facts.default_max_num_seqs(memory, H100_SKU),
    )


@pytest.mark.parametrize("model_id", sorted(MEASURED))
def test_the_plan_counts_the_encoder_peak_and_its_kv_matches_the_boot(model_id: str) -> None:
    measured_kv, measured_peak = MEASURED[model_id]
    result = _plan(_RecordedResolver(model_id))
    assert result.ok, result.error
    claim = result.claim.proposed_configuration
    # The encoder moment is the peak: the plan now predicts it ...
    without = _plan(_RecordedResolver(model_id, processors=False)).claim.proposed_configuration
    peak = claim["activation_estimate_bytes"]
    assert peak > without["activation_estimate_bytes"]
    assert abs(peak - measured_peak) <= 0.10 * measured_peak
    # ... and no longer hands its bytes to the KV cache.
    residual = claim["available_kv_cache_bytes"] - measured_kv
    segment = int(claim.get("compile_segment_bytes") or 0)
    assert -KV_UNDER_PREDICTION <= residual <= KV_BUDGET_SAFETY_BUFFER_BYTES + segment, (
        f"predicted - measured {residual / GIB:+.3f} GiB"
    )
    assert without["available_kv_cache_bytes"] - claim["available_kv_cache_bytes"] == (
        peak - without["activation_estimate_bytes"]
    )
    # Modelled: the plan carries no encoder note.
    assert not [n for n in result.notes if n.startswith("encoder startup peak")]


@pytest.mark.parametrize("model_id", sorted(MEASURED))
def test_the_planner_reads_the_processor_files_the_recorder_read(model_id: str) -> None:
    """The processor fields the plan's calculator gets are the recorded ones
    the calculator-vs-cohort activation test checks."""
    processor = download_processor_config(
        _RecordedResolver(model_id), model_id, WEIGHTS[model_id]["revision"]
    )
    derived = activation_config(CONFIGS[model_id], processor)
    recorded = WEIGHTS[model_id]["activation"]
    assert {k: derived[k] for k in ENCODER_FIELDS if k in derived} == {
        k: recorded[k] for k in ENCODER_FIELDS if k in recorded
    }
    assert "vision_family" in derived
    assert compile_segment_bytes({**derived, "torch_dtype": "bfloat16"}, _dummy_inputs(), 1) == 0


def _dummy_inputs() -> Any:
    from apron.domain.mechanisms import CalculatorInput, ComponentMechanism, TextWorkload

    return CalculatorInput(
        mechanism=ComponentMechanism(mechanism="autoregressive_decode", role="decoder"),
        workload=TextWorkload(kind="text", input_length=512, output_length=128),
        artifact_metadata={},
        hardware=H100,
        execution_spec_data={},
    )


def test_a_traced_tower_without_processor_files_is_noted() -> None:
    model_id = "Qwen/Qwen3.6-35B-A3B-FP8"
    result = _plan(_RecordedResolver(model_id, processors=False))
    assert result.ok, result.error
    (note,) = [n for n in result.notes if n.startswith("encoder startup peak")]
    assert "qwen3_5_moe vision tower" in note
    assert "no processor_config.json" in note
    assert note.endswith("activation may be under-predicted")


def test_an_untraced_vision_tower_is_noted_not_zeroed_silently() -> None:
    """Gemma 4's encoder is chunked by free memory (not traced): its recorded
    peak is the text forward, so the number keeps E_enc = 0; the note says the
    encoder is not modelled."""
    model_id = "google/gemma-4-31B-it"
    resolver = _RecordedResolver(model_id)
    result = _plan(resolver)
    assert result.ok, result.error
    assert "processor_config.json" in resolver.requested
    (note,) = [n for n in result.notes if n.startswith("encoder startup peak")]
    assert note == (
        "encoder startup peak not modelled for the gemma4 vision tower (not traced); "
        "activation may be under-predicted"
    )
    # No encoder field reaches the calculator: its moment stays 0.
    processor = download_processor_config(resolver, model_id, WEIGHTS[model_id]["revision"])
    assert not set(activation_config(CONFIGS[model_id], processor)) & set(ENCODER_FIELDS)


def test_a_text_only_model_fetches_no_processor_files() -> None:
    model_id = "Qwen/Qwen3-8B"
    resolver = _RecordedResolver(model_id)
    result = _plan(resolver)
    assert result.ok, result.error
    assert not set(resolver.requested) & {name for name, _ in PROCESSOR_FILES}
    assert not [n for n in result.notes if n.startswith("encoder startup peak")]


def test_encoder_note_cases() -> None:
    model_id = "Qwen/Qwen3.6-35B-A3B-FP8"
    qwen = CONFIGS[model_id]
    processor = download_processor_config(
        _RecordedResolver(model_id), model_id, WEIGHTS[model_id]["revision"]
    )
    assert encoder_peak_note(qwen, processor) is None
    assert encoder_peak_note({"model_type": "llama"}, None) is None
    # A processor that does not size the dummy item.
    note = encoder_peak_note(qwen, {"image_processor": {}, "video_processor": {}})
    assert note is not None and "do not give the dummy item's size" in note
    # An audio tower is never traced, even beside a traced vision tower.
    note = encoder_peak_note({**qwen, "audio_config": {"d_model": 8}}, processor)
    assert note == (
        "encoder startup peak not modelled for the qwen3_5_moe audio tower (not traced); "
        "activation may be under-predicted"
    )
