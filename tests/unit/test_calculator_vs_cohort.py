"""The calculator's weight prediction against every measured cohort record.

Measured: vLLM's per-rank "model weights" figure in the stored boot reports
(``_dev_notes/cohort-run/records``).  Predicted: the checkpoint's stored bytes
(``tests/fixtures/cohort/weight-bytes.json``, recorded from the safetensors
headers at the solution's revision by ``scripts/record_weight_bytes.py``)
through the same resolution and per-GPU sharding the planner uses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from apron.application.orchestration.plan_pipeline import _resolve_weight_bytes
from apron.domain.mechanisms.calculator import _per_gpu
from apron.interfaces.cohort_root import load_cohort_run

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "cohort" / "weight-bytes.json"
RUN = REPO / "_dev_notes" / "cohort-run"
# The FP8 checkpoint lands 5.4% under (0.70 vs 0.74 GiB); every other point
# within 1.5%.  Before the L5 fixes: +17% (tied lm_head), -22% (FP8), +100% (TP 2).
TOLERANCE = 0.06


class _Recorded:
    def __init__(self, entry: dict[str, Any]) -> None:
        self._entry = entry

    def _tensor_bytes(self, model_id: str, revision: str) -> dict[str, int]:
        lm_head = int(self._entry["lm_head_bytes"])
        rest = {"rest": int(self._entry["total_bytes"]) - lm_head}
        return {**rest, "lm_head.weight": lm_head} if lm_head else rest


class _Observation:
    resolved_revision = "recorded"
    publisher_metadata = None


def _measured_points() -> list[tuple[str, int, int]]:
    if not (RUN / "records").is_dir():
        return []
    records = load_cohort_run(RUN).records
    points = set()
    for report in records.reports.values():
        entry = records.solutions.get(report.solution_fingerprint or "")
        if (
            entry is None
            or report.claim_scope != "memory"
            or report.boot_outcome != "healthy"
            or not report.model_weight_memory
        ):
            continue
        points.add(
            (entry.model_id, entry.deployment_plan.tensor_parallel, report.model_weight_memory)
        )
    return sorted(points)


@pytest.mark.skipif(not (RUN / "records").is_dir(), reason="cohort records not present")
@pytest.mark.parametrize(("model_id", "tp", "measured"), _measured_points())
def test_weight_prediction_matches_the_measurement(model_id: str, tp: int, measured: int) -> None:
    recorded = json.loads(FIXTURE.read_text())
    assert model_id in recorded, "run scripts/record_weight_bytes.py"
    entry = recorded[model_id]
    config = {"tie_word_embeddings": entry["tie_word_embeddings"]}
    predicted = _per_gpu(
        _resolve_weight_bytes(_Recorded(entry), model_id, _Observation(), config), tp
    )
    assert abs(predicted - measured) / measured <= TOLERANCE, (
        f"{model_id} TP {tp}: predicted {predicted / 2**30:.2f} GiB, "
        f"measured {measured / 2**30:.2f} GiB"
    )


# ---------------------------------------------------------------------------
# Activation: the calculator's estimate against vLLM's profiled torch peak
# ---------------------------------------------------------------------------

# 20% of the measurement, or 0.08 GiB for the small ones (log values are
# printed to 0.01 GiB).
ACTIVATION_TOLERANCE = (0.20, int(0.08 * 2**30))


def _activation_points() -> list[tuple[str, str, int, int]]:
    if not (RUN / "records").is_dir():
        return []
    records = load_cohort_run(RUN).records
    points = set()
    for report in records.reports.values():
        entry = records.solutions.get(report.solution_fingerprint or "")
        if (
            entry is None
            or report.claim_scope != "memory"
            or report.boot_outcome != "healthy"
            or not report.transient_peak_headroom
        ):
            continue
        points.add(
            (
                entry.model_id,
                entry.requested_execution.gpu_sku,
                entry.deployment_plan.tensor_parallel,
                report.transient_peak_headroom,
            )
        )
    return sorted(points)


def _activation_params() -> list[Any]:
    return [
        pytest.param(
            *point,
            marks=pytest.mark.xfail(
                strict=True,
                reason="TP 2: measured 0.21 GiB against 0.61 predicted; one point, not modelled",
            ),
        )
        if point[2] > 1
        else pytest.param(*point)
        for point in _activation_points()
    ]


@pytest.mark.skipif(not (RUN / "records").is_dir(), reason="cohort records not present")
@pytest.mark.parametrize(("model_id", "gpu", "tp", "measured"), _activation_params())
def test_activation_estimate_matches_the_profiled_peak(
    model_id: str, gpu: str, tp: int, measured: int
) -> None:
    from apron.adapters.backends.vllm_quantization import default_max_num_batched_tokens
    from apron.domain.mechanisms import CalculatorInput, ComponentMechanism, TextWorkload
    from apron.domain.mechanisms.calculator import _activation_estimate
    from apron.interfaces.cohort_root import hardware_for

    entry = json.loads(FIXTURE.read_text())[model_id]
    hardware = hardware_for(gpu)
    tokens = default_max_num_batched_tokens(hardware.total_memory_bytes, gpu)
    inputs = CalculatorInput(
        mechanism=ComponentMechanism(mechanism="autoregressive_decode", role="decoder"),
        workload=TextWorkload(kind="text", input_length=512, output_length=128),
        artifact_metadata={},
        hardware=hardware,
        execution_spec_data={"max_num_batched_tokens": tokens},
    )
    metadata = {k: entry[k] for k in ("vocab_size", "hidden_size", "torch_dtype")}
    predicted = _activation_estimate(metadata, inputs, tp)
    assert predicted is not None
    relative, absolute = ACTIVATION_TOLERANCE
    assert abs(predicted - measured) <= max(relative * measured, absolute), (
        f"{model_id} on {gpu} TP {tp} ({tokens} tokens): predicted "
        f"{predicted / 2**30:.2f} GiB, measured {measured / 2**30:.2f} GiB"
    )
