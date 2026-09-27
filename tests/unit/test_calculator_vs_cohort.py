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

from apron.application.orchestration.plan_pipeline import recorded_loaded_bytes
from apron.domain.mechanisms.calculator import _per_gpu
from apron.interfaces.cohort_root import load_cohort_run

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "cohort" / "weight-bytes.json"
RUN = REPO / "_dev_notes" / "cohort-run"
# The FP8 checkpoint lands 5.4% under (0.70 vs 0.74 GiB); every other point
# within 1.5%.  Before the L5 fixes: +17% (tied lm_head), -22% (FP8), +100% (TP 2).
TOLERANCE = 0.06


def _measured_points() -> list[tuple[str, str, int, int]]:
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
            (
                entry.model_id,
                entry.deployment_plan.dtype or "bfloat16",
                entry.deployment_plan.tensor_parallel,
                report.model_weight_memory,
            )
        )
    return sorted(points)


@pytest.mark.skipif(not (RUN / "records").is_dir(), reason="cohort records not present")
@pytest.mark.parametrize(("model_id", "dtype", "tp", "measured"), _measured_points())
def test_weight_prediction_matches_the_measurement(
    model_id: str, dtype: str, tp: int, measured: int
) -> None:
    recorded = json.loads(FIXTURE.read_text())
    assert model_id in recorded, "run scripts/record_weight_bytes.py"
    predicted = _per_gpu(recorded_loaded_bytes(recorded[model_id], dtype), tp)
    assert abs(predicted - measured) / measured <= TOLERANCE, (
        f"{model_id} ({dtype}) TP {tp}: predicted {predicted / 2**30:.2f} GiB, "
        f"measured {measured / 2**30:.2f} GiB"
    )


# ---------------------------------------------------------------------------
# Activation: the calculator's estimate against vLLM's profiled torch peak
# ---------------------------------------------------------------------------

# 10% of the measurement, or 0.03 GiB for the small ones (log values are
# printed to 0.01 GiB).  Every point, TP 2 included, now fits within 0.02 GiB.
ACTIVATION_TOLERANCE = (0.10, int(0.03 * 2**30))


def _activation_points() -> list[tuple[str, str, int, str, int]]:
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
                entry.deployment_plan.dtype or "bfloat16",
                report.transient_peak_headroom,
            )
        )
    return sorted(points)


def _activation_params() -> list[Any]:
    return [pytest.param(*point) for point in _activation_points()]


@pytest.mark.skipif(not (RUN / "records").is_dir(), reason="cohort records not present")
@pytest.mark.parametrize(("model_id", "gpu", "tp", "dtype", "measured"), _activation_params())
def test_activation_estimate_matches_the_profiled_peak(
    model_id: str, gpu: str, tp: int, dtype: str, measured: int
) -> None:
    from apron.adapters.backends.vllm_quantization import (
        default_max_num_batched_tokens,
        default_max_num_seqs,
    )
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
        execution_spec_data={
            "max_num_batched_tokens": tokens,
            "max_num_seqs": default_max_num_seqs(hardware.total_memory_bytes, gpu),
        },
    )
    metadata = {"vocab_size": entry["vocab_size"], "hidden_size": entry["hidden_size"]}
    metadata["torch_dtype"] = dtype  # the plan's served dtype
    predicted = _activation_estimate(metadata, inputs, tp)
    assert predicted is not None
    relative, absolute = ACTIVATION_TOLERANCE
    assert abs(predicted - measured) <= max(relative * measured, absolute), (
        f"{model_id} on {gpu} TP {tp} ({tokens} tokens): predicted "
        f"{predicted / 2**30:.2f} GiB, measured {measured / 2**30:.2f} GiB"
    )


# ---------------------------------------------------------------------------
# Mamba-1 state: the calculator against the pool vLLM reports
# ---------------------------------------------------------------------------


def _state_points() -> list[tuple[str, str, int, float]]:
    if not (RUN / "records").is_dir():
        return []
    records = load_cohort_run(RUN).records
    ssm_models = {m for m, e in json.loads(FIXTURE.read_text()).items() if e.get("ssm")}
    points = set()
    for report in records.reports.values():
        entry = records.solutions.get(report.solution_fingerprint or "")
        if entry is None or not report.max_concurrency or not report.available_kv_cache_memory:
            continue
        if entry.model_id not in ssm_models:
            continue  # an attention model: its cache grows per token
        points.add(
            (
                entry.model_id,
                entry.deployment_plan.dtype or "bfloat16",
                report.available_kv_cache_memory,
                report.max_concurrency,
            )
        )
    return sorted(points)


@pytest.mark.skipif(not (RUN / "records").is_dir(), reason="cohort records not present")
@pytest.mark.parametrize(("model_id", "dtype", "available", "concurrency"), _state_points())
def test_mamba_state_matches_the_pool_vllm_reports(
    model_id: str, dtype: str, available: int, concurrency: float
) -> None:
    """available KV memory / max concurrency = the bytes one sequence reserves
    (v1/core/kv_cache_utils.py:1047-1069; no padding for a pure Mamba model)."""
    from apron.domain.mechanisms.calculator import DTYPE_BYTES

    ssm = json.loads(FIXTURE.read_text())[model_id]["ssm"]
    predicted = (
        ssm["num_hidden_layers"]
        * ssm["intermediate_size"]
        * ((ssm["conv_kernel"] - 1) + ssm["state_size"])
        * DTYPE_BYTES[dtype]
    )
    measured = available / concurrency  # the log rounds GiB to 0.01: about 0.5%
    assert abs(predicted - measured) / measured <= 0.01, (predicted, measured)
