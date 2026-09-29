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
# GLM-5.3-Flash on 2x B200: 150.28 GiB against 152.10 measured (-1.2%), the
# rest being the top-k index buffers vLLM builds at load (1.60 GiB at the
# boot's 16,384 tokens, layered.startup_buffer_bytes) and 0.23 GiB not traced.
TOLERANCE = 0.06


def _measured_points() -> list[tuple[str, str, str, int, int]]:
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
                entry.requested_execution.gpu_sku,
                entry.deployment_plan.dtype or "bfloat16",
                entry.deployment_plan.tensor_parallel,
                report.model_weight_memory,
            )
        )
    return sorted(points)


@pytest.mark.skipif(not (RUN / "records").is_dir(), reason="cohort records not present")
@pytest.mark.parametrize(("model_id", "gpu", "dtype", "tp", "measured"), _measured_points())
def test_weight_prediction_matches_the_measurement(
    model_id: str, gpu: str, dtype: str, tp: int, measured: int
) -> None:
    from apron.domain.mechanisms.layered import padded_weight_bytes
    from apron.interfaces.cohort_root import hardware_for

    recorded = json.loads(FIXTURE.read_text())
    assert model_id in recorded, "run scripts/record_weight_bytes.py"
    loaded = recorded_loaded_bytes(recorded[model_id], dtype)
    # Weights every rank holds whole (plan_pipeline.replicated_tensor_bytes);
    # the buffers vLLM builds at load (layered.startup_buffer_bytes) depend on
    # the plan's batch and are checked with the KV budget below.
    replicated = int(recorded[model_id].get("replicated_bytes") or 0)
    # Experts vLLM pads past the checkpoint on this GPU (layered.padded_weight_bytes).
    major, _, minor = hardware_for(gpu).compute_capability.partition(".")
    config = json.loads(CONFIGS.read_text())[model_id]
    padded = padded_weight_bytes(config, tp=tp, sm=int(major) * 10 + int(minor or 0))
    predicted = _per_gpu(loaded - replicated, tp) + replicated + padded
    assert abs(predicted - measured) / measured <= TOLERANCE, (
        f"{model_id} ({dtype}) TP {tp}: predicted {predicted / 2**30:.2f} GiB, "
        f"measured {measured / 2**30:.2f} GiB"
    )


# ---------------------------------------------------------------------------
# Activation: the calculator's estimate against vLLM's profiled torch peak
# ---------------------------------------------------------------------------

# 10% of the measurement, or 0.03 GiB for the small ones (log values are
# printed to 0.01 GiB).
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
    # The config fields the rule reads (text_config merged, towers as flags).
    metadata = {**entry["activation"], "torch_dtype": dtype}  # the plan's served dtype
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


# ---------------------------------------------------------------------------
# KV budget: the calculator's available KV memory against what vLLM left
# ---------------------------------------------------------------------------
#
# vLLM: available = requested - weights - torch peak - non-torch - CUDA-graph
# estimate (calculator.py, "The KV budget").  Each healthy memory record is
# predicted as it booted (cohort_root.recorded_prediction; recorded configs in
# tests/fixtures/cohort/configs.json).  Table and terms:
# _dev_notes/cohort-run/kv-budget-residuals.md.

CONFIGS = REPO / "tests" / "fixtures" / "cohort" / "configs.json"
_GIB = 1 << 30
# The startup log prints each term to 0.01 GiB: the five terms of the identity
# (requested, weights, peak, non-torch, CUDA graphs) and the available memory.
LOG_ROUNDING = int(0.005 * _GIB)
# The largest measured under-prediction: google/gemma-4-31B-it on an H100,
# -0.257 GiB (the per-layer CUDA-graph estimate 0.81 GiB over vLLM's, its
# weights 0.53 GiB under the measurement), plus the log's rounding.
KV_UNDER_PREDICTION = int(0.257 * _GIB) + LOG_ROUNDING


def _kv_records() -> list[str]:
    if not (RUN / "records").is_dir():
        return []
    records = load_cohort_run(RUN).records
    return sorted(
        digest
        for digest, report in records.reports.items()
        if report.claim_scope == "memory"
        and report.boot_outcome == "healthy"
        and records.solutions.get(report.solution_fingerprint or "") is not None
    )


def _kv_residual(digest: str) -> tuple[int, int]:
    """(predicted - measured available KV bytes, the plan's compile segment)."""
    from apron.interfaces.cohort_root import recorded_prediction

    records = load_cohort_run(RUN).records
    report = records.reports[digest]
    entry = records.solutions[report.solution_fingerprint or ""]
    configs = json.loads(CONFIGS.read_text())
    weights = json.loads(FIXTURE.read_text())
    predicted = recorded_prediction(entry, configs[entry.model_id], weights[entry.model_id])
    measured = int(report.available_kv_cache_memory or 0)
    return predicted["available_kv_cache_bytes"] - measured, predicted["compile_segment_bytes"]


@pytest.mark.skipif(not (RUN / "records").is_dir(), reason="cohort records not present")
@pytest.mark.parametrize("digest", _kv_records(), ids=lambda d: d[4:16])
def test_records_satisfy_vllms_kv_budget_identity(digest: str) -> None:
    """requested - weights - peak - non-torch - CUDA-graph estimate = available,
    to the log's rounding: the formula the calculator follows is vLLM's
    (v1/worker/gpu_worker.py:606-610; utils/mem_utils.py:317-326)."""
    report = load_cohort_run(RUN).records.reports[digest]
    left = (
        int(report.requested_memory or 0)
        - int(report.model_weight_memory or 0)
        - int(report.transient_peak_headroom or 0)
        - int(report.non_pytorch_increase or 0)
        - int(report.cuda_graph_estimate or 0)
    )
    assert abs(left - int(report.available_kv_cache_memory or 0)) <= 6 * LOG_ROUNDING


@pytest.mark.skipif(not (RUN / "records").is_dir(), reason="cohort records not present")
@pytest.mark.parametrize("digest", _kv_records(), ids=lambda d: d[4:16])
def test_kv_budget_prediction_matches_what_vllm_left(digest: str) -> None:
    """Over: at most the safety buffer plus the compile segment the plan may
    hold (``compile_segment_bytes``); under: at most the largest measured."""
    from apron.domain.mechanisms.layered import KV_BUDGET_SAFETY_BUFFER_BYTES

    residual, segment = _kv_residual(digest)
    assert -KV_UNDER_PREDICTION <= residual <= KV_BUDGET_SAFETY_BUFFER_BYTES + segment, (
        f"{digest[:12]}: predicted - measured {residual / _GIB:+.3f} GiB "
        f"(segment {segment / _GIB:.2f} GiB)"
    )


@pytest.mark.skipif(not (RUN / "records").is_dir(), reason="cohort records not present")
def test_the_safety_buffer_is_the_largest_measured_error() -> None:
    """The guard's buffer equals the largest over-prediction measured (with the
    plan's compile segment set apart), rounded up for the log's 0.005 GiB and to
    0.01 GiB: not more, so it names the measured error, not a margin."""
    from apron.domain.mechanisms.layered import KV_BUDGET_SAFETY_BUFFER_BYTES

    worst = max(residual - segment for residual, segment in map(_kv_residual, _kv_records()))
    assert worst + LOG_ROUNDING <= KV_BUDGET_SAFETY_BUFFER_BYTES
    assert KV_BUDGET_SAFETY_BUFFER_BYTES - worst <= LOG_ROUNDING + int(0.01 * _GIB)


@pytest.mark.skipif(not (RUN / "records").is_dir(), reason="cohort records not present")
def test_gpu_specs_are_the_byte_counts_the_pods_detected() -> None:
    """Every detected GPU SKU's GPU_SPECS total is a count some pod detected:
    the report's ``detected_hardware_fingerprint`` hashes it."""
    from apron.adapters.backends.runpod import GPU_SPECS
    from apron.domain.fingerprints import fingerprint_hex
    from apron.domain.schemas.primitives import HardwareSpec

    records = load_cohort_run(RUN).records
    detected: dict[str, set[str]] = {}
    for report in records.reports.values():
        entry = records.solutions.get(report.solution_fingerprint or "")
        if entry is not None and report.detected_hardware_fingerprint:
            sku = entry.requested_execution.gpu_sku
            detected.setdefault(sku, set()).add(report.detected_hardware_fingerprint)
    assert len(detected) >= 6
    for sku, fingerprints in detected.items():
        spec = GPU_SPECS[sku]
        hardware = HardwareSpec(
            gpu_sku=sku,
            total_memory_bytes=spec["total_memory_bytes"],
            compute_capability=spec["compute_capability"],
        )
        assert fingerprint_hex(hardware) in fingerprints, sku
