"""§10.3: the real L0-F logs, with the recorded classifier responses, correct each plan.

Each fixture (``tests/fixtures/l0f/classN.json``, written by
``scripts/record_l0f_classifier.py``) holds the log tail the fix proof reads
(the stored failed-boot record of the L0-F boot on RunPod), the production
classifier's recorded response and extraction, and the planning facts
diagnosis reads.  The pipeline, rules and correction strategies are the real
code; only the classifier call is replayed.  No GPU, no API call — this must
pass before any fixed boot is paid for.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from apron.adapters.backends.rule_loader import load_rules
from apron.adapters.backends.runpod import GPU_SPECS
from apron.application.cost_estimator import hourly_rate
from apron.application.orchestration.correction import CatalogEntry, CorrectionContext
from apron.application.orchestration.diagnosis_pipeline import run_diagnosis_pipeline
from apron.application.orchestration.remediation import SIX_CLASSES
from apron.domain.fingerprints import fingerprint_hex
from apron.interfaces.cohort_root import hardware_for, search_from_json

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "l0f"
RULES = load_rules(REPO / "rules", "vllm", "v0.29.0")
CATALOG = tuple(
    CatalogEntry(hardware_for(sku), hourly_rate("runpod", sku)[0]) for sku in GPU_SPECS
)


class RecordedEngine:
    """The production classifier's recorded answer for exactly this log."""

    def __init__(self, fixture: dict[str, Any]) -> None:
        self._fixture = fixture

    def classify(self, error: str) -> dict[str, Any]:
        assert error == self._fixture["log"], "replay only the log the response was recorded for"
        return dict(self._fixture["classification"])

    def extract(self, error: str, failure_class: str) -> dict[str, Any]:
        assert failure_class == self._fixture["classification"]["failure_class"]
        return dict(self._fixture["extraction"])


def _fixture(n: int) -> dict[str, Any]:
    return json.loads((FIXTURES / f"class{n}.json").read_text("utf-8"))


def _lineage(n: int) -> dict[str, Any] | None:
    """The recorded same-lineage search (``scripts/record_class6_lineage.py``)."""
    path = FIXTURES / f"class{n}-lineage.json"
    return json.loads(path.read_text("utf-8")) if path.exists() else None


def _diagnose(n: int) -> tuple[Any, Any, dict[str, Any]]:
    case = SIX_CLASSES[n - 1]
    fx = _fixture(n)
    assert fx["broken_plan_digest"] == fingerprint_hex(case.broken_plan), "fixture is stale"
    lineage = _lineage(n)
    search = search_from_json(lineage["search"]) if lineage and lineage["search"] else None
    result = run_diagnosis_pipeline(
        fx["log"],
        RecordedEngine(fx),
        case.broken_plan,
        fx["model_config"],
        hardware_for(fx["gpu_sku"]),
        RULES,
        correction_context=CorrectionContext(
            catalog=CATALOG,
            predicted_total_bytes=fx["predicted_total_bytes"],
            artifacts=lambda: search,
        ),
    )
    return case, result, fx


@pytest.mark.parametrize("n", range(1, 7))
def test_real_log_is_diagnosed_as_its_class_and_corrected(n: int) -> None:
    case, result, fx = _diagnose(n)
    assert case.expected_error in fx["log"], "the log must show the named failure (L0-F)"
    assert result.failure_class == case.expected_family  # gate A
    assert result.corrected_plan is not None and result.corrected_plan != case.broken_plan  # B


def _usable(sku: str) -> float:
    return hardware_for(sku).total_memory_bytes * 0.90


def test_class1_retargets_to_the_cheapest_gpu_that_holds_the_prediction() -> None:
    _, result, fx = _diagnose(1)
    predicted = fx["predicted_total_bytes"]
    fits = [e for e in CATALOG if _usable(e.hardware.gpu_sku) >= predicted]
    cheapest = min(fits, key=lambda e: e.hourly_rate).hardware.gpu_sku
    chosen = result.corrected_plan.resource_allocation["gpu_sku"]
    assert chosen == cheapest
    assert _usable(chosen) >= predicted > _usable(fx["gpu_sku"])


def test_class2_clamps_to_the_estimated_max_model_len_in_the_log() -> None:
    _, result, fx = _diagnose(2)
    assert result.corrected_plan.engine_configuration["max_model_len"] == str(
        fx["extraction"]["estimated_max_model_len"]
    )
    assert (
        f"estimated maximum model length is {fx['extraction']['estimated_max_model_len']}"
        in (fx["log"])
    )


def test_class3_clamps_to_the_derived_max() -> None:
    _, result, _ = _diagnose(3)
    assert result.corrected_plan.engine_configuration["max_model_len"] == "32768"


def test_class4_falls_back_to_bfloat16() -> None:
    _, result, _ = _diagnose(4)
    assert result.corrected_plan.dtype == "bfloat16"


def test_class5_reduces_tensor_parallel_to_a_divisor_that_fits() -> None:
    case, result, _ = _diagnose(5)
    tp = result.corrected_plan.tensor_parallel
    assert tp == 2
    assert 32 % tp == 0 and tp <= int(case.broken_plan.resource_allocation["gpu_count"])


def test_class6_substitutes_the_publishers_4bit_checkpoint_on_the_same_gpu() -> None:
    """The broken plan asks for a 4-bit fp_quant Qwen3-8B on an H100; the fix
    serves the same network as the base publisher's 4-bit AWQ copy, on the
    same H100 (booted 2026-09-27: 3/3, Fixed)."""
    case, result, _ = _diagnose(6)
    allocation = result.corrected_plan.resource_allocation
    assert allocation["model_id"] == "Qwen/Qwen3-8B-AWQ"
    assert allocation["gpu_sku"] == case.broken_plan.resource_allocation["gpu_sku"]
    assert result.correction_strategy == "substitute_artifact"


def test_class6_base_model_was_proposed_then_confirmed_by_the_hub() -> None:
    lineage = _lineage(6)
    assert lineage is not None
    base = lineage["evidence"]["base_model"]
    assert base["source"] == "proposed" and base["confirmed"] is True
    assert base["proposal"]["classifier_model_id"] and base["proposal"]["classifier_cost_usd"]
    search = search_from_json(lineage["search"])
    assert search.base_model_id == base["id"]
    assert all(dict(c.shape) == dict(search.requested_shape) for c in search.candidates)


def test_class1_never_retargets_to_a_gpu_without_capacity() -> None:
    """At run time the catalog holds only GPUs with image-compatible stock.

    RTX A6000 hosts run CUDA 12.8 only (stock map, 2026-09-27): the cheapest
    fitting GPU on paper cannot boot the image, so the fix goes to the next one.
    """
    from apron.interfaces.cohort_root import available_catalog

    listed = {"NVIDIA GeForce RTX 4090", "NVIDIA A100 80GB PCIe", "NVIDIA H100 80GB HBM3"}
    catalog = available_catalog({}, 1, stock=lambda sku: "Low" if sku in listed else None)
    assert {e.hardware.gpu_sku for e in catalog} == listed
    case, fx = SIX_CLASSES[0], _fixture(1)
    result = run_diagnosis_pipeline(
        fx["log"],
        RecordedEngine(fx),
        case.broken_plan,
        fx["model_config"],
        hardware_for(fx["gpu_sku"]),
        RULES,
        correction_context=CorrectionContext(
            catalog=catalog, predicted_total_bytes=fx["predicted_total_bytes"]
        ),
    )
    assert result.corrected_plan is not None
    assert result.corrected_plan.resource_allocation["gpu_sku"] == "NVIDIA A100 80GB PCIe"
