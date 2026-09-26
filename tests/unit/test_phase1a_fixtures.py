"""The Phase 1a run fixtures are strict domain documents (F2, §8).

``evaluation-protocol.json`` binds the protocol to the Phase 1a solution:
Qwen/Qwen3-8B, the plan the pipeline builds for an L4, on RunPod Secure with
the pinned runner image.  Its digests reproduce from the validated sibling
fixtures.  Set ``APRON_REGENERATE_FIXTURES=1`` to rewrite it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import apron.domain.mechanisms.calculator  # noqa: F401 — registers calculators
from apron.adapters.evidence.hf_hub import FixtureHFHubResolver
from apron.adapters.planning.calculator_source import CalculatorPlanningSource
from apron.adapters.runner_image import RUNNER_IMAGE_DIGEST
from apron.application.orchestration.evidence import (
    build_evaluation_protocol,
    decision_request_digest,
    protocol_template,
    solution_fingerprint,
)
from apron.application.orchestration.plan_pipeline import run_plan_pipeline
from apron.domain.fingerprints import fingerprint_hex
from apron.domain.schemas.authority import DecisionRequest
from apron.domain.schemas.primitives import HardwareSpec
from apron.domain.schemas.solutions import EvaluationProtocol, RequestedExecutionSpec
from apron.domain.schemas.tasks import ApplicationSpec, ServingWorkloadSpec, TaskSuiteSpec

RUN = Path(__file__).parents[1] / "fixtures" / "phase-1a-run"
HF = Path(__file__).parents[1] / "fixtures" / "external-formats" / "huggingface-hub"

_L4 = HardwareSpec(
    gpu_sku="NVIDIA L4", total_memory_bytes=25_769_803_776, compute_capability="8.9"
)


class _Clock:
    def now(self):  # type: ignore[no-untyped-def]
        from datetime import UTC, datetime

        return datetime(2026, 9, 25, tzinfo=UTC)


class _Ids:
    def generate(self) -> str:
        return "fixed"


def _phase1a_solution_fp() -> str:
    result = run_plan_pipeline(
        FixtureHFHubResolver(HF),
        CalculatorPlanningSource(clock=_Clock()),
        "Qwen/Qwen3-8B",
        _L4,
        clock=_Clock(),
        id_gen=_Ids(),
    )
    assert result.ok, result.error
    assert result.plan is not None and result.model_spec is not None
    requested = RequestedExecutionSpec(
        provider="runpod",
        gpu_sku="NVIDIA L4",
        gpu_count=1,
        cloud_type="SECURE",
        image_digest=RUNNER_IMAGE_DIGEST,
    )
    return solution_fingerprint(result.model_spec, result.plan, requested)


def _load() -> tuple[DecisionRequest, TaskSuiteSpec, ApplicationSpec]:
    return (
        DecisionRequest.model_validate_json((RUN / "decision-request.json").read_text()),
        TaskSuiteSpec.model_validate_json((RUN / "task-suite-spec.json").read_text()),
        ApplicationSpec.model_validate_json((RUN / "application-spec.json").read_text()),
    )


def _expected_protocol() -> EvaluationProtocol:
    request, suite, application = _load()
    template = EvaluationProtocol.model_validate_json(
        (RUN / "evaluation-protocol.json").read_text()
    )
    return build_evaluation_protocol(
        protocol_template(template),
        request=request,
        task_suite=suite,
        application=application,
        solution_fp=_phase1a_solution_fp(),
    )


def test_evaluation_protocol_digests_reproduce_from_objects() -> None:
    expected = _expected_protocol()
    path = RUN / "evaluation-protocol.json"
    if os.environ.get("APRON_REGENERATE_FIXTURES") == "1":
        path.write_text(json.dumps(expected.model_dump(mode="json"), indent=2) + "\n")
    stored = EvaluationProtocol.model_validate_json(path.read_text())
    assert stored == expected


def test_protocol_digests_are_the_sibling_fixtures() -> None:
    request, suite, application = _load()
    stored = EvaluationProtocol.model_validate_json((RUN / "evaluation-protocol.json").read_text())
    assert stored.decision_request_digest == decision_request_digest(request)
    assert stored.task_suite_fingerprint == fingerprint_hex(suite)
    assert stored.application_fingerprint == fingerprint_hex(application)
    assert stored.solution_fingerprint == _phase1a_solution_fp()


def test_serving_workload_declares_slo_fields() -> None:
    spec = ServingWorkloadSpec.model_validate_json(
        (RUN / "serving-workload-spec.json").read_text()
    )
    assert spec.input_sequence_length == 512
    assert spec.output_sequence_length == 128
    assert spec.p99_ttft_ms == 2000
    assert spec.p99_tpot_ms == 100
