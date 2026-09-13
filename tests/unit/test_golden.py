"""Step 12 tests: golden fixtures — full document chains, round-trip, remediation derivation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from apron.domain.canonical import canonicalize, record_digest_hex
from apron.domain.capabilities import CapabilitySignature
from apron.domain.fingerprints import fingerprint_hex
from apron.domain.mechanisms import ComponentMechanism
from apron.domain.schemas.authority import DecisionRequest
from apron.domain.schemas.models import (
    ArtifactSpec,
    ExecutionSpec,
    ModelSpec,
)
from apron.domain.schemas.primitives import HardwareSpec
from apron.domain.schemas.records import (
    DiagnosisRule,
    RemediationRecord,
    TaskAttemptRecord,
    VerificationReport,
    derive_remediation_result,
)
from apron.domain.schemas.reports import (
    CandidateEntry,
    DecisionReport,
    MaintainerBaselineAllocation,
)
from apron.domain.schemas.solutions import (
    DeploymentPlan,
    EvaluationProtocol,
)
from apron.domain.schemas.tasks import (
    ApplicationSpec,
    ServingWorkloadSpec,
    TaskSuiteSpec,
    WorkloadSpec,
)
from apron.domain.solutions import (
    DirectEndpoint,
    InferenceSolution,
    LogicalRoute,
    RoleBinding,
)

from apron.domain.artifacts.identity import ArtifactIdentity

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "golden"

_TEXT_CAP = CapabilitySignature(
    operation="text_generation",
    required_inputs=("text",),
    output_representation="generated_text",
)

_VISION_CAP = CapabilitySignature(
    operation="text_generation",
    required_inputs=("text", "image"),
    output_representation="generated_text",
)

_HW_4090 = HardwareSpec(
    gpu_sku="NVIDIA RTX 4090",
    total_memory_bytes=25_769_803_776,
    compute_capability="8.9",
    memory_bandwidth_gbps=1008.0,
    interconnect="PCIe",
)


def _write_fixture(directory: str, name: str, data: dict[str, Any]) -> None:
    path = FIXTURES_DIR / directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonicalize(data))


def _round_trip_check(model_cls, instance):
    canonical = canonicalize(instance.model_dump(mode="json"))
    restored = model_cls.model_validate_json(canonical)
    assert canonicalize(restored.model_dump(mode="json")) == canonical
    return restored


# ============================================================================
# BUILDERS — construct the full document chain for each golden category
# ============================================================================


def _build_self_hosted() -> dict[str, Any]:
    """Qwen3-8B BF16 on RTX 4090 via vLLM — single self-hosted solution."""
    decision_request = DecisionRequest(
        objective="deploy Qwen3-8B for text generation",
        success_criteria=("exact_match > 0.8",),
        quality_floor=0.7,
        serving_slos={"p99_ttft_ms": 2000.0},
        privacy_constraints=("no_external_destinations",),
        budget_limit=100.0,
    )

    task_suite = TaskSuiteSpec(
        name="simple-text-qa",
        version="1.0",
        required_capabilities=(_TEXT_CAP,),
        cases=(
            {"input": "What is 2+2?", "expected": "4"},
            {"input": "Capital of France?", "expected": "Paris"},
            {"input": "Largest planet?", "expected": "Jupiter"},
        ),
        success_criteria=("exact_match",),
        privacy_classification="public",
    )

    application = ApplicationSpec(name="minimal-qa", version="1.0")

    serving = ServingWorkloadSpec(
        concurrency=4,
        latency_p99_ms=2000.0,
        throughput_target_rps=10.0,
    )

    workload = WorkloadSpec(
        task_suite_fingerprint=fingerprint_hex(task_suite),
        serving_workload_fingerprint=fingerprint_hex(serving),
    )

    identity = ArtifactIdentity(content_digest="1220" + "a1" * 32)
    artifact_spec = ArtifactSpec(
        identity=identity,
        config_digest="1220" + "c1" * 32,
    )
    model_spec = ModelSpec(
        repository="Qwen/Qwen3-8B",
        immutable_revision="abc123def456",
        license="Apache-2.0",
        components=(
            ComponentMechanism(mechanism="autoregressive_decode", role="decoder"),
        ),
        component_bytes_dtype={"decoder": "bf16"},
    )
    execution_spec = ExecutionSpec(
        engine_image_digest="1220" + "e0" * 32,
        resolved_checkpoint_method="auto",
    )

    endpoint = DirectEndpoint(
        binding="direct_endpoint",
        model_spec_fingerprint=fingerprint_hex(model_spec),
        artifact_spec_fingerprint=fingerprint_hex(artifact_spec),
        execution_spec_fingerprint=fingerprint_hex(execution_spec),
        capabilities=(fingerprint_hex(_TEXT_CAP),),
        target_kind="local-container",
        deployment_plan_fingerprint=None,  # filled after plan is built
    )

    solution = InferenceSolution(endpoints=(endpoint,))

    plan = DeploymentPlan(
        tensor_parallel=1,
        dtype="bf16",
        batch_size=32,
        engine_configuration={"model": "Qwen/Qwen3-8B"},
        serve_command="vllm serve Qwen/Qwen3-8B --dtype bf16",
        docker_compose="services:\n  vllm:\n    image: vllm/vllm-openai:latest",
    )

    dr_digest = record_digest_hex(decision_request.model_dump(mode="json"))
    eval_protocol = EvaluationProtocol(
        decision_request_digest=dr_digest,
        task_suite_fingerprint=fingerprint_hex(task_suite),
        application_fingerprint=fingerprint_hex(application),
        solution_fingerprint=fingerprint_hex(solution),
        harness="apron_eval",
        harness_version="0.1.0",
        scorer="exact_match",
        seeds=(42,),
        repetitions=1,
    )

    attempt = TaskAttemptRecord(
        decision_fingerprint=fingerprint_hex(decision_request),
        task_suite_fingerprint=fingerprint_hex(task_suite),
        application_fingerprint=fingerprint_hex(application),
        evaluation_protocol_fingerprint=fingerprint_hex(eval_protocol),
        solution_fingerprint=fingerprint_hex(solution),
        case_id="case-1",
        attempt_id="attempt-1",
        criterion_scores={"exact_match": 1.0},
        accepted=True,
        claim_scope="task_outcome",
        production_mode=False,
        reason="evaluation_run",
        lifecycle="observed",
    )

    verification = VerificationReport(
        target_kind="local-container",
        operator="self",
        execution_fingerprint="1220" + "ef" * 32,
        initial_total_memory=25_769_803_776,
        initial_free_memory=20_000_000_000,
        requested_memory=16_000_000_000,
        model_weight_memory=8_500_000_000,
        persistent_consumption=10_200_000_000,
        transient_peak_headroom=2_000_000_000,
        non_pytorch_increase=500_000_000,
        cuda_graph_estimate=800_000_000,
        cuda_graph_applied=750_000_000,
        cuda_graph_actual=720_000_000,
        available_kv_cache_memory=6_500_000_000,
        safety_buffer=500_000_000,
        profiling_shape={"batch_size": 1, "seq_len": 1},
        claim_scope="boot",
        production_mode=False,
        reason="boot_profiling",
        lifecycle="observed",
    )

    report = DecisionReport(
        decision_request_digest=dr_digest,
        candidates=(
            CandidateEntry(
                solution_fingerprint=fingerprint_hex(solution),
                qualification_status="qualified",
            ),
        ),
        disclosed_comparable_set=(),
    )

    return {
        "decision_request": decision_request,
        "task_suite": task_suite,
        "application": application,
        "serving": serving,
        "workload": workload,
        "solution": solution,
        "plan": plan,
        "eval_protocol": eval_protocol,
        "attempt": attempt,
        "verification": verification,
        "report": report,
    }


def _build_managed_api() -> dict[str, Any]:
    """Managed API — provider_opaque, no plan, no verification."""
    decision_request = DecisionRequest(
        objective="evaluate gpt-4o for text generation",
        success_criteria=("accuracy > 0.9",),
        budget_limit=50.0,
    )

    task_suite = TaskSuiteSpec(
        name="managed-api-qa",
        version="1.0",
        required_capabilities=(_TEXT_CAP,),
        cases=({"input": "test", "expected": "response"},),
        success_criteria=("accuracy",),
    )

    application = ApplicationSpec(name="managed-qa", version="1.0")

    model_spec = ModelSpec(
        components=(
            ComponentMechanism(mechanism="autoregressive_decode", role="decoder"),
        ),
        publisher_claims=("gpt-4o-2024-08-06",),
    )

    endpoint = DirectEndpoint(
        binding="direct_endpoint",
        model_spec_fingerprint=fingerprint_hex(model_spec),
        provider="openai",
        provider_opaque=True,
    )

    solution = InferenceSolution(endpoints=(endpoint,))

    dr_digest = record_digest_hex(decision_request.model_dump(mode="json"))
    report = DecisionReport(
        decision_request_digest=dr_digest,
        candidates=(
            CandidateEntry(
                solution_fingerprint=fingerprint_hex(solution),
                qualification_status="task_evaluated",
                evidence_state="provider_opaque for boot/memory",
            ),
        ),
        disclosed_comparable_set=(),
    )

    return {
        "decision_request": decision_request,
        "task_suite": task_suite,
        "application": application,
        "solution": solution,
        "report": report,
    }


def _build_compound() -> dict[str, Any]:
    """Three-role compound: managed planner + self-hosted executor + vision."""
    decision_request = DecisionRequest(
        objective="compound planner/executor/vision for multimodal tasks",
        success_criteria=("vision_accuracy > 0.7",),
    )

    task_suite = TaskSuiteSpec(
        name="multimodal-qa",
        version="1.0",
        required_capabilities=(_TEXT_CAP, _VISION_CAP),
        cases=({"input": "describe image", "expected": "a cat"},),
        success_criteria=("vision_accuracy",),
    )

    application = ApplicationSpec(
        name="compound-app",
        version="1.0",
        logical_roles=("planner", "executor", "vision_analyzer"),
        invocation_conditions={"vision_analyzer": "input_has_image"},
    )

    planner_ep = DirectEndpoint(
        binding="direct_endpoint",
        model_spec_fingerprint="1220" + "b1" * 32,
        provider="openai",
        provider_opaque=True,
    )
    executor_ep = DirectEndpoint(
        binding="direct_endpoint",
        model_spec_fingerprint="1220" + "e0" * 32,
        target_kind="local-container",
        deployment_plan_fingerprint="1220" + "dd" * 32,
    )
    vision_ep = DirectEndpoint(
        binding="direct_endpoint",
        model_spec_fingerprint="1220" + "bb" * 32,
        target_kind="local-container",
        capabilities=(fingerprint_hex(_VISION_CAP),),
        deployment_plan_fingerprint="1220" + "d0" * 32,
    )

    solution = InferenceSolution(
        endpoints=(planner_ep, executor_ep, vision_ep),
        role_bindings=(
            RoleBinding(role="planner", endpoint_fingerprint=fingerprint_hex(planner_ep)),
            RoleBinding(role="executor", endpoint_fingerprint=fingerprint_hex(executor_ep)),
            RoleBinding(role="vision_analyzer", endpoint_fingerprint=fingerprint_hex(vision_ep)),
        ),
        routing_policy="role_based",
    )

    executor_plan = DeploymentPlan(tensor_parallel=1, dtype="bf16")
    vision_plan = DeploymentPlan(tensor_parallel=2, dtype="bf16")

    return {
        "decision_request": decision_request,
        "task_suite": task_suite,
        "application": application,
        "solution": solution,
        "executor_plan": executor_plan,
        "vision_plan": vision_plan,
    }


def _build_remediation() -> dict[str, Any]:
    """Remediation cycle: OOM → diagnose → correct → verify."""
    original_vr = VerificationReport(
        target_kind="local-container",
        operator="self",
        execution_fingerprint="1220" + "f1" * 32,
        initial_total_memory=25_769_803_776,
        initial_free_memory=20_000_000_000,
        requested_memory=22_000_000_000,
        model_weight_memory=8_500_000_000,
        persistent_consumption=18_000_000_000,
        transient_peak_headroom=5_000_000_000,
        non_pytorch_increase=1_000_000_000,
        cuda_graph_estimate=800_000_000,
        cuda_graph_applied=750_000_000,
        cuda_graph_actual=720_000_000,
        available_kv_cache_memory=1_000_000_000,
        safety_buffer=500_000_000,
        profiling_shape={"batch_size": 64, "seq_len": 512},
        claim_scope="boot",
        production_mode=False,
        reason="boot_profiling_oom",
        lifecycle="observed",
    )

    diagnosis = DiagnosisRule(
        exception_class="OutOfMemoryError",
        engine_callsite_module="vllm.worker",
        error_family="oom",
        correction_spec={"batch_size": "32"},
    )

    original_plan = DeploymentPlan(
        tensor_parallel=1, dtype="bf16", batch_size=64,
    )
    corrected_plan = DeploymentPlan(
        tensor_parallel=1, dtype="bf16", batch_size=32,
    )

    corrected_vr = VerificationReport(
        target_kind="local-container",
        operator="self",
        execution_fingerprint="1220" + "f2" * 32,
        initial_total_memory=25_769_803_776,
        initial_free_memory=20_000_000_000,
        requested_memory=12_000_000_000,
        model_weight_memory=8_500_000_000,
        persistent_consumption=10_200_000_000,
        transient_peak_headroom=2_000_000_000,
        non_pytorch_increase=500_000_000,
        cuda_graph_estimate=800_000_000,
        cuda_graph_applied=750_000_000,
        cuda_graph_actual=720_000_000,
        available_kv_cache_memory=6_500_000_000,
        safety_buffer=500_000_000,
        profiling_shape={"batch_size": 32, "seq_len": 512},
        claim_scope="boot",
        production_mode=False,
        reason="corrected_boot_profiling",
        lifecycle="observed",
    )

    original_digest = record_digest_hex(original_vr.model_dump(mode="json"))
    corrected_plan_digest = record_digest_hex(corrected_plan.model_dump(mode="json"))
    accepted_request_digest = "1220" + "aa" * 32

    fixed = RemediationRecord(
        mechanism_outcome="verified",
        request_outcome="satisfied",
        accepted_request_digest=accepted_request_digest,
        task_fingerprint="1220" + "cc" * 32,
        application_fingerprint="1220" + "a1" * 32,
        evaluation_fingerprint="1220" + "ee" * 32,
        corrected_plan_digest=corrected_plan_digest,
        corrects=original_digest,
        proving_record_fingerprints=(fingerprint_hex(corrected_vr),),
    )

    tradeoff = RemediationRecord(
        mechanism_outcome="verified",
        request_outcome="violated",
        accepted_request_digest=accepted_request_digest,
        task_fingerprint="1220" + "cc" * 32,
        application_fingerprint="1220" + "a1" * 32,
        evaluation_fingerprint="1220" + "ee" * 32,
        corrected_plan_digest=corrected_plan_digest,
        violated_constraints=("throughput_slo: batch_size reduction from 64 to 32",),
        corrects=original_digest,
    )

    unverified = RemediationRecord(
        mechanism_outcome="not_evaluated",
        request_outcome="not_evaluated",
        accepted_request_digest=accepted_request_digest,
        task_fingerprint="1220" + "cc" * 32,
        application_fingerprint="1220" + "a1" * 32,
        evaluation_fingerprint="1220" + "ee" * 32,
        corrected_plan_digest=corrected_plan_digest,
    )

    return {
        "original_verification": original_vr,
        "diagnosis": diagnosis,
        "original_plan": original_plan,
        "corrected_plan": corrected_plan,
        "corrected_verification": corrected_vr,
        "fixed": fixed,
        "tradeoff": tradeoff,
        "unverified": unverified,
        "original_digest": original_digest,
        "corrected_plan_digest": corrected_plan_digest,
    }


# ============================================================================
# ROUND-TRIP TESTS — every fixture survives serialization
# ============================================================================


def _assert_all_round_trip(fixtures: dict[str, Any]) -> None:
    for name, obj in fixtures.items():
        if isinstance(obj, str):
            continue
        canonical = canonicalize(obj.model_dump(mode="json"))
        restored = type(obj).model_validate_json(canonical)
        assert canonicalize(restored.model_dump(mode="json")) == canonical, f"{name} failed round-trip"


def test_self_hosted_round_trip():
    _assert_all_round_trip(_build_self_hosted())


def test_managed_api_round_trip():
    _assert_all_round_trip(_build_managed_api())


def test_compound_round_trip():
    _assert_all_round_trip(_build_compound())


def test_remediation_round_trip():
    _assert_all_round_trip(_build_remediation())


# ============================================================================
# SELF-HOSTED — full document chain
# ============================================================================


def test_self_hosted_single_endpoint():
    fixtures = _build_self_hosted()
    sol = fixtures["solution"]
    assert len(sol.endpoints) == 1
    assert sol.endpoints[0].binding == "direct_endpoint"
    assert sol.endpoints[0].target_kind == "local-container"


def test_self_hosted_verification_15_fields():
    fixtures = _build_self_hosted()
    vr = fixtures["verification"]
    dumped = vr.model_dump(mode="json")
    memory_fields = [
        "initial_total_memory", "initial_free_memory", "requested_memory",
        "model_weight_memory", "persistent_consumption", "transient_peak_headroom",
        "non_pytorch_increase", "cuda_graph_estimate", "cuda_graph_applied",
        "cuda_graph_actual", "available_kv_cache_memory", "safety_buffer",
    ]
    for f in memory_fields:
        assert dumped[f] is not None, f"missing: {f}"


def test_self_hosted_single_candidate_no_measured_efficient():
    fixtures = _build_self_hosted()
    report = fixtures["report"]
    assert len(report.candidates) == 1
    assert report.disclosed_comparable_set == ()
    assert report.candidates[0].qualification_status != "measured_efficient"


# ============================================================================
# MANAGED-API — provider_opaque, no plan, no verification
# ============================================================================


def test_managed_api_provider_opaque():
    fixtures = _build_managed_api()
    sol = fixtures["solution"]
    ep = sol.endpoints[0]
    assert ep.provider_opaque is True
    assert ep.provider == "openai"


def test_managed_api_no_plan_no_verification():
    fixtures = _build_managed_api()
    assert "plan" not in fixtures
    assert "verification" not in fixtures


# ============================================================================
# COMPOUND — three roles, routing, two plans
# ============================================================================


def test_compound_three_endpoints():
    fixtures = _build_compound()
    sol = fixtures["solution"]
    assert len(sol.endpoints) == 3
    assert len(sol.role_bindings) == 3
    roles = {rb.role for rb in sol.role_bindings}
    assert roles == {"planner", "executor", "vision_analyzer"}


def test_compound_changed_member_changes_fingerprint():
    fixtures = _build_compound()
    sol_a = fixtures["solution"]

    alt_vision = DirectEndpoint(
        binding="direct_endpoint",
        model_spec_fingerprint="1220" + "ae" * 32,
        target_kind="local-container",
    )
    sol_b = InferenceSolution(
        endpoints=(sol_a.endpoints[0], sol_a.endpoints[1], alt_vision),
        role_bindings=sol_a.role_bindings,
        routing_policy="role_based",
    )
    assert fingerprint_hex(sol_a) != fingerprint_hex(sol_b)


def test_compound_two_plans():
    fixtures = _build_compound()
    assert "executor_plan" in fixtures
    assert "vision_plan" in fixtures
    assert fingerprint_hex(fixtures["executor_plan"]) != fingerprint_hex(fixtures["vision_plan"])


# ============================================================================
# REMEDIATION — derivation logic and corrects-chain
# ============================================================================


def test_remediation_derivation_fixed():
    fixtures = _build_remediation()
    r = fixtures["fixed"]
    assert derive_remediation_result(r.mechanism_outcome, r.request_outcome) == "Fixed"


def test_remediation_derivation_tradeoff():
    fixtures = _build_remediation()
    r = fixtures["tradeoff"]
    assert derive_remediation_result(r.mechanism_outcome, r.request_outcome) == "Alternative with trade-offs"


def test_remediation_derivation_unverified():
    fixtures = _build_remediation()
    r = fixtures["unverified"]
    assert derive_remediation_result(r.mechanism_outcome, r.request_outcome) == "Unverified suggestion"


def test_remediation_corrects_chain():
    fixtures = _build_remediation()
    original_digest = fixtures["original_digest"]
    fixed = fixtures["fixed"]
    assert fixed.corrects == original_digest

    original_vr = fixtures["original_verification"]
    computed_digest = record_digest_hex(original_vr.model_dump(mode="json"))
    assert fixed.corrects == computed_digest


def test_remediation_corrected_plan_digest():
    fixtures = _build_remediation()
    corrected_plan = fixtures["corrected_plan"]
    computed = record_digest_hex(corrected_plan.model_dump(mode="json"))
    assert fixtures["fixed"].corrected_plan_digest == computed


def test_remediation_oom_structurally_correct():
    fixtures = _build_remediation()
    original = fixtures["original_verification"]
    assert original.requested_memory > original.initial_free_memory

    corrected = fixtures["corrected_verification"]
    assert corrected.requested_memory < corrected.initial_free_memory


# ============================================================================
# BUDGET — allocation covers run plan
# ============================================================================


def test_budget_feasibility():
    allocation = MaintainerBaselineAllocation(budget=500.0)
    run_cost = 350.0
    assert allocation.budget >= run_cost
    _round_trip_check(MaintainerBaselineAllocation, allocation)


# ============================================================================
# NEGATIVE — rejection scenarios
# ============================================================================


def test_negative_high_throughput_task_failure():
    report = DecisionReport(
        decision_request_digest="1220" + "dd" * 32,
        candidates=(
            CandidateEntry(
                solution_fingerprint="1220" + "a0" * 32,
                qualification_status="serving_verified",
                rejection_reason="task_outcome_failed",
                trade_off="high throughput but failed task",
            ),
        ),
    )
    assert report.candidates[0].qualification_status != "qualified"
    _round_trip_check(DecisionReport, report)


def test_negative_serving_slo_failure():
    report = DecisionReport(
        decision_request_digest="1220" + "d0" * 32,
        candidates=(
            CandidateEntry(
                solution_fingerprint="1220" + "b0" * 32,
                qualification_status="task_evaluated",
                rejection_reason="serving_slo_violated",
                trade_off="high task score but SLO miss",
            ),
        ),
    )
    assert report.candidates[0].qualification_status != "qualified"
    _round_trip_check(DecisionReport, report)


def test_negative_incomparable_cost_boundary():
    report = DecisionReport(
        decision_request_digest="1220" + "d3" * 32,
        candidates=(
            CandidateEntry(
                solution_fingerprint="1220" + "c0" * 32,
                qualification_status="qualified",
                evidence_state="managed_api",
            ),
            CandidateEntry(
                solution_fingerprint="1220" + "da" * 32,
                qualification_status="qualified",
                evidence_state="self_hosted",
            ),
        ),
        disclosed_comparable_set=(),
        exclusions=("cost boundary incomparable: managed vs self-hosted",),
    )
    assert report.disclosed_comparable_set == ()
    _round_trip_check(DecisionReport, report)


def test_negative_hardware_specific_measurement():
    fp_4090 = "1220" + "40" * 32
    fp_a100 = "1220" + "a1" * 32
    assert fp_4090 != fp_a100


def test_negative_deployment_substitution():
    env_permitted = DecisionRequest(
        objective="deploy on runpod",
        permitted_providers=("runpod",),
    )
    assert "lambda" not in env_permitted.permitted_providers


# ============================================================================
# WRITE representative fixtures to disk
# ============================================================================


def test_write_self_hosted_fixtures():
    fixtures = _build_self_hosted()
    for name, obj in fixtures.items():
        if isinstance(obj, str):
            continue
        _write_fixture("self-hosted", f"{name.replace('_', '-')}.json", obj.model_dump(mode="json"))


def test_write_remediation_fixtures():
    fixtures = _build_remediation()
    for name, obj in fixtures.items():
        if isinstance(obj, str):
            continue
        _write_fixture("remediation", f"{name.replace('_', '-')}.json", obj.model_dump(mode="json"))
