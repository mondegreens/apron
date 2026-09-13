"""Step 6 tests: Layer 4 — solutions, plans, evaluation, topology."""

from pydantic import TypeAdapter

from apron.domain.canonical import canonicalize, record_digest_hex
from apron.domain.fingerprints import assert_fully_classified, fingerprint_hex
from apron.domain.schemas.solutions import DeploymentPlan, EvaluationProtocol
from apron.domain.solutions import (
    DirectEndpoint,
    DistributedExecutionGroup,
    EndpointBinding,
    InferenceSolution,
    LogicalRoute,
    ReplicaPool,
    RoleBinding,
)

_FP = "1220" + "ab" * 32
_FP2 = "1220" + "cd" * 32
_FP3 = "1220" + "ef" * 32
_DIGEST = "1220" + "99" * 32


def _round_trip(model_cls, instance):
    canonical = canonicalize(instance.model_dump(mode="json"))
    restored = model_cls.model_validate_json(canonical)
    assert canonicalize(restored.model_dump(mode="json")) == canonical
    return restored


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


def test_endpoint_binding_variants_classified():
    assert_fully_classified(DirectEndpoint)
    assert_fully_classified(LogicalRoute)
    assert_fully_classified(ReplicaPool)
    assert_fully_classified(DistributedExecutionGroup)


def test_inference_solution_classified():
    assert_fully_classified(InferenceSolution)


def test_deployment_plan_classified():
    assert_fully_classified(DeploymentPlan)


def test_evaluation_protocol_classified():
    assert_fully_classified(EvaluationProtocol)


def test_role_binding_classified():
    assert_fully_classified(RoleBinding)


# ---------------------------------------------------------------------------
# round-trip
# ---------------------------------------------------------------------------


def _make_direct_endpoint(**overrides):
    defaults = {
        "binding": "direct_endpoint",
        "model_spec_fingerprint": _FP,
        "artifact_spec_fingerprint": _FP,
        "execution_spec_fingerprint": _FP,
        "capabilities": (_FP,),
        "provider": None,
        "target_kind": "local-container",
        "deployment_plan_fingerprint": _FP,
    }
    return DirectEndpoint(**(defaults | overrides))


def test_direct_endpoint_round_trip():
    _round_trip(DirectEndpoint, _make_direct_endpoint())


def test_logical_route_round_trip():
    lr = LogicalRoute(
        binding="logical_route",
        endpoints=(_FP, _FP2),
        routing_policy="round_robin",
    )
    _round_trip(LogicalRoute, lr)


def test_replica_pool_round_trip():
    rp = ReplicaPool(binding="replica_pool", replica_fingerprints=(_FP, _FP))
    _round_trip(ReplicaPool, rp)


def test_distributed_execution_group_round_trip():
    deg = DistributedExecutionGroup(
        binding="distributed_execution_group",
        worker_fingerprints=(_FP, _FP2),
        worker_roles=("prefill", "decode"),
        interconnect="NVLink",
    )
    _round_trip(DistributedExecutionGroup, deg)


def test_endpoint_binding_discriminated_union():
    adapter = TypeAdapter(EndpointBinding)
    de = _make_direct_endpoint()
    parsed = adapter.validate_json(canonicalize(de.model_dump(mode="json")))
    assert isinstance(parsed, DirectEndpoint)

    lr = LogicalRoute(binding="logical_route", endpoints=(_FP,), routing_policy="primary")
    parsed = adapter.validate_json(canonicalize(lr.model_dump(mode="json")))
    assert isinstance(parsed, LogicalRoute)


def test_deployment_plan_round_trip():
    dp = DeploymentPlan(
        tensor_parallel=2,
        dtype="bf16",
        batch_size=32,
        serve_command="vllm serve Qwen/Qwen3-8B --tp 2",
    )
    _round_trip(DeploymentPlan, dp)


def test_evaluation_protocol_round_trip():
    ep = EvaluationProtocol(
        decision_request_digest=_DIGEST,
        task_suite_fingerprint=_FP,
        application_fingerprint=_FP2,
        solution_fingerprint=_FP3,
        harness="inspect_ai",
        harness_version="0.4.0",
        scorer="exact_match",
        seeds=(42,),
        repetitions=1,
    )
    _round_trip(EvaluationProtocol, ep)


# ---------------------------------------------------------------------------
# InferenceSolution: 1 vs 3 endpoints, same API
# ---------------------------------------------------------------------------


def test_inference_solution_single_endpoint():
    ep = _make_direct_endpoint()
    sol = InferenceSolution(endpoints=(ep,))
    _round_trip(InferenceSolution, sol)


def test_inference_solution_three_endpoints():
    managed = _make_direct_endpoint(
        model_spec_fingerprint=_FP,
        provider="openai",
        target_kind=None,
        provider_opaque=True,
        deployment_plan_fingerprint=None,
    )
    executor = _make_direct_endpoint(model_spec_fingerprint=_FP2)
    vision = _make_direct_endpoint(model_spec_fingerprint=_FP3)

    sol = InferenceSolution(
        endpoints=(managed, executor, vision),
        role_bindings=(
            RoleBinding(role="planner", endpoint_fingerprint=fingerprint_hex(managed)),
            RoleBinding(role="executor", endpoint_fingerprint=fingerprint_hex(executor)),
            RoleBinding(role="vision", endpoint_fingerprint=fingerprint_hex(vision)),
        ),
        routing_policy="role_based",
    )
    restored = _round_trip(InferenceSolution, sol)
    assert len(restored.endpoints) == 3
    assert len(restored.role_bindings) == 3


# ---------------------------------------------------------------------------
# changed member changes fingerprint
# ---------------------------------------------------------------------------


def test_changed_member_changes_fingerprint():
    ep1 = _make_direct_endpoint(model_spec_fingerprint=_FP)
    ep2 = _make_direct_endpoint(model_spec_fingerprint=_FP2)

    sol_a = InferenceSolution(endpoints=(ep1,))
    sol_b = InferenceSolution(endpoints=(ep2,))

    assert fingerprint_hex(sol_a) != fingerprint_hex(sol_b)


def test_display_change_stable_fingerprint():
    dp1 = DeploymentPlan(tensor_parallel=2, serve_command="cmd1")
    dp2 = DeploymentPlan(tensor_parallel=2, serve_command="cmd2")
    assert fingerprint_hex(dp1) == fingerprint_hex(dp2)


# ---------------------------------------------------------------------------
# EvaluationProtocol mixed reference kinds
# ---------------------------------------------------------------------------


def test_evaluation_protocol_mixed_references():
    ep = EvaluationProtocol(
        decision_request_digest=_DIGEST,
        task_suite_fingerprint=_FP,
        application_fingerprint=_FP2,
        solution_fingerprint=_FP3,
        harness="harness",
        harness_version="1.0",
        scorer="exact_match",
    )
    assert ep.decision_request_digest == _DIGEST
    assert ep.task_suite_fingerprint == _FP
    assert ep.application_fingerprint == _FP2
    assert ep.solution_fingerprint == _FP3


# ---------------------------------------------------------------------------
# topology fixtures preserve distinct semantics
# ---------------------------------------------------------------------------


def test_topology_types_have_distinct_fingerprints():
    de = _make_direct_endpoint()
    lr = LogicalRoute(binding="logical_route", endpoints=(_FP,), routing_policy="primary")
    rp = ReplicaPool(binding="replica_pool", replica_fingerprints=(_FP,))
    deg = DistributedExecutionGroup(
        binding="distributed_execution_group",
        worker_fingerprints=(_FP,),
    )

    fps = {fingerprint_hex(de), fingerprint_hex(lr), fingerprint_hex(rp), fingerprint_hex(deg)}
    assert len(fps) == 4


def test_provider_opaque_endpoint():
    ep = _make_direct_endpoint(provider="openai", provider_opaque=True)
    assert ep.provider_opaque is True
    _round_trip(DirectEndpoint, ep)
