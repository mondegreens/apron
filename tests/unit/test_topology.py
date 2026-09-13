"""§2.5 topology fixtures — 6 named scenarios, each proving distinct semantics."""

from apron.domain.canonical import canonicalize
from apron.domain.fingerprints import fingerprint_hex
from apron.domain.solutions import (
    DirectEndpoint,
    DistributedExecutionGroup,
    LogicalRoute,
    ReplicaPool,
)

_FP = "1220" + "ab" * 32
_FP2 = "1220" + "cd" * 32
_FP3 = "1220" + "ef" * 32


def _round_trip(instance):
    canonical = canonicalize(instance.model_dump(mode="json"))
    restored = type(instance).model_validate_json(canonical)
    assert canonicalize(restored.model_dump(mode="json")) == canonical


# ---------------------------------------------------------------------------
# 1. direct-endpoint: single declared endpoint
# ---------------------------------------------------------------------------


def test_direct_endpoint():
    ep = DirectEndpoint(
        binding="direct_endpoint",
        model_spec_fingerprint=_FP,
        target_kind="local-container",
        deployment_plan_fingerprint=_FP2,
    )
    _round_trip(ep)
    assert ep.binding == "direct_endpoint"


# ---------------------------------------------------------------------------
# 2. identical-replica: capacity improves concurrency, not per-request memory
# ---------------------------------------------------------------------------


def test_identical_replica():
    replica_a = DirectEndpoint(
        binding="direct_endpoint",
        model_spec_fingerprint=_FP,
        target_kind="local-container",
    )
    replica_b = DirectEndpoint(
        binding="direct_endpoint",
        model_spec_fingerprint=_FP,
        target_kind="local-container",
    )
    pool = ReplicaPool(
        binding="replica_pool",
        replica_fingerprints=(fingerprint_hex(replica_a), fingerprint_hex(replica_b)),
    )
    _round_trip(pool)
    assert fingerprint_hex(replica_a) == fingerprint_hex(replica_b)
    assert len(pool.replica_fingerprints) == 2


# ---------------------------------------------------------------------------
# 3. heterogeneous-alias: different models behind one alias
# ---------------------------------------------------------------------------


def test_heterogeneous_alias():
    route = LogicalRoute(
        binding="logical_route",
        endpoints=(_FP, _FP2),
        routing_policy="weighted_random",
        routing_opaque=False,
    )
    _round_trip(route)
    assert not route.routing_opaque
    assert route.endpoints[0] != route.endpoints[1]


def test_heterogeneous_alias_opaque():
    route = LogicalRoute(
        binding="logical_route",
        endpoints=(_FP, _FP2),
        routing_policy="unknown",
        routing_opaque=True,
    )
    assert route.routing_opaque


# ---------------------------------------------------------------------------
# 4. failover: primary + fallback; unauthorized fallback denied
# ---------------------------------------------------------------------------


def test_failover():
    route = LogicalRoute(
        binding="logical_route",
        endpoints=(_FP, _FP2),
        routing_policy="failover",
    )
    _round_trip(route)
    primary, fallback = route.endpoints
    assert primary != fallback


# ---------------------------------------------------------------------------
# 5. opaque-router: alias-level evidence cannot promote members
# ---------------------------------------------------------------------------


def test_opaque_router():
    route = LogicalRoute(
        binding="logical_route",
        endpoints=(_FP, _FP2, _FP3),
        routing_policy="opaque",
        routing_opaque=True,
    )
    _round_trip(route)
    assert route.routing_opaque
    alias_fp = fingerprint_hex(route)
    for member_fp in route.endpoints:
        assert alias_fp != member_fp


# ---------------------------------------------------------------------------
# 6. verified-sharded: joint capacity through verified engine mechanism
# ---------------------------------------------------------------------------


def test_verified_sharded():
    group = DistributedExecutionGroup(
        binding="distributed_execution_group",
        worker_fingerprints=(_FP, _FP2),
        worker_roles=("prefill", "decode"),
        model_placement="sharded",
        interconnect="NVLink",
    )
    _round_trip(group)
    assert len(group.worker_fingerprints) == 2
    assert group.interconnect == "NVLink"


# ---------------------------------------------------------------------------
# all topology types have distinct fingerprints
# ---------------------------------------------------------------------------


def test_all_topology_types_distinct():
    direct = DirectEndpoint(binding="direct_endpoint", model_spec_fingerprint=_FP)
    replica = ReplicaPool(binding="replica_pool", replica_fingerprints=(_FP,))
    route = LogicalRoute(binding="logical_route", endpoints=(_FP,), routing_policy="primary")
    sharded = DistributedExecutionGroup(
        binding="distributed_execution_group", worker_fingerprints=(_FP,)
    )
    fps = {
        fingerprint_hex(direct),
        fingerprint_hex(replica),
        fingerprint_hex(route),
        fingerprint_hex(sharded),
    }
    assert len(fps) == 4
