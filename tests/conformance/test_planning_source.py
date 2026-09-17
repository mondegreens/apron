"""Conformance suite: PlanningSource Protocol."""

from apron.domain.canonical import canonicalize
from apron.domain.protocols import PlanningSource
from apron.domain.schemas.solutions import PlanningClaim


def test_satisfies_protocol(planning_source):
    assert isinstance(planning_source, PlanningSource)


def test_predict_returns_planning_claim(planning_source):
    claim = planning_source.predict(
        {"arch": "Qwen3ForCausalLM"},
        {"gpu_sku": "H100"},
        {"batch": 32},
    )
    assert isinstance(claim, PlanningClaim)


def test_claim_has_nonempty_proposed_configuration(planning_source):
    claim = planning_source.predict(
        {"arch": "Qwen3ForCausalLM"},
        {"gpu_sku": "H100"},
        {"batch": 32},
    )
    assert claim.proposed_configuration
    assert isinstance(claim.proposed_configuration, dict)


def test_unknown_mechanism_returns_claim(planning_source):
    claim = planning_source.predict(
        {"arch": "UnknownArch"},
        {"gpu_sku": "H100"},
        {"batch": 32},
    )
    assert isinstance(claim, PlanningClaim)


def test_determinism_via_injected_ports(planning_source):
    args = ({"arch": "Qwen3ForCausalLM"}, {"gpu_sku": "H100"}, {"batch": 32})
    claim1 = planning_source.predict(*args)
    claim2 = planning_source.predict(*args)
    bytes1 = canonicalize(claim1.model_dump(mode="json"))
    bytes2 = canonicalize(claim2.model_dump(mode="json"))
    assert bytes1 == bytes2
