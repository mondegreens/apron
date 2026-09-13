"""Layer 6 — Authority and automation schemas.

Authorization follows deny-overrides-permit: any deny in the envelope
overrides all permits.  Indeterminate never authorizes.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from apron.domain.fingerprints import DISPLAY, IDENTITY, FingerprintHex

# ---------------------------------------------------------------------------
# DecisionRequest (ADR-011 §1)
# ---------------------------------------------------------------------------


class DecisionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    objective: Annotated[str, IDENTITY]
    success_criteria: Annotated[tuple[str, ...], IDENTITY] = ()
    quality_floor: Annotated[float | None, IDENTITY] = None
    serving_slos: Annotated[dict[str, float], IDENTITY] = {}
    privacy_constraints: Annotated[tuple[str, ...], IDENTITY] = ()
    security_constraints: Annotated[tuple[str, ...], IDENTITY] = ()
    license_constraints: Annotated[tuple[str, ...], IDENTITY] = ()
    data_residency_constraints: Annotated[tuple[str, ...], IDENTITY] = ()
    budget_limit: Annotated[float | None, IDENTITY] = None
    time_bound_hours: Annotated[float | None, IDENTITY] = None
    permitted_providers: Annotated[tuple[str, ...], IDENTITY] = ()
    permitted_accounts: Annotated[tuple[str, ...], IDENTITY] = ()
    permitted_resources: Annotated[tuple[str, ...], IDENTITY] = ()
    optimization_objective: Annotated[str | None, IDENTITY] = None
    optimization_objective_description: Annotated[str | None, DISPLAY] = None


# ---------------------------------------------------------------------------
# ActionRequest (ADR-010 §1)
# ---------------------------------------------------------------------------


class ActionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    action_type: Annotated[str, IDENTITY]
    decision_request_digest: Annotated[str, IDENTITY]
    details: Annotated[dict[str, str], IDENTITY] = {}


# ---------------------------------------------------------------------------
# AuthorityContribution (ADR-010 §2)
# ---------------------------------------------------------------------------


class AuthorityContribution(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    source_type: Annotated[str, IDENTITY]
    source_version: Annotated[str, IDENTITY]
    principal: Annotated[str, IDENTITY]
    decision: Annotated[Literal["permit", "deny", "indeterminate"], IDENTITY]
    action: Annotated[str, IDENTITY]
    resource: Annotated[str | None, IDENTITY] = None
    context: Annotated[dict[str, str], IDENTITY] = {}


# ---------------------------------------------------------------------------
# AuthorizationDecision (ADR-010 §3)
# ---------------------------------------------------------------------------


class AuthorizationDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    action_request_digest: Annotated[str, IDENTITY]
    accepted_decision_request_digest: Annotated[str, IDENTITY]
    result: Annotated[Literal["authorized", "denied"], IDENTITY]
    contributions: Annotated[tuple[AuthorityContribution, ...], DISPLAY] = ()
    context_version: Annotated[str | None, DISPLAY] = None


# ---------------------------------------------------------------------------
# AuthorizationEnvelope (ADR-010 §3)
# ---------------------------------------------------------------------------


class AuthorizationEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    permitted_action_classes: Annotated[tuple[str, ...], IDENTITY] = ()
    permitted_providers: Annotated[tuple[str, ...], IDENTITY] = ()
    permitted_accounts: Annotated[tuple[str, ...], IDENTITY] = ()
    credential_scopes: Annotated[tuple[str, ...], IDENTITY] = ()
    task_data_destinations: Annotated[tuple[str, ...], IDENTITY] = ()
    hard_target_constraints: Annotated[dict[str, str], IDENTITY] = {}
    maximum_spend: Annotated[float | None, IDENTITY] = None
    baseline_allocation_digest: Annotated[str | None, IDENTITY] = None
    contributed_pool_digest: Annotated[str | None, IDENTITY] = None
    runtime_rules: Annotated[dict[str, str], IDENTITY] = {}
    lifecycle_rules: Annotated[dict[str, str], IDENTITY] = {}
    teardown_rules: Annotated[dict[str, str], IDENTITY] = {}
    security_rules: Annotated[dict[str, str], IDENTITY] = {}
    data_rules: Annotated[dict[str, str], IDENTITY] = {}
    publication_scope: Annotated[str | None, IDENTITY] = None
    adaptation_rules: Annotated[dict[str, str], IDENTITY] = {}


def evaluate_authorization(
    contributions: tuple[AuthorityContribution, ...],
) -> Literal["authorized", "denied"]:
    """Deny-overrides-permit: any deny overrides all permits.
    Indeterminate never authorizes."""
    if any(c.decision == "deny" for c in contributions):
        return "denied"
    if all(c.decision == "permit" for c in contributions) and contributions:
        return "authorized"
    return "denied"


# ---------------------------------------------------------------------------
# OrchestrationDecision (ADR-010 §7)
# ---------------------------------------------------------------------------


class OrchestrationDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    job_id: Annotated[str, IDENTITY]
    deduplication_key: Annotated[str, IDENTITY]
    policy_version: Annotated[str, IDENTITY]
    state_transition: Annotated[str, IDENTITY]
    inputs_digest: Annotated[str, IDENTITY]
    outputs_digest: Annotated[str | None, DISPLAY] = None


# ---------------------------------------------------------------------------
# ActionAttempt (ADR-010 §1, discriminated union)
# ---------------------------------------------------------------------------


class EvaluationAttempt(BaseModel):
    model_config = ConfigDict(frozen=True)
    attempt_type: Annotated[Literal["evaluation"], IDENTITY]
    protocol_fingerprint: Annotated[FingerprintHex, IDENTITY]
    status: Annotated[str, IDENTITY]


class ExecutionAttempt(BaseModel):
    model_config = ConfigDict(frozen=True)
    attempt_type: Annotated[Literal["execution"], IDENTITY]
    target_fingerprint: Annotated[str, IDENTITY]
    status: Annotated[str, IDENTITY]


class PublicationAttempt(BaseModel):
    model_config = ConfigDict(frozen=True)
    attempt_type: Annotated[Literal["publication"], IDENTITY]
    destination: Annotated[str, IDENTITY]
    status: Annotated[str, IDENTITY]


ActionAttempt = Annotated[
    EvaluationAttempt | ExecutionAttempt | PublicationAttempt,
    Field(discriminator="attempt_type"),
]


# ---------------------------------------------------------------------------
# AnomalyCase (phase-plan, ADR-010 §8)
# ---------------------------------------------------------------------------


class AnomalyCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    case_type: Annotated[str, IDENTITY]
    fingerprint: Annotated[str | None, IDENTITY] = None
    prediction_delta: Annotated[float | None, IDENTITY] = None
    details: Annotated[dict[str, str], DISPLAY] = {}


# ---------------------------------------------------------------------------
# ProposedExternalAction (ADR-010 §8)
# ---------------------------------------------------------------------------


class ProposedExternalAction(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    rendered_record: Annotated[dict[str, str], IDENTITY]
    upstream_shape: Annotated[str, IDENTITY]
    destination_policy: Annotated[str, IDENTITY]
    provenance_digest: Annotated[str, IDENTITY]
    deduplication_key: Annotated[str, IDENTITY]
    confidence: Annotated[float | None, DISPLAY] = None
    rate_controls: Annotated[dict[str, str], IDENTITY] = {}
