"""Layer 7 — Decision report and resource schemas.

DecisionReport preserves every considered solution with per-candidate
qualification status, economics, and evidence state.  Promotion gates
(INV-18) are enforced by ``can_promote``.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict

from apron.domain.fingerprints import DISPLAY, IDENTITY, FingerprintHex

QualificationStatus = Literal[
    "candidate",
    "capability_eligible",
    "identity_resolved",
    "task_evaluated",
    "serving_verified",
    "task_reproduced",
    "qualified",
    "measured_efficient",
]

_PROMOTION_ORDER = (
    "candidate",
    "capability_eligible",
    "identity_resolved",
    "task_evaluated",
    "serving_verified",
    "task_reproduced",
    "qualified",
    "measured_efficient",
)


# ---------------------------------------------------------------------------
# CandidateEntry — per-candidate data within a DecisionReport
# ---------------------------------------------------------------------------


class CandidateEconomics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    market_equivalent_price: Annotated[float | None, DISPLAY] = None
    gross_attributable_cost: Annotated[float | None, DISPLAY] = None
    subsidy_applied: Annotated[float | None, DISPLAY] = None
    project_out_of_pocket_cost: Annotated[float | None, DISPLAY] = None


class CandidateEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    solution_fingerprint: Annotated[FingerprintHex, IDENTITY]
    qualification_status: Annotated[QualificationStatus, IDENTITY]
    qualification_graph_state: Annotated[str | None, DISPLAY] = None
    rejection_reason: Annotated[str | None, DISPLAY] = None
    evidence_state: Annotated[str | None, DISPLAY] = None
    evaluation_coverage: Annotated[str | None, DISPLAY] = None
    economics: Annotated[CandidateEconomics | None, DISPLAY] = None
    trade_off: Annotated[str | None, DISPLAY] = None


# ---------------------------------------------------------------------------
# DecisionReport (ADR-011 §9)
# ---------------------------------------------------------------------------


class DecisionReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Annotated[int, DISPLAY] = 1
    decision_request_digest: Annotated[str, IDENTITY]
    candidates: Annotated[tuple[CandidateEntry, ...], IDENTITY]
    outcome_economics: Annotated[dict[str, float] | None, DISPLAY] = None
    uncertainty: Annotated[dict[str, float] | None, DISPLAY] = None
    disclosed_comparable_set: Annotated[tuple[FingerprintHex, ...], IDENTITY] = ()
    exclusions: Annotated[tuple[str, ...], DISPLAY] = ()


# ---------------------------------------------------------------------------
# MaintainerBaselineAllocation (ADR-010 §14)
# ---------------------------------------------------------------------------


class MaintainerBaselineAllocation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Annotated[int, DISPLAY] = 1
    budget: Annotated[float, IDENTITY]
    currency: Annotated[str, IDENTITY] = "USD"
    scope: Annotated[str, IDENTITY] = "phase_1a"
    constraints: Annotated[dict[str, str], IDENTITY] = {}


# ---------------------------------------------------------------------------
# ContributedResourcePool (ADR-010 §13)
# ---------------------------------------------------------------------------


class ContributedResourcePool(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Annotated[int, DISPLAY] = 1
    provenance: Annotated[str, IDENTITY]
    amount: Annotated[float, IDENTITY]
    currency: Annotated[str, IDENTITY] = "USD"
    scope: Annotated[str, IDENTITY] = "phase_1a"


# ---------------------------------------------------------------------------
# Promotion gates (INV-18)
# ---------------------------------------------------------------------------


def can_promote(
    current: QualificationStatus,
    target: QualificationStatus,
    *,
    has_task_evidence: bool = False,
    has_serving_evidence: bool = False,
    has_exact_reproduction: bool = False,
    has_comparable_set: bool = False,
) -> bool:
    """Check whether promotion from *current* to *target* is allowed (INV-18)."""
    cur_idx = _PROMOTION_ORDER.index(current)
    tgt_idx = _PROMOTION_ORDER.index(target)
    if tgt_idx <= cur_idx:
        return False

    if (
        target in ("task_evaluated", "serving_verified", "task_reproduced", "qualified")
        and not has_task_evidence
    ):
        return False
    if target in ("serving_verified", "task_reproduced", "qualified") and not has_serving_evidence:
        return False
    if target in ("task_reproduced", "qualified") and not has_exact_reproduction:
        return False
    return not (target == "measured_efficient" and not has_comparable_set)
