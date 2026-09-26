"""A cohort run's stored evidence, typed (exit gate §11, findings §12).

Two parts:

- the **identity manifest** — one entry per solution the run planned,
  holding the validated objects every digest on its records is computed
  from: model spec, deployment plan, requested execution (the solution
  fingerprint, F4) and the accepted request, task suite, application,
  evaluation protocol and serving workload (the INV-25 fingerprints, §8).
  Without it a digest can be compared with another digest, never
  reproduced from its object (§11 item 3);
- the **record store** — every record loaded through its own model on the
  migration path (F2, F10).  A record that fails is listed, never skipped.

No adapters here: the caller passes the store and the manifest entries.
"""

# No ``from __future__ import annotations``: pydantic resolves SolutionEntry's
# field types at runtime, so the schema imports below are runtime imports.
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from apron.domain.canonical import record_digest_hex
from apron.domain.protocols import RecordStore
from apron.domain.schemas.authority import DecisionRequest
from apron.domain.schemas.migrations import load_record
from apron.domain.schemas.models import ModelSpec
from apron.domain.schemas.records import (
    DiagnosisRule,
    RemediationRecord,
    TaskAttemptRecord,
    VerificationReport,
)
from apron.domain.schemas.reports import DecisionReport
from apron.domain.schemas.solutions import (
    DeploymentPlan,
    EvaluationProtocol,
    PlanningClaim,
    RequestedExecutionSpec,
)
from apron.domain.schemas.tasks import ApplicationSpec, ServingWorkloadSpec, TaskSuiteSpec


class SolutionEntry(BaseModel):
    """One identity-manifest line: a planned solution and the objects behind its digests."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    at: str
    solution_fingerprint: str
    label: str
    model_id: str
    status: str
    # Declared by the seed row (data, not measured): size_class, hardware_class,
    # quantized, features.  Fix-proof solutions have none.
    coverage: dict[str, Any] = {}
    # Observed in the resolved config.json (F5): quantization_config.quant_method.
    quantization_method: str | None = None
    model_spec: ModelSpec
    deployment_plan: DeploymentPlan
    requested_execution: RequestedExecutionSpec
    decision_request: DecisionRequest
    task_suite: TaskSuiteSpec
    application: ApplicationSpec
    evaluation_protocol: EvaluationProtocol
    serving_workload: ServingWorkloadSpec

    @property
    def mechanism(self) -> str:
        return self.model_spec.components[0].mechanism if self.model_spec.components else "unknown"


def identity_entry(
    *,
    at: str,
    solution_fingerprint: str,
    label: str,
    model_id: str,
    status: str,
    coverage: Mapping[str, Any],
    model_config: Mapping[str, Any],
    model_spec: ModelSpec,
    plan: DeploymentPlan,
    requested: RequestedExecutionSpec,
    request: DecisionRequest,
    task_suite: TaskSuiteSpec,
    application: ApplicationSpec,
    protocol: EvaluationProtocol,
    serving_workload: ServingWorkloadSpec,
) -> dict[str, Any]:
    """The manifest line for one solution, validated before it is written."""
    quant = model_config.get("quantization_config")
    method = quant.get("quant_method") if isinstance(quant, dict) else None
    entry = SolutionEntry(
        at=at,
        solution_fingerprint=solution_fingerprint,
        label=label,
        model_id=model_id,
        status=status,
        coverage=dict(coverage),
        quantization_method=str(method) if method else None,
        model_spec=model_spec,
        deployment_plan=plan,
        requested_execution=requested,
        decision_request=request,
        task_suite=task_suite,
        application=application,
        evaluation_protocol=protocol,
        serving_workload=serving_workload,
    )
    data = entry.model_dump(mode="json")
    SolutionEntry.model_validate(data)
    return data


@dataclass
class CohortRecords:
    """Every record of a run, by digest, typed; plus what failed to load."""

    claims: dict[str, PlanningClaim] = field(default_factory=dict)
    reports: dict[str, VerificationReport] = field(default_factory=dict)
    attempts: dict[str, TaskAttemptRecord] = field(default_factory=dict)
    remediations: dict[str, RemediationRecord] = field(default_factory=dict)
    decisions: dict[str, DecisionReport] = field(default_factory=dict)
    solutions: dict[str, SolutionEntry] = field(default_factory=dict)
    invalid: list[tuple[str, str]] = field(default_factory=list)  # (where, why)

    def reports_for(
        self, solution_fp: str, scope: str | None = None
    ) -> dict[str, VerificationReport]:
        return {
            d: r
            for d, r in self.reports.items()
            if r.solution_fingerprint == solution_fp and (scope is None or r.claim_scope == scope)
        }

    def attempts_for(self, solution_fp: str) -> dict[str, TaskAttemptRecord]:
        return {d: a for d, a in self.attempts.items() if a.solution_fingerprint == solution_fp}

    def measured(self) -> dict[str, VerificationReport]:
        """Healthy memory reports: solutions that booted and were measured."""
        return {
            d: r
            for d, r in self.reports.items()
            if r.claim_scope == "memory" and r.boot_outcome == "healthy"
        }


def _record_type(raw: Mapping[str, Any]) -> type[BaseModel] | None:
    if "producer" in raw and "proposed_configuration" in raw:
        return PlanningClaim
    if "candidates" in raw and "decision_request_digest" in raw:
        return DecisionReport
    scope = raw.get("claim_scope")
    if scope == "remediation":
        return RemediationRecord
    if scope in ("task_outcome", "outcome_economics"):
        return TaskAttemptRecord
    if scope in ("boot", "memory", "serving_performance"):
        return VerificationReport
    return None


_BUCKETS = {
    PlanningClaim: "claims",
    VerificationReport: "reports",
    TaskAttemptRecord: "attempts",
    RemediationRecord: "remediations",
    DecisionReport: "decisions",
}


def load_cohort_records(
    store: RecordStore, manifest: Iterable[Mapping[str, Any]]
) -> CohortRecords:
    """Load and validate every stored record and manifest entry.

    A record whose content does not hash to its storage key, whose shape
    matches no record type, or which fails strict validation is listed in
    ``invalid``.  Manifest entries that share a solution fingerprint must be
    identical apart from when they were written.
    """
    out = CohortRecords()
    for digest in store.search(""):
        raw = store.retrieve(digest)
        if raw is None:
            out.invalid.append((digest, "listed but not retrievable"))
            continue
        if record_digest_hex(raw) != digest:
            out.invalid.append((digest, "content does not hash to its storage key"))
            continue
        cls = _record_type(raw)
        if cls is None:
            out.invalid.append((digest, "matches no record type"))
            continue
        try:
            record = load_record(cls, dict(raw))
        except ValidationError as exc:
            out.invalid.append((digest, f"{cls.__name__}: {exc}"))
            continue
        getattr(out, _BUCKETS[cls])[digest] = record

    for n, raw in enumerate(manifest):
        try:
            entry = SolutionEntry.model_validate(raw)
        except ValidationError as exc:
            out.invalid.append((f"manifest line {n + 1}", str(exc)))
            continue
        known = out.solutions.get(entry.solution_fingerprint)
        if known is None:
            out.solutions[entry.solution_fingerprint] = entry
        elif _identity(known) != _identity(entry):
            out.invalid.append(
                (f"manifest line {n + 1}", f"conflicts with {entry.solution_fingerprint}")
            )
        elif entry.coverage and not known.coverage:
            out.solutions[entry.solution_fingerprint] = entry
    return out


_RUN_METADATA = ("at", "label", "status", "coverage")


def _identity(entry: SolutionEntry) -> dict[str, Any]:
    """The entry without run metadata: one solution may be planned twice (cohort
    and fix proof) under different labels, with or without a seed row."""
    data = entry.model_dump(mode="json")
    for key in _RUN_METADATA:
        data.pop(key)
    return data


@dataclass
class CohortRun:
    """A run directory as the exit gate and the findings read it."""

    records: CohortRecords
    rules: list[DiagnosisRule]  # every version: current and history
    rule_errors: list[str]
    ledger: list[dict[str, Any]]
    events: list[dict[str, Any]]
