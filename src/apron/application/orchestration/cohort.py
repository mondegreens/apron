"""Evidence cohort — the execution protocol per candidate (PLAN §9.1).

The same code runs every candidate, sequentially:

0. harness hygiene before every boot; environment errors (port in use, disk
   full, HF 401/403/404, network) are harness failures — stored as failed
   attempts, retried once, never sent to diagnosis;
1. plan (GPU-free; done by the caller, see ``SolutionPlan``) — prediction-
   error candidates stop there with a stored PlanningClaim;
2. provision — a budget hold is appended to the ledger first;
3. boot the rendered plan and verify memory;
4. task evaluation — one TaskAttemptRecord per case, failures and retries
   included;
5. serving measurement against the declared SLO (measurement only);
6. prediction delta (M5/M7);
7. settle cost and attach it in each schema's own fields;
8. teardown;
9. append machine events.

Layer rule (R-4): nothing here imports ``apron.adapters``.  Targets, the
engine, the evaluator, the store and the ledger arrive through ``CohortPorts``
built by a composition root.  Time and identifiers come from injected
``Clock``/``IdGenerator`` ports (INV-42).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

from apron.application.orchestration.budget import (
    PhaseTiming,
    attribute_costs,
    verification_cost_fields,
)
from apron.application.orchestration.errors import (
    HarnessError,
    PodLeakError,
    RunnerImageError,
    RunStopError,
    TokenLeakError,
)
from apron.application.orchestration.evidence import (
    EvidenceContext,
    build_task_attempt,
    chat_template_kwargs,
    load_typed,
    scorer_input,
    store_validated,
)
from apron.application.orchestration.serving import evaluate_serving_slos
from apron.application.sanitization import mask_secrets
from apron.domain.fingerprints import fingerprint_hex
from apron.domain.schemas.records import TaskAttemptRecord, VerificationReport
from apron.domain.schemas.solutions import PlanningClaim

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from apron.application.orchestration.budget import BudgetTracker
    from apron.domain.ports import Clock, IdGenerator
    from apron.domain.protocols import RecordStore
    from apron.domain.schemas.authority import AuthorizationEnvelope, DecisionRequest
    from apron.domain.schemas.models import ArtifactSpec, ModelSpec
    from apron.domain.schemas.reports import DecisionReport
    from apron.domain.schemas.solutions import DeploymentPlan, RequestedExecutionSpec
    from apron.domain.schemas.tasks import ApplicationSpec, ServingWorkloadSpec, TaskSuiteSpec

# ---------------------------------------------------------------------------
# Step 0: harness errors are not model failures
# ---------------------------------------------------------------------------

_HARNESS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("harness:port_in_use", re.compile(r"Address already in use|Errno 98", re.IGNORECASE)),
    ("harness:disk_full", re.compile(r"No space left on device|Errno 28", re.IGNORECASE)),
    (
        "harness:hf_auth",
        re.compile(
            r"401 Client Error|403 Client Error|GatedRepoError|Access to model .* is restricted",
            re.IGNORECASE,
        ),
    ),
    (
        "harness:hf_not_found",
        re.compile(r"404 Client Error|RepositoryNotFoundError|EntryNotFoundError", re.IGNORECASE),
    ),
    (
        "harness:network",
        re.compile(
            r"Temporary failure in name resolution|Max retries exceeded|ConnectionError|"
            r"Connection reset by peer|ReadTimeout|LocalEntryNotFoundError",
            re.IGNORECASE,
        ),
    ),
)


def classify_harness_error(log: str) -> str | None:
    """``harness:<kind>`` for an environment failure, ``None`` for a model failure."""
    for kind, pattern in _HARNESS_PATTERNS:
        if pattern.search(log):
            return kind
    return None


TOKEN_LEAK = "harness:token_in_vllm_environ"
RUNNER_IMAGE = "harness:runner_image_lacks_f7"
POD_LEAK = "harness:pod_not_terminated"

__all__ = [
    "HarnessError",
    "PodLeakError",
    "RunStopError",
    "RunnerImageError",
    "TokenLeakError",
]


def _failure_tag(exc: BaseException) -> str:
    if isinstance(exc, TokenLeakError):
        return TOKEN_LEAK
    if isinstance(exc, RunnerImageError):
        return RUNNER_IMAGE
    if isinstance(exc, PodLeakError):
        return POD_LEAK
    return f"harness:exception:{type(exc).__name__}"


# ---------------------------------------------------------------------------
# Ports (built by a composition root)
# ---------------------------------------------------------------------------


class EventLog(Protocol):
    def append(self, entry: dict[str, Any]) -> None: ...


class ExecutionEngine(Protocol):
    """The engine operations the protocol needs (the vLLM adapter implements them)."""

    def runner_supports_token_isolation(self, target: Any) -> bool: ...

    def prepare_boot(self, target: Any) -> dict[str, Any]: ...

    def download_weights(self, target: Any, model_id: str) -> dict[str, Any]: ...

    def boot(self, plan: DeploymentPlan, target: Any, *, health_timeout: int = 600) -> Any: ...

    def verify(
        self, plan: DeploymentPlan, target: Any, health_timeout: int = 600
    ) -> dict[str, Any]: ...

    def token_environ_check(self, target: Any) -> dict[str, Any]: ...

    def benchmark_serving(
        self,
        target: Any,
        *,
        model_id: str,
        input_len: int,
        output_len: int,
        concurrency: int,
    ) -> dict[str, Any]: ...

    def build_serving_report(self, bench: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class CohortPorts:
    target_factory: Callable[[RequestedExecutionSpec], Any]
    engine: ExecutionEngine
    evaluator: Any
    store: RecordStore
    budget: BudgetTracker
    clock: Clock
    ids: IdGenerator
    events: EventLog
    provision_env: Callable[[SolutionPlan], dict[str, str]]
    hourly_rate: Callable[[RequestedExecutionSpec], float]
    boot_timeout: int = 900


@dataclass(frozen=True)
class AcceptedInputs:
    """The accepted request every candidate is judged against."""

    request: DecisionRequest
    task_suite: TaskSuiteSpec
    application: ApplicationSpec
    protocol_template: Mapping[str, Any]
    serving_workload: ServingWorkloadSpec
    authorization: AuthorizationEnvelope

    def request_for(self, model_id: str) -> DecisionRequest:
        """The accepted request with the candidate model in scope (§8)."""
        from apron.domain.schemas.authority import DecisionRequest

        data = self.request.model_dump(mode="json")
        data["permitted_resources"] = [model_id]
        return DecisionRequest.model_validate(data)


# ---------------------------------------------------------------------------
# Step 1 result: a planned solution (GPU-free)
# ---------------------------------------------------------------------------

PlanStatus = Literal["planned", "unknown", "infeasible"]


@dataclass(frozen=True)
class SolutionPlan:
    label: str
    model_id: str
    model_spec: ModelSpec
    model_config: Mapping[str, Any]
    plan: DeploymentPlan
    requested: RequestedExecutionSpec
    claim: PlanningClaim
    solution_fp: str
    status: PlanStatus
    chat_template: str | None = None
    license_observed: str | None = None
    gating_observed: str | None = None
    estimate: float = 0.0
    notes: tuple[str, ...] = ()
    artifact_spec: ArtifactSpec | None = None

    @property
    def deployment_feasible(self) -> bool:
        """Whether the candidate can execute at all (qualification's first gate)."""
        return self.status == "planned"

    @property
    def predicted_total_bytes(self) -> int | None:
        value = self.claim.proposed_configuration.get("total_required_bytes")
        return int(value) if value else None


def predicted_feasible(claim: PlanningClaim, total_memory_bytes: int, gpu_count: int) -> bool:
    """The calculator's predicted total fits the requested GPUs at 0.90 utilization."""
    total = claim.proposed_configuration.get("total_required_bytes")
    if not total:
        return True
    return int(total) <= int(total_memory_bytes * 0.90) * gpu_count


def prediction_error_claim(
    claim: PlanningClaim, status: PlanStatus, solution_fp: str
) -> PlanningClaim:
    """The stored first-class record of a candidate that never executes (INV-32)."""
    config = dict(claim.proposed_configuration)
    config["status"] = status
    if status == "infeasible":
        config["reason"] = "predicted total exceeds the requested GPU memory"
    return PlanningClaim.model_validate(
        {
            **claim.model_dump(mode="json"),
            "proposed_configuration": config,
            "solution_fingerprint": solution_fp,
        }
    )


# ---------------------------------------------------------------------------
# Authorization (exit gate 7; D3/D4)
# ---------------------------------------------------------------------------


def authorize_execution(sp: SolutionPlan, envelope: AuthorizationEnvelope) -> None:
    """Refuse provider, cloud type or data destinations outside the envelope."""
    requested = sp.requested
    if envelope.permitted_providers and requested.provider not in envelope.permitted_providers:
        raise PermissionError(f"provider {requested.provider} is not permitted")
    required_cloud = envelope.hard_target_constraints.get("cloud_type")
    if required_cloud and requested.cloud_type != required_cloud:
        raise PermissionError(f"cloud type {requested.cloud_type} is not {required_cloud}")
    if "gpu_execution" not in envelope.permitted_action_classes:
        raise PermissionError("gpu_execution is not a permitted action class")


# ---------------------------------------------------------------------------
# Steps 2-9: execute one solution
# ---------------------------------------------------------------------------


@dataclass
class ExecutionOutcome:
    solution_fp: str
    healthy: bool
    boot_report_digest: str | None = None
    failed_boot_digests: list[str] = field(default_factory=list)
    attempt_digests: list[str] = field(default_factory=list)
    serving_digest: str | None = None
    log_tail: str = ""
    harness_failures: list[str] = field(default_factory=list)
    cost: float = 0.0
    provider_reported_cost: float | None = None
    token_check: dict[str, Any] | None = None


@dataclass(frozen=True)
class RunScope:
    """Which steps to run; resume passes only the missing ones."""

    task_evaluation: bool = True
    serving: bool = True


def execute_solution(
    sp: SolutionPlan,
    inputs: AcceptedInputs,
    ports: CohortPorts,
    *,
    scope: RunScope | None = None,
    reason: str = "cohort_measurement",
) -> ExecutionOutcome:
    """Provision, boot ``sp.plan``, measure, tear down, then store every record.

    Records are stored after teardown so each carries its cost (§7).  Whatever
    fails after the budget hold — provisioning, SSH, the engine, the scorer,
    the benchmark — the pod is torn down, the hold is settled, measurements
    already taken are stored, and the failure itself is stored as a failed
    VerificationReport carrying its cost share; then the error is re-raised.
    Only a process kill escapes this; ledger replay covers that case.
    """
    scope = scope or RunScope()
    authorize_execution(sp, inputs.authorization)
    engine, clock = ports.engine, ports.clock
    outcome = ExecutionOutcome(solution_fp=sp.solution_fp, healthy=False)
    timing = PhaseTiming()
    model_id = sp.plan.resource_allocation.get("model_id", sp.model_id)
    ctx = EvidenceContext.bind(
        request=inputs.request_for(model_id),
        task_suite=inputs.task_suite,
        application=inputs.application,
        protocol_template=inputs.protocol_template,
        solution_fp=sp.solution_fp,
    )
    rate = ports.hourly_rate(sp.requested)

    ports.budget.hold(sp.estimate, sp.label)
    _event(ports, "hold", sp, estimate=sp.estimate, rate=rate)
    target = ports.target_factory(sp.requested)
    report_fields: dict[str, Any] | None = None
    failed_boots: list[tuple[str, str]] = []  # (log tail, failure tag)
    scored: list[tuple[dict[str, Any], int]] = []
    bench: dict[str, Any] | None = None
    error: Exception | None = None
    leaked: PodLeakError | None = None
    try:
        timing.provision_start = _seconds(clock)
        target.provision(env=ports.provision_env(sp))
        ports.budget.annotate_hold(sp.label, target.pod_id)
        _event(ports, "provisioned", sp, pod_id=target.pod_id)
        if not engine.runner_supports_token_isolation(target):
            raise RunnerImageError("runner image lacks the F7 download step (apron-download)")

        boot = None
        for attempt in range(2):  # a harness failure is retried once
            hygiene = engine.prepare_boot(target)
            if not hygiene.get("clean"):
                failed_boots.append((str(hygiene), "harness:unclean_gpu"))
                continue
            download = engine.download_weights(target, model_id)
            if not download.get("ok"):
                tag = classify_harness_error(download["output_tail"]) or "harness:download"
                failed_boots.append((download["output_tail"], tag))
                continue
            boot = engine.boot(sp.plan, target, health_timeout=ports.boot_timeout)
            if boot.healthy:
                break
            tag = classify_harness_error(boot.log_tail)
            if tag is None:
                break  # a model failure: diagnosis material, not retried
            failed_boots.append((boot.log_tail, tag))
            _event(ports, "harness_retry", sp, failure=tag, attempt=attempt)
            boot = None

        if boot is not None and boot.healthy:
            outcome.healthy = True
            outcome.token_check = engine.token_environ_check(target)
            if not outcome.token_check.get("token_free"):
                # INV-13: a token in the vLLM process, or no process to check,
                # invalidates the run.  Nothing is measured; the run stops.
                outcome.healthy = False
                raise TokenLeakError(f"{sp.label}: {outcome.token_check}")
            report_fields = engine.verify(sp.plan, target, health_timeout=120)
            endpoint = target.proxy_url
            timing.task_eval_start = _seconds(clock)
            if scope.task_evaluation:
                scored = _run_task_suite(ctx, sp, ports, model_id, endpoint)
            timing.task_eval_end = _seconds(clock)
            if scope.serving and _declares_workload(inputs.serving_workload):
                timing.serving_start = _seconds(clock)
                workload = inputs.serving_workload
                bench = engine.benchmark_serving(
                    target,
                    model_id=model_id,
                    input_len=int(workload.input_sequence_length or 0),
                    output_len=int(workload.output_sequence_length or 0),
                    concurrency=int(workload.concurrency or 1),
                )
                timing.serving_end = _seconds(clock)
        elif boot is not None:
            outcome.log_tail = mask_secrets(boot.log_tail)
    except Exception as exc:  # every failure is settled and recorded
        error = exc
        failed_boots.append((f"{type(exc).__name__}: {exc}", _failure_tag(exc)))
        if isinstance(exc, TokenLeakError):
            report_fields, scored, bench = None, [], None  # nothing from this pod is evidence
    finally:
        try:
            outcome.provider_reported_cost = target.pod_reported_cost()
        except Exception:  # the cost API failing must not skip teardown
            outcome.provider_reported_cost = None
        try:
            target.teardown()
        except PodLeakError as exc:
            # The pod may still be billing: record it, keep an open-ended hold
            # for it (replay settles it at RunPod's reported cost), stop the run.
            failed_boots.append((str(exc), POD_LEAK))
            leaked = exc
        except Exception as exc:
            failed_boots.append((f"teardown: {exc}", "harness:teardown_failed"))
            error = error or exc
        timing.teardown_end = _seconds(clock)
        if timing.provision_start is None:
            timing.provision_start = timing.teardown_end
        _close_open_phases(timing)

    costs = attribute_costs(timing, rate, len(scored))
    ports.budget.settle(costs.total, sp.label)
    if leaked is not None:
        ports.budget.hold_open_ended(rate, f"leak:{leaked.pod_id}", leaked.pod_id)
    outcome.cost = costs.total
    outcome.harness_failures = [tag for _, tag in failed_boots]

    base = _report_base(sp, target, timing, rate)
    # Every boot report from this pod — failed harness attempts, retries and
    # the final boot — carries an equal share of the boot-phase cost (§7, exit
    # gate 4), so the parts still sum to the pod's total.
    final_report = (outcome.healthy and report_fields is not None) or bool(outcome.log_tail)
    boot_reports = len(failed_boots) + (1 if final_report else 0)
    # Paid time with no record of its own (a serving run that errored) is
    # carried by the boot reports, so every paid second lands on a record.
    boot_pool = costs.boot_report + (costs.serving_report if bench is None else 0.0)
    share = round(boot_pool / boot_reports, 6) if boot_reports else 0.0
    boot_cost = verification_cost_fields(share)
    for log, tag in failed_boots:
        report = VerificationReport.model_validate(
            {
                **base,
                **boot_cost,
                "claim_scope": "boot",
                "boot_outcome": "failed",
                "failures": (tag,),
                "log_tail": mask_secrets(log)[-8000:],
                "reason": "harness_failure",
            }
        )
        outcome.failed_boot_digests.append(store_validated(ports.store, report))

    if outcome.healthy and report_fields is not None:
        outcome.boot_report_digest = store_validated(
            ports.store,
            VerificationReport.model_validate(
                {
                    **base,
                    **_memory_fields(report_fields),
                    **boot_cost,
                    "provider_reported_cost": outcome.provider_reported_cost,
                    "execution_fingerprint": target.execution_fingerprint,
                    "detected_hardware_fingerprint": fingerprint_hex(target.hardware),
                    "claim_scope": "memory",
                    "boot_outcome": "healthy",
                    "predicted_minus_measured": prediction_delta(sp.claim, report_fields),
                    "prediction_notes": sp.notes,
                    "reason": reason,
                }
            ),
        )
    elif outcome.log_tail:  # a model failure (harness failures are stored above)
        outcome.failed_boot_digests.append(
            store_validated(
                ports.store,
                VerificationReport.model_validate(
                    {
                        **base,
                        **boot_cost,
                        "provider_reported_cost": outcome.provider_reported_cost,
                        "claim_scope": "boot",
                        "boot_outcome": "failed",
                        "failures": ("boot:model_failure",),
                        "log_tail": outcome.log_tail[-8000:],
                        "reason": reason,
                    }
                ),
            )
        )

    trace = tuple(d for d in (outcome.boot_report_digest,) if d)
    for attempt, retry in scored:
        record = build_task_attempt(
            ctx,
            attempt,
            attempt_id=ports.ids.generate(),
            retry=retry,
            infrastructure_cost=costs.per_attempt,
            trace_references=trace,
        )
        outcome.attempt_digests.append(store_validated(ports.store, record))

    if bench is not None:
        serving_fields = engine.build_serving_report(bench)
        draft = VerificationReport.model_validate(
            {
                **base,
                **serving_fields,
                **verification_cost_fields(costs.serving_report),
                "execution_fingerprint": target.execution_fingerprint,
                "claim_scope": "serving_performance",
                "serving_input_sequence_length": inputs.serving_workload.input_sequence_length,
                "serving_output_sequence_length": inputs.serving_workload.output_sequence_length,
                "reason": "serving_measurement",
            }
        )
        verdict = evaluate_serving_slos(draft, inputs.serving_workload)
        report = VerificationReport.model_validate(
            {
                **draft.model_dump(mode="json"),
                "serving_slo_verdict": verdict.verdict,
                "serving_slo_reasons": verdict.reasons,
            }
        )
        outcome.serving_digest = store_validated(ports.store, report)

    _event(
        ports,
        "executed",
        sp,
        healthy=outcome.healthy,
        cost=outcome.cost,
        provider_reported_cost=outcome.provider_reported_cost,
        harness_failures=outcome.harness_failures,
        attempts=len(outcome.attempt_digests),
        serving=outcome.serving_digest is not None,
        token_check=outcome.token_check,
    )
    if leaked is not None:
        _event(ports, "pod_leak", sp, pod_id=leaked.pod_id, error=str(leaked))
        raise leaked
    if error is not None:
        _event(ports, "candidate_error", sp, error=f"{type(error).__name__}: {error}")
        raise error
    return outcome


def _close_open_phases(timing: PhaseTiming) -> None:
    """A phase interrupted by an error ends at teardown (its time was still paid)."""
    if timing.task_eval_start is not None and timing.task_eval_end is None:
        timing.task_eval_end = timing.teardown_end
    if timing.serving_start is not None and timing.serving_end is None:
        timing.serving_end = timing.teardown_end


def _run_task_suite(
    ctx: EvidenceContext,
    sp: SolutionPlan,
    ports: CohortPorts,
    model_id: str,
    endpoint: str,
) -> list[tuple[dict[str, Any], int]]:
    """Score every case; a case whose request failed in transport is retried once.

    Returns (scored case, retry index) — both attempts of a retried case.
    """
    evaluator = ports.evaluator
    prepared = evaluator.prepare(
        {
            **scorer_input(
                ctx,
                model_id=model_id,
                chat_template_kwargs=chat_template_kwargs(sp.chat_template),
            ),
            "endpoint": endpoint,
        }
    )
    results: list[tuple[dict[str, Any], int]] = []
    first = evaluator.execute(prepared, endpoint)
    results.extend((a, 0) for a in first)
    failed_ids = {a.get("case_id") for a in first if a.get("status") == "failed"}
    if failed_ids:
        retry_cases = [c for c in prepared["cases"] if c.get("id") in failed_ids]
        second = evaluator.execute({**prepared, "cases": retry_cases}, endpoint)
        results.extend((a, 1) for a in second)
    return results


def _declares_workload(workload: ServingWorkloadSpec) -> bool:
    return bool(workload.input_sequence_length and workload.output_sequence_length)


def _report_base(
    sp: SolutionPlan, target: Any, timing: PhaseTiming, rate: float
) -> dict[str, Any]:
    # A pod that never answered has no observed execution; say so, never
    # borrow another identity.
    execution_fp = _safe_attr(target, "execution_fingerprint") or "not_provisioned"
    return {
        "target_kind": target.kind,
        "operator": target.operator,
        "provider": target.provider,
        "execution_fingerprint": execution_fp,
        "solution_fingerprint": sp.solution_fp,
        "deployment_plan_digest": fingerprint_hex(sp.plan),
        "hourly_rate": rate,
        "phase_seconds": timing.seconds(),
        "estimated_cost": sp.estimate,
        "production_mode": False,
        "lifecycle": "observed",
    }


_MEMORY_KEYS = (
    "initial_total_memory",
    "initial_free_memory",
    "requested_memory",
    "model_weight_memory",
    "persistent_consumption",
    "transient_peak_headroom",
    "non_pytorch_increase",
    "cuda_graph_estimate",
    "cuda_graph_applied",
    "cuda_graph_actual",
    "available_kv_cache_memory",
    "safety_buffer",
    "profiling_shape",
)


def _memory_fields(report: Mapping[str, Any]) -> dict[str, Any]:
    fields = {k: report.get(k) for k in _MEMORY_KEYS}
    applied = fields.get("cuda_graph_applied")
    if isinstance(applied, bool):
        fields["cuda_graph_applied"] = int(applied)
    return fields


def prediction_delta(claim: PlanningClaim, measured: Mapping[str, Any]) -> dict[str, int]:
    """Predicted minus measured bytes (M5 sign convention), M7 field map.

    weight -> weight, total -> persistent_consumption.  The KV cache figure is
    a consistency check (``kv_cache_consistency``), not a delta, because the
    engine sizes the KV cache from what is left.
    """
    predicted = claim.proposed_configuration
    delta: dict[str, int] = {}
    pairs = (
        ("weight_memory", "weight_memory_bytes", "model_weight_memory"),
        ("total", "total_required_bytes", "persistent_consumption"),
        ("kv_cache_consistency", "available_kv_cache_bytes", "available_kv_cache_memory"),
    )
    for name, p_key, m_key in pairs:
        p, m = predicted.get(p_key), measured.get(m_key)
        if isinstance(p, int | float) and isinstance(m, int | float):
            delta[name] = int(p) - int(m)
    return delta


def _seconds(clock: Clock) -> float:
    return clock.now().timestamp()


def _safe_attr(obj: Any, name: str) -> Any:
    try:
        return getattr(obj, name)
    except RuntimeError:
        return None


def _event(ports: CohortPorts, kind: str, sp: SolutionPlan, **details: Any) -> None:
    entry = {
        "at": ports.clock.now().isoformat(),
        "event": kind,
        "label": sp.label,
        "model_id": sp.model_id,
        "gpu_sku": sp.requested.gpu_sku,
        "gpu_count": sp.requested.gpu_count,
        "solution_fingerprint": sp.solution_fp,
    }
    entry.update({k: v for k, v in details.items() if v is not None})
    ports.events.append(_mask_tree(entry))


def _mask_tree(value: Any) -> Any:
    if isinstance(value, str):
        return mask_secrets(value)
    if isinstance(value, dict):
        return {k: _mask_tree(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_mask_tree(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Resume: what a solution already has on record
# ---------------------------------------------------------------------------


@dataclass
class RecordedEvidence:
    claims: list[str] = field(default_factory=list)
    memory_reports: list[str] = field(default_factory=list)
    failed_boots: list[str] = field(default_factory=list)
    attempts: list[str] = field(default_factory=list)
    serving_reports: list[str] = field(default_factory=list)

    def missing(self, sp: SolutionPlan) -> RunScope | None:
        """Steps still to run, or ``None`` when the solution is complete."""
        if sp.status != "planned":
            return None
        if self.memory_reports and self.attempts and self.serving_reports:
            return None
        if not self.memory_reports:
            return RunScope()
        return RunScope(task_evaluation=not self.attempts, serving=not self.serving_reports)


def recorded_evidence(store: RecordStore, solution_fp: str) -> RecordedEvidence:
    """Index stored records by solution fingerprint (single-operator scale)."""
    found = RecordedEvidence()
    for digest in store.search(""):
        raw = store.retrieve(digest)
        if raw is None or raw.get("solution_fingerprint") != solution_fp:
            continue
        if "producer" in raw and "proposed_configuration" in raw:
            found.claims.append(digest)
        elif raw.get("claim_scope") == "memory" and raw.get("boot_outcome") == "healthy":
            found.memory_reports.append(digest)
        elif raw.get("claim_scope") == "boot":
            found.failed_boots.append(digest)
        elif raw.get("claim_scope") == "task_outcome":
            found.attempts.append(digest)
        elif raw.get("claim_scope") == "serving_performance":
            found.serving_reports.append(digest)
    return found


# ---------------------------------------------------------------------------
# The cohort run
# ---------------------------------------------------------------------------


@dataclass
class CohortResult:
    executed: dict[str, ExecutionOutcome] = field(default_factory=dict)
    prediction_errors: dict[str, str] = field(default_factory=dict)  # label -> claim digest
    skipped: dict[str, str] = field(default_factory=dict)  # label -> reason
    stopped: str | None = None


# Circuit breaker: repeated unexpected failures are systematic, not per-candidate.
MAX_UNEXPECTED_FAILURES_IN_A_ROW = 2


def run_cohort(
    plans: Sequence[SolutionPlan],
    inputs: AcceptedInputs,
    ports: CohortPorts,
    *,
    stop_on_overrun: bool = True,
) -> CohortResult:
    """Run every planned solution in order; the loop survives candidate failures.

    Resume: a solution whose required records all exist is skipped; a
    partially recorded one re-runs only its missing steps.  Its earlier
    records stay stored.  Budget exhaustion or a >50% overrun stops the run.
    """
    from apron.application.orchestration.budget import BudgetExceededError

    result = CohortResult()
    unexpected_in_a_row = 0
    for sp in plans:
        recorded = recorded_evidence(ports.store, sp.solution_fp)
        if sp.status != "planned":
            if recorded.claims:
                result.prediction_errors[sp.label] = recorded.claims[0]
            else:
                claim = prediction_error_claim(sp.claim, sp.status, sp.solution_fp)
                result.prediction_errors[sp.label] = store_validated(ports.store, claim)
                _event(ports, "prediction_error", sp, status=sp.status)
            continue
        if not recorded.claims:
            claim = sp.claim.model_copy(update={"solution_fingerprint": sp.solution_fp})
            store_validated(ports.store, claim)
        scope = recorded.missing(sp)
        if scope is None:
            result.skipped[sp.label] = "already measured"
            continue
        if not ports.budget.can_afford(sp.estimate):
            result.skipped[sp.label] = f"estimate ${sp.estimate:.2f} exceeds remaining budget"
            continue
        try:
            result.executed[sp.label] = execute_solution(sp, inputs, ports, scope=scope)
            unexpected_in_a_row = 0
        except (BudgetExceededError, RunStopError) as exc:
            result.stopped = f"{type(exc).__name__}: {exc}"
            break
        except (HarnessError, PermissionError) as exc:
            result.skipped[sp.label] = f"{type(exc).__name__}: {exc}"
            _event(ports, "candidate_failed", sp, error=str(exc))
        except Exception as exc:  # settled and recorded inside
            result.skipped[sp.label] = f"{type(exc).__name__}: {exc}"
            unexpected_in_a_row += 1
            if unexpected_in_a_row >= MAX_UNEXPECTED_FAILURES_IN_A_ROW:
                result.stopped = f"{unexpected_in_a_row} unexpected failures in a row: {exc}"
                break
        if stop_on_overrun and ports.budget.overruns:
            result.stopped = f"cost overrun >50% on {ports.budget.overruns}"
            break
    return result


def load_attempts(store: RecordStore, digests: Sequence[str]) -> list[TaskAttemptRecord]:
    records: list[TaskAttemptRecord] = []
    for digest in digests:
        record, reason = load_typed(store, digest, TaskAttemptRecord)
        if record is None:
            raise ValueError(reason)
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Qualification from stored evidence (F3/F4)
# ---------------------------------------------------------------------------


def qualify_cohort(
    plans: Sequence[SolutionPlan],
    inputs: AcceptedInputs,
    store: RecordStore,
    clock: Clock,
    ids: IdGenerator,
) -> DecisionReport:
    """Advance every candidate through the qualification graph and report.

    The solution fingerprint was decided when the candidate was planned —
    model spec + deployment plan + requested execution (F4) — before
    feasibility is checked, so prediction-error candidates enter too.  The
    verdicts come from the records stored under that fingerprint (F3); no
    caller asserts a pass.
    """
    from apron.application.orchestration.decision import build_report
    from apron.application.orchestration.qualification import (
        AcceptedRequest,
        EvidenceDigests,
        QualificationGraph,
    )

    graph = QualificationGraph(clock=clock, store=store)
    accepted = AcceptedRequest(
        request=inputs.request,
        task_suite=inputs.task_suite,
        serving_workload=inputs.serving_workload,
    )
    entries = []
    for sp in plans:
        recorded = recorded_evidence(store, sp.solution_fp)
        evidence = EvidenceDigests(
            task_attempts=tuple(recorded.attempts) if recorded.attempts else None,
            serving_report=recorded.serving_reports[-1] if recorded.serving_reports else None,
        )
        entries.append(
            graph.advance(sp, accepted, solution_fingerprint=sp.solution_fp, evidence=evidence)
        )
    return build_report(inputs.request, entries, clock=clock, id_gen=ids)
