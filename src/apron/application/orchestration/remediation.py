"""Failure-fix proof (PLAN §10.2) — one function for all six ADR-005 classes.

D1: every fix is proven on the *broken* plan.  The only plan literals in this
module are the six ``P_bad`` below; there are no hand-written good settings.
The plan booted in step 5 is the very object diagnosis returned in step 2.

Protocol, identical for every class:

1. broken boot of ``P_bad`` (the L0-F boot is reused, never repeated);
2. diagnose the captured log with the resolved config.json (F5);
3. Gate A — the diagnosed family is the class's family;
4. Gate B — a corrected plan exists and differs from ``P_bad``;
5. fixed boot of ``P_fix`` (on the retargeted GPU for classes 1 and 6);
6. re-run the accepted task suite and serving measurement;
7. store a RemediationRecord (with the F6 classifier fields);
8. promote the rule to ``mechanism_verified`` when the mechanism is verified.

Labels come only from ``derive_remediation_result`` (INV-2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from apron.application.orchestration.cohort import (
    RunScope,
    execute_solution,
    load_attempts,
    recorded_evidence,
)
from apron.application.orchestration.diagnosis_pipeline import (
    diagnosis_model_config,
    run_diagnosis_pipeline,
)
from apron.application.orchestration.evidence import (
    EvidenceContext,
    decision_request_digest,
    load_typed,
    store_validated,
)
from apron.application.orchestration.serving import evaluate_serving_slos
from apron.domain.fingerprints import fingerprint_hex
from apron.domain.schemas.records import (
    DiagnosisRule,
    RemediationRecord,
    VerificationReport,
    derive_remediation_result,
)
from apron.domain.schemas.solutions import DeploymentPlan
from apron.domain.verdicts import task_verdict

if TYPE_CHECKING:
    from collections.abc import Sequence

    from apron.application.orchestration.cohort import (
        AcceptedInputs,
        CohortPorts,
        SolutionPlan,
    )

# ---------------------------------------------------------------------------
# The six broken plans (PLAN §10.1).  Rows are candidates: L0-F confirms each
# fails at the named vLLM call site before it counts.
# ---------------------------------------------------------------------------

_4090 = "NVIDIA GeForce RTX 4090"
_H100 = "NVIDIA H100 80GB HBM3"


@dataclass(frozen=True)
class BrokenCase:
    failure_class: int
    expected_family: str
    broken_plan: DeploymentPlan
    call_site: str  # the pinned vLLM line the failure must come from
    expected_error: str  # text that line prints; L0-F checks the captured log for it


SIX_CLASSES: tuple[BrokenCase, ...] = (
    BrokenCase(
        1,
        "oom_weight_load",
        DeploymentPlan(
            dtype="bfloat16",
            resource_allocation={"model_id": "Qwen/Qwen3-14B", "gpu_sku": _4090, "gpu_count": "1"},
        ),
        "v1/worker/gpu_model_runner.py:5460",
        "Failed to load model - not enough GPU memory",
    ),
    BrokenCase(
        2,
        "oom_kv_cache",
        DeploymentPlan(
            dtype="bfloat16",
            engine_configuration={"max_model_len": "40960"},
            resource_allocation={"model_id": "Qwen/Qwen3-8B", "gpu_sku": _4090, "gpu_count": "1"},
        ),
        "v1/core/kv_cache_utils.py:879",
        "To serve at least one request with the model's max seq len",
    ),
    BrokenCase(
        3,
        "max_model_len",
        DeploymentPlan(
            dtype="bfloat16",
            engine_configuration={"max_model_len": "999999"},
            resource_allocation={
                "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
                "gpu_sku": _4090,
                "gpu_count": "1",
            },
        ),
        "config/model.py:2502",
        "is greater than the derived max_model_len",
    ),
    BrokenCase(
        4,
        "dtype_incompatible",
        DeploymentPlan(
            dtype="float16",
            resource_allocation={
                "model_id": "google/gemma-2-2b-it",
                "gpu_sku": _4090,
                "gpu_count": "1",
            },
        ),
        "config/model.py:2262",
        "does not support float16",
    ),
    BrokenCase(
        5,
        "tp_divisibility",
        DeploymentPlan(
            tensor_parallel=3,
            dtype="bfloat16",
            resource_allocation={"model_id": "Qwen/Qwen3-8B", "gpu_sku": _4090, "gpu_count": "4"},
        ),
        "config/model.py:1414",
        "must be divisible by tensor parallel size",
    ),
    BrokenCase(
        6,
        "quant_compute_capability",
        DeploymentPlan(
            dtype="bfloat16",
            resource_allocation={
                "model_id": "ISTA-DASLab/Qwen3-0.6B-FPQuant-RTN-MXFP4",
                "gpu_sku": _H100,
                "gpu_count": "1",
            },
        ),
        "config/vllm.py:791",
        "is not supported for the current GPU. Minimum capability",
    ),
)


# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------


class RuleRepository(Protocol):
    """Where rules live (``rules/<engine>-v<major.minor>/``); an adapter writes files."""

    def current(self, family: str) -> DiagnosisRule: ...

    def write_version(self, rule: DiagnosisRule) -> str: ...


@dataclass(frozen=True)
class FixProofPorts:
    """What the fix proof needs beyond the cohort ports."""

    plan_solution: Any  # Callable[[DeploymentPlan, str], SolutionPlan] — GPU-free
    diagnosis_engine: Any  # classify / extract (the LLM classifier behind the engine)
    rules: Sequence[dict[str, Any]]
    rule_repository: RuleRepository
    correction_context: Any  # Callable[[SolutionPlan], CorrectionContext]
    hardware_for: Any  # Callable[[str], HardwareSpec]
    classifier_cost: float = 0.02


# ---------------------------------------------------------------------------
# Promotion (§10.2 step 8)
# ---------------------------------------------------------------------------


def promoted_rule(rule: DiagnosisRule, promoting_report_digest: str) -> DiagnosisRule:
    """A new rule version at ``mechanism_verified`` citing the boot that proved it."""
    return DiagnosisRule.model_validate(
        {
            **rule.model_dump(mode="json"),
            "rule_version": rule.rule_version + 1,
            "supersedes": fingerprint_hex(rule),
            "status": "mechanism_verified",
            "promoting_verification_fingerprint": promoting_report_digest,
        }
    )


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------


@dataclass
class FixProof:
    failure_class: int
    expected_family: str
    broken_solution_fp: str
    broken_failed: bool
    diagnosed_family: str = "unknown"
    gate_a: bool = False
    gate_b: bool = False
    corrected_plan: DeploymentPlan | None = None
    fixed_solution_fp: str | None = None
    mechanism_outcome: str = "not_evaluated"
    request_outcome: str = "not_evaluated"
    label: str = "Unverified suggestion"
    remediation_digest: str | None = None
    promoted_rule_digest: str | None = None
    violated_constraints: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)


def prove_fix(
    case: BrokenCase,
    inputs: AcceptedInputs,
    ports: CohortPorts,
    fix: FixProofPorts,
) -> FixProof:
    """Run §10.2 steps 1-8 for one class.  No per-class branches."""
    bad = fix.plan_solution(case.broken_plan, f"class{case.failure_class}-broken")
    proof = FixProof(
        case.failure_class, case.expected_family, bad.solution_fp, broken_failed=False
    )

    # 1. Broken boot — reuse the stored L0-F boot when it exists.
    log = _stored_failure_log(ports, bad.solution_fp)
    if log is None:
        outcome = execute_solution(
            bad, inputs, ports, scope=RunScope(False, False), reason="fix_proof_broken_boot"
        )
        if outcome.healthy:
            proof.notes.append("broken plan booted healthy — replace it; it counts for nothing")
            return proof
        log = outcome.log_tail
    proof.broken_failed = bool(log)
    if not log:
        proof.notes.append("broken boot produced no log (harness failure only)")
        return proof

    # 2. Diagnose (a paid classifier call, authorized and recorded — F6).
    _authorize_classification(inputs)
    ports.budget.record_spend(fix.classifier_cost, f"classifier:class{case.failure_class}")
    diagnosis = run_diagnosis_pipeline(
        log,
        fix.diagnosis_engine,
        bad.plan,
        diagnosis_model_config(dict(bad.model_config)),
        fix.hardware_for(bad.requested.gpu_sku),
        list(fix.rules),
        correction_context=fix.correction_context(bad),
    )
    proof.diagnosed_family = diagnosis.failure_class

    # 3-4. Gates A and B.
    proof.gate_a = diagnosis.failure_class == case.expected_family
    p_fix = diagnosis.corrected_plan
    proof.gate_b = p_fix is not None and p_fix != bad.plan
    proof.corrected_plan = p_fix
    if not (proof.gate_a and proof.gate_b) or p_fix is None:
        proof.remediation_digest = _store_record(
            ports, inputs, bad, None, diagnosis, proof, proving=(), reason="gate failed"
        )
        return proof

    # 5-6. Fixed boot of the object diagnosis returned, then the request re-run.
    fixed = fix.plan_solution(p_fix, f"class{case.failure_class}-fixed")
    proof.fixed_solution_fp = fixed.solution_fp
    recorded = recorded_evidence(ports.store, fixed.solution_fp)
    scope = recorded.missing(fixed)
    if scope is not None:
        execute_solution(fixed, inputs, ports, scope=scope, reason="fix_proof_fixed_boot")
        recorded = recorded_evidence(ports.store, fixed.solution_fp)
    boot_digest = recorded.memory_reports[0] if recorded.memory_reports else None
    proof.mechanism_outcome = "verified" if boot_digest else "failed"
    proving: tuple[str, ...] = ()
    if boot_digest:
        proof.request_outcome, proof.violated_constraints = _request_outcome(
            ports, inputs, fixed, recorded.attempts, recorded.serving_reports
        )
        proving = (boot_digest, *recorded.attempts, *recorded.serving_reports)

    # 7. Record.
    proof.label = derive_remediation_result(proof.mechanism_outcome, proof.request_outcome)
    proof.remediation_digest = _store_record(
        ports, inputs, bad, fixed, diagnosis, proof, proving=proving, reason="fix proof"
    )

    # 8. Promotion.
    if proof.mechanism_outcome == "verified" and boot_digest:
        rule = fix.rule_repository.current(case.expected_family)
        proof.promoted_rule_digest = fix.rule_repository.write_version(
            promoted_rule(rule, boot_digest)
        )
    return proof


def failed_as_named(case: BrokenCase, log: str) -> bool:
    """L0-F: the broken boot failed at the named call site (its printed text)."""
    return case.expected_error in log


def prove_all(
    inputs: AcceptedInputs,
    ports: CohortPorts,
    fix: FixProofPorts,
    cases: Sequence[BrokenCase] = SIX_CLASSES,
) -> list[FixProof]:
    return [prove_fix(case, inputs, ports, fix) for case in cases]


# ---------------------------------------------------------------------------


def _stored_failure_log(ports: CohortPorts, solution_fp: str) -> str | None:
    """The captured log of a stored model-failure boot for this solution, if any."""
    for digest in recorded_evidence(ports.store, solution_fp).failed_boots:
        report, _ = load_typed(ports.store, digest, VerificationReport)
        if report is not None and "boot:model_failure" in report.failures and report.log_tail:
            return report.log_tail
    return None


def _authorize_classification(inputs: AcceptedInputs) -> None:
    envelope = inputs.authorization
    if "diagnosis_classification" not in envelope.permitted_action_classes:
        raise PermissionError("diagnosis_classification is not a permitted action class")
    if "api.anthropic.com" not in envelope.task_data_destinations:
        raise PermissionError("api.anthropic.com is not a permitted data destination")


def _request_outcome(
    ports: CohortPorts,
    inputs: AcceptedInputs,
    fixed: SolutionPlan,
    attempt_digests: Sequence[str],
    serving_digests: Sequence[str],
) -> tuple[str, tuple[str, ...]]:
    """``satisfied`` only if the accepted task suite and the serving SLO both pass."""
    violated: list[str] = []
    attempts = load_attempts(ports.store, attempt_digests)
    task = task_verdict(
        attempts,
        solution_fingerprint=fixed.solution_fp,
        case_ids=[c.get("id", "") for c in inputs.task_suite.cases],
        quality_floor=inputs.request.quality_floor,
    )
    if not task.passed:
        violated.append(f"task: {task.reason}")
    if not serving_digests:
        violated.append("serving: not measured")
    else:
        report, reason = load_typed(ports.store, serving_digests[0], VerificationReport)
        if report is None:
            violated.append(f"serving: {reason}")
        else:
            serving = evaluate_serving_slos(report, inputs.serving_workload)
            if not serving.passed:
                violated.extend(f"serving: {r}" for r in serving.reasons)
    return ("violated" if violated else "satisfied"), tuple(violated)


def _store_record(
    ports: CohortPorts,
    inputs: AcceptedInputs,
    bad: SolutionPlan,
    fixed: SolutionPlan | None,
    diagnosis: Any,
    proof: FixProof,
    *,
    proving: tuple[str, ...],
    reason: str,
) -> str:
    solution = fixed if fixed is not None else bad
    request = inputs.request_for(solution.model_id)
    ctx = EvidenceContext.bind(
        request=request,
        task_suite=inputs.task_suite,
        application=inputs.application,
        protocol_template=inputs.protocol_template,
        solution_fp=solution.solution_fp,
    )
    corrected = proof.corrected_plan
    record = RemediationRecord(
        mechanism_outcome=proof.mechanism_outcome,  # type: ignore[arg-type]
        request_outcome=proof.request_outcome,  # type: ignore[arg-type]
        accepted_request_digest=decision_request_digest(request),
        task_fingerprint=ctx.task_suite_fingerprint,
        application_fingerprint=ctx.application_fingerprint,
        evaluation_fingerprint=ctx.evaluation_protocol_fingerprint,
        corrected_plan_digest=fingerprint_hex(corrected) if corrected is not None else "none",
        corrected_solution_digest=fixed.solution_fp if fixed is not None else None,
        violated_constraints=proof.violated_constraints,
        proving_record_fingerprints=proving,
        classifier_model_id=diagnosis.classifier_model_id,
        classifier_input_digest=diagnosis.classifier_input_digest,
        diagnosed_failure_class=diagnosis.failure_class,
        classifier_evidence_span=diagnosis.evidence_span or None,
        classifier_extraction=dict(diagnosis.extracted),
        correction_strategy=diagnosis.correction_strategy,
        reason=(
            f"class {proof.failure_class} ({proof.expected_family}): {reason}; "
            f"gate A {'pass' if proof.gate_a else 'fail'}, "
            f"gate B {'pass' if proof.gate_b else 'fail'}"
        ),
        corrects=fingerprint_hex(bad.plan),
    )
    return store_validated(ports.store, record)
