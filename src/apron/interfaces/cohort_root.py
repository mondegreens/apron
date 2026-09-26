"""Composition root for the Phase 1b evidence cohort (PLAN §9.1 layer rule).

The only place the real adapters (RunPod, vLLM, the deterministic scorer, the
HF resolver, the file ledger and rule repository) meet the application-layer
orchestrator.  No recommendation, ranking or promotion logic lives here
(INV-11): this module builds objects and hands them over.

Keys come from the environment only: ``RUNPOD_API_KEY``, ``ANTHROPIC_API_KEY``
(read by the Anthropic SDK), ``HF_TOKEN`` (gated downloads, F7).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import apron.domain.mechanisms.calculator  # noqa: F401 — registers calculators
from apron.adapters.backends.ledger_file import JsonlLedger
from apron.adapters.backends.local_store import LocalRecordStore
from apron.adapters.backends.rule_loader import load_rules
from apron.adapters.backends.rule_repository import FileRuleRepository
from apron.adapters.backends.runpod import CLOUD_TYPE, GPU_SPECS, RunPodTarget
from apron.adapters.backends.vllm_engine import VllmEngineAdapter
from apron.adapters.evaluations.deterministic_scorer import DeterministicScorer
from apron.adapters.evidence.hf_hub import HFHubResolver
from apron.adapters.planning.calculator_source import CalculatorPlanningSource
from apron.adapters.runner_image import RUNNER_IMAGE, RUNNER_IMAGE_DIGEST
from apron.application.cost_estimator import hourly_rate
from apron.application.orchestration.budget import BudgetTracker
from apron.application.orchestration.cohort import (
    AcceptedInputs,
    CohortPorts,
    SolutionPlan,
    predicted_feasible,
)
from apron.application.orchestration.correction import CatalogEntry, CorrectionContext
from apron.application.orchestration.evidence import protocol_template, solution_fingerprint
from apron.application.orchestration.plan_pipeline import run_plan_pipeline
from apron.application.orchestration.remediation import FixProofPorts
from apron.application.orchestration.scheduler import CandidateSeed, estimate_cost
from apron.application.sanitization import SecretMaskingFilter
from apron.domain.artifacts.identity import ArtifactIdentity
from apron.domain.ports import UuidIdGenerator, WallClock
from apron.domain.schemas.authority import AuthorizationEnvelope, DecisionRequest
from apron.domain.schemas.models import ArtifactSpec
from apron.domain.schemas.primitives import HardwareSpec
from apron.domain.schemas.solutions import (
    DeploymentPlan,
    EvaluationProtocol,
    RequestedExecutionSpec,
)
from apron.domain.schemas.tasks import ApplicationSpec, ServingWorkloadSpec, TaskSuiteSpec

if TYPE_CHECKING:
    from apron.domain.ports import Clock, IdGenerator

REPO = Path(__file__).resolve().parents[3]
RUN_DIR = REPO / "_dev_notes" / "cohort-run"
FIXTURES = REPO / "tests" / "fixtures" / "phase-1a-run"
SEED = REPO / "cohort" / "phase-1b-seed.json"
RULES_DIR = REPO / "rules"
AUTHORIZED_USD = 100.0  # D4: the owner's cap

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Accepted inputs and authorization
# ---------------------------------------------------------------------------


def cohort_envelope(maximum_spend: float = AUTHORIZED_USD) -> AuthorizationEnvelope:
    """D3/D4/D5: RunPod Secure only, $100 cap, classifier calls to Anthropic."""
    return AuthorizationEnvelope(
        permitted_action_classes=("gpu_execution", "diagnosis_classification"),
        permitted_providers=("runpod",),
        credential_scopes=("runpod:pods", "anthropic:messages", "huggingface:read"),
        task_data_destinations=("api.anthropic.com",),
        hard_target_constraints={"cloud_type": CLOUD_TYPE},
        maximum_spend=maximum_spend,
        teardown_rules={"orphan_max_age_seconds": "3600", "atexit": "terminate"},
    )


def load_inputs(
    fixtures: Path = FIXTURES, envelope: AuthorizationEnvelope | None = None
) -> AcceptedInputs:
    return AcceptedInputs(
        request=DecisionRequest.model_validate_json(
            (fixtures / "decision-request.json").read_text()
        ),
        task_suite=TaskSuiteSpec.model_validate_json(
            (fixtures / "task-suite-spec.json").read_text()
        ),
        application=ApplicationSpec.model_validate_json(
            (fixtures / "application-spec.json").read_text()
        ),
        protocol_template=protocol_template(
            EvaluationProtocol.model_validate_json(
                (fixtures / "evaluation-protocol.json").read_text()
            )
        ),
        serving_workload=ServingWorkloadSpec.model_validate_json(
            (fixtures / "serving-workload-spec.json").read_text()
        ),
        authorization=envelope or cohort_envelope(),
    )


_SEED_FIELDS = set(CandidateSeed.__dataclass_fields__)


def load_seed(path: Path = SEED) -> list[CandidateSeed]:
    """The seed list; metadata keys (fallback, licence_check) are kept aside, not dropped."""
    seeds = []
    for row in json.loads(path.read_text())["candidates"]:
        unknown = set(row) - _SEED_FIELDS - {"fallback", "licence_check"}
        if unknown:
            raise ValueError(f"seed row {row.get('model_id')}: unknown keys {sorted(unknown)}")
        fields = {k: v for k, v in row.items() if k in _SEED_FIELDS}
        fields["features"] = tuple(fields.get("features", ()))
        seeds.append(CandidateSeed(**fields))
    return seeds


# ---------------------------------------------------------------------------
# Step 1: GPU-free planning with the real resolver and calculator
# ---------------------------------------------------------------------------


def hardware_for(gpu_sku: str) -> HardwareSpec:
    spec = GPU_SPECS[gpu_sku]
    return HardwareSpec(
        gpu_sku=gpu_sku,
        total_memory_bytes=spec["total_memory_bytes"],
        compute_capability=spec["compute_capability"],
    )


@dataclass
class CohortPlanner:
    rates: dict[str, float]
    clock: Clock = field(default_factory=WallClock)
    ids: IdGenerator = field(default_factory=UuidIdGenerator)
    resolver: Any = field(default_factory=HFHubResolver)

    def plan_seed(self, seed: CandidateSeed) -> SolutionPlan:
        plan, pipeline = self._pipeline_plan(seed.model_id, seed.gpu_sku, seed.gpu_count)
        return self._solution(
            plan,
            pipeline,
            seed.key,
            check_feasibility=True,
            estimate=estimate_cost(seed, self.rates.get(seed.gpu_sku, 0.0)),
        )

    def plan_for(self, plan: DeploymentPlan, label: str) -> SolutionPlan:
        """Plan a given DeploymentPlan (fix proof): the object is kept, never rebuilt,
        and a predicted-infeasible broken plan is booted on purpose."""
        alloc = plan.resource_allocation
        _, pipeline = self._pipeline_plan(
            alloc["model_id"], alloc["gpu_sku"], int(alloc.get("gpu_count", "1"))
        )
        gpu = alloc["gpu_sku"]
        count = int(alloc.get("gpu_count", "1"))
        minutes = 25.0
        estimate = round(minutes / 60 * self.rates.get(gpu, 0.0) * count, 4)
        return self._solution(plan, pipeline, label, check_feasibility=False, estimate=estimate)

    # ------------------------------------------------------------------

    def _pipeline_plan(self, model_id: str, gpu: str, count: int) -> tuple[DeploymentPlan, Any]:
        pipeline = run_plan_pipeline(
            self.resolver,
            CalculatorPlanningSource(clock=self.clock),
            model_id,
            hardware_for(gpu),
            clock=self.clock,
            id_gen=self.ids,
        )
        if pipeline.model_spec is None or pipeline.claim is None:
            raise ValueError(f"{model_id}: planning failed: {pipeline.error}")
        base = pipeline.plan or DeploymentPlan()
        plan = base.model_copy(
            update={
                # one requested GPU means TP 1; the calculator's TP choice assumes more GPUs
                "tensor_parallel": base.tensor_parallel if count > 1 else 1,
                "resource_allocation": {
                    **base.resource_allocation,
                    "model_id": model_id,
                    "gpu_sku": gpu,
                    "gpu_count": str(count),
                },
            }
        )
        return DeploymentPlan.model_validate(plan.model_dump(mode="json")), pipeline

    def _solution(
        self,
        plan: DeploymentPlan,
        pipeline: Any,
        label: str,
        *,
        check_feasibility: bool,
        estimate: float,
    ) -> SolutionPlan:
        alloc = plan.resource_allocation
        count = int(alloc.get("gpu_count", "1"))
        requested = RequestedExecutionSpec(
            provider="runpod",
            gpu_sku=alloc["gpu_sku"],
            gpu_count=count,
            cloud_type=CLOUD_TYPE,
            image_digest=RUNNER_IMAGE_DIGEST,
        )
        claim = pipeline.claim
        status = "planned"
        if claim.proposed_configuration.get("status") == "unknown":
            status = "unknown"
        elif check_feasibility and not predicted_feasible(
            claim, hardware_for(alloc["gpu_sku"]).total_memory_bytes, count
        ):
            status = "infeasible"
        observation = pipeline.observation
        return SolutionPlan(
            label=label,
            model_id=alloc["model_id"],
            model_spec=pipeline.model_spec,
            model_config=pipeline.model_config or {},
            plan=plan,
            requested=requested,
            claim=claim,
            solution_fp=solution_fingerprint(pipeline.model_spec, plan, requested),
            status=status,  # type: ignore[arg-type]
            chat_template=pipeline.chat_template,
            license_observed=observation.license_observed if observation else None,
            gating_observed=observation.gating_observed if observation else None,
            estimate=estimate,
            artifact_spec=ArtifactSpec(
                identity=ArtifactIdentity.from_observation(observation),
                observations=(observation,),
            )
            if observation
            else None,
        )


# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------


def live_rates(api_key: str | None) -> dict[str, float]:
    """Secure per-GPU rates from the RunPod API at dispatch (GPU-dollar rule)."""
    if not api_key:
        return {}
    return {
        g["gpu_type_id"]: g["secure_price"] for g in RunPodTarget(api_key=api_key).discover_gpus()
    }


def _ssh_public_key() -> str | None:
    key = os.environ.get("RUNPOD_SSH_KEY_PATH") or RunPodTarget._detect_ssh_key()
    pub = Path(key + ".pub")
    return pub.read_text().strip() if pub.exists() else None


def install_log_masking() -> None:
    """F6: every log line passes through the secret mask."""
    root = logging.getLogger()
    for handler in root.handlers or [logging.StreamHandler()]:
        if not any(isinstance(f, SecretMaskingFilter) for f in handler.filters):
            handler.addFilter(SecretMaskingFilter())
        if handler not in root.handlers:
            root.addHandler(handler)


def build_ports(
    run_dir: Path = RUN_DIR,
    *,
    authorized: float = AUTHORIZED_USD,
    rates: dict[str, float] | None = None,
) -> CohortPorts:
    install_log_masking()
    api_key = os.environ.get("RUNPOD_API_KEY")
    rates = rates if rates is not None else live_rates(api_key)
    clock, ids = WallClock(), UuidIdGenerator()
    probe = RunPodTarget(api_key=api_key, leak_log=run_dir / "leaked_pods.json")
    terminated = probe.cleanup_orphaned_pods(max_age_seconds=3600)
    if terminated:
        logger.warning("orphan cleanup terminated %s", terminated)
    budget = BudgetTracker.replay(
        authorized=authorized,
        ledger=JsonlLedger(run_dir / "ledger.jsonl"),
        clock=clock,
        pod_cost=probe.pod_reported_cost,
    )
    public_key = _ssh_public_key()
    hf_token = os.environ.get("HF_TOKEN")

    def target_factory(requested: RequestedExecutionSpec) -> RunPodTarget:
        return RunPodTarget(
            api_key=api_key,
            image=RUNNER_IMAGE,
            gpu_type=requested.gpu_sku,
            gpu_count=requested.gpu_count,
            leak_log=run_dir / "leaked_pods.json",
        )

    def provision_env(sp: SolutionPlan) -> dict[str, str]:
        return RunPodTarget.build_env(ssh_public_key=public_key, hf_token=hf_token)

    def rate_for(requested: RequestedExecutionSpec) -> float:
        rate, _ = hourly_rate(
            "runpod", requested.gpu_sku, live_rates=rates, gpu_count=requested.gpu_count
        )
        return rate

    return CohortPorts(
        target_factory=target_factory,
        engine=VllmEngineAdapter(rules=load_rules(RULES_DIR, "vllm", "v0.29.0")),
        evaluator=DeterministicScorer(),
        store=LocalRecordStore(run_dir / "records"),
        budget=budget,
        clock=clock,
        ids=ids,
        events=JsonlLedger(run_dir / "events.jsonl"),
        provision_env=provision_env,
        hourly_rate=rate_for,
    )


def build_fix_ports(planner: CohortPlanner, rates: dict[str, float]) -> FixProofPorts:
    rules = load_rules(RULES_DIR, "vllm", "v0.29.0")
    catalog = tuple(
        CatalogEntry(hardware_for(sku), rate)
        for sku in GPU_SPECS
        for rate in [hourly_rate("runpod", sku, live_rates=rates)[0]]
    )

    def correction_context(sp: SolutionPlan) -> CorrectionContext:
        return CorrectionContext(catalog=catalog, predicted_total_bytes=sp.predicted_total_bytes)

    return FixProofPorts(
        plan_solution=planner.plan_for,
        diagnosis_engine=VllmEngineAdapter(rules=rules),
        rules=rules,
        rule_repository=FileRuleRepository(RULES_DIR / "vllm-v0.29"),
        correction_context=correction_context,
        hardware_for=hardware_for,
    )
