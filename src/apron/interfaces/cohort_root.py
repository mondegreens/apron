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
from apron.adapters.backends.llm_classifier import guess_base_model
from apron.adapters.backends.local_store import LocalRecordStore
from apron.adapters.backends.rule_loader import load_rules
from apron.adapters.backends.rule_repository import FileRuleRepository
from apron.adapters.backends.runpod import CLOUD_TYPE, GPU_SPECS, RunPodTarget
from apron.adapters.backends.vllm_engine import VllmEngineAdapter
from apron.adapters.backends.vllm_quantization import (
    default_max_num_batched_tokens,
    load_problems,
)
from apron.adapters.evaluations.deterministic_scorer import DeterministicScorer
from apron.adapters.evidence.hf_hub import HFHubResolver
from apron.adapters.evidence.hf_lineage import HubLineage
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
from apron.application.orchestration.cohort_records import CohortRun, load_cohort_records
from apron.application.orchestration.correction import (
    ArtifactCandidate,
    ArtifactSearch,
    CatalogEntry,
    CorrectionContext,
)
from apron.application.orchestration.evidence import solution_fingerprint
from apron.application.orchestration.plan_pipeline import run_plan_pipeline
from apron.application.orchestration.pods import TargetPool
from apron.application.orchestration.remediation import FixProofPorts
from apron.application.orchestration.scheduler import CandidateSeed, estimate_cost
from apron.application.sanitization import SecretMaskingFilter
from apron.domain.artifacts.identity import ArtifactIdentity
from apron.domain.canonical import digest_hex
from apron.domain.ports import UuidIdGenerator, WallClock
from apron.domain.schemas.authority import AuthorizationEnvelope, DecisionRequest
from apron.domain.schemas.migrations import load_record
from apron.domain.schemas.models import ArtifactSpec
from apron.domain.schemas.primitives import HardwareSpec
from apron.domain.schemas.records import DiagnosisRule
from apron.domain.schemas.solutions import (
    DeploymentPlan,
    RequestedExecutionSpec,
)
from apron.domain.schemas.tasks import ApplicationSpec, ServingWorkloadSpec, TaskSuiteSpec

if TYPE_CHECKING:
    from collections.abc import Callable

    from apron.domain.ports import Clock, IdGenerator

REPO = Path(__file__).resolve().parents[3]
RUN_DIR = REPO / "_dev_notes" / "cohort-run"
FIXTURES = REPO / "tests" / "fixtures" / "phase-1a-run"
SEED = REPO / "cohort" / "phase-1b-seed.json"
# Phase 1b's scoring rule: Phase 1a's whitespace-normalized exact match, plus a
# trailing full stop ignored ("Paris." answers "just the city name"; L5 review).
PROTOCOL = REPO / "cohort" / "phase-1b-evaluation-protocol.json"
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
    fixtures: Path = FIXTURES,
    envelope: AuthorizationEnvelope | None = None,
    protocol: Path = PROTOCOL,
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
        protocol_template=json.loads(protocol.read_text()),
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


def measurement_notes(run_dir: Path = RUN_DIR) -> tuple[str, ...]:
    """Notes every prediction delta carries, from L0-A3's repeat boots (§6.1).

    When the activation measurement varied more than 5% between two boots of
    the same solution, every activation-dependent delta is that uncertain.
    """
    path = run_dir / "l0a3-stability.json"
    if not path.exists():
        return ()
    result = json.loads(path.read_text("utf-8"))
    if not result.get("activation_uncertainty_note_required"):
        return ()
    spread = float(result["activation_relative_difference"])
    return (
        f"activation varied {spread:.1%} between two boots of one solution "
        "(L0-A3, above the 5% threshold); the total and activation deltas carry "
        "that uncertainty",
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
            coverage={
                "size_class": seed.size_class,
                "hardware_class": seed.hardware_class,
                "quantized": seed.quantized,
                "features": list(seed.features),
            },
        )

    def plan_for(self, plan: DeploymentPlan, label: str) -> SolutionPlan:
        """Plan a given DeploymentPlan (fix proof): the object is kept, never rebuilt,
        and a predicted-infeasible broken plan is booted on purpose."""
        alloc = plan.resource_allocation
        _, pipeline = self._pipeline_plan(
            alloc["model_id"],
            alloc["gpu_sku"],
            int(alloc.get("gpu_count", "1")),
            tensor_parallel=plan.tensor_parallel,
        )
        gpu = alloc["gpu_sku"]
        count = int(alloc.get("gpu_count", "1"))
        minutes = 25.0
        estimate = round(minutes / 60 * self.rates.get(gpu, 0.0) * count, 4)
        return self._solution(plan, pipeline, label, check_feasibility=False, estimate=estimate)

    # ------------------------------------------------------------------

    def _pipeline_plan(
        self, model_id: str, gpu: str, count: int, *, tensor_parallel: int = 1
    ) -> tuple[DeploymentPlan, Any]:
        pipeline = run_plan_pipeline(
            self.resolver,
            CalculatorPlanningSource(clock=self.clock),
            model_id,
            hardware_for(gpu),
            clock=self.clock,
            id_gen=self.ids,
            tensor_parallel=tensor_parallel,
            max_num_batched_tokens=default_max_num_batched_tokens(
                hardware_for(gpu).total_memory_bytes, gpu
            ),
            load_check=load_problems,
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
        coverage: dict[str, Any] | None = None,
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
        elif check_feasibility and pipeline.load_problems:
            status = "infeasible"  # the engine would refuse the checkpoint's tensors
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
                # The template turns every task prompt into what the model
                # reads; its digest says which one the measurement used.
                chat_template_identity=digest_hex(pipeline.chat_template.encode("utf-8"))
                if pipeline.chat_template
                else None,
            )
            if observation
            else None,
            coverage=coverage or {},
            notes=measurement_notes(),
            load_problems=tuple(pipeline.load_problems or ()),
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


def stock_waiter(
    api_key: str | None,
    *,
    poll_seconds: int = 120,
    max_wait_seconds: int = 2 * 3600,
    log: JsonlLedger | None = None,
) -> Any:
    """Wait for Secure stock (free read-only query) before a new pod; False on give-up."""
    import time

    def await_capacity(requested: RequestedExecutionSpec) -> bool:
        probe = RunPodTarget(
            api_key=api_key, gpu_type=requested.gpu_sku, gpu_count=requested.gpu_count
        )
        waited = 0
        while True:
            try:
                stock = probe.stock_status()
            except Exception:  # the stock API failing is not a reason to create blind
                stock = None
            if stock:
                if waited and log is not None:
                    log.append(
                        {
                            "gpu": requested.gpu_sku,
                            "count": requested.gpu_count,
                            "stock": stock,
                            "waited_s": waited,
                        }
                    )
                return True
            if waited >= max_wait_seconds:
                if log is not None:
                    log.append(
                        {
                            "gpu": requested.gpu_sku,
                            "count": requested.gpu_count,
                            "stock": None,
                            "gave_up_after_s": waited,
                        }
                    )
                return False
            time.sleep(poll_seconds)
            waited += poll_seconds

    return await_capacity


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
        identities=JsonlLedger(run_dir / "solutions.jsonl"),
        pool=TargetPool(factory=target_factory, budget=budget, clock=clock, hourly_rate=rate_for),
        await_capacity=stock_waiter(
            api_key,
            max_wait_seconds=int(os.environ.get("APRON_CAPACITY_WAIT", str(2 * 3600))),
            log=JsonlLedger(run_dir / "capacity-waits.jsonl"),
        ),
    )


def load_cohort_run(
    run_dir: Path = RUN_DIR, rules_dir: Path = RULES_DIR / "vllm-v0.29"
) -> CohortRun:
    """Read a run directory: records, identity manifest, ledger, events, rule versions."""
    manifest = JsonlLedger(run_dir / "solutions.jsonl").read_all()
    records = load_cohort_records(LocalRecordStore(run_dir / "records"), manifest)
    rules: list[DiagnosisRule] = []
    rule_errors: list[str] = []
    for path in sorted([*rules_dir.glob("*.json"), *rules_dir.glob("history/*.json")]):
        try:
            rules.append(load_record(DiagnosisRule, json.loads(path.read_text("utf-8"))))
        except ValueError as exc:  # pydantic's ValidationError is a ValueError
            rule_errors.append(f"{path.name}: {exc}")
    return CohortRun(
        records=records,
        rules=rules,
        rule_errors=rule_errors,
        ledger=JsonlLedger(run_dir / "ledger.jsonl").read_all(),
        events=JsonlLedger(run_dir / "events.jsonl").read_all(),
        billing=json.loads((run_dir / "billing-reconciliation.json").read_text())
        if (run_dir / "billing-reconciliation.json").exists()
        else None,
        recheck=json.loads((run_dir / "calculator-recheck.json").read_text())
        if (run_dir / "calculator-recheck.json").exists()
        else None,
    )


def available_catalog(
    rates: dict[str, float], gpu_count: int, stock: Any = None
) -> tuple[CatalogEntry, ...]:
    """GPUs a retarget may choose: only those with Secure stock on hosts that can
    run the image, now.  Availability removes an option (product definition
    §2b); an A6000 on CUDA 12.8 hosts only cannot boot the CUDA 13 image.
    """
    stock = stock or (lambda sku: RunPodTarget(gpu_type=sku, gpu_count=gpu_count).stock_status())
    entries = []
    for sku in GPU_SPECS:
        try:
            listed = stock(sku)
        except Exception:
            listed = None
        if listed:
            entries.append(
                CatalogEntry(hardware_for(sku), hourly_rate("runpod", sku, live_rates=rates)[0])
            )
    return tuple(entries)


LINEAGE_DIR = RUN_DIR / "artifact-lineage"


def lineage_search(
    model_id: str, record_spend: Callable[[float, str], None] | None = None
) -> ArtifactSearch | None:
    """Same-lineage checkpoints for *model_id* (class 6 fallback, PLAN §10.1).

    The base-model proposal is a paid classifier call (D5): its cost is
    recorded as ``classifier:base-model:<id>``.  Every step of the search is
    written to ``artifact-lineage/`` so the choice can be re-read later.
    """

    def propose(repo: str, config: dict[str, Any]) -> dict[str, Any]:
        answer = guess_base_model(repo, config)
        cost = answer.get("classifier_cost_usd")
        if record_spend is not None and cost:
            record_spend(float(cost), f"classifier:base-model:{repo}")
        return answer

    search, evidence = HubLineage().search(model_id, propose)
    LINEAGE_DIR.mkdir(parents=True, exist_ok=True)
    (LINEAGE_DIR / f"{model_id.replace('/', '--')}.json").write_text(
        json.dumps(
            {
                "evidence": evidence,
                "search": None if search is None else search_to_json(search),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return search


def search_to_json(search: ArtifactSearch) -> dict[str, Any]:
    return {
        "requested_model_id": search.requested_model_id,
        "base_model_id": search.base_model_id,
        "requested_weight_bits": search.requested_weight_bits,
        "requested_shape": dict(search.requested_shape),
        "candidates": [{**c.__dict__, "shape": dict(c.shape)} for c in search.candidates],
    }


def search_from_json(data: dict[str, Any]) -> ArtifactSearch:
    return ArtifactSearch(
        requested_model_id=data["requested_model_id"],
        base_model_id=data["base_model_id"],
        requested_weight_bits=data["requested_weight_bits"],
        requested_shape=data["requested_shape"],
        candidates=tuple(ArtifactCandidate(**c) for c in data["candidates"]),
    )


def build_fix_ports(
    planner: CohortPlanner,
    rates: dict[str, float],
    record_spend: Callable[[float, str], None] | None = None,
) -> FixProofPorts:
    rules = load_rules(RULES_DIR, "vllm", "v0.29.0")

    def correction_context(sp: SolutionPlan) -> CorrectionContext:
        return CorrectionContext(
            catalog=available_catalog(rates, sp.requested.gpu_count),
            predicted_total_bytes=sp.predicted_total_bytes,
            artifacts=lambda: lineage_search(sp.model_id, record_spend),
        )

    return FixProofPorts(
        plan_solution=planner.plan_for,
        diagnosis_engine=VllmEngineAdapter(rules=rules),
        rules=rules,
        rule_repository=FileRuleRepository(RULES_DIR / "vllm-v0.29"),
        correction_context=correction_context,
        hardware_for=hardware_for,
    )
