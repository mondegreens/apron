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
from apron.adapters.backends.runpod_storage import (
    RunPodStagerPod,
    RunPodStorage,
    storage_cost,
)
from apron.adapters.backends.vllm_engine import VllmEngineAdapter
from apron.adapters.backends.vllm_quantization import (
    VLLM_DEFAULT_GPU_MEMORY_UTILIZATION,
    engine_facts,
    engine_for,
)
from apron.adapters.evaluations.deployment_checks import DeploymentCheckScorer
from apron.adapters.evaluations.deterministic_scorer import DeterministicScorer
from apron.adapters.evidence.hf_hub import HFHubResolver
from apron.adapters.evidence.hf_lineage import HubLineage
from apron.adapters.planning.calculator_source import CalculatorPlanningSource
from apron.adapters.runner_image import (
    RUNNER_IMAGES,
    image_for_digest,
    newest_runner_image,
    runner_image,
)
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
from apron.application.orchestration.plan_builder import PLAN_GPU_MEMORY_UTILIZATION
from apron.application.orchestration.plan_pipeline import (
    ServingFeatures,
    StateBlockFacts,
    run_plan_pipeline,
)
from apron.application.orchestration.pods import TargetPool
from apron.application.orchestration.remediation import FixProofPorts
from apron.application.orchestration.scheduler import CandidateSeed, estimate_cost
from apron.application.orchestration.staging import STORAGE_PREFIX, StagingResult, stage_weights
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
    from collections.abc import Callable, Iterable

    from apron.domain.ports import Clock, IdGenerator

REPO = Path(__file__).resolve().parents[3]
RUN_DIR = REPO / "_dev_notes" / "cohort-run"
FIXTURES = REPO / "tests" / "fixtures" / "phase-1a-run"
SEED = REPO / "cohort" / "phase-1b-seed.json"
# Phase 1b's scoring rule: Phase 1a's whitespace-normalized exact match, plus a
# trailing full stop ignored ("Paris." answers "just the city name"; L5 review).
PROTOCOL = REPO / "cohort" / "phase-1b-evaluation-protocol.json"
RULES_DIR = REPO / "rules"
# The engine plans use when a caller does not choose one: the version the
# first cohort's records were made on.
RUNNER_IMAGES_DEFAULT = "v0.29.0"
# D4: the owner's cap for the whole Phase 1b cohort (all passes), not per day
# or run; raising it is the owner's call (group D).
AUTHORIZED_USD = 100.0

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Accepted inputs and authorization
# ---------------------------------------------------------------------------


def cohort_envelope(maximum_spend: float = AUTHORIZED_USD) -> AuthorizationEnvelope:
    """D3/D4/D5: RunPod Secure only, the owner's cap, classifier calls to Anthropic."""
    return AuthorizationEnvelope(
        permitted_action_classes=("gpu_execution", "diagnosis_classification"),
        permitted_providers=("runpod",),
        credential_scopes=("runpod:pods", "anthropic:messages", "huggingface:read"),
        task_data_destinations=("api.anthropic.com",),
        hard_target_constraints={"cloud_type": CLOUD_TYPE},
        maximum_spend=maximum_spend,
        teardown_rules={"orphan_max_age_seconds": "3600", "atexit": "terminate"},
    )


# The modern models (groups A-D) answer the same three questions with room to
# reason first: 512 tokens instead of 8 (owner, 2026-09-27).  A different suite
# is a different evaluation protocol; the first cohort's records keep theirs.
TASK_SUITE_V2 = REPO / "cohort" / "task-suite-v2.json"
# Deployment checks (PLAN §18.1 item 5): long-context needle, tool call, JSON
# output, image input, reasoning split.  Scored by DeploymentCheckScorer with
# each plan's deployment facts; v2 and its records are untouched.
TASK_SUITE_V3 = REPO / "cohort" / "task-suite-v3.json"
# The context a suite-v3 plan serves: the needle fills it.  32k tokens is a
# common serving length every model of groups C and D exceeds natively
# (owner-approved v3 proposal, 2026-09-28).
V3_MAX_MODEL_LEN = 32768


def load_inputs(
    fixtures: Path = FIXTURES,
    envelope: AuthorizationEnvelope | None = None,
    protocol: Path = PROTOCOL,
    task_suite: Path | None = None,
) -> AcceptedInputs:
    return AcceptedInputs(
        request=DecisionRequest.model_validate_json(
            (fixtures / "decision-request.json").read_text()
        ),
        task_suite=TaskSuiteSpec.model_validate_json(
            (task_suite or fixtures / "task-suite-spec.json").read_text()
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


def recorded_prediction(
    entry: Any, config: dict[str, Any], weights: dict[str, Any]
) -> dict[str, Any]:
    """The calculator's prediction for a recorded boot, GPU-free and offline.

    *entry* is the boot's solution (``solutions.jsonl``), *config* the
    checkpoint's ``config.json`` at the recorded revision
    (``tests/fixtures/cohort/configs.json``) and *weights* its row of
    ``weight-bytes.json`` (stored bytes, and the processor fields of the
    activation estimate).  Predicted as the plan booted: its GPU, tensor
    parallelism, dtype, utilization (vLLM's default when it set none) and
    max_num_seqs, and the engine version's batch defaults on that GPU.
    """
    from apron.application.orchestration.plan_pipeline import (
        calculator_metadata,
        recorded_loaded_bytes,
        runtime_dtype,
    )
    from apron.domain.mechanisms.model_spec_builder import build_model_spec

    version = next(
        tag
        for tag, image in RUNNER_IMAGES.items()
        if image.digest == entry.requested_execution.image_digest
    )
    facts = engine_facts(version)
    gpu = entry.requested_execution.gpu_sku
    hardware = hardware_for(gpu)
    plan = entry.deployment_plan
    engine = plan.engine_configuration
    dtype = plan.dtype or runtime_dtype(config)
    spec = build_model_spec(config, repository=entry.model_id)
    metadata = calculator_metadata(
        config, spec, total_weight_bytes=recorded_loaded_bytes(weights, dtype), dtype=dtype
    )
    # The processor-derived fields the encoder moment reads (config fields agree).
    metadata.update(
        {k: v for k, v in (weights.get("activation") or {}).items() if k not in metadata}
    )
    memory = hardware.total_memory_bytes
    shape = {
        "isl": 512,
        "osl": 128,
        "max_batch_size": 4,
        "tensor_parallel": plan.tensor_parallel,
        "gpu_memory_utilization": float(
            engine.get("gpu_memory_utilization") or VLLM_DEFAULT_GPU_MEMORY_UTILIZATION
        ),
        "max_num_batched_tokens": facts.default_max_num_batched_tokens(memory, gpu),
        "max_num_seqs": int(engine.get("max_num_seqs") or facts.default_max_num_seqs(memory, gpu)),
        **({"enforce_eager": True} if engine.get("enforce_eager") in ("true", True) else {}),
    }
    claim = CalculatorPlanningSource(clock=WallClock()).predict(metadata, hardware, shape)
    return dict(claim.proposed_configuration)


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
    # Plans for task suite v3: served at V3_MAX_MODEL_LEN with the tool-call
    # and reasoning parsers its deployment checks need (ServingFeatures).
    deployment_checks: bool = False

    def engine(self, model_id: str) -> str | None:
        """The vLLM version a plan for *model_id* runs on: the newest one with a
        pinned runner image whose model registry serves the checkpoint's
        architecture (owner, 2026-09-27: Apron follows vLLM releases).  None
        when no pinned version serves it."""
        raw = self.resolver._download_file(model_id, "config.json", "main")
        config = json.loads(raw) if raw else {}
        architecture = (config.get("architectures") or [""])[0]
        return engine_for(architecture, RUNNER_IMAGES)

    def plan_seed(self, seed: CandidateSeed) -> SolutionPlan:
        engine = self.engine(seed.model_id)
        # Several GPUs for one model: split it across them (TP = count), and
        # predict per GPU.  Unsplit, a 2-GPU MiniMax-M2.7 read as 214 GiB on
        # one H200 and was called infeasible (GPU-free check, 2026-09-27).
        plan, pipeline = self._pipeline_plan(
            seed.model_id,
            seed.gpu_sku,
            seed.gpu_count,
            tensor_parallel=seed.gpu_count,
            engine=engine or newest_runner_image().version,
        )
        return self._solution(
            plan,
            pipeline,
            seed.key,
            engine=engine,
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
        engine = plan.engine_configuration
        _, pipeline = self._pipeline_plan(
            alloc["model_id"],
            alloc["gpu_sku"],
            int(alloc.get("gpu_count", "1")),
            tensor_parallel=plan.tensor_parallel,
            # The given plan boots as it is: no max_num_seqs of the planner's own.
            state_block_guard=False,
            # ... and is predicted as it boots: its utilization (vLLM's default
            # when it sets none), dtype and max_num_seqs.  A float32 Mamba plan
            # was predicted at 16 bits and its KV budget 5.75 GiB high; plans
            # without a utilization were predicted at 0.90 and booted at 0.92.
            gpu_memory_utilization=float(
                engine.get("gpu_memory_utilization") or VLLM_DEFAULT_GPU_MEMORY_UTILIZATION
            ),
            dtype=plan.dtype,
            max_num_seqs=int(engine["max_num_seqs"]) if engine.get("max_num_seqs") else None,
        )
        gpu = alloc["gpu_sku"]
        count = int(alloc.get("gpu_count", "1"))
        minutes = 25.0
        estimate = round(minutes / 60 * self.rates.get(gpu, 0.0) * count, 4)
        return self._solution(plan, pipeline, label, check_feasibility=False, estimate=estimate)

    # ------------------------------------------------------------------

    def _pipeline_plan(
        self,
        model_id: str,
        gpu: str,
        count: int,
        *,
        tensor_parallel: int = 1,
        engine: str = RUNNER_IMAGES_DEFAULT,
        state_block_guard: bool = True,
        gpu_memory_utilization: float = PLAN_GPU_MEMORY_UTILIZATION,
        dtype: str | None = None,
        max_num_seqs: int | None = None,
    ) -> tuple[DeploymentPlan, Any]:
        facts = engine_facts(engine)
        memory = hardware_for(gpu).total_memory_bytes
        features = (
            ServingFeatures(
                max_model_len=V3_MAX_MODEL_LEN,
                tool_parsers=facts.tool_parsers,
                tool_parser_architectures=facts.tool_parser_architectures,
                recipe_checkpoints=facts.recipe_checkpoints,
            )
            if self.deployment_checks
            else None
        )
        pipeline = run_plan_pipeline(
            self.resolver,
            CalculatorPlanningSource(clock=self.clock),
            model_id,
            hardware_for(gpu),
            clock=self.clock,
            id_gen=self.ids,
            tensor_parallel=tensor_parallel,
            # The plan's engine version's own defaults for this GPU.
            max_num_batched_tokens=facts.default_max_num_batched_tokens(memory, gpu),
            max_num_seqs=max_num_seqs or facts.default_max_num_seqs(memory, gpu),
            load_check=facts.load_problems,
            unmodelled_architectures=facts.hybrid_architectures,
            tokenizer_modes=facts.tokenizer_modes,
            reasoning_parsers=facts.reasoning_parsers,
            reasoning_parser_architectures=facts.reasoning_parser_architectures,
            state_blocks=StateBlockFacts(
                engine_version=facts.version,
                check=facts.state_block_check,
                profiling=facts.state_block_profiling,
                default_source=facts.default_max_num_seqs_source(memory, gpu),
            )
            if state_block_guard
            else None,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            features=features,
        )
        if pipeline.model_spec is None or pipeline.claim is None:
            raise ValueError(f"{model_id}: planning failed: {pipeline.error}")
        base = pipeline.plan or DeploymentPlan()
        plan = base.model_copy(
            update={
                # One requested GPU means TP 1.  A plan split on purpose keeps
                # the split it was predicted for; otherwise the calculator's
                # TP choice applies when several GPUs are rented.
                "tensor_parallel": tensor_parallel
                if tensor_parallel > 1
                else (base.tensor_parallel if count > 1 else 1),
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
        engine: str | None = RUNNER_IMAGES_DEFAULT,
    ) -> SolutionPlan:
        alloc = plan.resource_allocation
        count = int(alloc.get("gpu_count", "1"))
        requested = RequestedExecutionSpec(
            provider="runpod",
            gpu_sku=alloc["gpu_sku"],
            gpu_count=count,
            cloud_type=CLOUD_TYPE,
            # The engine is part of the solution's identity through its image.
            image_digest=(runner_image(engine) if engine else newest_runner_image()).digest,
        )
        claim = pipeline.claim
        status = "planned"
        if engine is None:
            status = "unknown"  # no pinned vLLM serves the architecture: never booted
        elif claim.proposed_configuration.get("status") == "unknown":
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
            chat_renderer=pipeline.chat_renderer,
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
            # What the prediction needs besides the GPUs (host RAM for tables
            # the engine keeps off them) travels with the measurement notes.
            notes=measurement_notes() + tuple(getattr(pipeline, "notes", ()) or ()),
            load_problems=tuple(pipeline.load_problems or ()),
            engine_version=engine,
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
    data_center_id: str | None = None,
    poll_seconds: int = 120,
    max_wait_seconds: int = 2 * 3600,
    log: JsonlLedger | None = None,
) -> Any:
    """Wait for Secure stock (free read-only query) before a new pod; False on give-up."""
    import time

    def await_capacity(requested: RequestedExecutionSpec) -> bool:
        probe = RunPodTarget(
            api_key=api_key,
            gpu_type=requested.gpu_sku,
            gpu_count=requested.gpu_count,
            # With staged weights the pod can only start in the volume's datacenter.
            network_volume_id="probe" if data_center_id else None,
            data_center_id=data_center_id,
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


# ---------------------------------------------------------------------------
# Staged weights (phase plan: GPU dollar protection rule 1)
# ---------------------------------------------------------------------------

# A CPU stager's rate is not in the GPU price list; the hold uses this and the
# settle uses the pod's reported cost.
STAGER_RATE_ESTIMATE = 0.30
# Planning figure for the hold only: the low end of the volume's documented
# 200-400 MB/s, plus the runner image pull on a fresh host.
STAGER_BYTES_PER_SECOND = 200e6
STAGER_PULL_SECONDS = 15 * 60
VOLUMES_FILE = "volumes.json"


@dataclass(frozen=True)
class WeightsSite:
    """A network volume the GPU pods of a run attach, in its datacenter."""

    volume_id: str
    data_center_id: str
    size_gb: int


def staged_bytes(model_id: str) -> int:
    """Bytes the engine's download step writes for *model_id* (Hub listing)."""
    import fnmatch

    from huggingface_hub import HfApi

    from apron.adapters.backends.vllm_engine import DOWNLOAD_IGNORE

    info = HfApi().model_info(model_id, files_metadata=True, token=os.environ.get("HF_TOKEN"))
    return sum(
        int(s.size or 0)
        for s in info.siblings or []
        if not any(fnmatch.fnmatch(s.rfilename, g) for g in DOWNLOAD_IGNORE)
    )


def weights_site(
    storage: RunPodStorage,
    executions: list[RequestedExecutionSpec],
    model_ids: list[str],
    *,
    headroom: float = 1.15,
) -> WeightsSite | None:
    """Choose a datacenter with stock for every execution and size its volume.

    ``None`` when no storage datacenter has stock for all of them right now.
    An existing Apron volume's datacenter is preferred (its weights stay).
    """
    from apron.adapters.backends.runpod_storage import region_rank

    existing = tuple(
        str(v.get("dataCenterId")) for v in storage.list_volumes() if v.get("dataCenterId")
    )
    # Preferred region first (US before Europe before Asia): an existing
    # volume's weights are kept only when it is in the best region with stock.
    candidates = sorted(
        dict.fromkeys((*existing, *storage.storage_datacenters())),
        key=lambda dc: (region_rank(dc), dc not in existing),
    )
    with_volume = {str(v.get("dataCenterId")) for v in storage.list_volumes()}

    def staged_already(dc: str) -> bool:
        return dc in with_volume and set(model_ids) <= staged_ok_on(dc)

    chosen: str | None = None
    for dc in candidates:
        # GPU stock for every execution, and a CPU pod to stage from — unless
        # every model is already on that datacenter's volume (2026-09-28: no
        # CPU pod in US-CA-2 held back a run whose weights were all staged).
        if all(storage.stock_in(dc, e.gpu_sku, e.gpu_count) for e in executions) and (
            staged_already(dc) or storage.cpu_stock(dc)
        ):
            chosen = dc
            break
    if chosen is None:
        return None

    def need_gb(models: set[str]) -> int:
        return int(sum(staged_bytes(m) for m in sorted(models)) * headroom / 1e9) + 10

    volume = storage.ensure_volume(chosen, need_gb(set(model_ids)))
    # The volume keeps earlier groups' weights (and any partial download), so
    # it is sized for everything on it, not only this group's models.
    on_volume = staged_on(str(volume["id"])) | set(model_ids)
    if on_volume != set(model_ids):
        volume = storage.ensure_volume(chosen, need_gb(on_volume))
    size = int(volume.get("size") or need_gb(on_volume))
    return WeightsSite(str(volume["id"]), chosen, size)


def staged_ok_on(data_center_id: str, run_dir: Path | None = None) -> set[str]:
    """Models a staging run verified in *data_center_id* (``staged`` events;
    Apron keeps one weights volume per datacenter)."""
    events = (run_dir or RUN_DIR) / "events.jsonl"
    if not events.exists():
        return set()
    ok: set[str] = set()
    for line in events.read_text().splitlines():
        entry = json.loads(line) if line.strip() else {}
        if (
            entry.get("event") == "staged"
            and entry.get("ok")
            and entry.get("location") == data_center_id
        ):
            ok.add(str(entry["model_id"]))
    return ok


def staged_on(volume_id: str, run_dir: Path | None = None) -> set[str]:
    """Models earlier staging runs wrote to *volume_id* (finished or not: a
    partial download holds space too), from the ``prestage*.json`` records."""
    found: set[str] = set()
    for path in sorted((run_dir or RUN_DIR).glob("prestage*.json")):
        record = json.loads(path.read_text())
        if (record.get("site") or {}).get("volume_id") != volume_id:
            continue
        found |= {m["model_id"] for m in (record.get("staging") or {}).get("models") or []}
    return found


def stage_site(
    site: WeightsSite,
    model_ids: list[str],
    ports: CohortPorts,
    run_dir: Path = RUN_DIR,
) -> StagingResult:
    """Download *model_ids* onto the site's volume from a CPU pod in its datacenter.

    Models a staging run already verified there are not fetched again, and
    when all are, no stager pod is started at all.
    """
    from apron.application.orchestration.staging import StagedModel, StagingResult

    done = staged_ok_on(site.data_center_id, run_dir)
    if set(model_ids) <= done:
        return StagingResult(
            pod_id=None,
            cost=0.0,
            seconds=0.0,
            models=[StagedModel(m, True, 0.0, "already staged") for m in model_ids],
        )
    api_key = os.environ.get("RUNPOD_API_KEY")
    total = sum(staged_bytes(m) for m in model_ids)
    hours = (total / STAGER_BYTES_PER_SECOND + STAGER_PULL_SECONDS) / 3600
    stager = RunPodStagerPod(
        api_key=api_key,
        image=newest_runner_image().image,  # any runner image downloads the same
        network_volume_id=site.volume_id,
        data_center_id=site.data_center_id,
        leak_log=run_dir / "leaked_pods.json",
    )
    events = JsonlLedger(run_dir / "events.jsonl")
    return stage_weights(
        model_ids,
        stager=stager,
        engine=ports.engine,
        budget=ports.budget,
        clock=ports.clock,
        hourly_rate=STAGER_RATE_ESTIMATE,
        estimate=round(max(0.05, STAGER_RATE_ESTIMATE * hours * 1.5), 4),
        env=RunPodTarget.build_env(
            ssh_public_key=_ssh_public_key(), hf_token=os.environ.get("HF_TOKEN")
        ),
        event=lambda e: events.append({"at": ports.clock.now().isoformat(), **e}),
    )


def accrue_storage(site: WeightsSite, budget: BudgetTracker, run_dir: Path = RUN_DIR) -> float:
    """Spend the volume's storage since it was last accrued (created: now)."""
    path = run_dir / VOLUMES_FILE
    state: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    now = budget.clock.now().timestamp()
    entry = state.setdefault(site.volume_id, {"since": now, "size_gb": site.size_gb})
    hours = max(0.0, now - float(entry["since"])) / 3600
    amount = storage_cost(site.size_gb, hours, site.data_center_id)
    if amount > 0:
        budget.record_spend(
            amount, f"{STORAGE_PREFIX}{site.volume_id}:{site.size_gb}GB:{round(hours, 3)}h"
        )
    entry.update(since=now, size_gb=site.size_gb, data_center_id=site.data_center_id)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    return amount


def build_ports(
    run_dir: Path = RUN_DIR,
    *,
    authorized: float = AUTHORIZED_USD,
    rates: dict[str, float] | None = None,
    site: WeightsSite | None = None,
    evaluator: Any = None,
) -> CohortPorts:
    """The run's ports.  ``evaluator`` scores the task suite: the
    deterministic scorer unless given (suite v3's ``DeploymentCheckScorer``,
    built for the run's plans: ``v3_evaluator``)."""
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
            # The plan's engine: its image digest is in the solution's identity.
            image=image_for_digest(requested.image_digest),
            gpu_type=requested.gpu_sku,
            gpu_count=requested.gpu_count,
            leak_log=run_dir / "leaked_pods.json",
            network_volume_id=site.volume_id if site else None,
            data_center_id=site.data_center_id if site else None,
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
        engines={
            version: VllmEngineAdapter(
                engine_version=version, rules=load_rules(RULES_DIR, "vllm", version)
            )
            for version in RUNNER_IMAGES
        },
        evaluator=evaluator if evaluator is not None else DeterministicScorer(),
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
            data_center_id=site.data_center_id if site else None,
            max_wait_seconds=int(os.environ.get("APRON_CAPACITY_WAIT", str(2 * 3600))),
            log=JsonlLedger(run_dir / "capacity-waits.jsonl"),
        ),
    )


def v3_evaluator(plans: Iterable[SolutionPlan]) -> DeploymentCheckScorer:
    """Suite v3's scorer, knowing each runnable plan's deployment facts."""
    return DeploymentCheckScorer.for_plans(p for p in plans if p.status == "planned")


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
