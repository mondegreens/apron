"""Mock target, engine and evaluator for GPU-free cohort and fix-proof dry runs.

Only the provider and the GPU are simulated.  The orchestrator, the budget
ledger, record building, validation, storage, qualification verdicts and the
diagnosis pipeline under test are the real code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from apron.adapters.backends.local_store import LocalRecordStore
from apron.adapters.backends.vllm_engine import BootResult, VllmEngineAdapter
from apron.application.orchestration.budget import BudgetTracker
from apron.application.orchestration.cohort import (
    AcceptedInputs,
    CohortPorts,
    SolutionPlan,
    predicted_feasible,
)
from apron.application.orchestration.correction import CatalogEntry, CorrectionContext
from apron.application.orchestration.evidence import protocol_template, solution_fingerprint
from apron.domain.artifacts.identity import ArtifactIdentity
from apron.domain.canonical import canonicalize, digest_hex
from apron.domain.mechanisms.model_spec_builder import build_model_spec
from apron.domain.schemas.authority import AuthorizationEnvelope, DecisionRequest
from apron.domain.schemas.models import ArtifactSpec
from apron.domain.schemas.primitives import HardwareSpec
from apron.domain.schemas.records import DiagnosisRule
from apron.domain.schemas.solutions import (
    DeploymentPlan,
    EvaluationProtocol,
    PlanningClaim,
    RequestedExecutionSpec,
)
from apron.domain.schemas.tasks import ApplicationSpec, ServingWorkloadSpec, TaskSuiteSpec

RUN = Path(__file__).parents[1] / "fixtures" / "phase-1a-run"
IMAGE = "sha256:" + "c" * 64
GIB = 1 << 30

CATALOG = {
    "NVIDIA GeForce RTX 4090": (
        HardwareSpec(
            gpu_sku="NVIDIA GeForce RTX 4090",
            total_memory_bytes=24 * GIB,
            compute_capability="8.9",
        ),
        0.74,
    ),
    "NVIDIA RTX A6000": (
        HardwareSpec(
            gpu_sku="NVIDIA RTX A6000", total_memory_bytes=48 * GIB, compute_capability="8.6"
        ),
        0.53,
    ),
    "NVIDIA L4": (
        HardwareSpec(gpu_sku="NVIDIA L4", total_memory_bytes=24 * GIB, compute_capability="8.9"),
        0.49,
    ),
    "NVIDIA H100 80GB HBM3": (
        HardwareSpec(
            gpu_sku="NVIDIA H100 80GB HBM3", total_memory_bytes=80 * GIB, compute_capability="9.0"
        ),
        3.49,
    ),
    "NVIDIA B200": (
        HardwareSpec(
            gpu_sku="NVIDIA B200", total_memory_bytes=179 * GIB, compute_capability="10.0"
        ),
        6.79,
    ),
}

# Model facts for the fakes: config.json fields the pipeline reads, and weight bytes.
MODELS: dict[str, dict[str, Any]] = {
    "Qwen/Qwen3-1.7B": {
        "weights": 4_060_000_000,
        "config": {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "max_position_embeddings": 40960,
            "torch_dtype": "bfloat16",
        },
    },
    "Qwen/Qwen3-8B": {
        "weights": 16_380_000_000,
        "config": {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 40960,
            "torch_dtype": "bfloat16",
        },
    },
    "Qwen/Qwen3-14B": {
        "weights": 29_540_000_000,
        "config": {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "max_position_embeddings": 40960,
            "torch_dtype": "bfloat16",
        },
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "weights": 14_500_000_000,
        "config": {
            "architectures": ["MistralForCausalLM"],
            "model_type": "mistral",
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 32768,
            "torch_dtype": "bfloat16",
        },
    },
    "google/gemma-2-2b-it": {
        "weights": 5_230_000_000,
        "config": {
            "architectures": ["Gemma2ForCausalLM"],
            "model_type": "gemma2",
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "max_position_embeddings": 8192,
            "torch_dtype": "float32",
        },
    },
    "ISTA-DASLab/Qwen3-0.6B-FPQuant-RTN-MXFP4": {
        "weights": 550_000_000,
        "config": {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "max_position_embeddings": 40960,
            "quantization_config": {"quant_method": "fp_quant"},
            "torch_dtype": "bfloat16",
        },
    },
    "state-spaces/mamba-2.8b-hf": {
        "weights": 11_070_000_000,
        "config": {
            "architectures": ["MambaForCausalLM"],
            "model_type": "mamba",
            "hidden_size": 2560,
            "num_hidden_layers": 64,
        },
    },
    "Qwen/Qwen3-32B": {
        "weights": 65_520_000_000,
        "config": {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "max_position_embeddings": 40960,
            "torch_dtype": "bfloat16",
        },
    },
}


class StepClock:
    """Deterministic clock: every reading is 30 s after the previous one."""

    def __init__(self) -> None:
        self._t = datetime(2026, 9, 26, tzinfo=UTC)

    def now(self) -> datetime:
        self._t += timedelta(seconds=30)
        return self._t


class CountingIds:
    def __init__(self) -> None:
        self._n = 0

    def generate(self) -> str:
        self._n += 1
        return f"id-{self._n:04d}"


class MemoryLog:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def append(self, entry: dict[str, Any]) -> None:
        self.entries.append(dict(entry))

    def read_all(self) -> list[dict[str, Any]]:
        return [dict(e) for e in self.entries]


# Outcome of booting a plan: "healthy", a vLLM log text (model failure), or a
# ("harness", log) tuple for an environment failure.
Scenario = Callable[[DeploymentPlan, "FakeTarget"], Any]


@dataclass
class FakeTarget:
    requested: RequestedExecutionSpec
    kind: str = "rented-provider"
    operator: str = "apron"
    provider: str = "runpod"
    pod_id: str | None = None
    torn_down: bool = False
    provisioned_env: dict[str, str] = field(default_factory=dict)

    def provision(self, env: dict[str, str] | None = None) -> dict[str, Any]:
        self.pod_id = f"pod-{self.requested.gpu_sku[-4:]}-{id(self) % 1000}"
        self.provisioned_env = dict(env or {})
        return {"pod_id": self.pod_id}

    @property
    def hardware(self) -> HardwareSpec:
        return CATALOG[self.requested.gpu_sku][0]

    @property
    def execution_fingerprint(self) -> str:
        return digest_hex(
            canonicalize({"gpu": self.requested.gpu_sku, "n": self.requested.gpu_count})
        )

    @property
    def proxy_url(self) -> str:
        return f"https://{self.pod_id}-8000.proxy.runpod.net"

    def pod_reported_cost(self) -> float:
        return 0.123

    def teardown(self) -> None:
        self.torn_down = True


class FakeEngine:
    """Simulates vLLM on a pod; everything else is the real adapter code."""

    def __init__(self, scenario: Scenario, *, token_in_environ: bool = False) -> None:
        self._scenario = scenario
        self._token_in_environ = token_in_environ
        self.booted: list[DeploymentPlan] = []

    def runner_supports_token_isolation(self, target: Any) -> bool:
        return True

    def prepare_boot(self, target: Any) -> dict[str, Any]:
        return {"clean": True, "port_8000_busy": False, "gpu_memory_used_mib": [0]}

    def download_weights(self, target: Any, model_id: str) -> dict[str, Any]:
        return {"ok": True, "seconds": 60.0, "output_tail": ""}

    def boot(self, plan: DeploymentPlan, target: Any, *, health_timeout: int = 600) -> BootResult:
        self.booted.append(plan)
        result = self._scenario(plan, target)
        if result == "healthy":
            return BootResult(True, "", "vllm serve", 90.0)
        if isinstance(result, tuple):
            return BootResult(False, result[1], "vllm serve", 5.0)
        return BootResult(False, str(result), "vllm serve", 40.0)

    def verify(
        self, plan: DeploymentPlan, target: Any, health_timeout: int = 600
    ) -> dict[str, Any]:
        model = plan.resource_allocation.get("model_id", "")
        weights = MODELS.get(model, {"weights": 1_000_000_000})["weights"]
        return {
            "initial_total_memory": target.hardware.total_memory_bytes,
            "initial_free_memory": target.hardware.total_memory_bytes - GIB // 2,
            "requested_memory": int(target.hardware.total_memory_bytes * 0.9),
            "model_weight_memory": weights,
            "persistent_consumption": weights + 2 * GIB,
            "transient_peak_headroom": GIB,
            "non_pytorch_increase": GIB // 4,
            "cuda_graph_estimate": GIB // 2,
            "cuda_graph_applied": True,
            "cuda_graph_actual": GIB // 2,
            "available_kv_cache_memory": 4 * GIB,
            "safety_buffer": GIB,
            "profiling_shape": {"max_num_batched_tokens": 8192},
            "execution_fingerprint": target.execution_fingerprint,
            "target_kind": target.kind,
        }

    def token_environ_check(self, target: Any) -> dict[str, Any]:
        count = 1 if self._token_in_environ else 0
        return {"command": "pgrep", "processes": {101: count}, "token_free": count == 0}

    def benchmark_serving(self, target: Any, **_: Any) -> dict[str, Any]:
        return {
            "num_prompts": 50,
            "max_concurrency": 4,
            "completed": 50,
            "failed": 0,
            "request_throughput": 2.5,
            "output_throughput": 320.0,
            "p99_ttft_ms": 850.0,
            "p99_tpot_ms": 35.0,
            "p50_ttft_ms": 300.0,
        }

    def build_serving_report(self, bench: dict[str, Any]) -> dict[str, Any]:
        return VllmEngineAdapter.build_serving_report(bench)


class FakeEvaluator:
    """Answers every case correctly unless told which case ids fail."""

    harness_name = "deterministic_exact_match"
    harness_version = "0.1"

    def __init__(
        self,
        wrong: frozenset[str] = frozenset(),
        transport_fail_once: frozenset[str] = frozenset(),
    ) -> None:
        self._wrong = wrong
        self._transport = set(transport_fail_once)

    def accepts(self, protocol: dict[str, Any]) -> bool:
        return True

    def prepare(self, protocol: dict[str, Any]) -> dict[str, Any]:
        return dict(protocol)

    def execute(self, protocol: dict[str, Any], endpoint: str) -> list[dict[str, Any]]:
        out = []
        for case in protocol["cases"]:
            cid = case["id"]
            if cid in self._transport:
                self._transport.discard(cid)
                out.append(
                    {
                        "case_id": cid,
                        "score": 0,
                        "accepted": False,
                        "status": "failed",
                        "error": "ReadTimeout",
                    }
                )
            elif cid in self._wrong:
                out.append(
                    {
                        "case_id": cid,
                        "output": "?",
                        "score": 0,
                        "accepted": False,
                        "status": "completed",
                    }
                )
            else:
                out.append(
                    {
                        "case_id": cid,
                        "output": case["expected"],
                        "score": 1,
                        "accepted": True,
                        "status": "completed",
                    }
                )
        return out

    def collect(self, attempts: list[dict[str, Any]]) -> dict[str, Any]:
        return {"total_attempts": len(attempts)}


def envelope(**overrides: Any) -> AuthorizationEnvelope:
    data: dict[str, Any] = {
        "permitted_action_classes": ("gpu_execution", "diagnosis_classification"),
        "permitted_providers": ("runpod",),
        "task_data_destinations": ("api.anthropic.com",),
        "hard_target_constraints": {"cloud_type": "SECURE"},
        "maximum_spend": 100.0,
    }
    data.update(overrides)
    return AuthorizationEnvelope.model_validate(data)


def accepted_inputs(**envelope_overrides: Any) -> AcceptedInputs:
    return AcceptedInputs(
        request=DecisionRequest.model_validate_json((RUN / "decision-request.json").read_text()),
        task_suite=TaskSuiteSpec.model_validate_json((RUN / "task-suite-spec.json").read_text()),
        application=ApplicationSpec.model_validate_json(
            (RUN / "application-spec.json").read_text()
        ),
        protocol_template=protocol_template(
            EvaluationProtocol.model_validate_json((RUN / "evaluation-protocol.json").read_text())
        ),
        serving_workload=ServingWorkloadSpec.model_validate_json(
            (RUN / "serving-workload-spec.json").read_text()
        ),
        authorization=envelope(**envelope_overrides),
    )


def claim_for(model_id: str, hardware: HardwareSpec) -> PlanningClaim:
    """A calculator-shaped claim; unknown for attention-free models (F8)."""
    info = MODELS[model_id]
    spec = build_model_spec(info["config"], repository=model_id)
    if spec.components[0].mechanism == "unknown":
        return PlanningClaim(
            producer="apron-calculator",
            version="0.1",
            input_fingerprint="1220" + "11" * 32,
            proposed_configuration={"status": "unknown"},
            claim_scope="memory",
            producer_epistemic_tier="UNKNOWN",
        )
    weights = info["weights"]
    total = weights + 3 * GIB
    return PlanningClaim(
        producer="apron-calculator",
        version="0.1",
        input_fingerprint=digest_hex(canonicalize({"m": model_id, "hw": hardware.gpu_sku})),
        proposed_configuration={
            "weight_memory_bytes": weights,
            "total_required_bytes": total,
            "available_kv_cache_bytes": int(hardware.total_memory_bytes * 0.9) - total,
        },
        claim_scope="memory",
        producer_epistemic_tier="MECHANISM",
    )


def plan_solution(
    plan: DeploymentPlan, label: str, *, check_feasibility: bool = True
) -> SolutionPlan:
    """GPU-free step 1 for a given plan; keeps the plan *object* (identity guard).

    The fix proof passes ``check_feasibility=False``: it boots a broken plan on
    purpose, even when the calculator predicts it cannot fit.
    """
    model_id = plan.resource_allocation["model_id"]
    gpu = plan.resource_allocation["gpu_sku"]
    count = int(plan.resource_allocation.get("gpu_count", "1"))
    info = MODELS[model_id]
    spec = build_model_spec(info["config"], repository=model_id)
    hardware, rate = CATALOG[gpu]
    requested = RequestedExecutionSpec(
        provider="runpod", gpu_sku=gpu, gpu_count=count, cloud_type="SECURE", image_digest=IMAGE
    )
    claim = claim_for(model_id, hardware)
    status = "planned"
    if claim.proposed_configuration.get("status") == "unknown":
        status = "unknown"
    elif check_feasibility and not predicted_feasible(claim, hardware.total_memory_bytes, count):
        status = "infeasible"
    return SolutionPlan(
        label=label,
        model_id=model_id,
        model_spec=spec,
        model_config=info["config"],
        plan=plan,
        requested=requested,
        claim=claim,
        solution_fp=solution_fingerprint(spec, plan, requested),
        status=status,  # type: ignore[arg-type]
        chat_template="{% if enable_thinking %}{% endif %}"
        if model_id.startswith("Qwen/")
        else None,
        estimate=round(rate * count * 0.5, 4),
        artifact_spec=ArtifactSpec(
            identity=ArtifactIdentity(content_digest=digest_hex(canonicalize(info["config"])))
        ),
    )


def solution(
    model_id: str, gpu: str, *, count: int = 1, label: str | None = None, **plan_fields: Any
) -> SolutionPlan:
    plan = DeploymentPlan(
        dtype=plan_fields.pop("dtype", "bfloat16"),
        resource_allocation={"model_id": model_id, "gpu_sku": gpu, "gpu_count": str(count)},
        **plan_fields,
    )
    return plan_solution(plan, label or f"{model_id}@{gpu}x{count}")


def ports(
    tmp_path: Path,
    scenario: Scenario,
    *,
    authorized: float = 100.0,
    evaluator: FakeEvaluator | None = None,
) -> tuple[CohortPorts, FakeEngine, list[FakeTarget], MemoryLog]:
    engine = FakeEngine(scenario)
    targets: list[FakeTarget] = []
    events = MemoryLog()
    clock = StepClock()

    def target_factory(requested: RequestedExecutionSpec) -> FakeTarget:
        target = FakeTarget(requested)
        targets.append(target)
        return target

    budget = BudgetTracker(authorized=authorized, ledger=MemoryLog(), clock=clock)
    return (
        CohortPorts(
            target_factory=target_factory,
            engine=engine,
            evaluator=evaluator or FakeEvaluator(),
            store=LocalRecordStore(tmp_path / "records"),
            budget=budget,
            clock=clock,
            ids=CountingIds(),
            events=events,
            provision_env=lambda sp: {"VLLM_LOGGING_LEVEL": "DEBUG"},
            hourly_rate=lambda requested: CATALOG[requested.gpu_sku][1] * requested.gpu_count,
        ),
        engine,
        targets,
        events,
    )


def fix_plan_solution(plan: DeploymentPlan, label: str) -> SolutionPlan:
    return plan_solution(plan, label, check_feasibility=False)


def correction_context(sp: SolutionPlan) -> CorrectionContext:
    return CorrectionContext(
        catalog=tuple(CatalogEntry(hw, rate) for hw, rate in CATALOG.values()),
        predicted_total_bytes=sp.predicted_total_bytes,
    )


class MemoryRuleRepository:
    def __init__(self, rules: list[dict[str, Any]]) -> None:
        self.rules = {r["error_family"]: DiagnosisRule.model_validate(r) for r in rules}
        self.written: list[DiagnosisRule] = []

    def current(self, family: str) -> DiagnosisRule:
        return self.rules[family]

    def write_version(self, rule: DiagnosisRule) -> str:
        self.written.append(rule)
        self.rules[rule.error_family] = rule
        return digest_hex(canonicalize(rule.model_dump(mode="json")))
