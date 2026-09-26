"""A complete synthetic cohort run on disk — for the exit gate and findings tests.

The real orchestrator, fix-proof protocol, budget ledger, record store,
identity manifest and file rule repository, driven by the mock provider and
engine in ``_cohort_fakes``: nine seed candidates (two prediction errors),
then the six failure classes.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from conftest import FakeDiagnosisEngine
from unit._cohort_fakes import (
    CATALOG,
    MODELS,
    accepted_inputs,
    correction_context,
    fix_plan_solution,
    ports,
    solution,
)

from apron.adapters.backends.ledger_file import JsonlLedger
from apron.adapters.backends.rule_loader import load_rules
from apron.adapters.backends.rule_repository import FileRuleRepository
from apron.application.orchestration.cohort import run_cohort
from apron.application.orchestration.remediation import SIX_CLASSES, FixProofPorts, prove_all
from apron.application.orchestration.scheduler import (
    CandidateSeed,
    Coverage,
    rank_candidates,
    ranking_record,
)

REPO = Path(__file__).resolve().parents[2]
RULE_VERSION_DIR = REPO / "rules" / "vllm-v0.29"

_4090, _L4, _A6000, _H100 = (
    "NVIDIA GeForce RTX 4090",
    "NVIDIA L4",
    "NVIDIA RTX A6000",
    "NVIDIA H100 80GB HBM3",
)
_SEED = (
    ("Qwen/Qwen3-1.7B", _4090, "small", "consumer", ["gqa"]),
    ("Qwen/Qwen3-1.7B", _L4, "small", "datacenter", ["gqa"]),
    ("google/gemma-2-2b-it", _4090, "small", "consumer", ["gqa", "sliding_window"]),
    ("mistralai/Mistral-7B-Instruct-v0.3", _4090, "mid", "consumer", ["gqa", "sliding_window"]),
    ("deepseek-ai/DeepSeek-V2-Lite", _A6000, "large", "professional", ["mla", "moe"]),
    ("JunHowie/Qwen3-8B-GPTQ-Int4", _4090, "mid", "consumer", ["gqa", "gptq_int4"]),
    ("Qwen/Qwen3-32B", _H100, "large", "datacenter", ["gqa"]),
    ("Qwen/Qwen3-32B", _4090, "large", "consumer", ["gqa"]),  # predicted infeasible
    ("state-spaces/mamba-2.8b-hf", _4090, "mid", "consumer", ["ssm"]),  # unknown mechanism
)
_CLASS_LOGS = {
    1: "ERROR Failed to load model - not enough GPU memory. (original error: CUDA out of "
    "memory. Tried to allocate 1.50 GiB. GPU 0 has a total capacity of 23.52 GiB of which "
    "1.12 GiB is free.)",
    2: "ValueError: To serve at least one request with the model's max seq len (40960), "
    "(5.62 GiB KV cache is needed, which is larger than the available KV cache memory "
    "(2.40 GiB). Based on the available memory, the estimated maximum model length is 17472.\n"
    "RuntimeError: Engine core initialization failed. See root cause above.",
    3: "ValueError: User-specified max_model_len (999999) is greater than the derived "
    "max_model_len (max_position_embeddings=32768 or model_max_length=None in model's "
    "config.json).",
    4: "ValueError: The model type 'gemma2' does not support float16. Reason: Numerical "
    "instability. Please use bfloat16 or float32 instead.",
    5: "ValueError: Total number of attention heads (32) must be divisible by tensor "
    "parallel size (3).",
    6: "ValueError: The quantization method fp_quant is not supported for the current GPU. "
    "Minimum capability: 100. Current capability: 90.",
}


_MECHANISM = {
    "deepseek-ai/DeepSeek-V2-Lite": "mla_decode",
    "state-spaces/mamba-2.8b-hf": "unknown",
}


def _candidate(model: str, gpu: str, size: str, hw: str, features: list[str]) -> CandidateSeed:
    """The seed row the scheduler ranks; prediction errors as in cohort/phase-1b-seed.json."""
    infeasible = model == "Qwen/Qwen3-32B" and gpu == "NVIDIA GeForce RTX 4090"
    return CandidateSeed(
        model_id=model,
        gpu_sku=gpu,
        size_class=size,  # type: ignore[arg-type]
        hardware_class=hw,  # type: ignore[arg-type]
        mechanism=_MECHANISM.get(model, "autoregressive_decode"),
        weight_gb=round(MODELS[model]["weights"] / 1e9, 2),
        quantized="gptq_int4" in features,
        prediction_error=infeasible or model in ("state-spaces/mamba-2.8b-hf",),
        features=tuple(features),
    )


def _scenario(plan: Any, target: Any) -> str:
    for case in SIX_CLASSES:
        if plan is case.broken_plan:
            return _CLASS_LOGS[case.failure_class]
    return "healthy"


def build_synthetic_run(run_dir: Path) -> Path:
    """A complete run in *run_dir*; returns its rule directory."""
    rules_dir = run_dir / "rules" / "vllm-v0.29"
    shutil.copytree(RULE_VERSION_DIR, rules_dir)
    cohort_ports, _, _, _ = ports(
        run_dir,
        _scenario,
        events=JsonlLedger(run_dir / "events.jsonl"),
        ledger=JsonlLedger(run_dir / "ledger.jsonl"),
        identities=JsonlLedger(run_dir / "solutions.jsonl"),
    )
    inputs = accepted_inputs()
    seeds = [_candidate(*row) for row in _SEED]
    rates = {gpu: rate for gpu, (_, rate) in CATALOG.items()}
    ranking = rank_candidates(seeds, Coverage(), measured=[], remaining_budget=100.0, rates=rates)
    (run_dir / "cohort-ranking.json").write_text(
        json.dumps(
            {
                **ranking_record(
                    ranking,
                    candidates=seeds,
                    existing=Coverage(),
                    measured=[],
                    remaining_budget=100.0,
                    rates=rates,
                ),
                "authorization": inputs.authorization.model_dump(mode="json"),
            }
        )
    )
    rows = {_candidate(*row).key: row for row in _SEED}
    plans = [
        solution(
            model,
            gpu,
            coverage={
                "size_class": size,
                "hardware_class": hw,
                "quantized": "gptq_int4" in features,
                "features": features,
            },
        )
        for model, gpu, size, hw, features in (rows[r.seed.key] for r in ranking.ranked)
    ]
    result = run_cohort(plans, inputs, cohort_ports)
    assert result.stopped is None, result.stopped
    rules = load_rules(rules_dir.parent, "vllm", "v0.29")
    prove_all(
        inputs,
        cohort_ports,
        FixProofPorts(
            plan_solution=fix_plan_solution,
            diagnosis_engine=FakeDiagnosisEngine(),
            rules=rules,
            rule_repository=FileRuleRepository(rules_dir),
            correction_context=correction_context,
            hardware_for=lambda sku: CATALOG[sku][0],
        ),
    )
    (run_dir / "notebook.md").write_text("# Run notebook\n\nsynthetic run\n", "utf-8")
    return rules_dir
