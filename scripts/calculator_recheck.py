"""The calculator at run time, the calculator now, and the measurement, per boot.

For every healthy memory boot of the cohort: the weight and activation bytes
the stored PlanningClaim predicted when the run planned it, what the current
calculator predicts for the same checkpoint and GPU (stored bytes from
``tests/fixtures/cohort/weight-bytes.json``, recorded from the safetensors
headers at the solution's revision), and what vLLM measured.  GPU-free, no
network.  Writes ``_dev_notes/cohort-run/calculator-recheck.json``, which the
findings render.

    uv run python scripts/calculator_recheck.py
"""

from __future__ import annotations

import json
import sys

from apron.adapters.backends.vllm_quantization import default_max_num_batched_tokens
from apron.application.orchestration.plan_pipeline import _resolve_weight_bytes
from apron.domain.mechanisms import CalculatorInput, ComponentMechanism, TextWorkload
from apron.domain.mechanisms.calculator import _activation_estimate, _per_gpu
from apron.interfaces.cohort_root import REPO, RUN_DIR, hardware_for, load_cohort_run

FIXTURE = REPO / "tests" / "fixtures" / "cohort" / "weight-bytes.json"
OUT = RUN_DIR / "calculator-recheck.json"


class _Recorded:
    def __init__(self, entry: dict) -> None:
        self._entry = entry

    def _tensor_bytes(self, model_id: str, revision: str) -> dict[str, int]:
        lm_head = int(self._entry["lm_head_bytes"])
        rest = {"rest": int(self._entry["total_bytes"]) - lm_head}
        return {**rest, "lm_head.weight": lm_head} if lm_head else rest


class _Observation:
    resolved_revision = "recorded"
    publisher_metadata = None


def main() -> int:
    records = load_cohort_run().records
    recorded = json.loads(FIXTURE.read_text())
    claims = {c.solution_fingerprint: c for c in records.claims.values()}
    rows, seen = [], set()
    for digest, report in sorted(records.reports.items()):
        entry = records.solutions.get(report.solution_fingerprint or "")
        if entry is None or report.claim_scope != "memory" or report.boot_outcome != "healthy":
            continue
        gpu, tp = entry.requested_execution.gpu_sku, entry.deployment_plan.tensor_parallel
        key = (entry.model_id, gpu, tp)
        if key in seen or entry.model_id not in recorded:
            continue
        seen.add(key)
        fixture = recorded[entry.model_id]
        config = {"tie_word_embeddings": fixture["tie_word_embeddings"]}
        weights_now = _per_gpu(
            _resolve_weight_bytes(_Recorded(fixture), entry.model_id, _Observation(), config), tp
        )
        hardware = hardware_for(gpu)
        tokens = default_max_num_batched_tokens(hardware.total_memory_bytes, gpu)
        inputs = CalculatorInput(
            mechanism=ComponentMechanism(mechanism="autoregressive_decode", role="decoder"),
            workload=TextWorkload(kind="text", input_length=512, output_length=128),
            artifact_metadata={},
            hardware=hardware,
            execution_spec_data={"max_num_batched_tokens": tokens},
        )
        metadata = {k: fixture[k] for k in ("vocab_size", "hidden_size", "torch_dtype")}
        claim = claims.get(entry.solution_fingerprint)
        proposed = claim.proposed_configuration if claim else {}
        rows.append(
            {
                "model": entry.model_id,
                "gpu": gpu,
                "tensor_parallel": tp,
                "profiled_tokens": tokens,
                "weights_at_run": proposed.get("weight_memory_bytes"),
                "weights_now": weights_now,
                "weights_measured": report.model_weight_memory,
                "activation_at_run": proposed.get("activation_estimate_bytes"),
                "activation_now": _activation_estimate(metadata, inputs, tp),
                "activation_measured": report.transient_peak_headroom,
                "record": digest,
            }
        )
    OUT.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} boots -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
