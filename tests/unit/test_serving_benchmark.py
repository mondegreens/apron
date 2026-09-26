"""Serving measurement adapter: vllm bench serve command and report mapping."""

from __future__ import annotations

import json
from typing import Any

from apron.adapters.backends.vllm_engine import VllmEngineAdapter
from apron.application.orchestration.serving import evaluate_serving_slos
from apron.domain.schemas.records import VerificationReport
from apron.domain.schemas.tasks import ServingWorkloadSpec

# Shape of a vLLM v0.29 --save-result file (vllm/benchmarks/serve.py:1275,1372,2236).
BENCH = {
    "num_prompts": 50,
    "max_concurrency": 4,
    "completed": 50,
    "failed": 0,
    "request_throughput": 3.1,
    "output_throughput": 397.0,
    **{
        f"p{p}_{m}_ms": float(p) for p in (50, 90, 95, 99) for m in ("ttft", "tpot", "itl", "e2el")
    },
}


class _Target:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def execute(self, command: str) -> dict[str, Any]:
        self.commands.append(command)
        if command.startswith("cat /workspace/bench/"):
            return {"stdout": json.dumps(BENCH) + "\n", "exit_code": 0}
        return {"stdout": "", "exit_code": 0}


def test_benchmark_command_uses_declared_workload() -> None:
    target = _Target()
    VllmEngineAdapter().benchmark_serving(
        target, model_id="Qwen/Qwen3-1.7B", input_len=512, output_len=128, concurrency=4
    )
    command = target.commands[0]
    for flag in (
        "--dataset-name random",
        "--random-input-len 512 --random-output-len 128",
        "--max-concurrency 4",
        "--num-prompts 50 --num-warmups 5",
        "--percentile-metrics ttft,tpot,itl,e2el",
        "--metric-percentiles 50,90,95,99",
        "--save-result",
        "--model Qwen/Qwen3-1.7B",
        "--tokenizer /workspace/models/Qwen/Qwen3-1.7B",
    ):
        assert flag in command, flag


def test_report_fields_validate_and_feed_the_slo_verdict() -> None:
    fields = VllmEngineAdapter.build_serving_report(BENCH)
    assert fields["serving_latency_ms"]["p99_ttft_ms"] == 99.0
    report = VerificationReport(
        target_kind="rented-provider",
        operator="apron",
        execution_fingerprint="1220" + "ab" * 32,
        claim_scope="serving_performance",
        production_mode=False,
        reason="serving_measurement",
        lifecycle="observed",
        **fields,
    )
    verdict = evaluate_serving_slos(report, ServingWorkloadSpec(p99_ttft_ms=2000, p99_tpot_ms=100))
    assert verdict.verdict == "pass"


def test_missing_result_raises() -> None:
    import pytest

    class _Empty(_Target):
        def execute(self, command: str) -> dict[str, Any]:
            return {"stdout": "", "exit_code": 1}

    with pytest.raises(RuntimeError, match="no result"):
        VllmEngineAdapter().benchmark_serving(
            _Empty(), model_id="m", input_len=1, output_len=1, concurrency=1
        )
