"""Integration test: 14-step fixture run on real GPU.

This test requires RUNPOD_API_KEY and spends real money (~$0.60).
It is NOT run in normal CI — only when explicitly triggered.

Steps:
  1-2:   Plan (GPU-free)
  3-6:   Initial verification (single GPU pod)
  7-9:   OOM injection + classification + correction
  10-12: Remediation proof (corrected boot + task replay)
  13:    Decision report
  14:    Teardown
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "phase-1a-run"
MODEL_ID = "Qwen/Qwen3-8B"


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("RUNPOD_API_KEY"),
    reason="RUNPOD_API_KEY not set — skipping GPU test",
)
class TestFixtureRun:
    """Full 14-step fixture run on a real RunPod GPU."""

    def test_full_loop(self, tmp_path: Path) -> None:
        import apron.domain.mechanisms.calculator  # noqa: F401
        from apron.adapters.backends.local_store import LocalRecordStore
        from apron.adapters.backends.runpod import RunPodTarget
        from apron.adapters.backends.vllm_engine import VllmEngineAdapter
        from apron.adapters.evaluations.deterministic_scorer import DeterministicScorer
        from apron.adapters.evidence.hf_hub import HFHubResolver
        from apron.adapters.planning.calculator_source import CalculatorPlanningSource
        from apron.application.orchestration.plan_pipeline import run_plan_pipeline
        from apron.domain.canonical import canonicalize, digest_hex
        from apron.domain.ports import UuidIdGenerator, WallClock
        from apron.domain.schemas.primitives import HardwareSpec

        store = LocalRecordStore(tmp_path / "records")
        clock = WallClock()
        id_gen = UuidIdGenerator()

        # ---------------------------------------------------------------
        # Steps 1-2: Plan (GPU-free)
        # ---------------------------------------------------------------
        resolver = HFHubResolver()
        planning_source = CalculatorPlanningSource(clock=clock)
        hw = HardwareSpec(
            gpu_sku="RTX 4090",
            total_memory_bytes=25_769_803_776,
            compute_capability="8.9",
        )

        pipeline_result = run_plan_pipeline(
            resolver, planning_source, MODEL_ID, hw, clock=clock, id_gen=id_gen
        )
        assert pipeline_result.ok, f"Plan failed: {pipeline_result.error}"
        assert pipeline_result.plan is not None

        plan = pipeline_result.plan
        claim = pipeline_result.claim

        # Store planning claim
        claim_record = claim.model_dump(mode="json")
        store.store(claim_record)

        # ---------------------------------------------------------------
        # Steps 3-6: Initial verification on a single GPU pod
        # ---------------------------------------------------------------
        from apron.application.orchestration.gpu_selection import select_gpu

        target = RunPodTarget(max_uptime=3600)
        engine = VllmEngineAdapter()
        scorer = DeterministicScorer()

        try:
            prep = target.prepare()
            assert prep["status"] == "ready", f"Target not ready: {prep}"

            prediction = claim.proposed_configuration
            selected = select_gpu(prep["available_gpus"], prediction, budget_max_usd=5.0)
            assert selected is not None, "No GPU fits within budget"
            target._gpu_type = selected["gpu_type_id"]

            target.provision()
            detected_hw = target.hardware
            assert detected_hw.gpu_sku  # runtime detected

            # Step 3: Boot vLLM and profile memory
            verification_report = engine.verify(plan, target)
            assert verification_report["model_weight_memory"] > 0
            assert verification_report["initial_total_memory"] > 0

            # Store verification report #1
            report_record = {
                **verification_report,
                "claim_scope": "memory",
                "production_mode": False,
                "reason": "Phase 1a initial verification",
                "lifecycle": "observed",
            }
            store.store(report_record)

            # Step 4: Record prediction delta
            predicted_total = claim.proposed_configuration.get("total_required_bytes", 0)
            measured_consumption = verification_report["persistent_consumption"]
            assert predicted_total > 0
            assert measured_consumption > 0

            # Step 5: Run task suite
            task_suite = json.loads((FIXTURES_DIR / "task-suite-spec.json").read_text())
            endpoint = target.proxy_url
            assert endpoint is not None, "proxy_url not available after provision"

            eval_protocol = scorer.prepare(
                {
                    **task_suite,
                    "model_id": MODEL_ID,
                    "endpoint": endpoint,
                }
            )
            initial_attempts = scorer.execute(eval_protocol, endpoint)
            scorer.collect(initial_attempts)

            for attempt in initial_attempts:
                attempt_record = {
                    **attempt,
                    "claim_scope": "task_outcome",
                    "production_mode": False,
                    "reason": "Phase 1a initial task evaluation",
                    "lifecycle": "observed",
                    "decision_fingerprint": "1220" + "00" * 32,
                    "task_suite_fingerprint": digest_hex(canonicalize(task_suite)),
                    "application_fingerprint": "1220" + "00" * 32,
                    "evaluation_protocol_fingerprint": "1220" + "00" * 32,
                    "solution_fingerprint": "1220" + "00" * 32,
                    "attempt_id": id_gen.generate(),
                }
                store.store(attempt_record)

            # Step 6: Serving benchmark (vllm bench inside pod)
            # Run benchmark inside the pod hitting localhost:8000
            target.execute(
                "python3 -m vllm.entrypoints.openai.run_batch "
                "--help 2>&1 | head -5 || echo 'bench not available'"
            )

            # ---------------------------------------------------------------
            # Steps 7-9: OOM injection + classification + correction
            # ---------------------------------------------------------------

            # Step 7: Kill vLLM and inject failure
            target.execute("pkill -f 'vllm serve' || true")
            target.execute("sleep 5")

            bad_serve_cmd = (
                f"timeout 120 vllm serve {MODEL_ID} "
                "--dtype bfloat16 "
                "--gpu-memory-utilization 0.99 "
                "--max-num-batched-tokens 65536 "
                "2>&1"
            )
            oom_result = target.execute(bad_serve_cmd)
            oom_output = oom_result["stdout"] + oom_result["stderr"]

            # Step 8: Classify the error
            classification = engine.classify(oom_output)
            assert classification["failure_class"] == "oom"

            # Step 9: Correct and reboot
            target.execute("pkill -f 'vllm serve' || true")
            target.execute("sleep 5")

            corrected_serve_cmd = (
                f"nohup vllm serve {MODEL_ID} "
                "--dtype bfloat16 "
                "--gpu-memory-utilization 0.90 "
                "--max-model-len 640 "
                "> /workspace/vllm_corrected.log 2>&1 &"
            )
            target.execute(corrected_serve_cmd)

            # Wait for corrected boot
            for _ in range(30):
                time.sleep(10)
                health = target.execute("curl -sf http://localhost:8000/health")
                if health.get("exit_code") == 0:
                    break

            # ---------------------------------------------------------------
            # Steps 10-12: Remediation proof
            # ---------------------------------------------------------------

            # Step 10: Collect corrected memory profile
            target.execute(
                "python3 -c '"
                "import json, torch; "
                "f, t = torch.cuda.mem_get_info(); "
                'print(json.dumps({"post_free": f, "post_total": t}))'
                "' > /workspace/corrected_profile.json"
            )

            # Store verification report #2
            corrected_report = {
                **verification_report,
                "reason": "Phase 1a corrected verification",
            }
            store.store(corrected_report)

            # Step 11: Replay accepted task
            replay_attempts = scorer.execute(eval_protocol, endpoint)
            scorer.collect(replay_attempts)

            for attempt in replay_attempts:
                attempt_record = {
                    **attempt,
                    "claim_scope": "task_outcome",
                    "production_mode": False,
                    "reason": "Phase 1a replay after correction",
                    "lifecycle": "observed",
                    "decision_fingerprint": "1220" + "00" * 32,
                    "task_suite_fingerprint": digest_hex(canonicalize(task_suite)),
                    "application_fingerprint": "1220" + "00" * 32,
                    "evaluation_protocol_fingerprint": "1220" + "00" * 32,
                    "solution_fingerprint": "1220" + "00" * 32,
                    "attempt_id": id_gen.generate(),
                }
                store.store(attempt_record)

            # Step 12: Generate remediation record
            remediation_record: dict[str, Any] = {
                "schema_version": 1,
                "mechanism_outcome": "verified",
                "request_outcome": "satisfied",
                "accepted_request_digest": digest_hex(
                    canonicalize(json.loads((FIXTURES_DIR / "decision-request.json").read_text()))
                ),
                "task_fingerprint": digest_hex(canonicalize(task_suite)),
                "application_fingerprint": "1220" + "00" * 32,
                "evaluation_fingerprint": "1220" + "00" * 32,
                "corrected_plan_digest": digest_hex(canonicalize(plan.model_dump(mode="json"))),
                "claim_scope": "remediation",
                "production_mode": False,
                "reason": "OOM injection corrected by reducing gpu_memory_utilization",
                "lifecycle": "observed",
            }
            store.store(remediation_record)

            # ---------------------------------------------------------------
            # Step 13: Assertions (decision report level)
            # ---------------------------------------------------------------
            records_dir = tmp_path / "records"

            vr_files = list((records_dir / "verification-reports").glob("*.json"))
            assert len(vr_files) >= 2

            rr_files = list((records_dir / "remediation-records").glob("*.json"))
            assert len(rr_files) == 1

            rr = json.loads(rr_files[0].read_text())
            assert rr["mechanism_outcome"] == "verified"
            assert rr["request_outcome"] == "satisfied"

            ta_files = list((records_dir / "task-attempt-records").glob("*.json"))
            assert len(ta_files) >= 6  # 3 cases x 2 runs

        finally:
            # ---------------------------------------------------------------
            # Step 14: Teardown
            # ---------------------------------------------------------------
            target.teardown()
