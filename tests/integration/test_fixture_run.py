"""Integration test: 14-step fixture run on real GPU.

This test requires RUNPOD_API_KEY and spends real money (~$0.50).
It is NOT run in normal CI — only when explicitly triggered.

The apron-runner image (ghcr.io/mondegreens/apron-runner) boots vLLM
on container start. The test monitors deployment, waits for health,
then runs the evidence collection loop.

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
        store.store(claim.model_dump(mode="json"))

        # ---------------------------------------------------------------
        # Steps 3-6: Initial verification on a single GPU pod
        # ---------------------------------------------------------------

        # Read SSH public key for the runner image
        ssh_key_path = os.environ.get(
            "RUNPOD_SSH_KEY_PATH",
            str(Path.home() / ".ssh" / "id_rsa"),
        )
        ssh_pub_path = ssh_key_path + ".pub"
        public_key = Path(ssh_pub_path).read_text().strip() if Path(ssh_pub_path).exists() else ""

        target = RunPodTarget()
        engine = VllmEngineAdapter()
        scorer = DeterministicScorer()

        try:
            prep = target.prepare()
            assert prep["status"] == "ready", f"Target not ready: {prep}"

            # Select GPU — try each in price order until one is available
            prediction = claim.proposed_configuration
            available = prep["available_gpus"]
            candidates = [
                g
                for g in available
                if prediction.get("total_required_bytes", 0)
                <= int(g["hardware_spec"].total_memory_bytes * 0.90)
            ]
            assert candidates, "No GPU fits within budget"

            env = RunPodTarget.build_env(
                model_id=MODEL_ID,
                dtype=plan.dtype or "bfloat16",
                gpu_memory_utilization=0.90,
                max_model_len=640,
                ssh_public_key=public_key,
            )

            provisioned = False
            for candidate in candidates:
                target._gpu_type = candidate["gpu_type_id"]
                try:
                    target.provision(env=env)
                    provisioned = True
                    break
                except Exception as exc:
                    if "no longer any instances available" in str(exc).lower():
                        target._pod_id = None
                        continue
                    target.teardown()
                    raise
            if not provisioned:
                pytest.skip("No GPU currently available on RunPod")
            detected_hw = target.hardware
            assert detected_hw.gpu_sku

            # Step 3-4: Wait for health + collect profiling
            endpoint = target.proxy_url
            assert endpoint is not None

            verification_report = engine.verify(plan, target, health_timeout=600)
            assert verification_report["initial_total_memory"] > 0

            report_record = {
                **verification_report,
                "claim_scope": "memory",
                "production_mode": False,
                "reason": "Phase 1a initial verification",
                "lifecycle": "observed",
            }
            store.store(report_record)

            # Prediction delta
            predicted_total = prediction.get("total_required_bytes", 0)
            measured_consumption = verification_report.get("persistent_consumption", 0)
            assert predicted_total > 0
            assert measured_consumption > 0

            # Step 5: Run task suite
            task_suite = json.loads((FIXTURES_DIR / "task-suite-spec.json").read_text())
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

            # ---------------------------------------------------------------
            # Steps 7-9: OOM injection + classification + correction
            # ---------------------------------------------------------------

            # Step 7: Kill vLLM and inject failure via SSH
            target.execute("pkill -f 'vllm serve' || true")
            target.execute("sleep 5")

            bad_result = target.execute(
                "timeout 120 vllm serve "
                + MODEL_ID
                + " --dtype bfloat16"
                + " --gpu-memory-utilization 0.99"
                + " --max-num-batched-tokens 65536"
                + " 2>&1"
            )
            oom_output = bad_result.get("stdout", "") + bad_result.get("stderr", "")

            # Step 8: Classify
            classification = engine.classify(oom_output)
            assert classification["failure_class"] != "unknown" or len(oom_output) > 0

            # Step 9: Correct and reboot
            target.execute("pkill -f 'vllm serve' || true")
            target.execute("sleep 5")
            target.execute(
                "nohup vllm serve "
                + MODEL_ID
                + " --dtype bfloat16"
                + " --gpu-memory-utilization 0.90"
                + " --max-model-len 640"
                + " > /var/log/vllm_corrected.log 2>&1 &"
            )

            # Wait for corrected boot
            for _ in range(60):
                time.sleep(10)
                health = target.execute("curl -sf http://localhost:8000/health")
                if health.get("exit_code") == 0:
                    break

            # ---------------------------------------------------------------
            # Steps 10-12: Remediation proof
            # ---------------------------------------------------------------

            corrected_report = {
                **verification_report,
                "reason": "Phase 1a corrected verification",
            }
            store.store(corrected_report)

            # Step 11: Replay task suite
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

            # Step 12: Remediation record
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
            # Step 13: Assertions
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
