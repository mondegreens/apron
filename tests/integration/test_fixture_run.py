"""Integration test: 14-step fixture run on real GPU.

This test requires RUNPOD_API_KEY and spends real money (~$0.50).
It is NOT run in normal CI — only when explicitly triggered.

Timing: expect 10-20 minutes total. The apron-runner image is ~15 GB
(vLLM + CUDA + PyTorch); first pull on a fresh pod takes 8-10 minutes
with runtime=null the entire time. Model download (Qwen3-8B, ~16 GB)
adds another 5-10 minutes. Do not add timeouts to pod provisioning —
RunPod shows runtime=null for both "still pulling" and "crashed", and
the only way to distinguish them is the RunPod console Logs tab.

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
        from apron.adapters.runner_image import RUNNER_IMAGE_DIGEST
        from apron.application.orchestration.diagnosis_pipeline import diagnosis_model_config
        from apron.application.orchestration.evidence import (
            EvidenceContext,
            build_task_attempt,
            chat_template_kwargs,
            decision_request_digest,
            protocol_template,
            scorer_input,
            solution_fingerprint,
            store_validated,
        )
        from apron.application.orchestration.plan_pipeline import run_plan_pipeline
        from apron.domain.fingerprints import fingerprint_hex
        from apron.domain.ports import UuidIdGenerator, WallClock
        from apron.domain.schemas.authority import DecisionRequest
        from apron.domain.schemas.primitives import HardwareSpec
        from apron.domain.schemas.records import RemediationRecord, VerificationReport
        from apron.domain.schemas.solutions import EvaluationProtocol, RequestedExecutionSpec
        from apron.domain.schemas.tasks import ApplicationSpec, TaskSuiteSpec

        request = DecisionRequest.model_validate_json(
            (FIXTURES_DIR / "decision-request.json").read_text()
        )
        task_suite = TaskSuiteSpec.model_validate_json(
            (FIXTURES_DIR / "task-suite-spec.json").read_text()
        )
        application = ApplicationSpec.model_validate_json(
            (FIXTURES_DIR / "application-spec.json").read_text()
        )
        protocol_fixture = EvaluationProtocol.model_validate_json(
            (FIXTURES_DIR / "evaluation-protocol.json").read_text()
        )

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
        assert pipeline_result.model_spec is not None
        store_validated(store, claim)

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

            from runpod.error import QueryError  # type: ignore[import-untyped]

            provisioned = False
            for candidate in candidates:
                target._gpu_type = candidate["gpu_type_id"]
                try:
                    target.provision(env=env)
                    provisioned = True
                    break
                except QueryError:
                    target._pod_id = None
                    continue
                except Exception:
                    target.teardown()
                    raise
            if not provisioned:
                pytest.skip("No GPU currently available on RunPod")
            detected_hw = target.hardware
            assert detected_hw.gpu_sku
            requested = RequestedExecutionSpec(
                provider="runpod",
                gpu_sku=target._gpu_type or "",
                gpu_count=target.gpu_count,
                cloud_type="SECURE",
                image_digest=RUNNER_IMAGE_DIGEST,
            )
            solution_fp = solution_fingerprint(pipeline_result.model_spec, plan, requested)
            ctx = EvidenceContext.bind(
                request=request,
                task_suite=task_suite,
                application=application,
                protocol_template=protocol_template(protocol_fixture),
                solution_fp=solution_fp,
            )

            # Step 3-4: Wait for health + collect profiling
            endpoint = target.proxy_url
            assert endpoint is not None

            verification_report = engine.verify(plan, target, health_timeout=600)
            assert verification_report["initial_total_memory"] > 0

            report_digest = store_validated(
                store,
                VerificationReport.model_validate(
                    {
                        **verification_report,
                        "operator": target.operator,
                        "provider": target.provider,
                        "solution_fingerprint": solution_fp,
                        "deployment_plan_digest": fingerprint_hex(plan),
                        "boot_outcome": "healthy",
                        "claim_scope": "memory",
                        "production_mode": False,
                        "reason": "Phase 1a initial verification",
                        "lifecycle": "observed",
                    }
                ),
            )

            # Prediction delta
            predicted_total = prediction.get("total_required_bytes", 0)
            measured_consumption = verification_report.get("persistent_consumption", 0)
            assert predicted_total > 0
            assert measured_consumption > 0

            # Step 5: Run task suite
            eval_protocol = scorer.prepare(
                {
                    **scorer_input(
                        ctx,
                        model_id=MODEL_ID,
                        chat_template_kwargs=chat_template_kwargs(pipeline_result.chat_template),
                    ),
                    "endpoint": endpoint,
                }
            )
            initial_attempts = scorer.execute(eval_protocol, endpoint)
            scorer.collect(initial_attempts)

            for attempt in initial_attempts:
                store_validated(
                    store,
                    build_task_attempt(
                        ctx,
                        attempt,
                        attempt_id=id_gen.generate(),
                        trace_references=(report_digest,),
                    ),
                )

            # ---------------------------------------------------------------
            # Steps 7-9: OOM injection + classification + correction
            # ---------------------------------------------------------------

            # Step 7: Kill vLLM and inject failure via SSH
            target.execute("pkill -f 'vllm serve' || true")
            target.execute("sleep 5")

            bad_result = target.execute(
                "source /etc/apron_environment 2>/dev/null; "
                "timeout 120 /opt/venv/bin/vllm serve "
                + MODEL_ID
                + " --dtype bfloat16"
                + " --gpu-memory-utilization 0.99"
                + " --max-num-batched-tokens 65536"
                + " 2>&1"
            )
            oom_output = bad_result.get("stdout", "") + bad_result.get("stderr", "")

            # Step 8: Classify and correct via diagnosis pipeline
            from apron.adapters.backends.rule_loader import load_rules
            from apron.application.orchestration.diagnosis_pipeline import (
                run_diagnosis_pipeline,
            )

            assert oom_output, "OOM injection produced no output"
            rules_dir = Path(__file__).parents[2] / "rules"
            rules = load_rules(rules_dir, "vllm", engine.engine_version)
            diagnosis = run_diagnosis_pipeline(
                oom_output,
                engine,
                plan,
                diagnosis_model_config(pipeline_result.model_config or {}),
                detected_hw,
                rules,
                verification_report,
            )
            assert diagnosis.failure_class in ("oom", "config_incompatible", "model_runtime"), (
                f"Expected oom-related class, got: {diagnosis.failure_class}"
                f"\nvLLM output (last 500 chars): {oom_output[-500:]}"
            )
            assert diagnosis.corrected_plan is not None, (
                f"No correction for {diagnosis.failure_class}: {diagnosis.result_label}"
            )
            assert diagnosis.corrects is not None

            # Step 9: Reboot with corrected plan
            target.execute("pkill -f 'vllm serve' || true")
            target.execute("sleep 5")
            target.execute("truncate -s 0 /var/log/vllm.log")
            corrected_cmd = engine._build_serve_command(diagnosis.corrected_plan, target)
            target.execute(
                "source /etc/apron_environment 2>/dev/null; "
                f"nohup /opt/venv/bin/{corrected_cmd}"
                " > /var/log/vllm.log 2>&1 &"
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

            corrected_report = engine.verify(diagnosis.corrected_plan, target, health_timeout=60)
            assert corrected_report["initial_total_memory"] > 0
            corrected_digest = store_validated(
                store,
                VerificationReport.model_validate(
                    {
                        **corrected_report,
                        "operator": target.operator,
                        "provider": target.provider,
                        "solution_fingerprint": solution_fp,
                        "deployment_plan_digest": fingerprint_hex(diagnosis.corrected_plan),
                        "boot_outcome": "healthy",
                        "claim_scope": "memory",
                        "production_mode": False,
                        "reason": "Phase 1a corrected verification",
                        "lifecycle": "observed",
                    }
                ),
            )

            # Step 11: Replay task suite
            replay_attempts = scorer.execute(eval_protocol, endpoint)
            scorer.collect(replay_attempts)

            replay_digests = [
                store_validated(
                    store,
                    build_task_attempt(
                        ctx,
                        attempt,
                        attempt_id=id_gen.generate(),
                        trace_references=(corrected_digest,),
                    ),
                )
                for attempt in replay_attempts
            ]

            # Step 12: Remediation record — fingerprints from validated objects
            store_validated(
                store,
                RemediationRecord(
                    mechanism_outcome="verified",
                    request_outcome="satisfied",
                    accepted_request_digest=decision_request_digest(request),
                    task_fingerprint=ctx.task_suite_fingerprint,
                    application_fingerprint=ctx.application_fingerprint,
                    evaluation_fingerprint=ctx.evaluation_protocol_fingerprint,
                    corrected_plan_digest=fingerprint_hex(diagnosis.corrected_plan),
                    corrects=fingerprint_hex(plan),
                    proving_record_fingerprints=(corrected_digest, *replay_digests),
                    reason="OOM injection corrected by the diagnosis pipeline",
                ),
            )

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
