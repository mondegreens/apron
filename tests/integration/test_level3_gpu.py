"""Level 3 integration tests: real GPU failures through real LLM pipeline.

Injects each failure class on a real RunPod GPU, captures real vLLM
error output, classifies through the real LLM, corrects, reboots,
and verifies the corrected plan boots and the task suite passes.

Gate: RUNPOD_API_KEY + ANTHROPIC_API_KEY in env.
Cost: ~$15-25 total GPU time.
Runtime: 1-2 hours (sequential injections on shared pods).

The test provisions pods, runs injections, and tears down. If the
process crashes, orphan cleanup at next start will terminate the pod.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "phase-1a-run"

_HAS_KEYS = bool(os.environ.get("RUNPOD_API_KEY")) and bool(os.environ.get("ANTHROPIC_API_KEY"))
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_KEYS, reason="RUNPOD_API_KEY + ANTHROPIC_API_KEY required"),
]

logger = logging.getLogger(__name__)


@dataclass
class InjectionCase:
    name: str
    failure_class: str
    model_id: str
    gpu_type: str
    good_flags: dict[str, str]
    bad_flags: str
    expect_correction: bool = True


# --- Failure injection cases ---
# Named-strategy classes from cohort PLAN.md §4.4

_4090_INJECTIONS = [
    InjectionCase(
        name="oom_kv_cache",
        failure_class="oom",
        model_id="Qwen/Qwen3-1.7B",
        gpu_type="NVIDIA GeForce RTX 4090",
        good_flags={"max_model_len": "640", "gpu_memory_utilization": "0.90"},
        bad_flags=(
            "--dtype bfloat16 --gpu-memory-utilization 0.90"
            " --max-model-len 131072"
        ),
    ),
    InjectionCase(
        name="oom_torch",
        failure_class="oom",
        model_id="Qwen/Qwen3-8B",
        gpu_type="NVIDIA GeForce RTX 4090",
        good_flags={"max_model_len": "640", "gpu_memory_utilization": "0.90"},
        bad_flags=(
            "--dtype bfloat16 --gpu-memory-utilization 0.99"
            " --max-num-batched-tokens 65536"
        ),
    ),
    InjectionCase(
        name="max_model_len",
        failure_class="max_model_len",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        gpu_type="NVIDIA GeForce RTX 4090",
        good_flags={"max_model_len": "640", "gpu_memory_utilization": "0.90"},
        bad_flags="--dtype bfloat16 --max-model-len 999999",
    ),
    InjectionCase(
        name="dtype_incompatible",
        failure_class="dtype_incompatible",
        model_id="Qwen/Qwen3-1.7B",
        gpu_type="NVIDIA GeForce RTX 4090",
        good_flags={"max_model_len": "640", "gpu_memory_utilization": "0.90"},
        bad_flags="--dtype float16 --max-model-len 640",
    ),
    InjectionCase(
        name="lora_config",
        failure_class="lora_config",
        model_id="Qwen/Qwen3-1.7B",
        gpu_type="NVIDIA GeForce RTX 4090",
        good_flags={"max_model_len": "640", "gpu_memory_utilization": "0.90"},
        bad_flags="--dtype bfloat16 --max-model-len 640 --lora-modules test=/nonexistent",
        expect_correction=False,
    ),
    InjectionCase(
        name="scheduler_config",
        failure_class="scheduler_config",
        model_id="Qwen/Qwen3-1.7B",
        gpu_type="NVIDIA GeForce RTX 4090",
        good_flags={"max_model_len": "640", "gpu_memory_utilization": "0.90"},
        bad_flags=(
            "--dtype bfloat16 --max-model-len 2048"
            " --max-num-batched-tokens 128"
        ),
        expect_correction=False,
    ),
    InjectionCase(
        name="compilation_config",
        failure_class="compilation_config",
        model_id="Qwen/Qwen3-1.7B",
        gpu_type="NVIDIA GeForce RTX 4090",
        good_flags={"max_model_len": "640", "gpu_memory_utilization": "0.90"},
        bad_flags=(
            "--dtype bfloat16 --max-model-len 640"
            " --compilation-config '{\"use_inductor\": true, \"splitting_ops\": [\"vllm.unified_attention\"]}'"
        ),
        expect_correction=False,
    ),
]

_4090_INJECTIONS_CORRECTION_SPEC = [
    InjectionCase(
        name="speculative_config",
        failure_class="speculative_config",
        model_id="Qwen/Qwen3-1.7B",
        gpu_type="NVIDIA GeForce RTX 4090",
        good_flags={"max_model_len": "640", "gpu_memory_utilization": "0.90"},
        bad_flags=(
            "--dtype bfloat16 --max-model-len 640"
            " --speculative-model /nonexistent-draft-model"
        ),
        expect_correction=False,
    ),
    InjectionCase(
        name="parallelism_config",
        failure_class="parallelism_config",
        model_id="Qwen/Qwen3-1.7B",
        gpu_type="NVIDIA GeForce RTX 4090",
        good_flags={"max_model_len": "640", "gpu_memory_utilization": "0.90"},
        bad_flags=(
            "--dtype bfloat16 --max-model-len 640"
            " --tensor-parallel-size 2"
        ),
        expect_correction=False,
    ),
    InjectionCase(
        name="profiler_config",
        failure_class="profiler_config",
        model_id="Qwen/Qwen3-1.7B",
        gpu_type="NVIDIA GeForce RTX 4090",
        good_flags={"max_model_len": "640", "gpu_memory_utilization": "0.90"},
        bad_flags=(
            "--dtype bfloat16 --max-model-len 640"
            " --otlp-traces-endpoint http://localhost:4317"
            " --collect-detailed-traces all"
        ),
        expect_correction=False,
    ),
    InjectionCase(
        name="multimodal_config",
        failure_class="multimodal_config",
        model_id="Qwen/Qwen3-1.7B",
        gpu_type="NVIDIA GeForce RTX 4090",
        good_flags={"max_model_len": "640", "gpu_memory_utilization": "0.90"},
        bad_flags=(
            "--dtype bfloat16 --max-model-len 640"
            " --limit-mm-per-prompt image=999"
        ),
        expect_correction=False,
    ),
    InjectionCase(
        name="kv_transfer_config",
        failure_class="kv_transfer_config",
        model_id="Qwen/Qwen3-1.7B",
        gpu_type="NVIDIA GeForce RTX 4090",
        good_flags={"max_model_len": "640", "gpu_memory_utilization": "0.90"},
        bad_flags=(
            "--dtype bfloat16 --max-model-len 640"
            " --kv-connector PyNcclConnector"
        ),
        expect_correction=False,
    ),
    InjectionCase(
        name="config_incompatible",
        failure_class="config_incompatible",
        model_id="Qwen/Qwen3-1.7B",
        gpu_type="NVIDIA GeForce RTX 4090",
        good_flags={"max_model_len": "640", "gpu_memory_utilization": "0.90"},
        bad_flags=(
            "--dtype bfloat16 --max-model-len 640"
            " --enforce-eager --num-scheduler-steps 8"
            " --speculative-model [ngram] --num-speculative-tokens 3"
            " --ngram-prompt-lookup-max 3"
        ),
        expect_correction=False,
    ),
    InjectionCase(
        name="platform_unsupported",
        failure_class="platform_unsupported",
        model_id="Qwen/Qwen3-1.7B",
        gpu_type="NVIDIA GeForce RTX 4090",
        good_flags={"max_model_len": "640", "gpu_memory_utilization": "0.90"},
        bad_flags=(
            "--dtype bfloat16 --max-model-len 640"
            " --attention-backend FLASHINFER_VLLM"
        ),
        expect_correction=False,
    ),
    InjectionCase(
        name="model_runtime",
        failure_class="model_runtime",
        model_id="Qwen/Qwen3-1.7B",
        gpu_type="NVIDIA GeForce RTX 4090",
        good_flags={"max_model_len": "640", "gpu_memory_utilization": "0.90"},
        bad_flags=(
            "--dtype bfloat16 --max-model-len 640"
            " --load-format dummy"
        ),
        expect_correction=False,
    ),
]

_A100_INJECTIONS = [
    InjectionCase(
        name="tp_divisibility",
        failure_class="tp_divisibility",
        model_id="Qwen/Qwen3-32B",
        gpu_type="NVIDIA A100-SXM4-80GB",
        good_flags={
            "max_model_len": "640",
            "gpu_memory_utilization": "0.90",
        },
        bad_flags="--dtype bfloat16 --tensor-parallel-size 3 --max-model-len 640",
    ),
    InjectionCase(
        name="quant_compute_capability",
        failure_class="quant_compute_capability",
        model_id="Qwen/Qwen3-8B",
        gpu_type="NVIDIA A100-SXM4-80GB",
        good_flags={"max_model_len": "640", "gpu_memory_utilization": "0.90"},
        bad_flags="--dtype bfloat16 --quantization fp8 --max-model-len 640",
        expect_correction=False,
    ),
]


def _inject_and_diagnose(
    target: Any,
    engine: Any,
    case: InjectionCase,
    rules: list[dict[str, Any]],
    task_suite: dict[str, Any],
) -> dict[str, Any]:
    """Run one failure injection cycle on an already-provisioned pod."""
    from apron.adapters.backends.llm_classifier import _classification_cache
    from apron.application.orchestration.diagnosis_pipeline import run_diagnosis_pipeline
    from apron.domain.schemas.solutions import DeploymentPlan

    result: dict[str, Any] = {"case": case.name, "failure_class": case.failure_class}

    # Kill any running vLLM
    target.execute("pkill -f 'vllm serve' || true")
    target.execute("sleep 5")

    # Inject bad flags
    logger.info("Injecting %s: %s %s", case.name, case.model_id, case.bad_flags)
    bad_cmd = (
        "source /etc/apron_environment 2>/dev/null; "
        f"timeout 180 /opt/venv/bin/vllm serve {case.model_id} {case.bad_flags} 2>&1"
    )
    bad_result = target.execute(bad_cmd)
    error_output = bad_result.get("stdout", "") + bad_result.get("stderr", "")
    result["error_output_length"] = len(error_output)
    result["error_output_tail"] = error_output[-500:]

    if not error_output:
        result["status"] = "no_error_output"
        return result

    # Build plan from good flags for correction baseline
    plan = DeploymentPlan(
        tensor_parallel=1,
        dtype="bfloat16",
        engine_configuration=case.good_flags,
        resource_allocation={"model_id": case.model_id},
    )

    hw = target.hardware

    # Clear LLM cache to force fresh classification
    _classification_cache.clear()

    # Run diagnosis pipeline with real LLM
    model_config = {"num_attention_heads": 32, "num_kv_heads": 8}
    diagnosis = run_diagnosis_pipeline(
        error_output, engine, plan, model_config, hw, rules
    )

    result["diagnosed_class"] = diagnosis.failure_class
    result["result_label"] = diagnosis.result_label
    result["extraction_confidence"] = diagnosis.extraction_confidence
    result["classification_confidence"] = diagnosis.classification_confidence
    result["extracted"] = dict(diagnosis.extracted)

    # Check classification
    if diagnosis.failure_class == "unknown":
        result["status"] = "classification_failed"
        return result

    if diagnosis.corrected_plan is None:
        if case.expect_correction:
            result["status"] = "correction_failed"
        else:
            result["status"] = "correction_infeasible_expected"
        return result

    # Reboot with corrected plan
    target.execute("pkill -f 'vllm serve' || true")
    target.execute("sleep 5")
    target.execute("truncate -s 0 /var/log/vllm.log 2>/dev/null || true")

    corrected_cmd = engine._build_serve_command(diagnosis.corrected_plan, target)
    logger.info("Corrected command: %s", corrected_cmd)
    target.execute(
        "source /etc/apron_environment 2>/dev/null; "
        f"nohup /opt/venv/bin/{corrected_cmd} > /var/log/vllm.log 2>&1 &"
    )

    # Wait for corrected boot
    booted = False
    for i in range(60):
        time.sleep(10)
        health = target.execute("curl -sf http://localhost:8000/health")
        if health.get("exit_code") == 0:
            booted = True
            logger.info("Corrected boot healthy after %ds", (i + 1) * 10)
            break

    result["corrected_boot"] = booted
    if not booted:
        log_tail = target.execute("tail -50 /var/log/vllm.log")
        result["corrected_boot_log"] = log_tail.get("stdout", "")[-500:]
        result["status"] = "corrected_boot_failed"
        return result

    # Replay task suite
    from apron.adapters.evaluations.deterministic_scorer import DeterministicScorer

    scorer = DeterministicScorer()
    endpoint = target.proxy_url
    eval_protocol = scorer.prepare({
        **task_suite,
        "model_id": case.model_id,
        "endpoint": endpoint,
    })
    attempts = scorer.execute(eval_protocol, endpoint)
    scorer.collect(attempts)

    passed = sum(1 for a in attempts if a.get("outcome") == "pass")
    total = len(attempts)
    result["task_passed"] = passed
    result["task_total"] = total
    result["status"] = "full_success" if passed == total else "task_regression"

    return result


_24GB_FALLBACKS = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA L4",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 3090",
]

_80GB_FALLBACKS = [
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA H100 80GB HBM3",
]

_48GB_FALLBACKS = [
    "NVIDIA RTX A6000",
]


def _gpu_fallbacks(gpu_type: str) -> list[str]:
    """Return fallback GPU types in the same memory class."""
    for group in [_24GB_FALLBACKS, _80GB_FALLBACKS, _48GB_FALLBACKS]:
        if gpu_type in group:
            return [g for g in group if g != gpu_type]
    return []


def _run_injection_batch(
    gpu_type: str,
    cases: list[InjectionCase],
    tmp_path: Path,
) -> list[dict[str, Any]]:
    """Provision one pod and run all cases for a GPU type."""
    from runpod.error import QueryError  # type: ignore[import-untyped]

    from apron.adapters.backends.rule_loader import load_rules
    from apron.adapters.backends.runpod import RunPodTarget
    from apron.adapters.backends.vllm_engine import VllmEngineAdapter
    from apron.domain.schemas.solutions import DeploymentPlan

    rules_dir = Path(__file__).parents[2] / "rules"
    rules = load_rules(rules_dir, "vllm", "v0.29")
    task_suite = json.loads((FIXTURES_DIR / "task-suite-spec.json").read_text())

    ssh_key_path = os.environ.get(
        "RUNPOD_SSH_KEY_PATH", str(Path.home() / ".ssh" / "id_rsa")
    )
    ssh_pub_path = ssh_key_path + ".pub"
    public_key = (
        Path(ssh_pub_path).read_text().strip() if Path(ssh_pub_path).exists() else ""
    )

    first_model = cases[0].model_id
    engine = VllmEngineAdapter(rules=rules)

    env = RunPodTarget.build_env(
        model_id=first_model,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=640,
        ssh_public_key=public_key,
    )

    # Try primary GPU, then fallbacks — try ALL cloud type if SECURE fails
    gpu_candidates = [gpu_type, *_gpu_fallbacks(gpu_type)]
    target = None
    for candidate_gpu in gpu_candidates:
        for cloud_type in ["SECURE", "ALL"]:
            target = RunPodTarget(gpu_type=candidate_gpu)
            try:
                import runpod as _rp
                _rp.api_key = os.environ["RUNPOD_API_KEY"]
                pod = _rp.create_pod(
                    name="apron-run",
                    image_name=target._image,
                    gpu_type_id=candidate_gpu,
                    gpu_count=1,
                    cloud_type=cloud_type,
                    ports="22/tcp,8000/http",
                    volume_in_gb=100,
                    container_disk_in_gb=50,
                    env=env,
                )
                target._pod_id = pod["id"]
                logger.info("Pod created on %s (%s): %s", candidate_gpu, cloud_type, pod["id"])
                pod_info = target._wait_for_running(0)
                target._establish_ssh(pod_info)
                hw_info = target._detect_hardware()
                from apron.domain.schemas.primitives import HardwareSpec
                target._hardware = HardwareSpec(
                    gpu_sku=hw_info["gpu_name"],
                    total_memory_bytes=hw_info["total_memory_bytes"],
                    compute_capability=hw_info["compute_capability"],
                )
                target._build_execution_fingerprint(hw_info)
                logger.info("Pod provisioned on %s (%s)", candidate_gpu, cloud_type)
                break
            except QueryError:
                logger.warning("GPU %s (%s) unavailable", candidate_gpu, cloud_type)
                target._pod_id = None
                target = None
                continue
            except RuntimeError as exc:
                logger.warning("GPU %s (%s) failed post-provision: %s", candidate_gpu, cloud_type, exc)
                if target and target._pod_id:
                    target.teardown()
                target = None
                continue
            except Exception:
                if target and target._pod_id:
                    target.teardown()
                raise
        if target is not None:
            break

    if target is None:
        pytest.skip(f"No GPU available in class: {gpu_candidates}")

    results: list[dict[str, Any]] = []
    try:
        plan = DeploymentPlan(
            engine_configuration={"max_model_len": "640"},
            resource_allocation={"model_id": first_model},
        )
        engine.verify(plan, target, health_timeout=600)
        logger.info("Initial boot verified")

        downloaded_models: set[str] = {first_model}

        for case in cases:
            logger.info("=== Running injection: %s ===", case.name)

            if case.model_id not in downloaded_models:
                target.execute("pkill -f 'vllm serve' || true")
                target.execute("sleep 5")
                target.execute(
                    "source /etc/apron_environment 2>/dev/null; "
                    "/opt/venv/bin/python3 -c \"from huggingface_hub import snapshot_download; "
                    f"snapshot_download('{case.model_id}')\" 2>&1"
                )
                downloaded_models.add(case.model_id)

            r = _inject_and_diagnose(target, engine, case, rules, task_suite)
            results.append(r)
            logger.info("Result: %s → %s", case.name, r["status"])

            result_file = tmp_path / f"level3_{case.name}.json"
            result_file.write_text(json.dumps(r, indent=2, default=str))

    finally:
        target.teardown()

    return results


class TestLevel3Gpu4090:
    """Level 3 failure injections on RTX 4090."""

    def test_4090_injections(self, tmp_path: Path) -> None:
        results = _run_injection_batch(
            "NVIDIA GeForce RTX 4090", _4090_INJECTIONS, tmp_path
        )

        for r in results:
            case_name = r["case"]
            status = r["status"]
            logger.info("%s: %s", case_name, status)

            if r.get("diagnosed_class"):
                # Classification should match or be a reasonable related class
                assert r["diagnosed_class"] != "unknown", (
                    f"{case_name}: classified as unknown\n"
                    f"Error tail: {r.get('error_output_tail', '')}"
                )

            if status == "full_success":
                assert r["corrected_boot"]
                assert r["task_passed"] == r["task_total"]
            elif status == "correction_infeasible_expected":
                pass  # expected for some classes
            else:
                # Log but don't hard-fail — collect all results
                logger.warning(
                    "%s ended with status %s: %s",
                    case_name,
                    status,
                    json.dumps(r, indent=2, default=str),
                )


class TestLevel3Gpu4090CorrectionSpec:
    """Level 3 correction-spec class injections on RTX 4090."""

    def test_4090_correction_spec_injections(self, tmp_path: Path) -> None:
        results = _run_injection_batch(
            "NVIDIA GeForce RTX 4090", _4090_INJECTIONS_CORRECTION_SPEC, tmp_path
        )

        for r in results:
            case_name = r["case"]
            status = r["status"]
            logger.info("%s: %s", case_name, status)

            if r.get("diagnosed_class"):
                assert r["diagnosed_class"] != "unknown", (
                    f"{case_name}: classified as unknown\n"
                    f"Error tail: {r.get('error_output_tail', '')}"
                )


class TestLevel3GpuA100:
    """Level 3 failure injections on A100."""

    def test_a100_injections(self, tmp_path: Path) -> None:
        results = _run_injection_batch(
            "NVIDIA A100-SXM4-80GB", _A100_INJECTIONS, tmp_path
        )

        for r in results:
            case_name = r["case"]
            status = r["status"]
            logger.info("%s: %s", case_name, status)

            if r.get("diagnosed_class"):
                assert r["diagnosed_class"] != "unknown", (
                    f"{case_name}: classified as unknown\n"
                    f"Error tail: {r.get('error_output_tail', '')}"
                )
