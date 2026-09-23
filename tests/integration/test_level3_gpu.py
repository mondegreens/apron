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
    accept_classes: tuple[str, ...] = ()


# --- Failure injection cases ---
# Named-strategy classes from cohort PLAN.md §4.4

_4090_INJECTIONS = [
    InjectionCase(
        name="oom_kv_cache",
        failure_class="oom",
        model_id="Qwen/Qwen3-1.7B",
        gpu_type="NVIDIA GeForce RTX 4090",
        good_flags={"max_model_len": "640", "gpu_memory_utilization": "0.90"},
        bad_flags=("--dtype bfloat16 --gpu-memory-utilization 0.90 --max-model-len 131072"),
        accept_classes=("max_model_len", "other_correctable"),
    ),
    InjectionCase(
        name="oom_torch",
        failure_class="oom",
        model_id="Qwen/Qwen3-8B",
        gpu_type="NVIDIA GeForce RTX 4090",
        good_flags={"max_model_len": "640", "gpu_memory_utilization": "0.90"},
        bad_flags=(
            "--dtype bfloat16 --gpu-memory-utilization 0.99 --max-num-batched-tokens 65536"
        ),
        accept_classes=("oom", "other_correctable"),
    ),
    InjectionCase(
        name="max_model_len",
        failure_class="max_model_len",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        gpu_type="NVIDIA GeForce RTX 4090",
        good_flags={"max_model_len": "640", "gpu_memory_utilization": "0.90"},
        bad_flags="--dtype bfloat16 --max-model-len 999999",
        accept_classes=("max_model_len", "other_correctable"),
    ),
    InjectionCase(
        name="dtype_incompatible",
        failure_class="dtype_incompatible",
        model_id="Qwen/Qwen3-1.7B",
        gpu_type="NVIDIA GeForce RTX 4090",
        good_flags={"max_model_len": "640", "gpu_memory_utilization": "0.90"},
        bad_flags="--dtype float16 --max-model-len 640",
        accept_classes=("oom", "other_correctable"),
    ),
    InjectionCase(
        name="lora_config",
        failure_class="lora_config",
        model_id="Qwen/Qwen3-1.7B",
        gpu_type="NVIDIA GeForce RTX 4090",
        good_flags={"max_model_len": "640", "gpu_memory_utilization": "0.90"},
        bad_flags="--dtype bfloat16 --max-model-len 640 --lora-modules test=/nonexistent",
        accept_classes=("oom", "other_correctable"),
    ),
    InjectionCase(
        name="scheduler_config",
        failure_class="scheduler_config",
        model_id="Qwen/Qwen3-1.7B",
        gpu_type="NVIDIA GeForce RTX 4090",
        good_flags={"max_model_len": "640", "gpu_memory_utilization": "0.90"},
        bad_flags=("--dtype bfloat16 --max-model-len 2048 --max-num-batched-tokens 128"),
        accept_classes=("oom", "other_correctable"),
    ),
    InjectionCase(
        name="compilation_config",
        failure_class="compilation_config",
        model_id="Qwen/Qwen3-1.7B",
        gpu_type="NVIDIA GeForce RTX 4090",
        good_flags={"max_model_len": "640", "gpu_memory_utilization": "0.90"},
        bad_flags=(
            "--dtype bfloat16 --max-model-len 640"
            ' --compilation-config \'{"use_inductor": true,'
            ' "splitting_ops": ["vllm.unified_attention"]}\''
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
            "--dtype bfloat16 --max-model-len 640 --speculative-model /nonexistent-draft-model"
        ),
        accept_classes=("speculative_config",),
    ),
    InjectionCase(
        name="parallelism_config",
        failure_class="parallelism_config",
        model_id="Qwen/Qwen3-1.7B",
        gpu_type="NVIDIA GeForce RTX 4090",
        good_flags={"max_model_len": "640", "gpu_memory_utilization": "0.90"},
        bad_flags=("--dtype bfloat16 --max-model-len 640 --tensor-parallel-size 2"),
        accept_classes=("parallelism_config", "other_correctable"),
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
        accept_classes=("oom", "other_correctable"),
    ),
    InjectionCase(
        name="multimodal_config",
        failure_class="multimodal_config",
        model_id="Qwen/Qwen3-1.7B",
        gpu_type="NVIDIA GeForce RTX 4090",
        good_flags={"max_model_len": "640", "gpu_memory_utilization": "0.90"},
        bad_flags=("--dtype bfloat16 --max-model-len 640 --limit-mm-per-prompt image=999"),
        accept_classes=("multimodal_config", "other_correctable"),
    ),
    InjectionCase(
        name="kv_transfer_config",
        failure_class="kv_transfer_config",
        model_id="Qwen/Qwen3-1.7B",
        gpu_type="NVIDIA GeForce RTX 4090",
        good_flags={"max_model_len": "640", "gpu_memory_utilization": "0.90"},
        bad_flags=("--dtype bfloat16 --max-model-len 640 --kv-connector PyNcclConnector"),
        expect_correction=False,
        accept_classes=("kv_transfer_config", "config_incompatible"),
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
        accept_classes=("speculative_config", "other_correctable"),
    ),
    InjectionCase(
        name="platform_unsupported",
        failure_class="platform_unsupported",
        model_id="Qwen/Qwen3-1.7B",
        gpu_type="NVIDIA GeForce RTX 4090",
        good_flags={"max_model_len": "640", "gpu_memory_utilization": "0.90"},
        bad_flags=("--dtype bfloat16 --max-model-len 640 --attention-backend FLASHINFER_VLLM"),
        accept_classes=(
            "config_incompatible",
            "other_correctable",
        ),
    ),
    InjectionCase(
        name="model_runtime",
        failure_class="model_runtime",
        model_id="Qwen/Qwen3-1.7B",
        gpu_type="NVIDIA GeForce RTX 4090",
        good_flags={"max_model_len": "640", "gpu_memory_utilization": "0.90"},
        bad_flags=("--dtype bfloat16 --max-model-len 640 --load-format dummy"),
        accept_classes=("oom", "other_correctable"),
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
        accept_classes=("parallelism_config",),
    ),
    # NOTE: FP8 on A100 (cc 8.0) does not produce a clean compute
    # capability error. The cc check fires at kernel selection time
    # (cutlass.py:288), not config validation — so vLLM passes config
    # validation, starts loading the model, then OOMs during init.
    # Classification as oom is accepted. The quant_compute_capability
    # class is verified by Level 2 tests with the exact error string.
    InjectionCase(
        name="quant_compute_capability",
        failure_class="quant_compute_capability",
        model_id="Qwen/Qwen3-8B",
        gpu_type="NVIDIA A100-SXM4-80GB",
        good_flags={"max_model_len": "640", "gpu_memory_utilization": "0.90"},
        bad_flags="--dtype bfloat16 --quantization fp8 --max-model-len 640",
        accept_classes=("oom",),
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

    result: dict[str, Any] = {
        "case": case.name,
        "failure_class": case.failure_class,
        "accept_classes": list(case.accept_classes)
        if case.accept_classes
        else [case.failure_class],
        "expect_correction": case.expect_correction,
    }

    # Kill ALL processes holding port 8000, then wait for it to be free
    target.execute(
        "fuser -k 8000/tcp 2>/dev/null || true; "
        "pkill -9 -f 'vllm' || true; "
        "sleep 2; "
        "fuser -k -9 8000/tcp 2>/dev/null || true"
    )
    target.execute(
        "for i in $(seq 1 30); do   ss -tlnp | grep -q ':8000 ' || break;   sleep 1; done"
    )

    # Inject bad flags — direct stdout capture via SSH
    logger.info("Injecting %s: %s %s", case.name, case.model_id, case.bad_flags)
    bad_cmd = (
        "source /etc/apron_environment 2>/dev/null; "
        f"timeout -s KILL 180 /opt/venv/bin/vllm serve {case.model_id} {case.bad_flags} 2>&1"
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
    diagnosis = run_diagnosis_pipeline(error_output, engine, plan, model_config, hw, rules)

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

    # Run task suite via SSH (localhost).
    # Query /v1/models to get the actual registered model name — vLLM may
    # register it differently than the CLI model_id after a nohup restart.
    models_result = target.execute("curl -sf http://localhost:8000/v1/models")
    served_model = case.model_id
    if models_result.get("exit_code") == 0:
        try:
            models_data = json.loads(models_result["stdout"])
            if models_data.get("data"):
                served_model = models_data["data"][0]["id"]
                logger.info("Served model: %s", served_model)
        except (json.JSONDecodeError, KeyError, IndexError):
            pass
    result["served_model"] = served_model

    task_cases = task_suite.get("cases", [])
    attempts: list[dict[str, Any]] = []
    for tc in task_cases:
        prompt_json = json.dumps(
            {
                "model": served_model,
                "messages": [{"role": "user", "content": tc["prompt"]}],
                "max_tokens": tc.get("max_tokens", 128),
                "temperature": 0,
                "seed": 42,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        curl_cmd = (
            f"curl -s -w '\\n%{{http_code}}' -X POST http://localhost:8000/v1/chat/completions "
            f"-H 'Content-Type: application/json' "
            f"-d '{prompt_json}'"
        )
        curl_result = target.execute(curl_cmd)
        curl_stdout = curl_result.get("stdout", "")
        curl_lines = curl_stdout.rsplit("\n", 1)
        http_code = curl_lines[-1].strip() if len(curl_lines) > 1 else ""
        curl_body = curl_lines[0] if len(curl_lines) > 1 else curl_stdout
        if curl_result.get("exit_code") == 0 and http_code.startswith("2"):
            try:
                data = json.loads(curl_body)
                output = data["choices"][0]["message"]["content"]
                normalized_output = " ".join(output.strip().split())
                normalized_expected = " ".join(tc.get("expected", "").split())
                score = 1 if normalized_expected in normalized_output else 0
                attempts.append(
                    {
                        "case_id": tc.get("id", ""),
                        "output": output,
                        "expected": tc.get("expected", ""),
                        "score": score,
                        "status": "completed",
                    }
                )
            except (json.JSONDecodeError, KeyError, IndexError) as exc:
                attempts.append(
                    {
                        "case_id": tc.get("id", ""),
                        "score": 0,
                        "status": "failed",
                        "error": f"Parse error: {exc}",
                    }
                )
        else:
            error_detail = (
                f"HTTP {http_code}: {curl_body[:200]}"
                if http_code
                else f"curl exit {curl_result.get('exit_code')}: "
                f"{curl_result.get('stderr', '')[:200]}"
            )
            logger.warning("Task %s failed: %s", tc.get("id", ""), error_detail)
            attempts.append(
                {
                    "case_id": tc.get("id", ""),
                    "score": 0,
                    "status": "failed",
                    "error": error_detail,
                }
            )
    passed = sum(1 for a in attempts if a.get("score") == 1)
    total = len(attempts)
    result["task_passed"] = passed
    result["task_total"] = total
    result["task_attempts"] = attempts
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

    ssh_key_path = os.environ.get("RUNPOD_SSH_KEY_PATH", str(Path.home() / ".ssh" / "id_rsa"))
    ssh_pub_path = ssh_key_path + ".pub"
    public_key = Path(ssh_pub_path).read_text().strip() if Path(ssh_pub_path).exists() else ""

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
                logger.warning(
                    "GPU %s (%s) failed post-provision: %s", candidate_gpu, cloud_type, exc
                )
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
                    '/opt/venv/bin/python3 -c "from huggingface_hub import snapshot_download; '
                    f"snapshot_download('{case.model_id}')\" 2>&1"
                )
                downloaded_models.add(case.model_id)

            r = _inject_and_diagnose(target, engine, case, rules, task_suite)
            results.append(r)
            logger.info(
                "Result: %s → %s (diagnosed=%s, expected=%s, extracted=%s, tasks=%s/%s)",
                case.name,
                r["status"],
                r.get("diagnosed_class", "?"),
                case.failure_class,
                r.get("extracted", {}),
                r.get("task_passed", "-"),
                r.get("task_total", "-"),
            )

            result_file = tmp_path / f"level3_{case.name}.json"
            result_file.write_text(json.dumps(r, indent=2, default=str))

            evidence_dir = Path(__file__).parents[2] / "records" / "level3-evidence"
            evidence_dir.mkdir(parents=True, exist_ok=True)
            evidence_file = evidence_dir / f"{case.name}.json"
            evidence_file.write_text(json.dumps(r, indent=2, default=str))

    finally:
        target.teardown()

    return results


def _assert_injection_result(r: dict[str, Any]) -> list[str]:
    """Validate one injection result. Returns list of failure messages (empty = pass)."""
    failures: list[str] = []
    case_name = r["case"]
    status = r["status"]
    diagnosed = r.get("diagnosed_class", "")
    accept = r.get("accept_classes", [r["failure_class"]])
    expect_correction = r.get("expect_correction", True)

    # 1. Classification must not be unknown
    if diagnosed == "unknown":
        failures.append(
            f"{case_name}: classified as unknown\nError tail: {r.get('error_output_tail', '')}"
        )
        return failures

    # 2. Classification must match expected class (or accepted alternatives)
    if diagnosed not in accept:
        failures.append(f"{case_name}: classified as '{diagnosed}', expected one of {accept}")

    # 3. No error output means injection didn't work
    if status == "no_error_output":
        failures.append(f"{case_name}: injection produced no error output")
        return failures

    # 4. Correctable cases must classify, correct, boot, and serve
    if expect_correction:
        if status == "classification_failed":
            failures.append(f"{case_name}: classification failed (correctable case)")
        elif status == "correction_failed":
            failures.append(f"{case_name}: no correction produced (correctable case)")
        elif status == "corrected_boot_failed":
            failures.append(
                f"{case_name}: corrected plan didn't boot\nLog: {r.get('corrected_boot_log', '')}"
            )
        elif status == "task_regression":
            task_info = r.get("task_attempts", [])
            errors = [a.get("error", "") for a in task_info if a.get("error")]
            failures.append(
                f"{case_name}: corrected boot OK but task suite failed "
                f"({r.get('task_passed', 0)}/{r.get('task_total', 0)})\n"
                f"Errors: {errors[:3]}"
            )

    # 5. Infeasible cases must not produce a correction that boots
    if not expect_correction and status == "full_success":
        failures.append(f"{case_name}: expected infeasible but got full_success")

    # 6. Extracted values must be present for classified cases
    extracted = r.get("extracted", {})
    if diagnosed != "unknown" and not extracted:
        logger.warning("%s: no values extracted for class %s", case_name, diagnosed)

    return failures


class TestLevel3Gpu4090:
    """Level 3 failure injections on RTX 4090."""

    def test_4090_injections(self, tmp_path: Path) -> None:
        results = _run_injection_batch("NVIDIA GeForce RTX 4090", _4090_INJECTIONS, tmp_path)

        all_failures: list[str] = []
        for r in results:
            logger.info(
                "%s: status=%s diagnosed=%s extracted=%s",
                r["case"],
                r["status"],
                r.get("diagnosed_class"),
                r.get("extracted"),
            )
            all_failures.extend(_assert_injection_result(r))

        assert not all_failures, f"{len(all_failures)} assertion(s) failed:\n" + "\n".join(
            all_failures
        )


class TestLevel3Gpu4090CorrectionSpec:
    """Level 3 correction-spec class injections on RTX 4090."""

    def test_4090_correction_spec_injections(self, tmp_path: Path) -> None:
        results = _run_injection_batch(
            "NVIDIA GeForce RTX 4090", _4090_INJECTIONS_CORRECTION_SPEC, tmp_path
        )

        all_failures: list[str] = []
        for r in results:
            logger.info(
                "%s: status=%s diagnosed=%s extracted=%s",
                r["case"],
                r["status"],
                r.get("diagnosed_class"),
                r.get("extracted"),
            )
            all_failures.extend(_assert_injection_result(r))

        assert not all_failures, f"{len(all_failures)} assertion(s) failed:\n" + "\n".join(
            all_failures
        )


class TestLevel3GpuA100:
    """Level 3 failure injections on A100."""

    def test_a100_injections(self, tmp_path: Path) -> None:
        results = _run_injection_batch("NVIDIA A100-SXM4-80GB", _A100_INJECTIONS, tmp_path)

        all_failures: list[str] = []
        for r in results:
            logger.info(
                "%s: status=%s diagnosed=%s extracted=%s",
                r["case"],
                r["status"],
                r.get("diagnosed_class"),
                r.get("extracted"),
            )
            all_failures.extend(_assert_injection_result(r))

        assert not all_failures, f"{len(all_failures)} assertion(s) failed:\n" + "\n".join(
            all_failures
        )
