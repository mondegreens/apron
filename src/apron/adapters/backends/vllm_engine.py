"""vLLM EngineAdapter — resolve, validate, verify, classify, extract.

Implements the EngineAdapter Protocol (domain/protocols.py).
Works with any ExecutionTarget to boot vLLM, profile memory,
and classify errors.

Profiling strategy: vLLM logs its own MemoryProfilingResult at startup
(weights, non-torch, peak activation, CUDA graph, available KV cache).
We parse those logs rather than running separate profiling processes,
because vLLM's internal memory_stats/memory_reserved data is only
available inside its own process.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from apron.domain.schemas.solutions import DeploymentPlan

logger = logging.getLogger(__name__)

_RE_VLLM_VERSION = re.compile(r"vLLM\s+v?(\d+\.\d+\.\d+)")

# vLLM startup log patterns (from vllm/v1/worker/gpu_worker.py)
_RE_AVAILABLE_KV = re.compile(r"Available KV cache memory:\s*([\d.]+)\s*GiB")
_RE_CUDA_GRAPH = re.compile(
    r"CUDA graph pool memory:\s*([\d.]+)\s*GiB \(actual\),\s*([\d.]+)\s*GiB \(estimated\)"
)
_RE_STARTUP_MSG = re.compile(
    r"Free memory on device \(([\d.]+)/([\d.]+) GiB\) on startup\.\s*"
    r"Desired GPU memory utilization is \(([\d.]+), ([\d.]+) GiB\)\.\s*"
    r"Actual usage is ([\d.]+) GiB for consumed memory.*?"
    r"([\d.]+) GiB for peak activation.*?"
    r"([\d.]+) GiB for CUDAGraph memory",
    re.DOTALL,
)
_RE_PROFILING_RESULT = re.compile(
    r"Memory profiling takes [\d.]+ seconds\.\s*"
    r"Total non KV cache memory:\s*([\d.]+)GiB;\s*"
    r"torch peak memory increase:\s*([\d.]+)GiB;\s*"
    r"total consumed \(from mem_get_info\):\s*([\d.]+)GiB;\s*"
    r"weights memory:\s*([\d.]+)GiB"
)

GIB = 1 << 30


class VllmEngineAdapter:
    """EngineAdapter for vLLM inference engine."""

    _engine_name = "vllm"

    def __init__(self, engine_version: str = "v0.29.0") -> None:
        self._engine_version = engine_version

    @property
    def engine_name(self) -> str:
        return self._engine_name

    @property
    def engine_version(self) -> str:
        return self._engine_version

    def resolve_support(self, model_spec: Any, image_tag: str) -> dict[str, Any]:
        architecture = _extract_architecture(model_spec)
        supported_architectures = self._get_supported_architectures(image_tag)

        supported = architecture in supported_architectures
        tasks = self._get_task_registry(image_tag)

        return {
            "supported": supported,
            "architecture": architecture,
            "tasks": tasks,
            "engine": self._engine_name,
            "version": self._engine_version,
        }

    def validate(self, plan: DeploymentPlan, target: Any) -> list[str]:
        errors: list[str] = []

        if plan.tensor_parallel > 1:
            engine_config = plan.engine_configuration
            num_heads = int(engine_config.get("num_attention_heads", "0"))
            num_kv_heads = int(engine_config.get("num_kv_heads", str(num_heads)))
            if num_heads > 0 and num_heads % plan.tensor_parallel != 0:
                errors.append(
                    f"TP={plan.tensor_parallel} does not divide num_attention_heads={num_heads}"
                )
            if num_kv_heads > 0 and num_kv_heads % plan.tensor_parallel != 0:
                errors.append(
                    f"TP={plan.tensor_parallel} does not divide num_kv_heads={num_kv_heads}"
                )

        if target is not None and hasattr(target, "hardware"):
            hw = target.hardware
            if plan.dtype == "float16" and hw.compute_capability < "8.0":
                errors.append(
                    f"FP16 requires compute capability >= 8.0, got {hw.compute_capability}"
                )

        return errors

    def render(self, plan: DeploymentPlan) -> dict[str, Any]:
        args: list[str] = []
        if plan.dtype:
            args.extend(["--dtype", plan.dtype])
        if plan.tensor_parallel > 1:
            args.extend(["--tensor-parallel-size", str(plan.tensor_parallel)])
        for key, value in plan.engine_configuration.items():
            args.extend([f"--{key.replace('_', '-')}", str(value)])
        return {"args": args, "engine": self._engine_name}

    def verify(
        self,
        plan: DeploymentPlan,
        target: Any,
        health_timeout: int = 600,
    ) -> dict[str, Any]:
        """Collect profiling data from a running vLLM instance.

        The apron runner image boots vLLM on container start and tees
        logs to /var/log/vllm.log. This method waits for health, reads
        the logs for profiling data, and queries GPU memory via SSH.
        """
        self._wait_for_health(target, timeout=health_timeout)

        log_result = target.execute("cat /var/log/vllm.log 2>/dev/null || echo ''")
        log_text = log_result.get("stdout", "") if isinstance(log_result, dict) else ""

        parsed = self.parse_profiling_logs(log_text)

        mem_result = target.execute(
            "/opt/venv/bin/python3 -c 'import torch; f,t=torch.cuda.mem_get_info(); "
            'print(f"{{\\"post_free\\":{f},\\"post_total\\":{t}}}")\''
        )
        post = _parse_json_output(mem_result)

        pre: dict[str, Any] = {}
        if parsed.get("startup_total_gib"):
            pre["pre_total"] = int(parsed["startup_total_gib"] * GIB)
            pre["pre_free"] = int(parsed["startup_free_gib"] * GIB)
        else:
            pre["pre_total"] = post.get("post_total", 0)
            pre["pre_free"] = post.get("post_total", 0)

        return self._build_verification_report(pre, post, parsed, plan, target)

    def classify(self, error: str) -> dict[str, Any]:
        from apron.adapters.backends.llm_classifier import classify_and_extract

        result = classify_and_extract(error)
        return {
            "failure_class": result.get("failure_class", "unknown"),
            "confidence": result.get("confidence", 0.0),
            "evidence_span": result.get("evidence_span", ""),
            "_full_extraction": result,
        }

    def extract(self, error: str, failure_class: str) -> dict[str, int | float | str]:
        """Extract typed values — delegates to LLM classifier.

        The LLM classify_and_extract call returns both classification and
        extraction in one pass. When called after classify(), uses the
        cached extraction from the classify result. When called standalone,
        makes a fresh LLM call.
        """
        from apron.adapters.backends.llm_classifier import classify_and_extract
        from apron.domain.diagnosis import EXTRACTION_SCHEMAS

        full = classify_and_extract(error)
        schema = EXTRACTION_SCHEMAS.get(failure_class, [])
        result: dict[str, int | float | str] = {}
        for field_name, field_type in schema:
            value = full.get(field_name)
            if value is not None:
                try:
                    result[field_name] = field_type(value)
                except (ValueError, TypeError):
                    continue
        return result

    def detect_engine_version(self, output: str) -> str | None:
        """Try to detect the vLLM version from process output."""
        m = _RE_VLLM_VERSION.search(output)
        return m.group(1) if m else None

    def extract_schema(self, image_tag: str) -> dict[str, Any]:
        architectures = self._get_supported_architectures(image_tag)
        tasks = self._get_task_registry(image_tag)
        return {
            "architectures": architectures,
            "tasks": tasks,
            "image_tag": image_tag,
        }

    # ------------------------------------------------------------------
    # Log parsing — extracts vLLM's own profiling output
    # ------------------------------------------------------------------

    @staticmethod
    def parse_profiling_logs(log_text: str) -> dict[str, Any]:
        """Parse vLLM startup logs for memory profiling data.

        vLLM logs MemoryProfilingResult at DEBUG and a startup summary
        at INFO. We extract both when available.
        """
        parsed: dict[str, Any] = {}

        m = _RE_PROFILING_RESULT.search(log_text)
        if m:
            parsed["non_kv_cache_memory"] = int(float(m.group(1)) * GIB)
            parsed["torch_peak_increase"] = int(float(m.group(2)) * GIB)
            parsed["total_consumed"] = int(float(m.group(3)) * GIB)
            parsed["weights_memory"] = int(float(m.group(4)) * GIB)

        m = _RE_STARTUP_MSG.search(log_text)
        if m:
            parsed["startup_free_gib"] = float(m.group(1))
            parsed["startup_total_gib"] = float(m.group(2))
            parsed["gpu_memory_utilization"] = float(m.group(3))
            parsed["requested_gib"] = float(m.group(4))
            parsed["consumed_gib"] = float(m.group(5))
            parsed["peak_activation_gib"] = float(m.group(6))
            parsed["cudagraph_gib"] = float(m.group(7))

        m = _RE_AVAILABLE_KV.search(log_text)
        if m:
            parsed["available_kv_cache_gib"] = float(m.group(1))

        m = _RE_CUDA_GRAPH.search(log_text)
        if m:
            parsed["cuda_graph_actual_gib"] = float(m.group(1))
            parsed["cuda_graph_estimate_gib"] = float(m.group(2))

        return parsed

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_supported_architectures(self, image_tag: str) -> list[str]:
        from apron.adapters.backends.vllm_constraints import extract_supported_architectures

        fixture_dir = Path("tests/fixtures/external-formats/vllm")
        supported_models = fixture_dir / "supported_models.md"
        if supported_models.exists():
            return extract_supported_architectures(supported_models)
        return []

    def _get_task_registry(self, image_tag: str) -> list[str]:
        from apron.adapters.backends.vllm_constraints import extract_task_registry

        fixture_dir = Path("tests/fixtures/external-formats/vllm")
        tasks_file = fixture_dir / "tasks.py"
        if tasks_file.exists():
            return extract_task_registry(tasks_file)
        return ["generate"]

    def _build_serve_command(self, plan: DeploymentPlan, target: Any) -> str:
        resource = plan.resource_allocation
        model_id = resource.get("model_id", "")

        parts = ["vllm", "serve", model_id]
        if plan.dtype:
            parts.extend(["--dtype", plan.dtype])
        if plan.tensor_parallel > 1:
            parts.extend(["--tensor-parallel-size", str(plan.tensor_parallel)])

        for key, value in plan.engine_configuration.items():
            parts.extend([f"--{key.replace('_', '-')}", str(value)])

        if resource.get("remote_code_required") == "true":
            parts.append("--trust-remote-code")

        return " ".join(parts)

    def _wait_for_health(self, target: Any, timeout: int = 300, poll_interval: int = 10) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = target.execute("curl -sf http://localhost:8000/health")
            if result.get("exit_code") == 0:
                logger.info("vLLM health check passed")
                return
            time.sleep(poll_interval)

        log_result = target.execute("tail -100 /workspace/vllm.log")
        log_tail = log_result.get("stdout", "")
        raise RuntimeError(f"vLLM did not become healthy within {timeout}s\n{log_tail}")

    def _build_verification_report(
        self,
        pre: dict[str, Any],
        post: dict[str, Any],
        parsed_logs: dict[str, Any],
        plan: DeploymentPlan,
        target: Any,
    ) -> dict[str, Any]:
        initial_total = pre.get("pre_total", 0)
        initial_free = pre.get("pre_free", 0)
        post_free = post.get("post_free", 0)

        if parsed_logs.get("weights_memory"):
            model_weight_memory = parsed_logs["weights_memory"]
        else:
            model_weight_memory = initial_free - post_free

        total_consumed = parsed_logs.get("total_consumed", initial_free - post_free)
        non_torch_increase = (
            parsed_logs.get("non_kv_cache_memory", 0)
            - model_weight_memory
            - parsed_logs.get("torch_peak_increase", 0)
        )
        non_torch_increase = max(0, non_torch_increase)

        torch_peak_increase = parsed_logs.get("torch_peak_increase", 0)

        if "cuda_graph_actual_gib" in parsed_logs:
            cuda_graph_actual = int(parsed_logs["cuda_graph_actual_gib"] * GIB)
            cuda_graph_estimate = int(parsed_logs.get("cuda_graph_estimate_gib", 0) * GIB)
        else:
            cuda_graph_actual = 0
            cuda_graph_estimate = 0

        if "available_kv_cache_gib" in parsed_logs:
            available_kv_cache = int(parsed_logs["available_kv_cache_gib"] * GIB)
        else:
            available_kv_cache = post_free

        persistent_consumption = total_consumed
        safety_buffer = int(initial_total * 0.05)
        requested_memory = int(parsed_logs.get("requested_gib", initial_total * 0.9 / GIB) * GIB)

        return {
            "initial_total_memory": initial_total,
            "initial_free_memory": initial_free,
            "requested_memory": requested_memory,
            "model_weight_memory": model_weight_memory,
            "persistent_consumption": persistent_consumption,
            "transient_peak_headroom": torch_peak_increase,
            "non_pytorch_increase": non_torch_increase,
            "cuda_graph_estimate": cuda_graph_estimate,
            "cuda_graph_applied": cuda_graph_actual > 0,
            "cuda_graph_actual": cuda_graph_actual,
            "available_kv_cache_memory": available_kv_cache,
            "safety_buffer": safety_buffer,
            "profiling_shape": {"batch_size": 1, "seq_len": 1},
            "execution_fingerprint": target.execution_fingerprint,
            "target_kind": target.kind,
        }


def _extract_architecture(model_spec: Any) -> str:
    if isinstance(model_spec, dict):
        if "architecture" in model_spec:
            return model_spec["architecture"]
        components = model_spec.get("components", [])
        if components:
            comp = components[0] if isinstance(components[0], dict) else {}
            return comp.get("architecture", "")
        return model_spec.get("architectures", [""])[0] if "architectures" in model_spec else ""

    if hasattr(model_spec, "components") and model_spec.components:
        return model_spec.components[0].architecture
    return ""


def _parse_json_output(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        stdout = result.get("stdout", "")
        for line in stdout.strip().splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
    return {}
