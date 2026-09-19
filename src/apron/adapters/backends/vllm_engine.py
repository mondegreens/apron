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

ERROR_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # oom — torch OOM during weight loading or warmup
    # Source: gpu_model_runner.py:5460, 6370, 6473
    (
        re.compile(
            r"OutOfMemoryError|CUDA out of memory|torch\.cuda\.OutOfMemoryError",
            re.IGNORECASE,
        ),
        "oom",
    ),
    # oom — KV cache memory insufficient (ValueError, not torch OOM)
    # Source: kv_cache_utils.py:858, 879
    (
        re.compile(
            r"larger than the available KV cache memory|"
            r"No available memory for the cache blocks",
            re.IGNORECASE,
        ),
        "oom",
    ),
    # max_model_len — user value exceeds model-derived maximum
    # Source: config/model.py:2517
    (
        re.compile(
            r"max_model_len.*is greater than|"
            r"max_model_len must be a positive integer",
            re.IGNORECASE,
        ),
        "max_model_len",
    ),
    # dtype_incompatible — dtype not supported by model type, platform, or quant
    # Sources: config/model.py:2261, config/vllm.py:799
    (
        re.compile(
            r"does not support float16|"
            r"is not supported for quantization method|"
            r"bfloat16 KV cache is not supported|"
            r"dtype.*not supported",
            re.IGNORECASE,
        ),
        "dtype_incompatible",
    ),
    # tp_divisibility — attention heads not divisible by TP size
    # Source: config/model.py:1414
    (
        re.compile(
            r"must be divisible by tensor parallel|"
            r"is not divisible by",
            re.IGNORECASE,
        ),
        "tp_divisibility",
    ),
    # quant_compute_capability — GPU too old for quantization method
    # Source: config/vllm.py:791
    (
        re.compile(
            r"quantization.*not supported for the current GPU|"
            r"Minimum capability:.*Current capability:",
            re.IGNORECASE,
        ),
        "quant_compute_capability",
    ),
    # engine_init — correctable config RuntimeErrors only
    # Sources: config/model.py (unsupported task/model),
    #   config/device.py:56, lora_model_runner_mixin.py:87,
    #   gpu_worker.py:960
    (
        re.compile(
            r"Unsupported task|"
            r"LoRA is not enabled|"
            r"no draft model is configured|"
            r"Failed to infer device type",
            re.IGNORECASE,
        ),
        "engine_init",
    ),
]

# -------------------------------------------------------------------
# Extraction: typed value capture from classified error messages
# -------------------------------------------------------------------

_RE_TRACEBACK = re.compile(r"Traceback \(most recent call last\):")
_RE_BLANK_LINE = re.compile(r"\n\s*\n")


def _find_error_block(error: str, failure_class: str) -> str:
    """Return the contiguous error block around the match.

    Searches for the enclosing traceback frame (delimited by
    ``Traceback (most recent call last):`` or blank lines) and
    returns the text within it. Falls back to the full error string
    when no traceback boundary is found.
    """
    boundaries = sorted(
        {m.start() for m in _RE_TRACEBACK.finditer(error)}
        | {m.start() for m in _RE_BLANK_LINE.finditer(error)}
    )
    if not boundaries:
        return error

    for pattern, fc in ERROR_PATTERNS:
        if fc != failure_class:
            continue
        m = pattern.search(error)
        if m:
            frame_start = 0
            frame_end = len(error)
            for s in boundaries:
                if s <= m.start():
                    frame_start = s
                else:
                    frame_end = s
                    break
            return error[frame_start:frame_end]
    return error


_ExtractorEntry = tuple[str, re.Pattern[str], type]

_EXTRACTORS: dict[str, list[_ExtractorEntry]] = {
    "oom": [
        ("estimated_max_model_len", re.compile(r"estimated maximum model length is (\d+)"), int),
        ("max_model_len", re.compile(r"max seq len\s*\((\d+)\)"), int),
        ("needed_gib", re.compile(r"([\d.]+)\s*GiB KV\s*cache is needed"), float),
        ("available_gib", re.compile(r"available KV cache\s*memory \(([\d.]+)\s*GiB\)"), float),
        ("max_num_seqs_attempted", re.compile(r"warming up (?:sampler|pooler) with\s*(\d+)"), int),
    ],
    "max_model_len": [
        ("requested", re.compile(r"max_model_len \((\d+)\)"), int),
        ("derived_max", re.compile(r"derived max_model_len \([^=]+=(\d+)"), int),
        ("max_len_key", re.compile(r"derived max_model_len \((\w+)="), str),
    ],
    "dtype_incompatible": [
        ("model_type", re.compile(r"model type '(\w+)'"), str),
        ("unsupported_dtype", re.compile(r"does not support (\w+)"), str),
        ("supported_list", re.compile(r"Supported dtypes:\s*(.+)"), str),
    ],
    "tp_divisibility": [
        ("num_heads", re.compile(r"attention heads \((\d+)\)"), int),
        ("tp_size", re.compile(r"tensor parallel size\s*\((\d+)\)"), int),
    ],
    "quant_compute_capability": [
        ("method", re.compile(r"quantization method (\w+)"), str),
        ("min_cap", re.compile(r"Minimum capability:\s*(\d+)"), int),
        ("cur_cap", re.compile(r"Current capability:\s*(\d+)"), int),
    ],
    "engine_init": [
        ("suggested_fix", re.compile(r"(?:Use |please set )(.+?)(?:\.|$)", re.IGNORECASE), str),
    ],
}

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
        for pattern, failure_class in ERROR_PATTERNS:
            m = pattern.search(error)
            if m:
                return {
                    "failure_class": failure_class,
                    "matched_pattern": pattern.pattern,
                    "match_start": m.start(),
                    "match_end": m.end(),
                }
        return {"failure_class": "unknown", "raw_error": error[:500]}

    def extract(self, error: str, failure_class: str) -> dict[str, int | float | str]:
        """Extract typed values from classified error for deterministic correction."""
        block = _find_error_block(error, failure_class)
        extractors = _EXTRACTORS.get(failure_class)
        if extractors is None:
            return {}
        result: dict[str, int | float | str] = {}
        for key, pattern, converter in extractors:
            m = pattern.search(block)
            if m:
                result[key] = converter(m.group(1))
        return result

    def extraction_confidence(self, failure_class: str, extracted: dict[str, Any]) -> float:
        """Fraction of expected fields that were actually extracted (0.0-1.0)."""
        extractors = _EXTRACTORS.get(failure_class)
        if not extractors:
            return 1.0
        expected = len(extractors)
        found = sum(1 for key, _, _ in extractors if key in extracted)
        return found / expected

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
