"""vLLM EngineAdapter — resolve, validate, verify, classify, extract.

Implements the EngineAdapter Protocol (domain/protocols.py).
Works with any ExecutionTarget to boot vLLM, profile memory,
and classify errors.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from apron.domain.canonical import canonicalize, digest_hex

if TYPE_CHECKING:
    from apron.domain.schemas.solutions import DeploymentPlan

logger = logging.getLogger(__name__)

ERROR_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"torch\.OutOfMemoryError|CUDA out of memory", re.IGNORECASE), "oom"),
    (re.compile(r"RuntimeError.*engine", re.IGNORECASE), "engine_init"),
    (re.compile(r"ValueError.*max_model_len", re.IGNORECASE), "max_model_len"),
    (
        re.compile(r"dtype.*not supported|BFloat16 is not supported", re.IGNORECASE),
        "dtype_incompatible",
    ),
    (re.compile(r"not divisible by tensor_parallel", re.IGNORECASE), "tp_divisibility"),
    (
        re.compile(r"quantization.*compute capability", re.IGNORECASE),
        "quant_compute_capability",
    ),
]

_PRE_LOAD_SCRIPT = r"""
import json, torch
free, total = torch.cuda.mem_get_info()
with open("/workspace/apron_pre_load.json", "w") as f:
    json.dump({"pre_free": free, "pre_total": total}, f)
"""

_POST_LOAD_SCRIPT = r"""
import json, torch
free, total = torch.cuda.mem_get_info()
with open("/workspace/apron_post_load.json", "w") as f:
    json.dump({"post_free": free, "post_total": total}, f)
"""

_POST_CUDA_GRAPH_SCRIPT = r"""
import json, torch
free, total = torch.cuda.mem_get_info()
with open("/workspace/apron_post_cuda_graph.json", "w") as f:
    json.dump({"post_cg_free": free, "post_cg_total": total}, f)
"""


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
                    f"TP={plan.tensor_parallel} does not divide "
                    f"num_attention_heads={num_heads}"
                )
            if num_kv_heads > 0 and num_kv_heads % plan.tensor_parallel != 0:
                errors.append(
                    f"TP={plan.tensor_parallel} does not divide "
                    f"num_kv_heads={num_kv_heads}"
                )

        if target is not None and hasattr(target, "hardware"):
            hw = target.hardware
            if plan.dtype == "float16" and hw.compute_capability < "8.0":
                errors.append(
                    f"FP16 requires compute capability >= 8.0, "
                    f"got {hw.compute_capability}"
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

    def verify(self, plan: DeploymentPlan, target: Any) -> dict[str, Any]:
        # Pre-load memory snapshot
        target.execute(f"python3 -c {_shell_quote(_PRE_LOAD_SCRIPT)}")

        serve_cmd = self._build_serve_command(plan, target)
        target.execute(f"nohup {serve_cmd} > /workspace/vllm.log 2>&1 &")

        self._wait_for_health(target)

        # Post-load memory snapshot
        target.execute(f"python3 -c {_shell_quote(_POST_LOAD_SCRIPT)}")

        # Allow CUDA graph capture to complete, then snapshot
        time.sleep(5)
        target.execute(f"python3 -c {_shell_quote(_POST_CUDA_GRAPH_SCRIPT)}")

        # Collect profiling data via execute (Protocol-compliant)
        pre_result = target.execute("cat /workspace/apron_pre_load.json")
        post_result = target.execute("cat /workspace/apron_post_load.json")
        cg_result = target.execute("cat /workspace/apron_post_cuda_graph.json")

        pre = json.loads(pre_result.get("stdout", "{}") if isinstance(pre_result, dict) else "{}")
        post = json.loads(
            post_result.get("stdout", "{}") if isinstance(post_result, dict) else "{}"
        )
        cg = json.loads(cg_result.get("stdout", "{}") if isinstance(cg_result, dict) else "{}")

        return self._build_verification_report(pre, post, cg, plan, target)

    def classify(self, error: str) -> dict[str, Any]:
        for pattern, failure_class in ERROR_PATTERNS:
            if pattern.search(error):
                return {
                    "failure_class": failure_class,
                    "matched_pattern": pattern.pattern,
                }
        return {"failure_class": "unknown", "raw_error": error[:500]}

    def extract_schema(self, image_tag: str) -> dict[str, Any]:
        architectures = self._get_supported_architectures(image_tag)
        tasks = self._get_task_registry(image_tag)
        return {
            "architectures": architectures,
            "tasks": tasks,
            "image_tag": image_tag,
        }

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

    def _wait_for_health(
        self, target: Any, timeout: int = 300, poll_interval: int = 10
    ) -> None:
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
        cg: dict[str, Any],
        plan: DeploymentPlan,
        target: Any,
    ) -> dict[str, Any]:
        initial_total = pre.get("pre_total", 0)
        initial_free = pre.get("pre_free", 0)
        post_free = post.get("post_free", 0)
        post_cg_free = cg.get("post_cg_free", post_free)

        model_weight_memory = initial_free - post_free
        cuda_graph_actual = post_free - post_cg_free if post_cg_free < post_free else 0
        requested_memory = initial_total - post_cg_free
        non_pytorch_increase = max(0, requested_memory - model_weight_memory - cuda_graph_actual)
        persistent_consumption = requested_memory
        transient_peak = 0
        available_kv_cache = post_cg_free
        safety_buffer = int(initial_total * 0.05)

        cuda_graph_estimate = int(plan.engine_configuration.get("cuda_graph_estimate", "0"))
        cuda_graph_applied = cuda_graph_actual > 0

        return {
            "initial_total_memory": initial_total,
            "initial_free_memory": initial_free,
            "requested_memory": requested_memory,
            "model_weight_memory": model_weight_memory,
            "persistent_consumption": persistent_consumption,
            "transient_peak_headroom": transient_peak,
            "non_pytorch_increase": non_pytorch_increase,
            "cuda_graph_estimate": cuda_graph_estimate,
            "cuda_graph_applied": cuda_graph_applied,
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


def _shell_quote(script: str) -> str:
    escaped = script.replace("'", "'\\''")
    return f"'{escaped}'"
