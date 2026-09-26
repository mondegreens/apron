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
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from apron.domain.schemas.solutions import DeploymentPlan

logger = logging.getLogger(__name__)

_RE_VLLM_VERSION = re.compile(r"vLLM\s+v?(\d+\.\d+\.\d+)")

# vLLM startup log patterns (from vllm/v1/worker/gpu_worker.py)
_RE_AVAILABLE_KV = re.compile(r"Available KV cache memory:\s*([\d.]+)\s*GiB")
# vllm/config/scheduler.py:277 — the profile run's token count (gpu_model_runner.py:6574)
_RE_MAX_BATCHED_TOKENS = re.compile(r"max_num_batched_tokens=(\d+)")
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

MODELS_DIR = "/workspace/models"
VLLM_LOG = "/var/log/vllm.log"
TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
# vLLM API server and its engine-core children.  The bracket keeps the
# pattern from matching the shell that runs pgrep/pkill with it.
VLLM_PROCESS_PATTERN = "bin/[v]llm serve|[V]LLM::"

# Same logic as docker/apron-download: the only process that reads the token
# file (F7).  The file is written by the runner image's start.sh; an image
# without F7 support never writes it (see runner_supports_token_isolation).
#
# Native duplicates are skipped: Mistral repos ship consolidated.safetensors
# beside the HF shards (Mistral-7B-Instruct-v0.3: 29 GB instead of 14.5) and
# Llama repos ship original/*.pth (16 GB).  vLLM loads the HF-format files.
DOWNLOAD_IGNORE = ("original/*", "consolidated*", "*.pth", "*.pt", "*.gguf")
_DOWNLOAD_PY = (
    "import pathlib,sys\n"
    "from huggingface_hub import snapshot_download\n"
    "f=pathlib.Path('/run/apron/hf_token')\n"
    "t=f.read_text().strip() if f.exists() else None\n"
    "snapshot_download(repo_id=sys.argv[1],local_dir=sys.argv[2],token=t or None,"
    f"ignore_patterns={list(DOWNLOAD_IGNORE)!r})\n"
)


@dataclass(frozen=True)
class BootResult:
    """Outcome of one vLLM boot from a rendered plan."""

    healthy: bool
    log_tail: str
    command: str
    seconds: float


class VllmEngineAdapter:
    """EngineAdapter for vLLM inference engine."""

    _engine_name = "vllm"

    def __init__(
        self,
        engine_version: str = "v0.29.0",
        rules: list[dict[str, Any]] | None = None,
    ) -> None:
        self._engine_version = engine_version
        self._rules = rules or []

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
            # Head counts are plan metadata (resource_allocation), not engine
            # flags; engine_configuration is rendered onto the command line.
            meta = {**plan.engine_configuration, **plan.resource_allocation}
            num_heads = int(meta.get("num_attention_heads", "0"))
            num_kv_heads = int(meta.get("num_kv_heads", str(num_heads)))
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
        return {"args": args, "engine": self._engine_name, "tp": plan.tensor_parallel}

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

        result = classify_and_extract(error, rules=self._rules)
        return {
            "failure_class": result.get("failure_class", "unknown"),
            "confidence": result.get("confidence", 0.0),
            "evidence_span": result.get("evidence_span", ""),
            "classifier_model_id": result.get("classifier_model_id"),
            "classifier_input_digest": result.get("classifier_input_digest"),
            "_full_extraction": result,
        }

    def extract(self, error: str, failure_class: str) -> dict[str, int | float | str]:
        """Extract typed values — delegates to LLM classifier."""
        from apron.adapters.backends.llm_classifier import classify_and_extract
        from apron.domain.diagnosis import build_extraction_enums, build_extraction_schemas

        full = classify_and_extract(error, rules=self._rules)
        schemas = build_extraction_schemas(self._rules)
        enums = build_extraction_enums(self._rules).get(failure_class, {})
        schema = schemas.get(failure_class, [])
        result: dict[str, int | float | str] = {}
        for field_name, field_type in schema:
            value = full.get(field_name)
            if value is not None:
                try:
                    typed = field_type(value)
                except (ValueError, TypeError):
                    continue
                if field_name in enums and typed not in enums[field_name]:
                    logger.warning(
                        "Rejecting %s=%r: not in enum %s", field_name, typed, enums[field_name]
                    )
                    continue
                result[field_name] = typed
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

        m = _RE_MAX_BATCHED_TOKENS.search(log_text)
        if m:
            parsed["max_num_batched_tokens"] = int(m.group(1))

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

    def _build_serve_command(
        self, plan: DeploymentPlan, target: Any, *, model_path: str | None = None
    ) -> str:
        """Render ``vllm serve`` from a DeploymentPlan — never from raw flags.

        With ``model_path`` the weights are served from local disk under the
        plan's model id (``--served-model-name``), so no token is needed.
        """
        resource = plan.resource_allocation
        model_id = resource.get("model_id", "")

        parts = ["vllm", "serve", model_path or model_id]
        if model_path:
            parts.extend(["--served-model-name", model_id])
        if plan.dtype:
            parts.extend(["--dtype", plan.dtype])
        if plan.tensor_parallel > 1:
            parts.extend(["--tensor-parallel-size", str(plan.tensor_parallel)])

        for key, value in plan.engine_configuration.items():
            parts.extend([f"--{key.replace('_', '-')}", str(value)])

        if resource.get("remote_code_required") == "true":
            parts.append("--trust-remote-code")

        return " ".join(shlex.quote(p) for p in parts)

    # ------------------------------------------------------------------
    # Boot from a plan (§9.1 steps 0 and 3; F7)
    # ------------------------------------------------------------------

    @staticmethod
    def model_dir(model_id: str) -> str:
        return f"{MODELS_DIR}/{model_id}"

    def prepare_boot(self, target: Any, timeout: int = 120) -> dict[str, Any]:
        """Harness hygiene before every boot: no vLLM running, port 8000
        free, GPU memory used below 1 GiB (the Part 1 dtype evidence was a
        port collision from a previous vLLM)."""
        target.execute(f"pkill -9 -f '{VLLM_PROCESS_PATTERN}' || true")
        deadline = time.monotonic() + timeout
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            # bash /dev/tcp: no dependency on ss/netstat being in the image
            port = target.execute(
                "bash -c '(exec 3<>/dev/tcp/127.0.0.1/8000) 2>/dev/null && echo 1 || echo 0'"
            )
            mem = target.execute(
                "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
            )
            port_busy = str(port.get("stdout", "0")).strip() not in ("", "0")
            used = [int(x) for x in str(mem.get("stdout", "")).split() if x.strip().isdigit()]
            last = {"port_8000_busy": port_busy, "gpu_memory_used_mib": used}
            if not port_busy and used and max(used) < 1024:
                return {"clean": True, **last}
            time.sleep(3)
        return {"clean": False, **last}

    @staticmethod
    def runner_supports_token_isolation(target: Any) -> bool:
        """True if the runner image has the F7 download step.

        An older image exports the HF token into every SSH session, so a
        gated download there either gets no token or leaks it.  The cohort
        refuses to boot on such an image.
        """
        result = target.execute("test -x /usr/local/bin/apron-download && echo yes || echo no")
        return "yes" in str(result.get("stdout", ""))

    def download_weights(self, target: Any, model_id: str, timeout: int = 3600) -> dict[str, Any]:
        """Fetch weights in a separate step — the only one that reads the token."""
        dest = self.model_dir(model_id)
        script = shlex.quote(_DOWNLOAD_PY)
        command = (
            f"mkdir -p {shlex.quote(dest)} && "
            f"env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN /opt/venv/bin/python3 -c {script} "
            f"{shlex.quote(model_id)} {shlex.quote(dest)} 2>&1 | tail -20"
        )
        start = time.monotonic()
        result = target.execute(command)
        return {
            "ok": result.get("exit_code", 1) == 0,
            "seconds": round(time.monotonic() - start, 1),
            "output_tail": str(result.get("stdout", ""))[-2000:],
        }

    def boot(self, plan: DeploymentPlan, target: Any, *, health_timeout: int = 600) -> BootResult:
        """Boot vLLM from the rendered plan with no token in its environment.

        Returns when ``/health`` passes or the process exits.  A failed boot
        returns the log tail — the evidence diagnosis reads.
        """
        model_id = plan.resource_allocation.get("model_id", "")
        serve = self._build_serve_command(plan, target, model_path=self.model_dir(model_id))
        command = (
            f"cd /workspace && env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN HF_HUB_OFFLINE=1 "
            f"nohup /opt/venv/bin/{serve} > {VLLM_LOG} 2>&1 &"
        )
        start = time.monotonic()
        target.execute(command)
        deadline = start + health_timeout
        while time.monotonic() < deadline:
            if target.execute("curl -sf http://localhost:8000/health").get("exit_code") == 0:
                return BootResult(True, "", serve, round(time.monotonic() - start, 1))
            alive = target.execute(
                f"pgrep -f '{VLLM_PROCESS_PATTERN}' >/dev/null && echo up || echo down"
            )
            if "down" in str(alive.get("stdout", "")):
                break
            time.sleep(10)
        tail = target.execute(f"tail -200 {VLLM_LOG}")
        return BootResult(
            False, str(tail.get("stdout", "")), serve, round(time.monotonic() - start, 1)
        )

    # ------------------------------------------------------------------
    # Serving measurement (§9.1 step 5; measurement only, D2)
    # ------------------------------------------------------------------

    SERVING_PERCENTILES = (50, 90, 95, 99)
    SERVING_METRICS = ("ttft", "tpot", "itl", "e2el")

    def benchmark_serving(
        self,
        target: Any,
        *,
        model_id: str,
        input_len: int,
        output_len: int,
        concurrency: int,
        num_prompts: int = 50,
        num_warmups: int = 5,
    ) -> dict[str, Any]:
        """Run ``vllm bench serve`` (random dataset) against the running endpoint.

        ISL, OSL and concurrency come from the accepted ServingWorkloadSpec.
        ``--ignore-eos`` makes every request produce the declared OSL, so the
        measured TPOT is over the workload the spec declares.  Returns the
        saved result JSON; raises if the run produced none.
        """
        name = f"bench-{int(time.time())}.json"
        command = " ".join(
            [
                "cd /workspace && HF_HUB_OFFLINE=1 /opt/venv/bin/vllm bench serve",
                "--backend vllm --base-url http://localhost:8000",
                f"--model {shlex.quote(model_id)}",
                f"--tokenizer {shlex.quote(self.model_dir(model_id))}",
                "--dataset-name random",
                f"--random-input-len {int(input_len)} --random-output-len {int(output_len)}",
                f"--max-concurrency {int(concurrency)}",
                f"--num-prompts {int(num_prompts)} --num-warmups {int(num_warmups)}",
                "--ignore-eos",
                "--percentile-metrics " + ",".join(self.SERVING_METRICS),
                "--metric-percentiles " + ",".join(str(p) for p in self.SERVING_PERCENTILES),
                f"--save-result --result-dir /workspace/bench --result-filename {name}",
                "> /workspace/bench.log 2>&1; tail -5 /workspace/bench.log",
            ]
        )
        run = target.execute(command)
        result = target.execute(f"cat /workspace/bench/{name}")
        parsed = _parse_json_output(result)
        if not parsed:
            raise RuntimeError(f"vllm bench serve produced no result: {run.get('stdout', '')}")
        return {**parsed, "_command": command}

    @classmethod
    def build_serving_report(cls, bench: dict[str, Any]) -> dict[str, Any]:
        """Map a ``vllm bench serve`` result to VerificationReport serving fields."""
        latency: dict[str, float] = {}
        for metric in cls.SERVING_METRICS:
            for pct in cls.SERVING_PERCENTILES:
                key = f"p{pct}_{metric}_ms"
                if bench.get(key) is not None:
                    latency[key] = float(bench[key])
        return {
            "serving_num_prompts": bench.get("num_prompts"),
            "serving_concurrency": bench.get("max_concurrency"),
            "serving_completed": bench.get("completed"),
            "serving_failed": bench.get("failed"),
            "serving_latency_ms": latency or None,
            "serving_request_throughput": bench.get("request_throughput"),
            "serving_output_token_throughput": bench.get("output_throughput"),
        }

    def token_environ_check(self, target: Any) -> dict[str, Any]:
        """INV-13 evidence: which token variables the vLLM processes carry.

        Reads ``/proc/<pid>/environ`` of every ``vllm serve`` process and
        reports variable *names* only — values never leave the pod.
        """
        names = "|".join(TOKEN_VARS)
        command = (
            f"for p in $(pgrep -f '{VLLM_PROCESS_PATTERN}'); do "
            f"n=$(tr '\\0' '\\n' < /proc/$p/environ | cut -d= -f1 | grep -c -E '^({names})$'); "
            'echo "$p $n"; done'
        )
        result = target.execute(command)
        rows = [
            line.split() for line in str(result.get("stdout", "")).splitlines() if line.strip()
        ]
        processes = {int(pid): int(count) for pid, count in rows if pid.isdigit()}
        return {
            "command": command,
            "processes": processes,
            "token_free": bool(processes) and all(c == 0 for c in processes.values()),
        }

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
            "profiling_shape": profiling_shape(parsed_logs, plan),
            "execution_fingerprint": target.execution_fingerprint,
            "target_kind": target.kind,
        }


def profiling_shape(parsed_logs: dict[str, Any], plan: DeploymentPlan) -> dict[str, int] | None:
    """The shape vLLM's memory profile run used, or ``None`` when unknown.

    vLLM v0.29 profiles one dummy forward of ``max_num_batched_tokens`` tokens
    (``v1/worker/gpu_model_runner.py:6574``) spread over at most
    ``max_num_seqs`` requests (``_dummy_run``).  The token count is read from
    vLLM's own log (``config/scheduler.py:277``) or the plan; ``max_num_seqs``
    is not logged, so it is recorded only when the plan sets it.
    """
    shape: dict[str, int] = {}
    tokens = parsed_logs.get("max_num_batched_tokens") or plan.engine_configuration.get(
        "max_num_batched_tokens"
    )
    if tokens is not None:
        shape["max_num_batched_tokens"] = int(tokens)
    seqs = plan.engine_configuration.get("max_num_seqs")
    if seqs is not None:
        shape["max_num_seqs"] = int(seqs)
    return shape or None


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
