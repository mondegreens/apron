"""RunPod ExecutionTarget — rented-provider adapter for RunPod Secure Cloud GPUs.

Implements the ExecutionTarget Protocol (domain/schemas/primitives.py).

Control plane: ``runpod`` Python SDK (create_pod / terminate_pod).
Status polling: raw GraphQL (SDK doesn't expose runtime/uptime fields).
Management plane: SSH via paramiko (kill/restart vLLM, run profiling).
Data plane: HTTPS proxy ``https://{pod_id}-8000.proxy.runpod.net``.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from apron.adapters.runner_image import RUNNER_IMAGE
from apron.application.orchestration.errors import PodLeakError
from apron.application.sanitization import mask_secrets
from apron.domain.canonical import canonicalize, digest_hex
from apron.domain.schemas.primitives import HardwareSpec

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

GRAPHQL_URL = "https://api.runpod.io/graphql"
DEFAULT_IMAGE = RUNNER_IMAGE
DEFAULT_MAX_UPTIME = 3600
POD_NAME_PREFIX = "apron-run"
CLOUD_TYPE = "SECURE"  # D4: RunPod Secure only; Community is never used
TEARDOWN_BACKOFF_SECONDS = (2, 4, 8)
DEFAULT_LEAK_LOG = Path("_dev_notes/cohort-run/leaked_pods.json")

GPU_SPECS: dict[str, dict[str, Any]] = {
    "NVIDIA GeForce RTX 4090": {
        "total_memory_bytes": 25_769_803_776,
        "compute_capability": "8.9",
    },
    "NVIDIA RTX A5000": {
        "total_memory_bytes": 25_769_803_776,
        "compute_capability": "8.6",
    },
    "NVIDIA L4": {
        "total_memory_bytes": 25_769_803_776,
        "compute_capability": "8.9",
    },
    "NVIDIA RTX A6000": {
        "total_memory_bytes": 51_539_607_552,
        "compute_capability": "8.6",
    },
    "NVIDIA A100 80GB PCIe": {
        "total_memory_bytes": 85_899_345_920,
        "compute_capability": "8.0",
    },
    "NVIDIA A100-SXM4-80GB": {
        "total_memory_bytes": 85_899_345_920,
        "compute_capability": "8.0",
    },
    "NVIDIA H100 80GB HBM3": {
        "total_memory_bytes": 85_899_345_920,
        "compute_capability": "9.0",
    },
    # Phase 1b additions (§6.2).  Type IDs and memory are re-checked against
    # runpod.get_gpus() and the detected hardware at L0.
    "NVIDIA GeForce RTX 3090": {
        "total_memory_bytes": 25_769_803_776,
        "compute_capability": "8.6",
    },
    "NVIDIA L40": {
        # 48 GB nominal; 46,068 MiB usable as reported by the driver
        "total_memory_bytes": 48_305_799_168,
        "compute_capability": "8.9",
    },
    "NVIDIA B200": {
        # class 6 retarget target: SM100 (compute capability 10.0)
        "total_memory_bytes": 192_265_846_784,
        "compute_capability": "10.0",
    },
}

_POD_COST_QUERY = """query Pod {{
  pod(input: {{podId: "{pod_id}"}}) {{
    id
    costPerHr
    runtime {{ uptimeInSeconds }}
  }}
}}"""

_POD_STATUS_QUERY = """query Pod {{
  pod(input: {{podId: "{pod_id}"}}) {{
    id
    name
    runtime {{
      uptimeInSeconds
      ports {{ ip isIpPublic privatePort publicPort type }}
      gpus {{ id gpuUtilPercent memoryUtilPercent }}
    }}
  }}
}}"""


def _default_leak_log() -> Path:
    # Read at call time so tests can redirect it (tests/conftest.py).
    return DEFAULT_LEAK_LOG


class _EphemeralHostKeyPolicy:
    """Accept SSH host keys from ephemeral cloud containers.

    RunPod pods generate host keys at boot (start.sh line 12-14).
    No prior known key exists. Logs the key fingerprint for audit.
    """

    def missing_host_key(self, client, hostname, key):  # type: ignore[no-untyped-def]
        logger.info("Ephemeral host key %s for %s", key.get_fingerprint().hex(), hostname)


class RunPodTarget:
    """ExecutionTarget for RunPod Secure Cloud GPUs."""

    _kind = "rented-provider"
    _operator = "apron"
    _provider = "runpod"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        ssh_key_path: str | None = None,
        image: str = DEFAULT_IMAGE,
        gpu_type: str | None = None,
        gpu_count: int = 1,
        max_uptime: int = DEFAULT_MAX_UPTIME,
        ssh_timeout: int = 30,
        command_timeout: int = 600,
        leak_log: Path | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        self._ssh_key_path = ssh_key_path or os.environ.get(
            "RUNPOD_SSH_KEY_PATH",
            self._detect_ssh_key(),
        )
        self._image = image
        self._gpu_type = gpu_type
        if gpu_count < 1:
            raise ValueError(f"gpu_count must be >= 1, got {gpu_count}")
        self._gpu_count = gpu_count
        self._max_uptime = max_uptime
        self._leak_log = leak_log or _default_leak_log()
        self._atexit_guard: Callable[[], None] | None = None
        self._ssh_timeout = ssh_timeout
        self._command_timeout = command_timeout

        self._pod_id: str | None = None
        self._ssh: Any | None = None
        self._ssh_host: str | None = None
        self._ssh_port: int | None = None
        self._hardware: HardwareSpec | None = None
        self._execution_fingerprint_hex: str | None = None

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def operator(self) -> str:
        return self._operator

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def hardware(self) -> HardwareSpec:
        if self._hardware is None:
            raise RuntimeError("hardware not detected — call provision() first")
        return self._hardware

    @property
    def execution_fingerprint(self) -> str:
        if self._execution_fingerprint_hex is None:
            raise RuntimeError("execution fingerprint not built — call provision() first")
        return self._execution_fingerprint_hex

    @property
    def gpu_count(self) -> int:
        return self._gpu_count

    @property
    def pod_id(self) -> str | None:
        return self._pod_id

    @property
    def proxy_url(self) -> str | None:
        if self._pod_id is None:
            return None
        return f"https://{self._pod_id}-8000.proxy.runpod.net"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def prepare(self) -> dict[str, Any]:
        if not self._api_key:
            return {"status": "hardware_unavailable", "reason": "RUNPOD_API_KEY not set"}

        try:
            available = self.discover_gpus()
            if not available:
                return {"status": "hardware_unavailable", "reason": "No GPUs available"}
        except Exception as exc:
            return {"status": "hardware_unavailable", "reason": f"RunPod API error: {exc}"}

        return {"status": "ready", "available_gpus": available}

    def discover_gpus(self) -> list[dict[str, Any]]:
        """Query RunPod for available GPU types with specs and pricing.

        Uses get_gpu() per known type because get_gpus() omits pricing.
        """
        import runpod as _runpod  # type: ignore[import-untyped]

        _runpod.api_key = self._api_key

        available: list[dict[str, Any]] = []
        for gpu_id, specs in GPU_SPECS.items():
            try:
                gpu = _runpod.get_gpu(gpu_id)
            except Exception:
                continue
            # D4: Secure only.  A GPU without a Secure price is unavailable;
            # the Community price is never used, not even as a fallback.
            secure_price = gpu.get("securePrice") or 0
            if not secure_price:
                continue
            available.append(
                {
                    "gpu_type_id": gpu_id,
                    "hourly_rate_usd": float(secure_price),
                    "secure_price": float(secure_price),
                    "hardware_spec": HardwareSpec(
                        gpu_sku=gpu_id,
                        total_memory_bytes=specs["total_memory_bytes"],
                        compute_capability=specs["compute_capability"],
                    ),
                }
            )
        return sorted(available, key=lambda g: g["hourly_rate_usd"])

    def provision(
        self,
        wait_timeout: int = 0,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if not self._api_key:
            raise RuntimeError("Cannot provision without RUNPOD_API_KEY")
        if not self._gpu_type:
            raise RuntimeError("No gpu_type set — call select_gpu() or pass gpu_type")

        if self._pod_id is not None:
            # Never orphan a live pod by overwriting its ID and guard.
            logger.warning("provision() with live pod %s — tearing it down first", self._pod_id)
            self.teardown()

        import runpod as _runpod  # type: ignore[import-untyped]

        _runpod.api_key = self._api_key

        pod = _runpod.create_pod(
            name=POD_NAME_PREFIX,
            image_name=self._image,
            gpu_type_id=self._gpu_type,
            gpu_count=self._gpu_count,
            cloud_type=CLOUD_TYPE,
            ports="22/tcp,8000/http",
            volume_in_gb=100,
            container_disk_in_gb=50,
            env=env or {},
        )
        self._pod_id = pod["id"]
        logger.info("Pod created: %s", self._pod_id)
        self._register_teardown_guard()

        pod_info = self._wait_for_running(wait_timeout)

        self._establish_ssh(pod_info)

        hw_info = self._detect_hardware()
        self._hardware = HardwareSpec(
            gpu_sku=hw_info["gpu_name"],
            total_memory_bytes=hw_info["total_memory_bytes"],
            compute_capability=hw_info["compute_capability"],
        )

        self._build_execution_fingerprint(hw_info)

        return {
            "status": "provisioned",
            "pod_id": self._pod_id,
            "hardware": self._hardware.model_dump(mode="json"),
            "detected_gpu_count": hw_info.get("gpu_count", 1),
        }

    # ------------------------------------------------------------------
    # Pod safety net (§6.2)
    # ------------------------------------------------------------------

    def _register_teardown_guard(self) -> None:
        """Terminate the pod at interpreter exit if teardown never ran.

        Captures the pod ID and key by value (H14): the guard must not
        depend on ``self``, which may be mid-teardown or gone.  SIGKILL is
        not covered — the next start's ``cleanup_orphaned_pods`` is.
        """
        import runpod as _sdk  # type: ignore[import-untyped]

        pod_id, api_key = self._pod_id, self._api_key

        # The SDK module is captured too: importing during interpreter
        # shutdown can fail, and the guard runs exactly then.
        def _teardown_guard(
            pid: str | None = pod_id, key: str | None = api_key, _rp: Any = _sdk
        ) -> None:
            _rp.api_key = key
            try:
                _rp.terminate_pod(pid)
                logger.warning("atexit: terminated pod %s", pid)
            except Exception as exc:
                logger.error("atexit: failed to terminate pod %s: %s", pid, mask_secrets(str(exc)))

        self._atexit_guard = _teardown_guard
        atexit.register(_teardown_guard)

    def _unregister_teardown_guard(self) -> None:
        if self._atexit_guard is not None:
            atexit.unregister(self._atexit_guard)
            self._atexit_guard = None

    def _record_leaked_pod(self, pod_id: str, error: str) -> None:
        path = self._leak_log
        path.parent.mkdir(parents=True, exist_ok=True)
        leaked: list[dict[str, Any]] = []
        if path.exists():
            leaked = json.loads(path.read_text() or "[]")
        leaked.append({"pod_id": pod_id, "error": mask_secrets(error), "at": time.time()})
        path.write_text(json.dumps(leaked, indent=2) + "\n")

    def cleanup_orphaned_pods(self, max_age_seconds: int = 3600) -> list[str]:
        """Terminate Apron pods older than *max_age_seconds* (H10).

        Runs at orchestrator start: a pod whose process was SIGKILLed has no
        atexit guard.  Only pods named ``apron-run*`` with a reported uptime
        above the age are touched.  Returns the terminated pod IDs.
        """
        if not self._api_key:
            return []
        import runpod as _runpod  # type: ignore[import-untyped]

        _runpod.api_key = self._api_key
        terminated: list[str] = []
        for pod in _runpod.get_pods() or []:
            name = str(pod.get("name") or "")
            runtime = pod.get("runtime") or {}
            uptime = int(runtime.get("uptimeInSeconds") or 0)
            if not name.startswith(POD_NAME_PREFIX) or uptime <= max_age_seconds:
                continue
            try:
                _runpod.terminate_pod(pod["id"])
                terminated.append(pod["id"])
                logger.warning("Orphan cleanup: terminated %s (uptime %ds)", pod["id"], uptime)
            except Exception as exc:
                logger.error("Orphan cleanup failed for %s: %s", pod["id"], mask_secrets(str(exc)))
                self._record_leaked_pod(pod["id"], str(exc))
        return terminated

    def list_apron_pods(self) -> list[dict[str, Any]]:
        """Pods named ``apron-run*`` on the account — the before/after session check."""
        if not self._api_key:
            return []
        import runpod as _runpod  # type: ignore[import-untyped]

        _runpod.api_key = self._api_key
        return [
            {
                "id": pod.get("id"),
                "name": pod.get("name"),
                "uptime_seconds": (pod.get("runtime") or {}).get("uptimeInSeconds"),
            }
            for pod in _runpod.get_pods() or []
            if str(pod.get("name") or "").startswith(POD_NAME_PREFIX)
        ]

    def pod_reported_cost(self, pod_id: str | None = None) -> float | None:
        """Cost RunPod reports for a pod (default: the current one): costPerHr x uptime (M3).

        Read before teardown, or by ledger replay for a pod a crashed run left
        behind; ``None`` when the API does not answer.
        """
        pod_id = pod_id or self._pod_id
        if pod_id is None:
            return None
        try:
            data = self._gql_status(_POD_COST_QUERY.format(pod_id=pod_id))
        except Exception:
            logger.debug("pod cost query failed", exc_info=True)
            return None
        pod = data.get("pod") or {}
        rate = pod.get("costPerHr")
        uptime = (pod.get("runtime") or {}).get("uptimeInSeconds")
        if rate is None or uptime is None:
            return None
        return round(float(rate) * float(uptime) / 3600, 6)

    def execute(self, command: str, retries: int = 2) -> dict[str, Any]:
        import paramiko as _paramiko

        for attempt in range(retries + 1):
            try:
                self._ensure_ssh()
                assert self._ssh is not None
                _, stdout, stderr = self._ssh.exec_command(command, timeout=self._command_timeout)
                exit_code = stdout.channel.recv_exit_status()
                return {
                    "stdout": stdout.read().decode(),
                    "stderr": stderr.read().decode(),
                    "exit_code": exit_code,
                }
            except (_paramiko.SSHException, OSError, EOFError):
                if attempt == retries:
                    raise
                logger.warning("SSH connection lost, reconnecting (attempt %d)", attempt + 1)
                time.sleep(2**attempt)
                self._ssh = None

        raise RuntimeError("execute() exhausted retries")

    def observe(self) -> dict[str, Any]:
        result = self.execute(
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
            "--format=csv,noheader,nounits"
        )
        if result["exit_code"] != 0:
            return {"error": result["stderr"]}

        parts = result["stdout"].strip().split(",")
        if len(parts) >= 3:
            return {
                "gpu_utilization_pct": float(parts[0].strip()),
                "memory_used_mib": float(parts[1].strip()),
                "memory_total_mib": float(parts[2].strip()),
            }
        return {"raw": result["stdout"]}

    def collect(self, remote_paths: list[str] | None = None) -> dict[str, Any]:
        import paramiko as _paramiko

        self._ensure_ssh()
        assert self._ssh is not None

        try:
            sftp = self._ssh.open_sftp()
        except _paramiko.SSHException as exc:
            return {"error": f"SFTP failed: {exc}"}

        results: dict[str, Any] = {}
        paths = remote_paths or ["/workspace"]

        for path in paths:
            try:
                content = sftp.file(path, "r").read()
                results[path] = content.decode() if isinstance(content, bytes) else content
            except FileNotFoundError:
                results[path] = None
            except Exception as exc:
                results[path] = f"error: {exc}"

        sftp.close()
        return results

    def teardown(self) -> None:
        """Terminate the pod: 3 attempts with 2/4/8 s backoff, then record a leak.

        Idempotent.  On final failure the pod ID is logged at ERROR, appended
        to the leaked-pods list and ``PodLeakError`` is raised so the caller
        stops (a pod that cannot be terminated is an owner stop condition).
        The atexit guard is unregistered either way (it would retry the same
        failing call).
        """
        if self._ssh is not None:
            with contextlib.suppress(Exception):
                self._ssh.close()
            self._ssh = None

        if self._pod_id is None:
            self._unregister_teardown_guard()
            return

        pod_id = self._pod_id
        import runpod as _runpod  # type: ignore[import-untyped]

        _runpod.api_key = self._api_key
        last_error = ""
        for attempt, delay in enumerate(TEARDOWN_BACKOFF_SECONDS, start=1):
            try:
                _runpod.terminate_pod(pod_id)
                logger.info("Pod %s terminated", pod_id)
                break
            except Exception as exc:
                last_error = mask_secrets(str(exc))
                logger.warning(
                    "Pod %s termination attempt %d failed: %s", pod_id, attempt, last_error
                )
                if attempt < len(TEARDOWN_BACKOFF_SECONDS):
                    time.sleep(delay)
        else:
            logger.error("Pod %s could not be terminated: %s", pod_id, last_error)
            self._record_leaked_pod(pod_id, last_error)
            self._pod_id = None
            self._unregister_teardown_guard()
            # Owner stop condition: the orchestrator must see this and stop.
            raise PodLeakError(pod_id, last_error)
        self._pod_id = None
        self._unregister_teardown_guard()

    # ------------------------------------------------------------------
    # Environment variable builder for apron runner image
    # ------------------------------------------------------------------

    @staticmethod
    def build_env(
        model_id: str | None = None,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 640,
        tensor_parallel: int = 1,
        trust_remote_code: bool = False,
        ssh_public_key: str | None = None,
        hf_token: str | None = None,
    ) -> dict[str, str]:
        """Build env vars for the apron runner image.

        With ``model_id`` the image boots vLLM itself at container start.
        Without it the container waits and the orchestrator boots vLLM over
        SSH from a rendered DeploymentPlan (the cohort path).

        ``hf_token`` (from the caller's environment, never from code or
        files) reaches only the image's download step: ``start.sh`` moves it
        into a root-only file and unsets it before anything else runs (F7).
        """
        env: dict[str, str] = {"VLLM_LOGGING_LEVEL": "DEBUG"}
        if model_id:
            env.update(
                {
                    "VLLM_MODEL": model_id,
                    "VLLM_TOKENIZER": model_id,
                    "VLLM_DTYPE": dtype,
                    "VLLM_GPU_MEMORY_UTILIZATION": str(gpu_memory_utilization),
                    "VLLM_MAX_MODEL_LEN": str(max_model_len),
                    "VLLM_TENSOR_PARALLEL_SIZE": str(tensor_parallel),
                }
            )
            if trust_remote_code:
                env["VLLM_TRUST_REMOTE_CODE"] = "1"
        if ssh_public_key:
            env["PUBLIC_KEY"] = ssh_public_key
        if hf_token:
            env["HF_TOKEN"] = hf_token
        return env

    # ------------------------------------------------------------------
    # Internal — status polling (raw GraphQL, SDK lacks runtime fields)
    # ------------------------------------------------------------------

    def _gql_status(self, query: str) -> dict[str, Any]:
        """Raw GraphQL for pod status — auth via query param per RunPod convention."""
        resp = httpx.post(
            f"{GRAPHQL_URL}?api_key={self._api_key}",
            json={"query": query},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # The message embeds the request URL, which carries the key.
            raise RuntimeError(mask_secrets(str(exc))) from None
        data = resp.json()
        if "errors" in data:
            raise RuntimeError(f"GraphQL errors: {json.dumps(data['errors'])}")
        return data.get("data", {})

    def _wait_for_running(self, timeout: int = 0) -> dict[str, Any]:
        """Poll until pod reaches RUNNING. No timeout by default — image
        pulls and model downloads can take arbitrarily long."""
        start = time.monotonic()
        while True:
            if timeout > 0 and time.monotonic() - start > timeout:
                raise TimeoutError(f"Pod {self._pod_id} did not reach RUNNING within {timeout}s")
            query = _POD_STATUS_QUERY.format(pod_id=self._pod_id)
            data = self._gql_status(query)
            pod = data.get("pod")
            if pod is None:
                time.sleep(10)
                continue
            runtime = pod.get("runtime")
            if runtime and runtime.get("uptimeInSeconds", 0) > 0:
                return pod
            time.sleep(10)

        raise TimeoutError(f"Pod {self._pod_id} did not reach RUNNING within {timeout}s")

    # ------------------------------------------------------------------
    # Internal — SSH management
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_ssh_key() -> str:
        ssh_dir = Path.home() / ".ssh"
        for name in ("id_ed25519", "id_rsa", "id_ecdsa"):
            if (ssh_dir / name).exists():
                return str(ssh_dir / name)
        return str(ssh_dir / "id_rsa")

    def _establish_ssh(
        self, pod_info: dict[str, Any], retries: int = 6, backoff: int = 10
    ) -> None:
        import paramiko as _paramiko

        ports = pod_info["runtime"]["ports"]
        ssh_port_info = next((p for p in ports if p["privatePort"] == 22), None)
        if ssh_port_info is None:
            raise RuntimeError("No SSH port mapping found on pod")

        ssh_host: str = ssh_port_info["ip"]
        ssh_port: int = int(ssh_port_info["publicPort"])
        self._ssh_host = ssh_host
        self._ssh_port = ssh_port

        for attempt in range(retries):
            try:
                client = _paramiko.SSHClient()
                client.set_missing_host_key_policy(_EphemeralHostKeyPolicy())  # type: ignore[arg-type]
                client.connect(
                    ssh_host,
                    port=ssh_port,
                    username="root",
                    key_filename=self._ssh_key_path,
                    timeout=self._ssh_timeout,
                )
                self._ssh = client
                return
            except (OSError, _paramiko.SSHException) as exc:
                if attempt == retries - 1:
                    raise RuntimeError(
                        f"SSH to {ssh_host}:{ssh_port} failed after {retries} attempts: {exc}"
                    ) from exc
                logger.info("SSH not ready, retrying in %ds (attempt %d)", backoff, attempt + 1)
                time.sleep(backoff)

    def _ensure_ssh(self) -> None:
        import paramiko as _paramiko

        if self._ssh is not None:
            transport = self._ssh.get_transport()
            if transport is not None and transport.is_active():
                return

        ssh_host = self._ssh_host
        ssh_port = self._ssh_port
        if ssh_host is None or ssh_port is None:
            raise RuntimeError("SSH host/port not set — call provision() first")

        client = _paramiko.SSHClient()
        client.set_missing_host_key_policy(_EphemeralHostKeyPolicy())  # type: ignore[arg-type]
        client.connect(
            ssh_host,
            port=ssh_port,
            username="root",
            key_filename=self._ssh_key_path,
            timeout=self._ssh_timeout,
        )
        self._ssh = client

    # ------------------------------------------------------------------
    # Internal — hardware detection + fingerprint
    # ------------------------------------------------------------------

    def _detect_hardware(self) -> dict[str, Any]:
        detection_script = (
            "import torch, subprocess, json; "
            "props = torch.cuda.get_device_properties(0); "
            "nvsmi = subprocess.run("
            "['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],"
            "capture_output=True,text=True); "
            "print(json.dumps({"
            "'gpu_name': torch.cuda.get_device_name(),"
            "'total_memory_bytes': props.total_memory,"
            "'compute_capability': f'{props.major}.{props.minor}',"
            "'driver_version': nvsmi.stdout.strip(),"
            "'cuda_version': torch.version.cuda,"
            "'pytorch_version': torch.__version__,"
            "'gpu_count': torch.cuda.device_count()"
            "}))"
        )
        result = self.execute(
            f"source /etc/apron_environment 2>/dev/null; "
            f'/opt/venv/bin/python3 -c "{detection_script}"'
        )
        if result["exit_code"] != 0:
            raise RuntimeError(f"Hardware detection failed: {result['stderr']}")
        return json.loads(result["stdout"].strip())

    def _build_execution_fingerprint(self, hw_info: dict[str, Any]) -> None:
        fp_data = {
            "gpu_sku": hw_info["gpu_name"],
            "total_memory_bytes": hw_info["total_memory_bytes"],
            "compute_capability": hw_info["compute_capability"],
            "driver_version": hw_info.get("driver_version", ""),
            "cuda_version": hw_info.get("cuda_version", ""),
            "pytorch_version": hw_info.get("pytorch_version", ""),
            "image": self._image,
            "provider": self._provider,
            "gpu_count": hw_info.get("gpu_count", self._gpu_count),
        }
        canonical = canonicalize(fp_data)
        self._execution_fingerprint_hex = digest_hex(canonical)
