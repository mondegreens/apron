"""RunPod ExecutionTarget — rented-provider adapter for RunPod Secure Cloud GPUs.

Implements the ExecutionTarget Protocol (domain/schemas/primitives.py).

Control plane: ``runpod`` Python SDK (create_pod / terminate_pod).
Status polling: raw GraphQL (SDK doesn't expose runtime/uptime fields).
Management plane: SSH via paramiko (kill/restart vLLM, run profiling).
Data plane: HTTPS proxy ``https://{pod_id}-8000.proxy.runpod.net``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

from apron.domain.canonical import canonicalize, digest_hex
from apron.domain.schemas.primitives import HardwareSpec

logger = logging.getLogger(__name__)

GRAPHQL_URL = "https://api.runpod.io/graphql"
DEFAULT_IMAGE = "ghcr.io/mondegreens/apron-runner:v0.29.0-rc6"
DEFAULT_MAX_UPTIME = 3600

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
}

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
        max_uptime: int = DEFAULT_MAX_UPTIME,
        ssh_timeout: int = 30,
        command_timeout: int = 600,
    ) -> None:
        self._api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        self._ssh_key_path = ssh_key_path or os.environ.get(
            "RUNPOD_SSH_KEY_PATH",
            self._detect_ssh_key(),
        )
        self._image = image
        self._gpu_type = gpu_type
        self._max_uptime = max_uptime
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
        import runpod as _runpod

        _runpod.api_key = self._api_key

        available: list[dict[str, Any]] = []
        for gpu_id, specs in GPU_SPECS.items():
            try:
                gpu = _runpod.get_gpu(gpu_id)
            except Exception:
                continue
            secure_price = gpu.get("securePrice") or 0
            community_price = gpu.get("communityPrice") or 0
            price = secure_price or community_price
            if not price:
                continue
            available.append(
                {
                    "gpu_type_id": gpu_id,
                    "hourly_rate_usd": float(price),
                    "secure_price": float(secure_price),
                    "community_price": float(community_price),
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
            raise RuntimeError(
                "No gpu_type set — call select_gpu() or pass gpu_type"
            )

        import runpod as _runpod

        _runpod.api_key = self._api_key

        pod = _runpod.create_pod(
            name="apron-run",
            image_name=self._image,
            gpu_type_id=self._gpu_type,
            gpu_count=1,
            cloud_type="SECURE",
            ports="22/tcp,8000/http",
            volume_in_gb=100,
            container_disk_in_gb=50,
            env=env or {},
        )
        self._pod_id = pod["id"]
        logger.info("Pod created: %s", self._pod_id)

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
        }

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
        if self._ssh is not None:
            with contextlib.suppress(Exception):
                self._ssh.close()
            self._ssh = None

        if self._pod_id is not None:
            try:
                import runpod as _runpod

                _runpod.api_key = self._api_key
                _runpod.terminate_pod(self._pod_id)
                logger.info("Pod %s terminated", self._pod_id)
            except Exception as exc:
                logger.warning("Pod termination failed (maxUptime safety net active): %s", exc)
            self._pod_id = None

    # ------------------------------------------------------------------
    # Environment variable builder for apron runner image
    # ------------------------------------------------------------------

    @staticmethod
    def build_env(
        model_id: str,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 640,
        tensor_parallel: int = 1,
        trust_remote_code: bool = False,
        ssh_public_key: str | None = None,
    ) -> dict[str, str]:
        """Build env vars for the apron runner image."""
        env: dict[str, str] = {
            "VLLM_MODEL": model_id,
            "VLLM_TOKENIZER": model_id,
            "VLLM_DTYPE": dtype,
            "VLLM_GPU_MEMORY_UTILIZATION": str(gpu_memory_utilization),
            "VLLM_MAX_MODEL_LEN": str(max_model_len),
            "VLLM_TENSOR_PARALLEL_SIZE": str(tensor_parallel),
            "VLLM_LOGGING_LEVEL": "DEBUG",
        }
        if trust_remote_code:
            env["VLLM_TRUST_REMOTE_CODE"] = "1"
        if ssh_public_key:
            env["PUBLIC_KEY"] = ssh_public_key
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
        resp.raise_for_status()
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
                raise TimeoutError(
                    f"Pod {self._pod_id} did not reach RUNNING "
                    f"within {timeout}s"
                )
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
                client.set_missing_host_key_policy(_paramiko.AutoAddPolicy())
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
        client.set_missing_host_key_policy(_paramiko.AutoAddPolicy())
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
            "'pytorch_version': torch.__version__"
            "}))"
        )
        result = self.execute(
            f'source /etc/apron_environment 2>/dev/null; '
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
        }
        canonical = canonicalize(fp_data)
        self._execution_fingerprint_hex = digest_hex(canonical)
