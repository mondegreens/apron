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
DEFAULT_IMAGE = "vllm/vllm-openai:v0.29.0"
DEFAULT_GPU_TYPE = "NVIDIA GeForce RTX 4090"
DEFAULT_MAX_UPTIME = 3600

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
        gpu_type: str = DEFAULT_GPU_TYPE,
        max_uptime: int = DEFAULT_MAX_UPTIME,
        ssh_timeout: int = 30,
        command_timeout: int = 600,
    ) -> None:
        self._api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        self._ssh_key_path = ssh_key_path or os.environ.get(
            "RUNPOD_SSH_KEY_PATH",
            str(Path.home() / ".ssh" / "id_ed25519"),
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
            import runpod as _runpod

            _runpod.api_key = self._api_key
            gpus = _runpod.get_gpus()
            if not gpus:
                return {"status": "hardware_unavailable", "reason": "No GPUs available"}
        except Exception as exc:
            return {"status": "hardware_unavailable", "reason": f"RunPod API error: {exc}"}

        return {"status": "ready"}

    def provision(self, wait_timeout: int = 300) -> dict[str, Any]:
        if not self._api_key:
            raise RuntimeError("Cannot provision without RUNPOD_API_KEY")

        import runpod as _runpod

        _runpod.api_key = self._api_key

        pod = _runpod.create_pod(
            name="apron-run",
            image_name=self._image,
            gpu_type_id=self._gpu_type,
            gpu_count=1,
            cloud_type="SECURE",
            ports="22/tcp,8000/http",
            volume_in_gb=50,
            container_disk_in_gb=20,
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
            except (_paramiko.SSHException, OSError):
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

    def _wait_for_running(self, timeout: int) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
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

    def _establish_ssh(self, pod_info: dict[str, Any]) -> None:
        import paramiko as _paramiko

        ports = pod_info["runtime"]["ports"]
        ssh_port_info = next((p for p in ports if p["privatePort"] == 22), None)
        if ssh_port_info is None:
            raise RuntimeError("No SSH port mapping found on pod")

        ssh_host: str = ssh_port_info["ip"]
        ssh_port: int = int(ssh_port_info["publicPort"])
        self._ssh_host = ssh_host
        self._ssh_port = ssh_port

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
            "'total_memory_bytes': props.total_mem,"
            "'compute_capability': f'{props.major}.{props.minor}',"
            "'driver_version': nvsmi.stdout.strip(),"
            "'cuda_version': torch.version.cuda,"
            "'pytorch_version': torch.__version__"
            "}))"
        )
        result = self.execute(f'python3 -c "{detection_script}"')
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
