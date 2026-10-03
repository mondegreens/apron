"""Modal ExecutionTarget — rented-provider adapter for Modal Sandbox GPUs.

Implements the ExecutionTarget Protocol (domain/schemas/primitives.py).

Control plane: ``modal`` Python SDK (Sandbox.create / Sandbox.terminate).
Management plane: ``Sandbox.exec()`` (replaces SSH — stdout/stderr streaming).
Data plane: HTTPS via ``curl`` inside the sandbox (localhost:8000).

Unlike RunPod, Modal sandboxes:
- do not use SSH (exec replaces it);
- auto-terminate on timeout (no orphan pods);
- support Volumes for pre-downloaded weights;
- run on gvisor (GPU sandboxes use gvisor runtime).
"""

from __future__ import annotations

import atexit
import logging
import os
import shlex
import time
from typing import Any

from apron.adapters.runner_image import RUNNER_IMAGE_REPOSITORY, RunnerImage
from apron.domain.canonical import canonicalize, digest_hex
from apron.domain.schemas.primitives import HardwareSpec

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10800  # 3 hours — frontier models need 30+ min for DeepGEMM JIT
VOLUME_NAME = "apron-models"
CLOUD_TYPE = "SANDBOX"


# ---------------------------------------------------------------------------
# GPU specs — same structure as runpod.GPU_SPECS
# ---------------------------------------------------------------------------

MODAL_GPU_SPECS: dict[str, dict[str, Any]] = {
    "NVIDIA H200": {
        # The H200 count RunPod detected (runpod.GPU_SPECS): the Modal 8x H200
        # records report the same initial total as the RunPod 4x H200 ones
        # (150,109,106,995 B).  The 143,771 MiB nvidia-smi figure used before
        # was 0.60 GiB high and put the KV budget 0.54 GiB high per GPU.
        "total_memory_bytes": 150_110_011_392,
        "compute_capability": "9.0",
    },
    "NVIDIA H100": {
        "total_memory_bytes": 85_017_493_504,
        "compute_capability": "9.0",
    },
}

# Modal GPU string → SDK gpu= parameter
_MODAL_GPU_MAP: dict[str, str] = {
    "NVIDIA H200": "H200",
    "NVIDIA H100": "H100",
}


class ModalTarget:
    """ExecutionTarget for Modal Sandbox GPUs."""

    _kind = "rented-provider"
    _operator = "apron"
    _provider = "modal"
    models_dir = "/runpod-volume/models"

    def __init__(
        self,
        *,
        token_id: str | None = None,
        token_secret: str | None = None,
        image: str | None = None,
        runner: RunnerImage | None = None,
        gpu_type: str | None = None,
        gpu_count: int = 1,
        timeout: int = DEFAULT_TIMEOUT,
        volume_name: str = VOLUME_NAME,
    ) -> None:
        self._token_id = token_id or os.environ.get("MODAL_TOKEN_ID")
        self._token_secret = token_secret or os.environ.get("MODAL_TOKEN_SECRET")
        self._runner = runner
        if image:
            self._image_ref = image
        elif runner:
            self._image_ref = f"{RUNNER_IMAGE_REPOSITORY}@{runner.digest}"
        else:
            from apron.adapters.runner_image import RUNNER_IMAGE

            self._image_ref = RUNNER_IMAGE
        self._gpu_type = gpu_type
        self._gpu_count = gpu_count
        self._timeout = timeout
        self._volume_name = volume_name

        self._sandbox = None
        self._sandbox_id: str | None = None
        self._hardware: HardwareSpec | None = None
        self._provision_start: float | None = None
        self._volume: Any = None

    # -- Protocol properties --------------------------------------------------

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
            raise RuntimeError("hardware not set — call provision() first")
        return self._hardware

    @property
    def pod_id(self) -> str | None:
        return self._sandbox_id

    weights_persist = True
    weights_source: str | None = None

    @property
    def start_attempts(self) -> list[dict[str, Any]]:
        return []

    def pod_reported_cost(self) -> float | None:
        return None

    @property
    def proxy_url(self) -> str:
        if self._sandbox is None:
            raise RuntimeError("No sandbox — call provision() first")
        tunnels = self._sandbox.tunnels()
        if 8000 in tunnels:
            return tunnels[8000].url
        raise RuntimeError("No tunnel for port 8000 — provision with encrypted_ports=[8000]")

    @property
    def execution_fingerprint(self) -> str:
        parts = {
            "provider": self._provider,
            "gpu_type": self._gpu_type,
            "gpu_count": self._gpu_count,
            "image": self._image_ref,
        }
        return digest_hex(canonicalize(parts))

    # -- Protocol methods ------------------------------------------------------

    def prepare(self) -> None:
        """Validate config before provisioning."""
        if not self._gpu_type:
            raise RuntimeError("No gpu_type set")
        if self._gpu_type not in MODAL_GPU_SPECS:
            raise RuntimeError(
                f"Unknown GPU type {self._gpu_type!r}; known: {sorted(MODAL_GPU_SPECS)}"
            )

    def provision(
        self,
        wait_timeout: int = 0,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Create a Modal Sandbox with GPU."""
        import modal  # type: ignore[import-untyped]

        if self._gpu_type is None:
            raise RuntimeError("No gpu_type set")
        gpu_type: str = self._gpu_type

        if self._sandbox is not None:
            logger.warning("provision() with live sandbox — tearing down first")
            self.teardown()

        self._provision_start = time.monotonic()

        gpu_str = _MODAL_GPU_MAP.get(gpu_type, gpu_type)
        gpu_spec = f"{gpu_str}:{self._gpu_count}" if self._gpu_count > 1 else gpu_str

        app = modal.App.lookup("apron-cohort", create_if_missing=True)
        image = modal.Image.from_registry(self._image_ref, add_python="3.12")
        volume = modal.Volume.from_name(self._volume_name)
        self._volume = volume

        secrets: dict[str, str | None] = {}
        if env:
            secrets = {k: v for k, v in env.items()}
        sb = modal.Sandbox.create(
            app=app,
            image=image,
            gpu=gpu_spec,
            timeout=self._timeout,
            volumes={"/runpod-volume": volume},
            encrypted_ports=[8000],
            secrets=[modal.Secret.from_dict(secrets)] if secrets else [],
        )
        self._sandbox = sb
        self._sandbox_id = sb.object_id
        logger.info("Modal sandbox %s created (gpu=%s)", self._sandbox_id, gpu_spec)

        sandbox_ref = sb

        def _atexit_guard() -> None:
            try:
                sandbox_ref.terminate()
                logger.warning("atexit: terminated sandbox %s", self._sandbox_id)
            except Exception as exc:
                logger.error("atexit: failed to terminate: %s", exc)

        atexit.register(_atexit_guard)
        self._atexit_guard = _atexit_guard

        self.weights_source = f"modal_volume:{self._volume_name}"
        self._init_environment(env or {})

        spec = MODAL_GPU_SPECS[gpu_type]
        self._hardware = HardwareSpec(
            gpu_sku=gpu_type,
            total_memory_bytes=spec["total_memory_bytes"],
            compute_capability=spec["compute_capability"],
        )

        return {"sandbox_id": self._sandbox_id, "gpu": gpu_spec}

    def _init_environment(self, env: dict[str, str]) -> None:
        """Replicate what docker/start.sh does: HF token, env file, NCCL tuning."""
        init = [
            "mkdir -p /run/apron && chmod 700 /run/apron",
        ]
        if env.get("HF_TOKEN"):
            init.append(
                "printf '%s' \"$HF_TOKEN\" > /run/apron/hf_token && chmod 600 /run/apron/hf_token"
            )
        init.extend(
            [
                "printenv | grep -E '^VLLM_|^NVIDIA_|^NCCL_|^CUDA_|^PATH=|^LD_LIBRARY_PATH='"
                " | grep -v -E '^(HF_TOKEN|HUGGING_FACE_HUB_TOKEN)='"
                " | sed \"s/^/export /; s/=/='/; s/$/'/\""
                " > /etc/apron_environment",
                "echo 'export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-DEBUG}'"
                " >> /etc/apron_environment",
                "echo 'export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}' >> /etc/apron_environment",
                "echo 'export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}'"
                " >> /etc/apron_environment",
            ]
        )
        cmd = " && ".join(init)
        p = self._sandbox.exec("bash", "-c", cmd)  # type: ignore[union-attr]
        p.wait()
        rc = p.returncode
        if rc != 0:
            err = p.stderr.read()
            raise RuntimeError(f"_init_environment failed (rc={rc}): {err[:500]}")

    def execute(
        self,
        command: str,
        retries: int = 2,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Run command via Sandbox.exec(); return stdout/stderr/exit_code."""
        if self._sandbox is None:
            raise RuntimeError("No sandbox — call provision() first")

        for attempt in range(retries + 1):
            try:
                p = self._sandbox.exec("bash", "-c", command, timeout=timeout or 0)
                p.wait()
                out = p.stdout.read()
                err = p.stderr.read()
                return {
                    "stdout": out,
                    "stderr": err,
                    "exit_code": p.returncode,
                }
            except Exception as exc:
                if attempt == retries:
                    raise
                logger.warning("exec failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(2**attempt)

        raise RuntimeError("execute() exhausted retries")

    def observe(self) -> dict[str, Any]:
        """Read GPU utilization via nvidia-smi."""
        result = self.execute(
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
            "--format=csv,noheader,nounits"
        )
        if result.get("exit_code", 1) != 0:
            return {"raw": result.get("stderr", ""), "exit_code": result["exit_code"]}
        gpus = []
        for line in result["stdout"].strip().splitlines():
            parts = line.split(",")
            if len(parts) >= 3:
                try:
                    gpus.append(
                        {
                            "gpu_utilization_pct": float(parts[0].strip()),
                            "memory_used_mib": float(parts[1].strip()),
                            "memory_total_mib": float(parts[2].strip()),
                        }
                    )
                except ValueError:
                    gpus.append({"raw": line.strip()})
        if len(gpus) == 1:
            return gpus[0]
        return {"gpus": gpus} if gpus else {"raw": result["stdout"]}

    def collect(self, remote_paths: list[str] | None = None) -> dict[str, Any]:
        """Read files from sandbox via exec cat."""
        results: dict[str, Any] = {}
        paths = remote_paths or []
        for path in paths:
            result = self.execute(f"cat {shlex.quote(path)} 2>/dev/null")
            content = result["stdout"]
            results[path] = content if content else None
        return results

    def commit_volume(self) -> None:
        """Persist Volume writes so other sandboxes see them."""
        if self._volume is not None:
            try:
                self._volume.commit()
                logger.info("Volume %s committed", self._volume_name)
            except Exception as exc:
                logger.error("Volume commit failed: %s", exc)

    def teardown(self) -> None:
        """Terminate the sandbox. Only call when owner says."""
        if self._sandbox is None:
            return
        self.commit_volume()
        try:
            self._sandbox.terminate()
            logger.info("Terminated sandbox %s", self._sandbox_id)
        except Exception as exc:
            logger.error("Failed to terminate sandbox %s: %s", self._sandbox_id, exc)
        finally:
            if hasattr(self, "_atexit_guard"):
                atexit.unregister(self._atexit_guard)
            self._sandbox = None
            self._sandbox_id = None
