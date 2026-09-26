"""Unit tests for RunPod ExecutionTarget adapter.

Collects the ExecutionTarget conformance suite via pytest_plugins: the
suite's tests are imported below and run against the real RunPodTarget.
The execution_target fixture provides a provisioned RunPodTarget with only
network I/O mocked — the RunPod SDK, the GraphQL status call and SSH.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
from conformance.test_execution_target import *  # noqa: F403 — the shared suite, on the real adapter

from apron.adapters.backends.runpod import RunPodTarget
from apron.domain.schemas.primitives import ExecutionTarget

pytest_plugins = ["conformance.plugin"]

if TYPE_CHECKING:
    from collections.abc import Iterator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeSSHClient:
    def __init__(self) -> None:
        self._connected = False
        self._commands: list[str] = []

    def set_missing_host_key_policy(self, policy: Any) -> None:
        pass

    def connect(self, *args: Any, **kwargs: Any) -> None:
        self._connected = True

    def exec_command(
        self, command: str, timeout: int | None = None
    ) -> tuple[Any, MagicMock, MagicMock]:
        self._commands.append(command)
        stdout = MagicMock()
        stderr = MagicMock()
        stdout.read.return_value = b'{"exit_code": 0}'
        stderr.read.return_value = b""
        stdout.channel.recv_exit_status.return_value = 0
        return MagicMock(), stdout, stderr

    def get_transport(self) -> MagicMock:
        transport = MagicMock()
        transport.is_active.return_value = self._connected
        return transport

    def open_sftp(self) -> MagicMock:
        sftp = MagicMock()
        return sftp

    def close(self) -> None:
        self._connected = False


@pytest.fixture()
def runpod_target(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> RunPodTarget:
    monkeypatch.setattr("apron.adapters.backends.runpod.time.sleep", lambda s: None)
    return RunPodTarget(
        api_key="test-key", ssh_key_path="/tmp/test_key", leak_log=tmp_path / "leaked.json"
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_execution_target_protocol(runpod_target: RunPodTarget) -> None:
    assert isinstance(runpod_target, ExecutionTarget)


def test_kind_is_rented_provider(runpod_target: RunPodTarget) -> None:
    assert runpod_target.kind == "rented-provider"


def test_operator_is_apron(runpod_target: RunPodTarget) -> None:
    assert runpod_target.operator == "apron"


def test_provider_is_runpod(runpod_target: RunPodTarget) -> None:
    assert runpod_target.provider == "runpod"


# ---------------------------------------------------------------------------
# prepare()
# ---------------------------------------------------------------------------


def test_prepare_without_api_key() -> None:
    target = RunPodTarget(api_key="")
    result = target.prepare()
    assert result["status"] == "hardware_unavailable"
    assert "RUNPOD_API_KEY" in result["reason"]


def test_prepare_with_api_key_checks_api(runpod_target: RunPodTarget) -> None:
    mock_runpod = MagicMock()
    mock_runpod.get_gpu.return_value = {
        "id": "NVIDIA GeForce RTX 4090",
        "securePrice": 0.74,
        "communityPrice": 0.34,
        "memoryInGb": 24,
    }
    with patch.dict("sys.modules", {"runpod": mock_runpod}):
        result = runpod_target.prepare()
    assert result["status"] == "ready"
    assert len(result["available_gpus"]) >= 1


def test_prepare_api_failure(runpod_target: RunPodTarget) -> None:
    mock_runpod = MagicMock()
    mock_runpod.get_gpu.side_effect = RuntimeError("connection refused")
    with patch.dict("sys.modules", {"runpod": mock_runpod}):
        result = runpod_target.prepare()
    assert result["status"] == "hardware_unavailable"


# ---------------------------------------------------------------------------
# hardware property
# ---------------------------------------------------------------------------


def test_hardware_before_provision_raises(runpod_target: RunPodTarget) -> None:
    with pytest.raises(RuntimeError, match="call provision"):
        _ = runpod_target.hardware


def test_execution_fingerprint_before_provision_raises(runpod_target: RunPodTarget) -> None:
    with pytest.raises(RuntimeError, match="call provision"):
        _ = runpod_target.execution_fingerprint


# ---------------------------------------------------------------------------
# teardown()
# ---------------------------------------------------------------------------


def test_teardown_idempotent(runpod_target: RunPodTarget) -> None:
    runpod_target.teardown()
    runpod_target.teardown()


def test_teardown_with_ssh(runpod_target: RunPodTarget) -> None:
    runpod_target._ssh = FakeSSHClient()
    runpod_target._pod_id = "pod-123"
    mock_runpod = MagicMock()
    with patch.dict("sys.modules", {"runpod": mock_runpod}):
        runpod_target.teardown()
    assert runpod_target._ssh is None
    assert runpod_target._pod_id is None
    mock_runpod.terminate_pod.assert_called_once_with("pod-123")


def test_teardown_with_closed_ssh(runpod_target: RunPodTarget) -> None:
    ssh = FakeSSHClient()
    ssh.close()
    runpod_target._ssh = ssh
    runpod_target._pod_id = "pod-123"
    mock_runpod = MagicMock()
    with patch.dict("sys.modules", {"runpod": mock_runpod}):
        runpod_target.teardown()
    assert runpod_target._ssh is None


def test_teardown_sdk_failure_raises_pod_leak_after_recording(runpod_target: RunPodTarget) -> None:
    """A pod that cannot be terminated is an owner stop condition: the caller sees it."""
    from apron.application.orchestration.errors import PodLeakError

    runpod_target._pod_id = "pod-123"
    mock_runpod = MagicMock()
    mock_runpod.terminate_pod.side_effect = RuntimeError("network")
    with patch.dict("sys.modules", {"runpod": mock_runpod}), pytest.raises(PodLeakError) as exc:
        runpod_target.teardown()
    assert exc.value.pod_id == "pod-123"
    assert mock_runpod.terminate_pod.call_count == 3
    assert runpod_target._pod_id is None
    runpod_target.teardown()  # idempotent afterwards


# ---------------------------------------------------------------------------
# execute()
# ---------------------------------------------------------------------------


def test_execute_returns_structured_result(runpod_target: RunPodTarget) -> None:
    stdout_mock = MagicMock()
    stdout_mock.read.return_value = b"hello world"
    stdout_mock.channel.recv_exit_status.return_value = 0
    stderr_mock = MagicMock()
    stderr_mock.read.return_value = b""

    ssh_mock = MagicMock()
    ssh_mock.get_transport.return_value.is_active.return_value = True
    ssh_mock.exec_command.return_value = (MagicMock(), stdout_mock, stderr_mock)

    runpod_target._ssh = ssh_mock
    runpod_target._ssh_host = "test"
    runpod_target._ssh_port = 22

    result = runpod_target.execute("echo hello")
    assert result["stdout"] == "hello world"
    assert result["exit_code"] == 0


# ---------------------------------------------------------------------------
# proxy_url
# ---------------------------------------------------------------------------


def test_proxy_url_without_pod() -> None:
    target = RunPodTarget(api_key="key")
    assert target.proxy_url is None


def test_proxy_url_with_pod() -> None:
    target = RunPodTarget(api_key="key")
    target._pod_id = "abc123"
    assert target.proxy_url == "https://abc123-8000.proxy.runpod.net"


# ---------------------------------------------------------------------------
# Conformance suite fixture (F9) — network mocked, adapter real
# ---------------------------------------------------------------------------

_HW = {
    "gpu_name": "NVIDIA GeForce RTX 4090",
    "total_memory_bytes": 25_386_352_640,
    "compute_capability": "8.9",
    "driver_version": "570.124.06",
    "cuda_version": "12.8",
    "pytorch_version": "2.9.0",
}


class _PodSSH(FakeSSHClient):
    """SSH to a pod: answers hardware detection, nvidia-smi and plain commands."""

    def exec_command(self, command: str, timeout: int | None = None):  # type: ignore[no-untyped-def]
        self._commands.append(command)
        if "get_device_properties" in command:
            out = json.dumps(_HW).encode()
        elif "nvidia-smi --query-gpu=utilization" in command:
            out = b"3, 512, 24564\n"
        else:
            out = b"hello\n"
        stdout, stderr = MagicMock(), MagicMock()
        stdout.read.return_value = out
        stderr.read.return_value = b""
        stdout.channel.recv_exit_status.return_value = 0
        return MagicMock(), stdout, stderr

    def open_sftp(self) -> MagicMock:
        sftp = MagicMock()
        sftp.file.return_value.read.return_value = b"log"
        return sftp


def _graphql_response(*args: Any, **kwargs: Any) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {
        "data": {
            "pod": {
                "id": "pod-conf",
                "runtime": {
                    "uptimeInSeconds": 12,
                    "ports": [
                        {"ip": "203.0.113.7", "privatePort": 22, "publicPort": 40022},
                    ],
                },
            }
        }
    }
    return response


@pytest.fixture()
def execution_target() -> Iterator[RunPodTarget]:
    sdk = SimpleNamespace(
        api_key=None,
        get_gpu=lambda gpu_id: {"securePrice": 0.69, "communityPrice": 0.34},
        create_pod=lambda **kwargs: {"id": "pod-conf"},
        terminate_pod=lambda pod_id: None,
    )
    with (
        patch.dict(sys.modules, {"runpod": sdk}),
        patch("apron.adapters.backends.runpod.httpx.post", side_effect=_graphql_response),
        patch("paramiko.SSHClient", _PodSSH),
        patch("apron.adapters.backends.runpod.time.sleep"),
    ):
        target = RunPodTarget(
            api_key="test-key",
            ssh_key_path="/tmp/test_key",
            gpu_type="NVIDIA GeForce RTX 4090",
        )
        target.provision()
        yield target
        target.teardown()
