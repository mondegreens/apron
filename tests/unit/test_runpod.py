"""Unit tests for RunPod ExecutionTarget adapter.

Collects the ExecutionTarget conformance suite via pytest_plugins.
The execution_target fixture below provides a pre-provisioned RunPodTarget
that satisfies conformance without real API calls.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from apron.adapters.backends.runpod import RunPodTarget
from apron.domain.schemas.primitives import ExecutionTarget

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
def runpod_target() -> RunPodTarget:
    return RunPodTarget(api_key="test-key", ssh_key_path="/tmp/test_key")


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


def test_teardown_sdk_failure_does_not_raise(runpod_target: RunPodTarget) -> None:
    runpod_target._pod_id = "pod-123"
    mock_runpod = MagicMock()
    mock_runpod.terminate_pod.side_effect = RuntimeError("network")
    with patch.dict("sys.modules", {"runpod": mock_runpod}):
        runpod_target.teardown()
    assert runpod_target._pod_id is None


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
