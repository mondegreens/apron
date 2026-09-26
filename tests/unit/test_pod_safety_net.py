"""§6.2 pod safety net: atexit guard (by value), teardown retry, leak list, orphans."""

from __future__ import annotations

import atexit
import json
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from apron.adapters.backends import runpod as runpod_module
from apron.adapters.backends.runpod import TEARDOWN_BACKOFF_SECONDS, RunPodTarget
from apron.application.orchestration.errors import PodLeakError

if TYPE_CHECKING:
    from pathlib import Path


class _Sdk(SimpleNamespace):
    def __init__(self, *, fail_terminate: int = 0, pods: list[dict[str, Any]] | None = None):
        super().__init__(api_key=None)
        self.terminated: list[str] = []
        self.created: list[dict[str, Any]] = []
        self._fail = fail_terminate
        self._pods = pods or []

    def create_pod(self, **kwargs: Any) -> dict[str, Any]:
        self.created.append(kwargs)
        return {"id": f"pod-{len(self.created)}"}

    def terminate_pod(self, pod_id: str) -> None:
        if self._fail > 0:
            self._fail -= 1
            raise RuntimeError("503 from api?api_key=rpa_" + "S" * 30)
        self.terminated.append(pod_id)

    def get_pods(self) -> list[dict[str, Any]]:
        return self._pods


@pytest.fixture()
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(runpod_module.time, "sleep", recorded.append)
    return recorded


def _target(tmp_path: Path, **kwargs: Any) -> RunPodTarget:
    return RunPodTarget(
        api_key="k", ssh_key_path="/tmp/k", leak_log=tmp_path / "leaked.json", **kwargs
    )


def test_guard_registered_after_pod_creation_and_unregistered_by_teardown(
    tmp_path: Path, sleeps: list[float]
) -> None:
    sdk = _Sdk()
    target = _target(tmp_path)
    registered: list[Any] = []
    unregistered: list[Any] = []
    with (
        patch.dict(sys.modules, {"runpod": sdk}),
        patch.object(atexit, "register", registered.append),
        patch.object(atexit, "unregister", unregistered.append),
    ):
        target._pod_id = "pod-7"
        target._register_teardown_guard()
        target.teardown()
    assert len(registered) == 1
    assert unregistered == registered
    assert sdk.terminated == ["pod-7"]


def test_guard_captures_pod_and_key_by_value(tmp_path: Path) -> None:
    sdk = _Sdk()
    target = _target(tmp_path)
    with patch.dict(sys.modules, {"runpod": sdk}), patch.object(atexit, "register"):
        target._pod_id = "pod-a"
        target._register_teardown_guard()
    guard = target._atexit_guard
    assert guard is not None
    target._pod_id = None  # self no longer knows the pod
    target._api_key = None
    guard()
    assert sdk.terminated == ["pod-a"]
    assert sdk.api_key == "k"


def test_teardown_retries_with_backoff_then_succeeds(tmp_path: Path, sleeps: list[float]) -> None:
    sdk = _Sdk(fail_terminate=2)
    target = _target(tmp_path)
    target._pod_id = "pod-r"
    with patch.dict(sys.modules, {"runpod": sdk}):
        target.teardown()
    assert sdk.terminated == ["pod-r"]
    assert sleeps == [2, 4]
    assert not (tmp_path / "leaked.json").exists()


def test_teardown_final_failure_appends_to_leaked_list_masked(
    tmp_path: Path, sleeps: list[float]
) -> None:
    sdk = _Sdk(fail_terminate=len(TEARDOWN_BACKOFF_SECONDS))
    target = _target(tmp_path)
    target._pod_id = "pod-x"
    unregistered: list[Any] = []
    with (
        patch.dict(sys.modules, {"runpod": sdk}),
        patch.object(atexit, "unregister", unregistered.append),
    ):
        target._atexit_guard = lambda: None
        with pytest.raises(PodLeakError, match="pod-x"):
            target.teardown()
    leaked = json.loads((tmp_path / "leaked.json").read_text())
    assert [entry["pod_id"] for entry in leaked] == ["pod-x"]
    assert "rpa_" not in leaked[0]["error"]
    assert sleeps == [2, 4]
    assert target._pod_id is None
    assert len(unregistered) == 1


def test_orphan_cleanup_filters_by_name_and_age(tmp_path: Path) -> None:
    pods = [
        {"id": "old", "name": "apron-run", "runtime": {"uptimeInSeconds": 7200}},
        {"id": "young", "name": "apron-run", "runtime": {"uptimeInSeconds": 60}},
        {"id": "other", "name": "someone-else", "runtime": {"uptimeInSeconds": 99999}},
        {"id": "starting", "name": "apron-run", "runtime": None},
    ]
    sdk = _Sdk(pods=pods)
    with patch.dict(sys.modules, {"runpod": sdk}):
        terminated = _target(tmp_path).cleanup_orphaned_pods(max_age_seconds=3600)
    assert terminated == ["old"]
    assert sdk.terminated == ["old"]


def test_second_provision_tears_down_the_live_pod_first(
    tmp_path: Path, sleeps: list[float]
) -> None:
    sdk = _Sdk()
    target = _target(tmp_path, gpu_type="NVIDIA GeForce RTX 4090")
    target._pod_id = "pod-live"
    with (
        patch.dict(sys.modules, {"runpod": sdk}),
        patch.object(target, "_wait_for_running", side_effect=RuntimeError("stop here")),
        patch.object(atexit, "register"),
        pytest.raises(RuntimeError, match="stop here"),
    ):
        target.provision()
    assert sdk.terminated == ["pod-live"]


def test_graphql_error_message_does_not_leak_the_key(tmp_path: Path) -> None:
    import httpx

    request = httpx.Request("POST", "https://api.runpod.io/graphql?api_key=rpa_" + "Z" * 30)
    response = httpx.Response(500, request=request)
    target = _target(tmp_path)
    with (
        patch.object(runpod_module.httpx, "post", return_value=response),
        pytest.raises(RuntimeError) as exc,
    ):
        target._gql_status("query {}")
    assert "rpa_" not in str(exc.value)


def test_pod_reported_cost_uses_cost_per_hour_and_uptime(tmp_path: Path) -> None:
    target = _target(tmp_path)
    target._pod_id = "pod-c"
    reply = {"pod": {"costPerHr": 0.74, "runtime": {"uptimeInSeconds": 1800}}}
    with patch.object(target, "_gql_status", return_value=reply):
        assert target.pod_reported_cost() == pytest.approx(0.37)
    with patch.object(target, "_gql_status", side_effect=RuntimeError("down")):
        assert target.pod_reported_cost() is None


def test_mock_is_not_leaking_into_other_tests() -> None:
    assert not isinstance(runpod_module.time.sleep, MagicMock)
