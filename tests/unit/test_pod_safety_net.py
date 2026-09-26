"""§6.2 pod safety net: atexit guard (by value), teardown retry, leak list, orphans."""

from __future__ import annotations

import atexit
import json
import re
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


# get_pods() as L0-A3 saw it: uptimeSeconds 0 for a pod up 810 s, runtime
# holding ports only.  Ages come from the single-pod query (lastStartedAt).
_LISTED = [
    {"id": "old", "name": "apron-run", "uptimeSeconds": 0, "runtime": {"ports": []}},
    {"id": "young", "name": "apron-run", "uptimeSeconds": 0, "runtime": {"ports": []}},
    {"id": "pulling", "name": "apron-run", "uptimeSeconds": 0, "runtime": None},
    {"id": "other", "name": "someone-else", "uptimeSeconds": 0, "runtime": None},
]
_NOW = 1_790_000_000.0


def _started(seconds_ago: int) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(_NOW - seconds_ago, UTC).isoformat().replace("+00:00", "Z")


_AGES = {
    "old": {"lastStartedAt": _started(7200), "runtime": {"uptimeInSeconds": 6600}},
    "young": {"lastStartedAt": _started(60), "runtime": None},
    "pulling": {"lastStartedAt": _started(4000), "runtime": None},  # never came up; bills
    "other": {"lastStartedAt": _started(99999), "runtime": None},
}


def _age_reply(query: str) -> dict[str, Any]:
    pod_id = re.search(r'podId: "([^"]+)"', query).group(1)  # type: ignore[union-attr]
    return {"pod": _AGES[pod_id]}


def test_orphan_cleanup_filters_by_name_and_age(tmp_path: Path) -> None:
    sdk = _Sdk(pods=_LISTED)
    target = _target(tmp_path)
    with (
        patch.dict(sys.modules, {"runpod": sdk}),
        patch.object(target, "_gql_status", side_effect=_age_reply),
        patch.object(runpod_module.time, "time", return_value=_NOW),
    ):
        terminated = target.cleanup_orphaned_pods(max_age_seconds=3600)
    # "pulling" is aged from when it was rented, although it has no runtime yet.
    assert terminated == ["old", "pulling"]
    assert sdk.terminated == ["old", "pulling"]


def test_listed_uptime_is_never_the_age_source(tmp_path: Path) -> None:
    """get_pods() said 0 s for a pod up 810 s; trusting it made the cleanup blind."""
    sdk = _Sdk(pods=[{"id": "old", "name": "apron-run", "uptimeSeconds": 0, "runtime": None}])
    target = _target(tmp_path)
    with (
        patch.dict(sys.modules, {"runpod": sdk}),
        patch.object(target, "_gql_status", side_effect=_age_reply),
        patch.object(runpod_module.time, "time", return_value=_NOW),
    ):
        listed = target.list_apron_pods()
    assert listed[0]["uptime_seconds"] == 7200


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
    # Rented 1800 s ago, container up only 1200 s (600 s of image pull): the
    # pull is billed, so the cost follows lastStartedAt.
    reply = {
        "pod": {
            "costPerHr": 0.74,
            "lastStartedAt": _started(1800),
            "runtime": {"uptimeInSeconds": 1200},
        }
    }
    with (
        patch.object(target, "_gql_status", return_value=reply),
        patch.object(runpod_module.time, "time", return_value=_NOW),
    ):
        assert target.pod_reported_cost() == pytest.approx(0.37)
    runtime_only = {"pod": {"costPerHr": 0.74, "runtime": {"uptimeInSeconds": 1800}}}
    with patch.object(target, "_gql_status", return_value=runtime_only):
        assert target.pod_reported_cost() == pytest.approx(0.37)
    with patch.object(target, "_gql_status", side_effect=RuntimeError("down")):
        assert target.pod_reported_cost() is None


def test_mock_is_not_leaking_into_other_tests() -> None:
    assert not isinstance(runpod_module.time.sleep, MagicMock)


def test_wait_for_running_is_bounded_when_a_timeout_is_given() -> None:
    """A pod that never reports uptime raises instead of polling forever."""
    target = RunPodTarget(api_key="rp_test", gpu_type="NVIDIA GeForce RTX 4090")
    target._pod_id = "pod-stuck"
    clock = iter([0.0, 0.0, 5.0, 11.0])
    with (
        patch.object(target, "_gql_status", return_value={"pod": {"runtime": None}}),
        patch.object(runpod_module.time, "monotonic", side_effect=lambda: next(clock)),
        patch.object(runpod_module.time, "sleep"),
        pytest.raises(TimeoutError, match="pod-stuck"),
    ):
        target._wait_for_running(10)


def test_age_zero_cleans_every_apron_pod_even_one_not_yet_running(
    tmp_path: Path, sleeps: list[float]
) -> None:
    sdk = _Sdk(
        pods=[
            {"id": "pulling", "name": "apron-run", "uptimeSeconds": 0, "runtime": None},
            {"id": "other", "name": "someone-else", "uptimeSeconds": 0, "runtime": None},
        ]
    )
    target = _target(tmp_path)
    with (
        patch.dict(sys.modules, {"runpod": sdk}),
        patch.object(target, "_gql_status", return_value={"pod": {"runtime": None}}),
    ):
        assert target.cleanup_orphaned_pods(max_age_seconds=0) == ["pulling"]
    assert sdk.terminated == ["pulling"]


def test_stock_status_asks_secure_stock_under_the_image_cuda(tmp_path: Path) -> None:
    target = _target(tmp_path, gpu_type="NVIDIA GeForce RTX 4090", gpu_count=4)
    reply = {"gpuTypes": [{"id": "x", "lowestPrice": {"stockStatus": "Low"}}]}
    with patch.object(target, "_gql_status", return_value=reply) as gql:
        assert target.stock_status() == "Low"
    query = gql.call_args.args[0]
    assert 'id: "NVIDIA GeForce RTX 4090"' in query
    assert "gpuCount: 4" in query and "secureCloud: true" in query
    assert 'allowedCudaVersions: ["13.0"]' in query
    empty = {"gpuTypes": [{"id": "x", "lowestPrice": {"stockStatus": None}}]}
    with patch.object(target, "_gql_status", return_value=empty):
        assert target.stock_status() is None
