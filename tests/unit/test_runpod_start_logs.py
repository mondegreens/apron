"""A rented RunPod pod is watched through its own logs, not blind.

The frames below are the shape the log API sent for the runner image's pull
on hn1e7xkwbg6iml (US-CA-2, 2026-09-29).  B200 pods whose host never began
(2026-09-29) wrote no system line for 12-20 minutes and were waited on blind.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from apron.adapters.backends import runpod as runpod_module
from apron.adapters.backends.runpod import PodStartError, RunPodTarget
from apron.adapters.backends.runpod_logs import (
    FIRST_LINE_WITHIN,
    SILENCE_LIMIT,
    LogLine,
    PodLogReader,
    StartWatch,
    parse_sse,
)

IMG = "ghcr.io/mondegreens/apron-runner@sha256:faed"
SSE = [
    "id: 1",
    f'data: {{"source":"system","line":"create container {IMG}","ts":"2026-09-29T00:47:26Z"}}',
    "",
    ": keepalive",
    "id: 2",
    "",
    "id: 3",
    'data: {"source":"system","line":"9b61b0826a9a Pulling fs layer","ts":"2026-09-29T00:47:27Z"}',
    "",
    "data: not json",
    "",
]


def test_frames_are_parsed_with_cursors_keepalives_and_raw_payloads() -> None:
    frames = list(parse_sse(SSE))
    assert frames == [
        ("1", LogLine("system", f"create container {IMG}", "2026-09-29T00:47:26Z")),
        ("2", None),
        ("3", LogLine("system", "9b61b0826a9a Pulling fs layer", "2026-09-29T00:47:27Z")),
        (None, LogLine("raw", "not json", "")),
    ]


class _Resp:
    def __init__(self, status: int, lines: list[str]) -> None:
        self.status_code = status
        self._lines = lines

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *a: object) -> None:
        return None

    def iter_lines(self) -> Any:
        return iter(self._lines)

    def read(self) -> bytes:
        return b'{"detail":"pod not found"}'


class _Client:
    def __init__(self, responses: list[_Resp]) -> None:
        self.responses = responses
        self.headers: list[dict[str, str]] = []

    def stream(self, method: str, url: str, **kw: Any) -> _Resp:
        self.headers.append(kw["headers"])
        return self.responses.pop(0)


def test_reader_resumes_after_the_last_frame_and_reports_errors() -> None:
    client = _Client([_Resp(200, SSE), _Resp(404, [])])
    reader = PodLogReader("rpa_secret", "hn1e", client=client)
    lines, error = reader.read("system")
    assert [ln.line for ln in lines][:2] == [f"create container {IMG}", "9b61b0826a9a Pulling fs layer"]
    assert error is None
    lines, error = reader.read("system")
    assert lines == [] and error is not None and "404" in error
    assert client.headers[1]["Last-Event-ID"] == "3"  # no line is read twice
    assert "Last-Event-ID" not in client.headers[0]


SYS = [LogLine("system", "Downloading", "")]
OUT = [LogLine("container", "Starting OpenBSD Secure Shell server", "")]


def test_a_host_that_writes_nothing_is_abandoned_after_the_first_line_window() -> None:
    w = StartWatch(rented_at=0.0)
    w.observe(FIRST_LINE_WITHIN - 1, [], [], None)
    assert w.verdict(FIRST_LINE_WITHIN - 1) is None
    w.observe(FIRST_LINE_WITHIN + 1, [], [], None)
    assert w.verdict(FIRST_LINE_WITHIN + 1) == "never_started"


def test_a_pull_that_keeps_logging_is_never_cut() -> None:
    w = StartWatch(rented_at=0.0)
    for t in range(10, 4 * 3600, 10):  # four hours of pull progress
        w.observe(float(t), SYS, [], None)
        assert w.verdict(float(t)) is None


def test_a_start_that_goes_silent_is_stalled_and_container_lines_count() -> None:
    w = StartWatch(rented_at=0.0)
    w.observe(5.0, SYS, [], None)
    w.observe(300.0, [], OUT, None)  # the container talks after the pull ends
    w.observe(300.0 + SILENCE_LIMIT - 1, [], [], None)
    assert w.verdict(300.0 + SILENCE_LIMIT - 1) is None
    w.observe(300.0 + SILENCE_LIMIT + 1, [], [], None)
    assert w.verdict(300.0 + SILENCE_LIMIT + 1) == "stalled"
    assert w.report(301.0 + SILENCE_LIMIT)["first_container_line_s"] == 300.0


def test_failed_reads_are_no_evidence_of_silence_but_are_bounded() -> None:
    w = StartWatch(rented_at=0.0)
    w.observe(10.0, [], [], None)
    for t in range(20, SILENCE_LIMIT + 10, 10):
        w.observe(float(t), [], [], "HTTP 503")
    # Silence is counted only to the last good read (10 s): not never_started.
    assert w.verdict(float(SILENCE_LIMIT)) is None
    assert w.verdict(float(SILENCE_LIMIT + 11)) == "logs_unreadable"


class _Reader:
    """Per pod: a host that never logs, or one that pulls and starts."""

    def __init__(self, starts: bool, clock: list[float]) -> None:
        self.starts = starts
        self.clock = clock

    def read(self, source: str) -> tuple[list[LogLine], None]:
        self.clock[0] += 5.0
        if self.starts and source == "system":
            return SYS, None
        return [], None


def test_provision_rents_another_pod_when_the_host_never_starts(tmp_path: Any) -> None:
    clock = [0.0]
    target = RunPodTarget(
        api_key="rp_test", gpu_type="NVIDIA B200", gpu_count=4, pod_log_dir=tmp_path
    )
    created: list[str] = []
    terminated: list[str] = []

    def create_pod(**kw: Any) -> dict[str, str]:
        created.append(f"pod{len(created)}")
        return {"id": created[-1]}

    def status(query: str) -> dict[str, Any]:
        pid = created[-1]
        up = pid == "pod1" and clock[0] > 60
        return {
            "pod": {
                "machineId": f"m-{pid}",
                "machine": {"dataCenterId": "US-CA-2"},
                "runtime": {"uptimeInSeconds": 5} if up else None,
            }
        }

    with (
        patch("runpod.create_pod", side_effect=create_pod),
        patch("runpod.terminate_pod", side_effect=terminated.append),
        patch.object(target, "_gql_status", side_effect=status),
        patch.object(
            target,
            "_log_reader_factory",
            side_effect=lambda pid: _Reader(pid == "pod1", clock),
        ),
        patch.object(runpod_module.time, "monotonic", side_effect=lambda: clock[0]),
        patch.object(runpod_module.time, "sleep", side_effect=lambda s: None),
        patch.object(target, "_establish_ssh"),
        patch.object(
            target,
            "_detect_hardware",
            return_value={
                "gpu_name": "NVIDIA B200",
                "total_memory_bytes": 191_495_471_104,
                "compute_capability": "10.0",
                "gpu_count": 4,
            },
        ),
        patch.object(target, "_build_execution_fingerprint"),
    ):
        result = target.provision()
        assert terminated == ["pod0"]
        target.teardown()
    assert terminated == ["pod0", "pod1"]
    assert [a["outcome"] for a in result["start_attempts"]] == ["never_started", "started"]
    assert result["start_attempts"][0]["machine_id"] == "m-pod0"
    assert result["start_attempts"][1]["first_system_line_s"] is not None
    assert (tmp_path / "pod1.log").read_text().count("[system] Downloading") >= 1


def test_provision_gives_up_after_every_rental_is_abandoned() -> None:
    clock = [0.0]
    target = RunPodTarget(api_key="rp_test", gpu_type="NVIDIA B200", start_attempts=2)
    n = iter(range(10))
    terminated: list[str] = []
    with (
        patch("runpod.create_pod", side_effect=lambda **kw: {"id": f"pod{next(n)}"}),
        patch("runpod.terminate_pod", side_effect=terminated.append),
        patch.object(target, "_gql_status", return_value={"pod": {"runtime": None}}),
        patch.object(target, "_log_reader_factory", side_effect=lambda pid: _Reader(False, clock)),
        patch.object(runpod_module.time, "monotonic", side_effect=lambda: clock[0]),
        patch.object(runpod_module.time, "sleep", side_effect=lambda s: None),
        pytest.raises(PodStartError) as info,
    ):
        target.provision()
    assert terminated == ["pod0", "pod1"]
    assert [a["outcome"] for a in info.value.attempts] == ["never_started", "never_started"]
    assert target.pod_id is None
