"""RunPod pod logs: what the host is doing while a rented pod starts.

RunPod reports nothing machine-readable between renting a pod and its
container running: ``Pod.runtime`` stays null through the image pull, the
container create and the boot (runpodctl ``internal/podstate/podstate.go``,
commit 4351fca).  The host's own narration exists as log text:
``GET https://api.runpod.io/v2/pods/{id}/logs`` with ``source=system`` (image
pull, container create and start) or ``source=container`` (the workload's
stdout and stderr).  The answer is a ``text/event-stream`` of
``data: {"source", "line", "ts"}`` frames with ``id:`` cursors that never
closes on its own; ``Last-Event-ID`` resumes after a frame
(runpodctl ``internal/api/logs.go``).

Measured on a CPU pod with the runner image (hn1e7xkwbg6iml, US-CA-2,
2026-09-29): the first system line (``create container <image>``) came 1 s
after the pod was rented; 1,592 pull lines followed over 4.6 minutes
(``Pulling fs layer``, ``Downloading``, ``Extracting``, ``Pull complete``,
``create container: still fetching image``); the container's first line came
at 297 s and ``runtime`` at 316 s.  A terminated pod's logs are gone (404).

Before this, a rented pod was watched blind: B200 pods in US-CA-2 were cut at
15 minutes without knowing whether they were pulling (2026-09-28), and pods
whose host never began were waited on for up to 20 minutes (2026-09-29).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from apron.application.sanitization import mask_secrets

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

LOGS_URL = "https://api.runpod.io/v2/pods/{pod_id}/logs"
# The largest ``tail`` the API accepts (more is a 422; runpodctl LogTailMax).
TAIL_MAX = 5000

# Healthy hosts wrote their first system line 1 s (CPU pod hn1e7xkwbg6iml,
# US-CA-2) and at most 42 s (L4 pod pqhitzvsj8xpuq, EU-RO-1, through
# RunPodTarget.provision; first read at 41.9 s) after the pod was rented,
# 2026-09-29.  Pods whose host never began showed no line for 12-20 minutes.
# Three minutes is four times the slowest start measured and costs 3 minutes
# of rent per host that never begins.
FIRST_LINE_WITHIN = 180
# Once the host is working, a line arrives every few seconds while the image
# pulls; the longest gap measured was 11 s (last ``Extracting`` to the
# container's first line).  Ten minutes of no line from either source is a
# stuck start, not a slow one.  The same window bounds how long the pod is
# waited on while its logs cannot be read at all.
SILENCE_LIMIT = 600


@dataclass(frozen=True)
class LogLine:
    source: str  # "system", "container", or "raw" for a frame that was not the documented object
    line: str
    ts: str


def parse_sse(lines: Iterable[str]) -> Iterator[tuple[str | None, LogLine | None]]:
    """(event id, entry) per frame; the entry is None for a keepalive.

    Only ``data:`` and ``id:`` fields are read; a frame ends at a blank line.
    A payload that is not the documented object is kept as a ``raw`` line, so
    a change on RunPod's side shows up instead of vanishing.
    """
    data: list[str] = []
    event_id: str | None = None
    for raw in lines:
        text = raw.rstrip("\r\n")
        if text.startswith("data:"):
            data.append(text[5:].lstrip(" "))
        elif text.startswith("id:"):
            event_id = text[3:].strip()
        elif text == "":
            if data or event_id:
                yield event_id, _entry("\n".join(data))
            data, event_id = [], None
    if data or event_id:
        yield event_id, _entry("\n".join(data))


def _entry(payload: str) -> LogLine | None:
    if not payload.strip():
        return None
    try:
        obj = json.loads(payload)
    except ValueError:
        return LogLine("raw", payload, "")
    if not isinstance(obj, dict) or not ({"source", "line", "ts"} & obj.keys()):
        return LogLine("raw", payload, "")
    return LogLine(str(obj.get("source", "")), str(obj.get("line", "")), str(obj.get("ts", "")))


class PodLogReader:
    """New log lines of one pod, per source, since the last read."""

    def __init__(
        self,
        api_key: str,
        pod_id: str,
        *,
        quiet: float = 1.5,
        max_wait: float = 12.0,
        client: Any = None,
    ) -> None:
        self._api_key = api_key
        self._pod_id = pod_id
        # The stream never closes: a read ends ``quiet`` seconds after the last
        # frame (the backlog arrives in one burst) or at ``max_wait``.
        self._quiet = quiet
        self._max_wait = max_wait
        self._client = client or httpx
        self._cursor: dict[str, str] = {}

    def read(self, source: str) -> tuple[list[LogLine], str | None]:
        """(new lines, error); the error names why the logs could not be read."""
        headers = {"Authorization": f"Bearer {self._api_key}", "Accept": "text/event-stream"}
        if source in self._cursor:
            headers["Last-Event-ID"] = self._cursor[source]
        params = {"source": source, "tail": TAIL_MAX}
        got: list[LogLine] = []
        end = time.monotonic() + self._max_wait
        try:
            with self._client.stream(
                "GET",
                LOGS_URL.format(pod_id=self._pod_id),
                params=params,
                headers=headers,
                timeout=httpx.Timeout(self._max_wait, read=self._quiet),
            ) as resp:
                if resp.status_code != 200:
                    body = resp.read()[:200].decode(errors="replace")
                    return got, mask_secrets(f"HTTP {resp.status_code}: {body}")
                for event_id, entry in parse_sse(_until(resp.iter_lines(), end)):
                    if entry is not None:
                        got.append(entry)
                    if event_id:
                        self._cursor[source] = event_id
        except httpx.ReadTimeout:
            pass  # quiet for ``quiet`` seconds: the backlog is drained
        except httpx.HTTPError as exc:
            return got, mask_secrets(f"{type(exc).__name__}: {exc}")
        return got, None


def _until(lines: Iterable[str], end: float) -> Iterator[str]:
    for line in lines:
        yield line
        if time.monotonic() > end:
            yield ""  # close the frame in progress
            return


@dataclass
class StartWatch:
    """Decides, from a rented pod's logs, whether its start is progressing.

    ``verdict`` is None while the pod should be waited on, else why it
    should be abandoned: ``never_started`` (no system line since the rent:
    the host did not begin creating the container), ``stalled`` (lines came,
    then none from either source for ``silence_limit``) or
    ``logs_unreadable`` (no successful read for ``silence_limit``).  There is
    no limit on a start that keeps logging.
    """

    rented_at: float
    first_line_within: float = FIRST_LINE_WITHIN
    silence_limit: float = SILENCE_LIMIT
    first_system_at: float | None = None
    first_container_at: float | None = None
    last_line_at: float | None = None
    last_read_ok_at: float | None = None
    system_lines: int = 0
    container_lines: int = 0
    read_errors: list[str] = field(default_factory=list)

    def observe(
        self, now: float, system: list[LogLine], container: list[LogLine], error: str | None
    ) -> None:
        if error is None:
            self.last_read_ok_at = now
        else:
            self.read_errors = [*self.read_errors[-4:], error]
        if system:
            self.system_lines += len(system)
            self.first_system_at = self.first_system_at or now
            self.last_line_at = now
        if container:
            self.container_lines += len(container)
            self.first_container_at = self.first_container_at or now
            self.last_line_at = now

    def verdict(self, now: float) -> str | None:
        seen_ok = self.last_read_ok_at if self.last_read_ok_at is not None else self.rented_at
        if now - seen_ok > self.silence_limit:
            return "logs_unreadable"
        if self.last_read_ok_at is None:
            return None  # nothing read yet: silence is not evidence
        # Silence counts only up to the last read that succeeded: a failed
        # read is no evidence that the host is quiet.
        if self.first_system_at is None:
            quiet = self.last_read_ok_at - self.rented_at
            return "never_started" if quiet > self.first_line_within else None
        if self.last_read_ok_at - (self.last_line_at or self.rented_at) > self.silence_limit:
            return "stalled"
        return None

    def report(self, now: float) -> dict[str, Any]:
        def since(t: float | None) -> float | None:
            return None if t is None else round(t - self.rented_at, 1)

        return {
            "seconds": round(now - self.rented_at, 1),
            "first_system_line_s": since(self.first_system_at),
            "first_container_line_s": since(self.first_container_at),
            "system_lines": self.system_lines,
            "container_lines": self.container_lines,
            "read_errors": list(self.read_errors),
        }
