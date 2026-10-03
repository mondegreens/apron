"""The boot wait has no time limit: it ends on /health, on the process
exiting, or when vLLM stops writing to its log (a hung engine).

A fixed 30-minute deadline cut a loading DeepSeek-V4-Flash-0731 off at
shard 42 of 48 on 2xB200 and recorded it as a model failure (2026-09-28).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from apron.adapters.backends.vllm_engine import BOOT_STALLED, VllmEngineAdapter
from apron.application.orchestration.cohort import classify_harness_error
from apron.domain.schemas.solutions import DeploymentPlan

PLAN = DeploymentPlan(
    engine_configuration={"max_model_len": "640"},
    resource_allocation={"model_id": "org/m", "gpu_sku": "NVIDIA B200", "gpu_count": "2"},
)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _Pod:
    """A pod whose vLLM log grows while loading, then is healthy or hangs."""

    def __init__(self, clock: _Clock, *, grows_until: float, healthy_at: float | None) -> None:
        self.clock = clock
        self.grows_until = grows_until
        self.healthy_at = healthy_at

    def execute(self, command: str, **_: Any) -> dict[str, Any]:
        now = self.clock.now
        if command.startswith("curl -sf http://localhost:8000/health"):
            ok = self.healthy_at is not None and now >= self.healthy_at
            return {"exit_code": 0 if ok else 7, "stdout": ""}
        if command.startswith("pgrep"):
            return {"exit_code": 0, "stdout": "up\n"}
        if command.startswith("stat -c %s"):
            return {"exit_code": 0, "stdout": f"{int(min(now, self.grows_until))}\n"}
        if command.startswith("tail -400"):
            return {"exit_code": 0, "stdout": "Loading safetensors checkpoint shards: 88%\n"}
        return {"exit_code": 0, "stdout": ""}


def _boot(pod: _Pod, clock: _Clock, stall: int) -> Any:
    adapter = VllmEngineAdapter()
    with (
        patch("apron.adapters.backends.vllm_engine.time.monotonic", clock.monotonic),
        patch("apron.adapters.backends.vllm_engine.time.sleep", clock.sleep),
        patch.object(adapter, "_build_serve_command", return_value="vllm serve /m"),
        patch.object(adapter, "load_args", return_value=[]),
    ):
        return adapter.boot(PLAN, pod, health_timeout=stall)


def test_a_long_load_that_keeps_logging_is_waited_for() -> None:
    """A 50-minute load is waited out: the window is for silence, not length."""
    clock = _Clock()
    pod = _Pod(clock, grows_until=3000, healthy_at=3000)
    result = _boot(pod, clock, stall=1200)
    assert result.healthy
    assert result.seconds >= 3000


def test_a_silent_engine_is_abandoned_and_says_so() -> None:
    clock = _Clock()
    pod = _Pod(clock, grows_until=600, healthy_at=None)
    result = _boot(pod, clock, stall=1200)
    assert not result.healthy
    assert BOOT_STALLED in result.log_tail
    # Abandoned 1200 s after the log last grew, not at a fixed deadline.
    assert 1800 <= result.seconds <= 1820
    assert classify_harness_error(result.log_tail) == "harness:boot_stalled"
