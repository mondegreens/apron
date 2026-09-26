"""Harness errors shared by the orchestrator and the adapters it drives.

A ``RunStopError`` is a condition that affects every candidate — the run
stops instead of paying for the next pod.  Adapters may raise these (the
adapters layer may import application; the reverse never happens).
"""

from __future__ import annotations


class HarnessError(RuntimeError):
    """The harness, not the model, failed (image, token, provider)."""


class RunStopError(HarnessError):
    """A harness condition that affects every candidate: the run stops."""


class TokenLeakError(RunStopError):
    """INV-13 violated: the HF token reached the vLLM process (or could not be
    checked).  Stops the whole run, not just the candidate."""


class RunnerImageError(RunStopError):
    """The runner image cannot isolate the HF token (no F7 download step)."""


class PodLeakError(RunStopError):
    """A pod could not be terminated after every retry; it may still be billing.

    Owner stop condition ("a pod cannot be terminated").  Raised by the target
    after it has recorded the pod in the leaked-pods list.
    """

    def __init__(self, pod_id: str, detail: str) -> None:
        super().__init__(f"pod {pod_id} could not be terminated: {detail}")
        self.pod_id = pod_id
