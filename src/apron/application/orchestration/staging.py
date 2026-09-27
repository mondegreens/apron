"""Stage model weights before any GPU bills (phase plan: GPU dollar protection rule 1).

Provider-neutral: a *stager* is any target that can reach the persistent
storage the GPU target will mount (on RunPod, a CPU pod with the datacenter's
network volume; for a user's own GPU, the local disk itself).  The engine's
own download step runs there — the same verified, token-isolated download
a GPU pod runs — so when the GPU pod starts, its download step finds every
file present and only verifies it.

Every stager pod is paid for through the budget like a GPU pod: an estimate
is held first, settled at the provider-reported cost (or the rate times the
time) when the pod is gone.  A failed download is reported, never hidden:
the solution that needed the model then runs its own download on the GPU pod
and the record says where its weights came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from apron.application.orchestration.budget import BudgetTracker
    from apron.domain.ports import Clock

STAGE_PREFIX = "stage:"
# Storage of staged weights, accrued per window by the composition root.
STORAGE_PREFIX = "storage:"


@dataclass(frozen=True)
class StagedModel:
    model_id: str
    ok: bool
    seconds: float
    output_tail: str


@dataclass
class StagingResult:
    pod_id: str | None
    cost: float
    seconds: float
    models: list[StagedModel] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and all(m.ok for m in self.models)


def stage_weights(
    model_ids: Sequence[str],
    *,
    stager: Any,
    engine: Any,
    budget: BudgetTracker,
    clock: Clock,
    hourly_rate: float,
    estimate: float,
    env: dict[str, str] | None = None,
    event: Callable[[dict[str, Any]], None] | None = None,
    download_timeout: int = 4 * 3600,
) -> StagingResult:
    """Download *model_ids* onto the stager's persistent storage, then tear it down."""
    label = f"{STAGE_PREFIX}{','.join(model_ids)}"
    budget.hold(estimate, label)
    start = clock.now().timestamp()
    result = StagingResult(pod_id=None, cost=0.0, seconds=0.0)
    try:
        stager.provision(env=env or {})
        result.pod_id = stager.pod_id
        budget.annotate_hold(label, stager.pod_id)
        if not engine.runner_supports_token_isolation(stager):
            raise RuntimeError("runner image lacks the F7 download step (apron-download)")
        for model_id in model_ids:
            got = engine.download_weights(stager, model_id, timeout=download_timeout)
            staged = StagedModel(
                model_id=model_id,
                ok=bool(got.get("ok")),
                seconds=float(got.get("seconds") or 0.0),
                output_tail=str(got.get("output_tail", ""))[-2000:],
            )
            result.models.append(staged)
            if event is not None:
                event(
                    {
                        "event": "staged",
                        "model_id": model_id,
                        "ok": staged.ok,
                        "seconds": staged.seconds,
                        "pod_id": stager.pod_id,
                        # Where it ran: the speed per datacenter is measured here.
                        "location": getattr(stager, "location", None),
                    }
                )
    except Exception as exc:  # reported in the result; the pod is still torn down
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        # Tear the pod down first: an interrupt during a cost query must not
        # leave it billing (the atexit guard is the last resort, not the path).
        pod_id = stager.pod_id or result.pod_id
        reported = None
        try:
            reported = stager.pod_reported_cost() if pod_id else None
        except Exception:
            reported = None
        finally:
            stager.teardown()
        if pod_id and result.pod_id is None:  # stopped before provision returned
            result.pod_id = pod_id
            budget.annotate_hold(label, pod_id)
        result.seconds = round(clock.now().timestamp() - start, 1)
        result.cost = (
            float(reported)
            if reported is not None
            else round(hourly_rate * result.seconds / 3600, 6)
        )
        budget.settle(result.cost, label)
    return result
