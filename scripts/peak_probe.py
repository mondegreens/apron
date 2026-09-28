"""Boot staged models once each on one H100 with the startup peak probe.

vLLM measures ``torch_peak_increase`` around ``profile_run``; two measured
H100 values are not explained by source reading (GLM-4.7-Flash, gemma-4-31B-it).
This driver boots each model with exactly the serve command the cohort used
(rebuilt from ``_dev_notes/cohort-run/solutions.jsonl`` through the engine
adapter), with ``scripts/peak_probe/`` on ``PYTHONPATH`` so the probe records
which tensors are alive at the peak, and saves the probe's JSON to
``_dev_notes/cohort-run/peak-probe/``.

Without ``--run`` it only prints the plan (commands, estimate): nothing is
created and nothing is spent.  With ``--run`` it spends money:

    uv run python scripts/peak_probe.py --run zai-org/GLM-4.7-Flash \\
        google/gemma-4-31B-it --variant 'google/gemma-4-31B-it=--language-model-only'

One pod for all boots, on the staged volume's datacenter (US-CA-2), with the
v0.29.0 runner image.  Paid like a cohort pod: a ``probe:`` hold in the cohort
ledger before the pod is created, annotated with its pod id, settled at the
provider-reported cost after teardown (``staging.py`` does the same).
Needs ``RUNPOD_API_KEY``; ``HF_TOKEN`` for the verification step of gated
repositories (Gemma).  The token reaches only the download step, never vLLM.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import re
import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apron.adapters.backends.ledger_file import JsonlLedger
from apron.adapters.backends.runpod import RunPodTarget
from apron.adapters.backends.vllm_engine import (
    VLLM_LOG,
    VLLM_PROCESS_PATTERN,
    VllmEngineAdapter,
    collapse_repeats,
    launch_command,
)
from apron.adapters.runner_image import image_for_digest
from apron.application.cost_estimator import hourly_rate
from apron.application.orchestration.budget import BudgetTracker
from apron.application.orchestration.findings import PROBE_PREFIX
from apron.application.sanitization import mask_secrets
from apron.domain.fingerprints import fingerprint_hex
from apron.domain.ports import WallClock
from apron.domain.schemas.solutions import DeploymentPlan
from apron.interfaces.cohort_root import (
    AUTHORIZED_USD,
    RUN_DIR,
    WeightsSite,
    _ssh_public_key,
    install_log_masking,
    live_rates,
)

HERE = Path(__file__).resolve().parent
PROBE_SRC = HERE / "peak_probe"
PROBE_FILES = ("sitecustomize.py", "analyze.py")
OUT_DIR = RUN_DIR / "peak-probe"
MANIFEST = RUN_DIR / "solutions.jsonl"
LEDGER = RUN_DIR / "ledger.jsonl"
REPORTS = RUN_DIR / "records" / "verification-reports"

GPU = "NVIDIA H100 80GB HBM3"  # H100 SXM
# The volume the group A weights were staged on (volumes.json, prestage-groupA.json).
SITE = WeightsSite(volume_id="7jiuum2nk6", data_center_id="US-CA-2", size_gb=228)
LABEL_PREFIX = PROBE_PREFIX
LOG_TAIL_CHARS = 20_000

REMOTE_DIR = "/workspace/peak_probe"
BIN = "/opt/venv/bin/"
# vLLM's own profiling lines (DEBUG repr, mem_utils.py:220-231; INFO, gpu_worker.py:625)
_PROFILED = re.compile(r"Memory profiling takes|Available KV cache memory")

# Planning figures for the hold (events.jsonl: H100 provisioning took 280-708 s;
# the recorded boots 104 s and 268 s, plus the probe's snapshot and a cold compile).
PROVISION_MINUTES = 12.0
BOOT_MINUTES = 10.0
HOLD_FACTOR = 1.5


# ---------------------------------------------------------------------------
# What to boot (no network)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Boot:
    model_id: str
    extra: tuple[str, ...] = ()  # extra serve flags of a variant

    @property
    def name(self) -> str:
        slug = self.model_id.replace("/", "--")
        if not self.extra:
            return slug
        flags = re.sub(r"[^A-Za-z0-9]+", "-", " ".join(self.extra)).strip("-")
        return f"{slug}__{flags[:60]}"


@dataclass(frozen=True)
class Solution:
    model_id: str
    fingerprint: str
    plan: DeploymentPlan
    image_digest: str


def recorded_solution(model_id: str, gpu: str = GPU, manifest: Path = MANIFEST) -> Solution:
    """The last identity the cohort recorded for *model_id* on one *gpu*; its
    image (the engine version it ran) is what the probe boots."""
    found: dict[str, Any] | None = None
    for line in manifest.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        requested = entry.get("requested_execution") or {}
        if (
            entry.get("model_id") == model_id
            and requested.get("gpu_sku") == gpu
            and int(requested.get("gpu_count", 0)) == 1
        ):
            found = entry
    if found is None:
        raise LookupError(f"{model_id}: no solution on {gpu} x1 in {manifest}")
    digest = found["requested_execution"]["image_digest"]
    image_for_digest(digest)  # a pinned runner image, or KeyError
    return Solution(
        model_id=model_id,
        fingerprint=found["solution_fingerprint"],
        plan=DeploymentPlan.model_validate(found["deployment_plan"]),
        image_digest=digest,
    )


def recorded_measurements(fingerprint: str, reports: Path = REPORTS) -> list[dict[str, Any]]:
    """The healthy memory reports of a solution: what the probe is compared with."""
    rows = []
    for path in sorted(reports.glob("*.json")) if reports.exists() else []:
        r = json.loads(path.read_text("utf-8"))
        if r.get("solution_fingerprint") != fingerprint or r.get("claim_scope") != "memory":
            continue
        if r.get("boot_outcome") != "healthy":
            continue
        rows.append(
            {
                "record": path.stem,
                "torch_peak_increase": r.get("transient_peak_headroom"),
                "profiling_shape": r.get("profiling_shape"),
                "deployment_plan_digest": r.get("deployment_plan_digest"),
                "weights_source": r.get("weights_source"),
            }
        )
    return rows


def serve_command(
    engine: VllmEngineAdapter, plan: DeploymentPlan, load_args: list[str], extra: tuple[str, ...]
) -> str:
    """``VllmEngineAdapter.boot``'s serve string (vllm_engine.py:551-555) plus variant flags."""
    model_id = plan.resource_allocation.get("model_id", "")
    serve = engine._build_serve_command(plan, None, model_path=engine.model_dir(model_id))
    added = [*load_args, *extra]
    if added:
        serve = " ".join([serve, *(shlex.quote(a) for a in added)])
    return serve


def probe_env(out: str, cache_root: str) -> dict[str, str]:
    return {
        "APRON_PEAK_PROBE": "1",
        "APRON_PEAK_PROBE_OUT": out,
        "PYTHONPATH": REMOTE_DIR,
        # A fresh compile cache: the recorded boots compiled these models cold.
        "VLLM_CACHE_ROOT": cache_root,
    }


def probe_launch(serve: str, env: dict[str, str]) -> str:
    """``launch_command`` with the probe's variables placed before the binary.

    ``env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN HF_HUB_OFFLINE=1`` stays first,
    so the token removal and offline mode are the cohort's (vllm_engine.py:131-134).
    """
    assigns = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    return launch_command(f"{assigns} {BIN}{serve}", bin_dir="")


def estimate_usd(rate: float, boots: int) -> float:
    minutes = PROVISION_MINUTES + BOOT_MINUTES * boots
    return round(rate * minutes / 60 * HOLD_FACTOR, 4)


def open_holds(ledger: Path = LEDGER) -> list[str]:
    """Holds nobody has settled yet: another process may be running one right now."""
    holds: dict[str, bool] = {}
    for entry in JsonlLedger(ledger).read_all():
        if entry["op"] == "hold":
            holds[entry["label"]] = True
        elif entry["op"] in ("settle", "release"):
            holds.pop(entry["label"], None)
    return sorted(holds)


# ---------------------------------------------------------------------------
# On the pod
# ---------------------------------------------------------------------------


def upload_probe(target: Any, src: Path = PROBE_SRC) -> None:
    for name in PROBE_FILES:
        payload = base64.b64encode((src / name).read_bytes()).decode("ascii")
        result = target.execute(
            f"mkdir -p {REMOTE_DIR} && echo {payload} | base64 -d > {REMOTE_DIR}/{name}"
        )
        if result.get("exit_code") != 0:
            raise RuntimeError(f"upload of {name} failed: {result.get('stderr', '')[-500:]}")


@dataclass
class BootOutcome:
    status: str
    seconds: float
    probe: dict[str, Any] | None = None
    parsed_log: dict[str, Any] = field(default_factory=dict)
    log_tail: str = ""
    token_check: dict[str, Any] | None = None


def wait_and_collect(
    target: Any, engine: VllmEngineAdapter, out: str, timeout: int
) -> BootOutcome:
    """Wait for the probe JSON and vLLM's profiling line; stop at a dead process or timeout."""
    start = time.monotonic()
    status = "timeout"
    json_seen: float | None = None
    while time.monotonic() - start < timeout:
        have_json = "yes" in str(
            target.execute(f"test -s {shlex.quote(out)} && echo yes || echo no").get("stdout")
        )
        logged = target.execute(f"grep -c -E '{_PROFILED.pattern}' {VLLM_LOG} || true")
        profiled = str(logged.get("stdout", "0")).strip() not in ("", "0")
        if have_json and json_seen is None:
            json_seen = time.monotonic()
        if have_json and (profiled or time.monotonic() - (json_seen or 0) > 300):
            status = "profiled" if profiled else "probe_only"
            break
        alive = target.execute(
            f"pgrep -f '{VLLM_PROCESS_PATTERN}' >/dev/null && echo up || echo down"
        )
        if "down" in str(alive.get("stdout", "")):
            status = "exited_with_probe" if have_json else "exited"
            break
        time.sleep(10)
    outcome = BootOutcome(status=status, seconds=round(time.monotonic() - start, 1))
    if status in ("profiled", "probe_only"):
        outcome.token_check = engine.token_environ_check(target)
    raw = target.collect([out]).get(out)
    if isinstance(raw, str) and raw.startswith("{"):
        outcome.probe = json.loads(raw)
    log = str(target.execute(f"cat {VLLM_LOG} 2>/dev/null || true").get("stdout", ""))
    outcome.parsed_log = engine.parse_profiling_logs(log)
    # Last 200 lines, and at most LOG_TAIL_CHARS: a progress bar redrawn with
    # carriage returns is one "line" of tens of MB (Muse-Glimmer, 2026-09-28).
    tail = collapse_repeats("\n".join(log.replace("\r", "\n").splitlines()[-200:]))
    outcome.log_tail = mask_secrets(tail[-LOG_TAIL_CHARS:])
    return outcome


def run_boot(
    target: Any,
    engine: VllmEngineAdapter,
    boot: Boot,
    solution: Solution,
    index: int,
    timeout: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model_id": boot.model_id,
        "variant_flags": list(boot.extra),
        "solution_fingerprint": solution.fingerprint,
        "deployment_plan_digest": fingerprint_hex(solution.plan),
        "recorded": recorded_measurements(solution.fingerprint),
        "pod_id": target.pod_id,
        "image": image_for_digest(solution.image_digest),
        "gpu": GPU,
    }
    hygiene = engine.prepare_boot(target)
    result["hygiene"] = hygiene
    if not hygiene.get("clean"):
        result["status"] = "unclean_gpu"
        return result
    download = engine.download_weights(target, boot.model_id)
    result["weights_check"] = {k: download.get(k) for k in ("ok", "seconds")}
    if not download.get("ok"):
        result["status"] = "weights_incomplete"
        result["weights_output"] = mask_secrets(str(download.get("output_tail", "")))
        return result
    out = f"/workspace/peak-probe-{index}.json"
    cache_root = f"/workspace/probe-cache-{index}"
    target.execute(f"rm -rf {cache_root} {out} && mkdir -p {cache_root}")
    load = engine.load_args(target, boot.model_id)
    serve = serve_command(engine, solution.plan, load, boot.extra)
    command = probe_launch(serve, probe_env(out, cache_root))
    result.update(serve=serve, load_args=load, launch=command)
    target.execute(command)
    outcome = wait_and_collect(target, engine, out, timeout)
    target.execute(f"pkill -9 -f '{VLLM_PROCESS_PATTERN}' || true")
    result.update(
        status=outcome.status,
        seconds=outcome.seconds,
        vllm_logged=outcome.parsed_log,
        token_check=outcome.token_check,
        probe=outcome.probe,
        log_tail=outcome.log_tail,
    )
    return result


# ---------------------------------------------------------------------------


def parse_boots(models: list[str], variants: list[str]) -> list[Boot]:
    """Base boots, then variants.  FLAGS are split like a shell line, so a JSON
    value needs its own quotes:
    ``MODEL=--limit-mm-per-prompt '{"image":0,"video":0,"audio":0}'``."""
    boots = [Boot(m) for m in models]
    for v in variants:
        model_id, sep, flags = v.partition("=")
        if not sep or model_id not in models:
            raise SystemExit(f"--variant {v!r}: expected MODEL=FLAGS for a listed model")
        extra = tuple(shlex.split(flags))
        for arg in extra:
            if arg.startswith(("{", "[")):
                try:
                    json.loads(arg)
                except ValueError:
                    raise SystemExit(
                        f"--variant {v!r}: {arg!r} is not JSON (quote JSON values inside FLAGS)"
                    ) from None
        boots.append(Boot(model_id, extra))
    # Keep a model's variants right after its base boot.
    return sorted(boots, key=lambda b: (models.index(b.model_id), bool(b.extra)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("models", nargs="+", help="model ids staged on the volume")
    parser.add_argument("--variant", action="append", default=[], help="MODEL=EXTRA SERVE FLAGS")
    parser.add_argument("--run", action="store_true", help="create the pod and spend money")
    parser.add_argument("--boot-timeout", type=int, default=1500)
    parser.add_argument(
        "--allow-open-holds",
        action="store_true",
        help="replay (and settle) other open holds; only when nothing else is running",
    )
    args = parser.parse_args(argv)

    boots = parse_boots(args.models, args.variant)
    solutions = {m: recorded_solution(m) for m in args.models}
    engine = VllmEngineAdapter()
    for i, b in enumerate(boots):
        serve = serve_command(engine, solutions[b.model_id].plan, [], b.extra)
        print(f"[{i}] {b.name}\n    {serve}\n    (+ load args decided on the pod)")

    api_key = os.environ.get("RUNPOD_API_KEY")
    rate, source = hourly_rate("runpod", GPU, live_rates=live_rates(api_key) if args.run else None)
    estimate = estimate_usd(rate, len(boots))
    print(f"rate ${rate}/h ({source}); hold ${estimate} for {len(boots)} boot(s)")
    if not args.run:
        print("dry run: nothing created (pass --run to boot)")
        return 0
    if not api_key:
        print("RUNPOD_API_KEY is not set", file=sys.stderr)
        return 2
    pending = open_holds(LEDGER)
    if pending and not args.allow_open_holds:
        print(f"open holds in the ledger (another run?): {pending}", file=sys.stderr)
        return 3

    install_log_masking()
    # One pod runs one image: every model must have been recorded on the same one.
    digests = {sol.image_digest for sol in solutions.values()}
    if len(digests) != 1:
        print(f"models were recorded on different images: {sorted(digests)}", file=sys.stderr)
        return 5
    image = image_for_digest(digests.pop())
    target = RunPodTarget(
        api_key=api_key,
        image=image,
        gpu_type=GPU,
        gpu_count=1,
        leak_log=RUN_DIR / "leaked_pods.json",
        network_volume_id=SITE.volume_id,
        data_center_id=SITE.data_center_id,
    )
    if not target.stock_status():
        print(f"no Secure {GPU} stock in {SITE.data_center_id} now", file=sys.stderr)
        return 4
    budget = BudgetTracker.replay(
        authorized=AUTHORIZED_USD,
        ledger=JsonlLedger(LEDGER),
        clock=WallClock(),
        pod_cost=target.pod_reported_cost,
    )
    label = f"{LABEL_PREFIX}peak:{','.join(args.models)}"
    budget.hold(estimate, label)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    pod_id: str | None = None
    try:
        target.provision(
            env=RunPodTarget.build_env(
                ssh_public_key=_ssh_public_key(), hf_token=os.environ.get("HF_TOKEN")
            ),
            wait_timeout=1800,
        )
        pod_id = target.pod_id
        budget.annotate_hold(label, str(pod_id))
        print(f"pod {pod_id} up after {time.monotonic() - start:.0f}s")
        if not engine.runner_supports_token_isolation(target):
            raise RuntimeError("runner image lacks the F7 download step (apron-download)")
        upload_probe(target)
        for i, b in enumerate(boots):
            try:
                result = run_boot(target, engine, b, solutions[b.model_id], i, args.boot_timeout)
            except Exception as exc:  # recorded; the next boot still runs on this pod
                result = {"model_id": b.model_id, "variant_flags": list(b.extra)}
                result["status"] = "harness_error"
                result["error"] = mask_secrets(f"{type(exc).__name__}: {exc}")
                with contextlib.suppress(Exception):
                    target.execute(f"pkill -9 -f '{VLLM_PROCESS_PATTERN}' || true")
            path = OUT_DIR / f"{b.name}.json"
            path.write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
            analysis = ((result.get("probe") or {}).get("runs") or [{}])[0].get("analysis") or {}
            print(
                f"[{i}] {b.name}: {result.get('status')}; vLLM logged "
                f"{(result.get('vllm_logged') or {}).get('torch_peak_increase')}; replay peak "
                f"{analysis.get('peak_increase_bytes')} -> {path}"
            )
    finally:
        # Cost first, then teardown, then settle (staging.py:103-122).
        pod_id = target.pod_id or pod_id
        reported = None
        try:
            reported = target.pod_reported_cost() if pod_id else None
        except Exception:
            reported = None
        finally:
            target.teardown()
        if pod_id is None:
            budget.release_hold(label)  # no pod was created: nothing was spent (M2)
            print("no pod was created; hold released")
        else:
            if budget.holds[label].pod_id is None:
                budget.annotate_hold(label, pod_id)
            seconds = time.monotonic() - start
            cost = float(reported) if reported is not None else round(rate * seconds / 3600, 6)
            budget.settle(cost, label)
            source = "provider" if reported is not None else "rate x time"
            print(f"pod {pod_id} torn down; settled ${cost} ({source})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
