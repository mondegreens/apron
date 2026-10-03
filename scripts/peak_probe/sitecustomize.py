"""Startup peak probe: record which tensors are alive at vLLM's profile-run peak.

Put this directory first on ``PYTHONPATH`` of ``vllm serve``: Python imports
``sitecustomize`` in every process that inherits the variable, so the engine
core (where the worker and its model runner live) loads it too.  With
``APRON_PEAK_PROBE=1`` it installs an import hook; without it nothing happens
beyond chaining to any other ``sitecustomize`` the environment has.

When ``vllm.v1.worker.gpu.model_runner`` (Model Runner V2, the CUDA default
in v0.29: ``config/vllm.py:645-679``, chosen at ``v1/worker/gpu_worker.py:457-476``)
or ``vllm.v1.worker.gpu_model_runner`` (V1) is imported, ``GPUModelRunner.profile_run``
is wrapped.  vLLM calls it inside ``memory_profiling`` right after
``reset_peak_memory_stats`` (``v1/worker/gpu_worker.py:553-557``,
``utils/mem_utils.py:289-311``), so the wrapper's baseline is the ``before``
snapshot vLLM subtracts; ``torch_peak_increase`` below is the same quantity
vLLM logs as "torch peak memory increase".

The wrapper switches on the allocator's history
(``torch.cuda.memory._record_memory_history(enabled="all", context="all",
stacks="python", max_entries=...)``, torch 2.13 ``cuda/memory.py:892-1006``),
runs the original, takes ``torch.cuda.memory._snapshot()``, switches history
off, replays the trace (``analyze.py``) and writes one JSON document to
``APRON_PEAK_PROBE_OUT`` (default ``/workspace/peak-probe.json``).

It must never stop a boot: every failure is caught and written into the JSON;
an exception from vLLM's own ``profile_run`` is recorded and re-raised as is.
"""

from __future__ import annotations

import contextlib
import functools
import importlib.util
import json
import os
import sys
import time
import traceback
from typing import Any

ENV_FLAG = "APRON_PEAK_PROBE"
ENV_OUT = "APRON_PEAK_PROBE_OUT"
ENV_MAX_ENTRIES = "APRON_PEAK_PROBE_MAX_ENTRIES"
DEFAULT_OUT = "/workspace/peak-probe.json"
# Each entry carries a Python stack; a cold-compile profile run of a ~60 GB
# model is expected in the tens of thousands of events.  A full ring buffer
# is reported as ``truncated`` (the replay is then not trustworthy).
DEFAULT_MAX_ENTRIES = 1_000_000

# module -> class whose profile_run is wrapped
TARGETS = {
    "vllm.v1.worker.gpu.model_runner": "GPUModelRunner",
    "vllm.v1.worker.gpu_model_runner": "GPUModelRunner",
}
_HERE = os.path.dirname(os.path.abspath(__file__))
_MARK = "__apron_peak_probe__"
_STATE: dict[str, Any] = {"active": False, "runs": [], "errors": []}


def enabled() -> bool:
    return os.environ.get(ENV_FLAG) == "1"


def out_path() -> str:
    return os.environ.get(ENV_OUT) or DEFAULT_OUT


def max_entries() -> int:
    try:
        return int(os.environ.get(ENV_MAX_ENTRIES) or DEFAULT_MAX_ENTRIES)
    except ValueError:
        return DEFAULT_MAX_ENTRIES


def _write() -> None:
    """Write everything recorded so far (atomic: never a half-written file)."""
    try:
        path = out_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        doc = {
            "probe": "apron-startup-peak-probe",
            "schema": 1,
            "pid": os.getpid(),
            "argv": list(sys.argv),
            "errors": list(_STATE["errors"]),
            "runs": list(_STATE["runs"]),
        }
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True, default=str)
        os.replace(tmp, path)
    except Exception:  # the probe never breaks the boot
        sys.stderr.write(f"[apron-peak-probe] write failed: {traceback.format_exc()}\n")


def _note_error(where: str) -> None:
    _STATE["errors"].append({"where": where, "error": traceback.format_exc()})
    _write()


def _load_analyze() -> Any:
    spec = importlib.util.spec_from_file_location(
        "apron_peak_probe_analyze", os.path.join(_HERE, "analyze.py")
    )
    if spec is None or spec.loader is None:
        raise ImportError("analyze.py not found beside sitecustomize.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _probed(original: Any, runner: Any, args: Any, kwargs: Any, label: str) -> Any:
    if _STATE["active"]:  # a nested call is part of the outer recording
        return original(runner, *args, **kwargs)
    run: dict[str, Any] = {"runner": label, "errors": [], "max_entries": max_entries()}
    _STATE["runs"].append(run)
    torch: Any = None
    device = 0
    recording = False
    try:
        torch = importlib.import_module("torch")  # the engine's own torch
        device = torch.cuda.current_device()
        torch.cuda.synchronize(device)
        run.update(
            device=device,
            device_name=torch.cuda.get_device_name(device),
            torch_version=str(torch.__version__),
            baseline_allocated=int(torch.cuda.memory_allocated(device)),
            baseline_max_allocated=int(torch.cuda.max_memory_allocated(device)),
            baseline_reserved=int(torch.cuda.memory_reserved(device)),
        )
        try:
            vllm = importlib.import_module("vllm")
            run["vllm_version"] = str(getattr(vllm, "__version__", "?"))
        except Exception:
            run["vllm_version"] = None
        torch.cuda.memory._record_memory_history(
            enabled="all",
            context="all",
            stacks="python",
            max_entries=max_entries(),
            clear_history=True,
        )
        recording = True
    except Exception:
        run["errors"].append({"where": "start", "error": traceback.format_exc()})

    _STATE["active"] = True
    started = time.monotonic()
    try:
        return original(runner, *args, **kwargs)
    except BaseException as exc:
        run["profile_run_error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _STATE["active"] = False
        run["profile_run_seconds"] = round(time.monotonic() - started, 3)
        _finish(run, torch, device, recording)


def _finish(run: dict[str, Any], torch: Any, device: int, recording: bool) -> None:
    snapshot = None
    try:
        if torch is not None and "baseline_max_allocated" in run:
            torch.cuda.synchronize(device)
            run["end_allocated"] = int(torch.cuda.memory_allocated(device))
            run["end_max_allocated"] = int(torch.cuda.max_memory_allocated(device))
            run["end_reserved"] = int(torch.cuda.memory_reserved(device))
            # vLLM: after.torch_peak - before.torch_peak (mem_utils.py:310)
            run["torch_peak_increase"] = run["end_max_allocated"] - run["baseline_max_allocated"]
            run["torch_max_minus_baseline_allocated"] = (
                run["end_max_allocated"] - run["baseline_allocated"]
            )
        if recording and torch is not None:
            began = time.monotonic()
            try:
                snapshot = torch.cuda.memory._snapshot()
            finally:
                torch.cuda.memory._record_memory_history(enabled=None)
            run["snapshot_seconds"] = round(time.monotonic() - began, 3)
    except Exception:
        run["errors"].append({"where": "snapshot", "error": traceback.format_exc()})
    try:
        if snapshot is not None:
            began = time.monotonic()
            analysis = _load_analyze().analyze(
                snapshot, device=device, max_entries=run["max_entries"]
            )
            run["analysis_seconds"] = round(time.monotonic() - began, 3)
            run["analysis"] = analysis
            if "torch_peak_increase" in run:
                run["replay_minus_torch_peak_bytes"] = (
                    analysis["peak_increase_bytes"] - run["torch_peak_increase"]
                )
    except Exception:
        run["errors"].append({"where": "analysis", "error": traceback.format_exc()})
    _write()


def patch_module(module: Any, class_name: str) -> bool:
    """Wrap ``class_name.profile_run`` in *module*; False when there is nothing to wrap."""
    cls = getattr(module, class_name, None)
    original = getattr(cls, "__dict__", {}).get("profile_run")
    if cls is None or original is None or getattr(original, _MARK, False):
        return False
    label = f"{module.__name__}.{class_name}"

    @functools.wraps(original)
    def profile_run(self: Any, *args: Any, **kwargs: Any) -> Any:
        return _probed(original, self, args, kwargs, label)

    setattr(profile_run, _MARK, True)
    cls.profile_run = profile_run
    _STATE.setdefault("patched", []).append(label)
    return True


class ProbeFinder:
    """A meta-path finder that only decorates the real loader of two modules."""

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        if fullname not in TARGETS:
            return None
        try:
            spec = None
            for finder in sys.meta_path:
                find = getattr(finder, "find_spec", None)
                if finder is self or find is None:
                    continue
                spec = find(fullname, path, target)
                if spec is not None:
                    break
            loader = getattr(spec, "loader", None)
            if spec is None or loader is None or not hasattr(loader, "exec_module"):
                return spec
            original = loader.exec_module
            class_name = TARGETS[fullname]

            def exec_module(module: Any) -> None:
                original(module)  # vLLM's own import errors propagate unchanged
                try:
                    patch_module(module, class_name)
                except Exception:
                    _note_error(f"patch {fullname}")

            setattr(loader, "exec_module", exec_module)  # noqa: B010 — this loader instance only
            return spec
        except Exception:
            _note_error(f"find_spec {fullname}")
            return None  # the normal import machinery takes over


def install() -> ProbeFinder | None:
    """Install the hook when the probe is switched on; the finder, or None."""
    if not enabled():
        return None
    for name, class_name in TARGETS.items():  # already imported: patch in place
        if name in sys.modules:
            patch_module(sys.modules[name], class_name)
    finder = ProbeFinder()
    sys.meta_path.insert(0, finder)
    return finder


def _chain_next_sitecustomize() -> None:
    """Run the ``sitecustomize`` this file shadows (Ubuntu's Python ships one)."""
    for entry in sys.path:
        if not entry or os.path.abspath(entry) == _HERE:
            continue
        candidate = os.path.join(entry, "sitecustomize.py")
        if os.path.isfile(candidate):
            spec = importlib.util.spec_from_file_location(
                "_apron_chained_sitecustomize", candidate
            )
            if spec is not None and spec.loader is not None:
                spec.loader.exec_module(importlib.util.module_from_spec(spec))
            return


try:
    install()
except Exception:
    _note_error("install")
with contextlib.suppress(Exception):  # a foreign sitecustomize never stops the interpreter
    _chain_next_sitecustomize()
