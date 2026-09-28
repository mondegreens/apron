"""Which tensors are alive at the startup profile run's peak (pure; no CUDA).

Input: the dict ``torch.cuda.memory._snapshot()`` returns after
``torch.cuda.memory._record_memory_history(enabled="all", ...)`` was switched
on at the start of vLLM's ``profile_run``.  Its ``device_traces[d]`` is the
allocator's event list for device *d*, oldest first.

Replay rules (torch 2.13, ``c10/cuda/CUDACachingAllocator.cpp`` at the wheel's
git_version cf30153c):

- ``alloc`` is recorded with the requested size (``orig_size``, :2147-2149);
  ``allocated_bytes`` grows by the block size (:2161).
- ``free_requested`` is recorded in ``free_locked`` with the block's requested
  size (:2433-2435), right where ``allocated_bytes`` shrinks (:2424).
  ``free_completed`` (:3513) only returns the block to the cache when a
  cross-stream use ends; it does not change ``allocated_bytes`` and is ignored.
- segment events (cudaMalloc/cudaFree/map/unmap) are reserved memory, not
  allocated memory, and are ignored.

So the running sum of ``+alloc - free_requested`` tracks
``torch.cuda.memory_allocated()`` minus its value when recording started,
except for the allocator's rounding: a block is at least the request rounded
up to 512 bytes (``round_size``, :3062-3074) and an unsplit large block can
be up to 1 MiB larger (``should_split``, :3680-3697).  The replay is therefore
done with sizes rounded to 512, and the report gives torch's own counter
beside it so the difference is visible.

A ``free_requested`` for an address not allocated during recording frees a
block from before (weights, buffers): it lowers the running total by its size.

Frames in a snapshot are innermost first (``torch/cuda/_memory_viz.py``
reverses them to draw flame graphs root first, :130-133).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

MIN_BLOCK = 512  # kMinBlockSize

# Frames worth naming in an allocation site: the engine, the model code and
# the compiler.  Inductor's generated code lives in vLLM's cache directory
# (``.../torch_compile_cache/.../inductor_cache/xx/c....py``).
SITE_MARKERS = (
    "/vllm/",
    "/transformers/",
    "/torch/_inductor/",
    "inductor_cache/",
    "/triton/",
    "/flashinfer/",
)
SITE_DEPTH = 6

# Phase markers, checked in this order over every frame of an allocation.
# compile: Dynamo/Inductor compiling or autotuning (Triton benchmarks clone
#   their arguments and allocate a cache-flush buffer).
_COMPILE_FILES = (
    "/torch/_dynamo/convert_frame.py",
    "/torch/_inductor/compile_fx.py",
    "/torch/_inductor/select_algorithm.py",
    "/torch/_inductor/autotune_process.py",
    "/torch/_inductor/runtime/benchmarking.py",
    "/torch/_inductor/codecache.py",
    "/triton/testing.py",
    "/vllm/compilation/compiler_interface.py",
)
_COMPILE_FUNCTIONS = (
    "benchmark_all_configs",
    "autotune_to_one_config",
    "clone_args",
    "do_bench",
    "precompile",
    "compile_fx",
    "fx_codegen_and_compile",
    "standalone_compile",
)
_ENCODER = (
    "profile_encoder_cache",
    "encoder_runner",
    "_execute_mm_encoder",
    "embed_multimodal",
    "get_multimodal_embeddings",
    "vision_tower",
    "audio_tower",
)
_SAMPLER_FUNCTIONS = ("_dummy_sampler_run", "_dummy_pooler_run")
_SAMPLER_FILES = ("/vllm/v1/sample/", "/vllm/v1/worker/gpu/sample/")
_FORWARD_FUNCTIONS = ("_dummy_run",)
_FORWARD_FILES = ("/vllm/model_executor/", "inductor_cache/")

_IGNORED = frozenset(
    {"free_completed", "segment_alloc", "segment_free", "segment_map", "segment_unmap"}
)


def rounded(size: int) -> int:
    """The allocator's minimum block size for a request (round_size, no divisions)."""
    if size < MIN_BLOCK:
        return MIN_BLOCK
    return MIN_BLOCK * ((size + MIN_BLOCK - 1) // MIN_BLOCK)


def frame_str(frame: dict[str, Any]) -> str:
    return f"{frame.get('filename', '?')}:{frame.get('line', 0)}:{frame.get('name', '?')}"


def _short(path: str) -> str:
    """Path from the first interesting package down (site-packages prefix dropped)."""
    for marker in ("/site-packages/", "/dist-packages/"):
        if marker in path:
            return path.split(marker, 1)[1]
    if "inductor_cache/" in path:
        return "<inductor>/" + path.split("inductor_cache/", 1)[1]
    return path


def site_of(frames: list[dict[str, Any]], depth: int = SITE_DEPTH) -> tuple[str, ...]:
    """The innermost *depth* frames under the engine, model or compiler code."""

    def short(f: dict[str, Any]) -> str:
        return f"{_short(str(f.get('filename', '?')))}:{f.get('line', 0)}:{f.get('name', '?')}"

    picked: list[str] = []
    for f in frames:
        if any(m in str(f.get("filename", "")) for m in SITE_MARKERS):
            picked.append(short(f))
            if len(picked) == depth:
                break
    if not picked and frames:
        picked.append(short(frames[0]))  # nothing under the engine: say where it was
    return tuple(picked) or ("<no python frames>",)


def phase_of(frames: list[dict[str, Any]]) -> str:
    """compile / encoder / sampler / forward / other, from the whole stack."""
    files = [str(f.get("filename", "")) for f in frames]
    names = [str(f.get("name", "")) for f in frames]
    if any(c in fn for fn in files for c in _COMPILE_FILES) or any(
        n in _COMPILE_FUNCTIONS for n in names
    ):
        return "compile"
    if any(e in n for n in names for e in _ENCODER) or any(
        "/vllm/v1/worker/gpu/mm/" in fn for fn in files
    ):
        return "encoder"
    if any(n in _SAMPLER_FUNCTIONS for n in names) or any(
        s in fn for fn in files for s in _SAMPLER_FILES
    ):
        return "sampler"
    if any(n in _FORWARD_FUNCTIONS for n in names) or any(
        s in fn for fn in files for s in _FORWARD_FILES
    ):
        return "forward"
    return "other"


def _group(blocks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_site: dict[tuple[str, ...], dict[str, Any]] = {}
    for b in blocks:
        g = by_site.setdefault(
            b["site"],
            {
                "site": list(b["site"]),
                "bytes": 0,
                "requested_bytes": 0,
                "count": 0,
                "phases": defaultdict(int),
            },
        )
        g["bytes"] += b["size"]
        g["requested_bytes"] += b["requested_size"]
        g["count"] += 1
        g["phases"][b["phase"]] += b["size"]
    groups = sorted(by_site.values(), key=lambda g: (-g["bytes"], g["site"]))
    for g in groups:
        g["phases"] = dict(sorted(g["phases"].items()))
    return groups[:limit]


def _phase_totals(blocks: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for b in blocks:
        totals[b["phase"]] += b["size"]
    return dict(sorted(totals.items(), key=lambda kv: -kv[1]))


def _block(event: dict[str, Any], index: int) -> dict[str, Any]:
    frames = list(event.get("frames") or [])
    size = int(event.get("size", 0))
    return {
        "index": index,
        "addr": event.get("addr"),
        "requested_size": size,
        "size": rounded(size),
        "stream": event.get("stream"),
        "pool_id": event.get("pool_id"),
        "site": site_of(frames),
        "phase": phase_of(frames),
        "innermost": frame_str(frames[0]) if frames else None,
    }


def analyze(
    snapshot: dict[str, Any],
    *,
    device: int = 0,
    max_entries: int | None = None,
    groups: int = 40,
    top_blocks: int = 30,
) -> dict[str, Any]:
    """Replay one device's trace; report the peak increase and what was alive then."""
    traces = snapshot.get("device_traces") or []
    events: list[dict[str, Any]] = list(traces[device]) if device < len(traces) else []

    live: dict[Any, dict[str, Any]] = {}
    running = 0  # rounded bytes over the start of recording
    running_requested = 0
    peak, peak_requested, peak_index = 0, 0, -1
    peak_live: list[dict[str, Any]] = []
    freed_before = 0  # bytes of pre-recording blocks freed during recording
    freed_before_count = 0
    actions: dict[str, int] = defaultdict(int)
    oom: list[dict[str, Any]] = []

    # First pass finds the peak; the live set is rebuilt at the peak in a
    # second pass so the first stays O(n) without copying dicts.
    for i, e in enumerate(events):
        action = e.get("action")
        actions[str(action)] += 1
        if action == "alloc":
            b = {"size": rounded(int(e.get("size", 0))), "requested_size": int(e.get("size", 0))}
            live[e.get("addr")] = b
            running += b["size"]
            running_requested += b["requested_size"]
            if running > peak:
                peak, peak_requested, peak_index = running, running_requested, i
        elif action == "free_requested":
            b = live.pop(e.get("addr"), None)
            if b is None:
                size = int(e.get("size", 0))
                running -= rounded(size)
                running_requested -= size
                freed_before += rounded(size)
                freed_before_count += 1
            else:
                running -= b["size"]
                running_requested -= b["requested_size"]
        elif action == "oom":
            oom.append({"index": i, "size": e.get("size"), "device_free": e.get("device_free")})
        elif action in _IGNORED:
            continue
    end_running = running

    alive: dict[Any, dict[str, Any]] = {}
    if peak_index >= 0:
        for i, e in enumerate(events[: peak_index + 1]):
            action = e.get("action")
            if action == "alloc":
                alive[e.get("addr")] = _block(e, i)
            elif action == "free_requested":
                alive.pop(e.get("addr"), None)
        peak_live = sorted(alive.values(), key=lambda b: (-b["size"], b["index"]))

    end_alive: dict[Any, dict[str, Any]] = {}
    for i, e in enumerate(events):
        action = e.get("action")
        if action == "alloc":
            end_alive[e.get("addr")] = _block(e, i)
        elif action == "free_requested":
            end_alive.pop(e.get("addr"), None)
    end_live = list(end_alive.values())

    peak_event = None
    if peak_index >= 0:
        e = events[peak_index]
        peak_event = {
            "index": peak_index,
            "requested_size": int(e.get("size", 0)),
            "phase": phase_of(list(e.get("frames") or [])),
            "frames": [frame_str(f) for f in list(e.get("frames") or [])[:25]],
        }

    live_bytes = sum(b["size"] for b in peak_live)
    return {
        "device": device,
        "events": len(events),
        "actions": dict(sorted(actions.items())),
        "truncated": bool(max_entries) and len(events) >= int(max_entries or 0),
        "peak_increase_bytes": peak,
        "peak_increase_requested_bytes": peak_requested,
        "peak_event_index": peak_index,
        "peak_event": peak_event,
        # At the peak, the blocks allocated during recording and still alive
        # sum to the peak plus what pre-recording frees had released by then.
        "peak_live_bytes": live_bytes,
        "peak_live_count": len(peak_live),
        "freed_pre_recording_bytes": freed_before,
        "freed_pre_recording_count": freed_before_count,
        "end_increase_bytes": end_running,
        "phase_at_peak": _phase_totals(peak_live),
        "dominant_phase_at_peak": next(iter(_phase_totals(peak_live)), None),
        "sites_at_peak": _group(peak_live, groups),
        "largest_blocks_at_peak": [
            {k: (list(v) if k == "site" else v) for k, v in b.items()}
            for b in peak_live[:top_blocks]
        ],
        "live_at_end_bytes": sum(b["size"] for b in end_live),
        "sites_live_at_end": _group(end_live, 15),
        "oom": oom,
    }
