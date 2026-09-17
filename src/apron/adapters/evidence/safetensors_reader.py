"""Read safetensors shard binary headers for per-tensor metadata.

Manual parse (~20 lines) with struct + json. No dependency on the
safetensors package for header-only reading.
"""

from __future__ import annotations

import json
import struct
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


def read_index(index_path: Path) -> dict[str, str]:
    """Parse model.safetensors.index.json and return tensor→shard mapping."""
    data = json.loads(index_path.read_bytes())
    return data["weight_map"]


def read_index_metadata(index_path: Path) -> dict[str, Any]:
    """Parse the metadata block from the index file."""
    data = json.loads(index_path.read_bytes())
    return data.get("metadata", {})


def read_shard_header(shard_path: Path) -> dict[str, dict[str, Any]]:
    """Read a safetensors shard's binary header without loading tensor data.

    Returns a dict mapping tensor names to their metadata:
    ``{name: {"dtype": str, "shape": list[int], "byte_count": int}}``.
    The ``__metadata__`` key is excluded.
    """
    with open(shard_path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) < 8:
            raise ValueError(f"File too small to contain a safetensors header: {shard_path}")
        (header_len,) = struct.unpack("<Q", raw_len)
        header_bytes = f.read(header_len)

    header = json.loads(header_bytes)
    tensors: dict[str, dict[str, Any]] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        offsets = info["data_offsets"]
        tensors[name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "byte_count": offsets[1] - offsets[0],
        }
    return tensors


def read_all_tensors(
    index_path: Path,
    shard_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Read tensor metadata from all shards referenced by the index.

    Falls back to ``index_path.parent`` as the shard directory.
    """
    if shard_dir is None:
        shard_dir = index_path.parent

    weight_map = read_index(index_path)
    unique_shards = set(weight_map.values())

    all_tensors: dict[str, dict[str, Any]] = {}
    for shard_name in sorted(unique_shards):
        shard_path = shard_dir / shard_name
        all_tensors.update(read_shard_header(shard_path))

    return all_tensors
