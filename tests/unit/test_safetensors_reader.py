"""Tests for safetensors reader — index parsing and binary header reading."""

from pathlib import Path

import pytest

from apron.adapters.evidence.safetensors_reader import (
    read_all_tensors,
    read_index,
    read_index_metadata,
    read_shard_header,
)

FIXTURE_DIR = Path("tests/fixtures/external-formats/huggingface-hub")


# ---------------------------------------------------------------------------
# Index parsing
# ---------------------------------------------------------------------------


def test_read_index_returns_weight_map():
    weight_map = read_index(FIXTURE_DIR / "model.safetensors.index.json")
    assert isinstance(weight_map, dict)
    assert len(weight_map) == 399
    assert weight_map["lm_head.weight"] == "model-00005-of-00005.safetensors"


def test_read_index_metadata():
    meta = read_index_metadata(FIXTURE_DIR / "model.safetensors.index.json")
    assert meta["total_size"] == 16381470720


def test_unique_shards_count():
    weight_map = read_index(FIXTURE_DIR / "model.safetensors.index.json")
    unique_shards = set(weight_map.values())
    assert len(unique_shards) == 5


# ---------------------------------------------------------------------------
# Binary header parsing (synthetic fixture)
# ---------------------------------------------------------------------------


def test_read_shard_header_synthetic():
    tensors = read_shard_header(FIXTURE_DIR / "synthetic.safetensors")
    assert "weight_a" in tensors
    assert "weight_b" in tensors
    assert "__metadata__" not in tensors


def test_synthetic_tensor_metadata():
    tensors = read_shard_header(FIXTURE_DIR / "synthetic.safetensors")
    a = tensors["weight_a"]
    assert a["dtype"] == "BF16"
    assert a["shape"] == [4, 8]
    assert a["byte_count"] == 64

    b = tensors["weight_b"]
    assert b["dtype"] == "F32"
    assert b["shape"] == [2, 3]
    assert b["byte_count"] == 24


def test_read_shard_header_too_small(tmp_path):
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"\x00" * 4)
    with pytest.raises(ValueError, match="too small"):
        read_shard_header(bad)


# ---------------------------------------------------------------------------
# read_all_tensors (requires shard files — tested with synthetic only)
# ---------------------------------------------------------------------------


def test_read_all_tensors_synthetic(tmp_path):
    """Create a mini index + shard and verify read_all_tensors."""
    import json
    import struct

    header = json.dumps(
        {
            "__metadata__": {},
            "t1": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]},
        },
        sort_keys=True,
    ).encode()

    shard_path = tmp_path / "shard-00001.safetensors"
    with open(shard_path, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        f.write(b"\x00" * 4)

    index = {"metadata": {"total_size": 4}, "weight_map": {"t1": "shard-00001.safetensors"}}
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index))

    tensors = read_all_tensors(index_path)
    assert "t1" in tensors
    assert tensors["t1"]["dtype"] == "F16"
    assert tensors["t1"]["byte_count"] == 4
