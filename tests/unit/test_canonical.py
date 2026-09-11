"""Tests for RFC 8785 canonicalization and multihash-prefixed SHA-256 digest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apron.domain.canonical import (
    MULTIHASH_SHA2_256,
    canonicalize,
    digest,
    digest_hex,
    record_digest,
    record_digest_hex,
)

# ---------------------------------------------------------------------------
# RFC 8785 canonicalization vectors
# Sourced from cyberphone/json-canonicalization testdata/
# ---------------------------------------------------------------------------

RFC8785_VECTORS: list[tuple[object, bytes]] = [
    # Sorted object keys
    ({"e": "f", "a": "b", "c": "d"}, b'{"a":"b","c":"d","e":"f"}'),
    # Integers
    ({"val": 1}, b'{"val":1}'),
    ({"val": 0}, b'{"val":0}'),
    ({"val": -1}, b'{"val":-1}'),
    # Booleans and null
    ({"t": True, "f": False, "n": None}, b'{"f":false,"n":null,"t":true}'),
    # Nested objects
    (
        {"b": {"z": 1, "a": 2}, "a": []},
        b'{"a":[],"b":{"a":2,"z":1}}',
    ),
    # String escaping
    ({"esc": "\n\t\r"}, b'{"esc":"\\n\\t\\r"}'),
    # Empty structures
    ({}, b"{}"),
    ([], b"[]"),
    # Array with mixed types
    ([1, "two", True, None], b'[1,"two",true,null]'),
    # Unicode - BMP characters pass through
    ({"key": "é"}, b'{"key":"\xc3\xa9"}'),
]


@pytest.mark.parametrize("input_val,expected", RFC8785_VECTORS, ids=range(len(RFC8785_VECTORS)))
def test_rfc8785_canonicalization(input_val: object, expected: bytes) -> None:
    result = canonicalize(input_val)
    assert result == expected, f"Expected {expected!r}, got {result!r}"


# Float serialization per RFC 8785 §3.2.2.3
FLOAT_VECTORS: list[tuple[float, str]] = [
    (0.0, "0"),
    (-0.0, "0"),
    (1.0, "1"),
    (-1.0, "-1"),
    (1.5, "1.5"),
    (1e20, "100000000000000000000"),
    (1e-7, "1e-7"),
    (1e-6, "0.000001"),
    (1e21, "1e+21"),
]


@pytest.mark.parametrize("value,expected", FLOAT_VECTORS, ids=[str(v) for v, _ in FLOAT_VECTORS])
def test_float_serialization(value: float, expected: str) -> None:
    result = canonicalize(value)
    assert result == expected.encode("utf-8")


def test_nan_raises() -> None:
    with pytest.raises(ValueError, match="NaN"):
        canonicalize(float("nan"))


def test_infinity_raises() -> None:
    with pytest.raises(ValueError, match="Infinity"):
        canonicalize(float("inf"))


# ---------------------------------------------------------------------------
# Digest vectors (project-owned)
# ---------------------------------------------------------------------------

VECTOR_FILE = Path(__file__).parent.parent / "vectors" / "digest" / "vectors.json"


def _load_digest_vectors() -> list[dict]:
    data = json.loads(VECTOR_FILE.read_text())
    return data["vectors"]


@pytest.mark.parametrize("vector", _load_digest_vectors(), ids=lambda v: v["description"])
def test_digest_against_vectors(vector: dict) -> None:
    canonical_bytes = canonicalize(vector["input"])
    assert canonical_bytes.hex() == vector["canonical_utf8_hex"]
    assert digest_hex(canonical_bytes) == vector["digest_hex"]


def test_multihash_prefix() -> None:
    d = digest(b"test")
    assert d[:2] == MULTIHASH_SHA2_256
    assert len(d) == 34  # 2 prefix + 32 hash


def test_record_digest_round_trip() -> None:
    obj = {"model": "qwen3-8b", "memory_gb": 16}
    d1 = record_digest(obj)
    d2 = record_digest(obj)
    assert d1 == d2


def test_record_digest_hex_matches() -> None:
    obj = {"x": 1}
    assert record_digest(obj).hex() == record_digest_hex(obj)


def test_key_order_does_not_affect_digest() -> None:
    obj1 = {"b": 2, "a": 1}
    obj2 = {"a": 1, "b": 2}
    assert record_digest(obj1) == record_digest(obj2)
