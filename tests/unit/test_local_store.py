"""Tests for local filesystem record store."""

import json

import pytest
from apron.adapters.backends.local_store import LocalRecordStore
from apron.domain.canonical import canonicalize
from apron.domain.protocols import RecordStore


@pytest.fixture()
def store(tmp_path):
    return LocalRecordStore(base_dir=tmp_path)


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------


def test_satisfies_record_store_protocol(store):
    assert isinstance(store, RecordStore)


# ---------------------------------------------------------------------------
# Store and retrieve
# ---------------------------------------------------------------------------


def test_store_and_retrieve_by_full_digest(store):
    record = {"type": "test", "value": 42}
    digest = store.store(record)
    assert digest.startswith("1220")
    assert len(digest) == 68
    retrieved = store.retrieve(digest)
    assert retrieved == record


def test_retrieve_by_8_char_prefix(store):
    record = {"type": "test", "value": 99}
    digest = store.store(record)
    retrieved = store.retrieve(digest[:8])
    assert retrieved == record


def test_retrieve_nonexistent_returns_none(store):
    assert store.retrieve("1220" + "00" * 32) is None


def test_retrieve_short_prefix_returns_none(store):
    assert store.retrieve("abc") is None


# ---------------------------------------------------------------------------
# Idempotent store
# ---------------------------------------------------------------------------


def test_store_same_record_twice_is_idempotent(store):
    record = {"type": "test", "value": 1}
    d1 = store.store(record)
    d2 = store.store(record)
    assert d1 == d2
    assert store.retrieve(d1) == record


# ---------------------------------------------------------------------------
# Canonical storage
# ---------------------------------------------------------------------------


def test_stored_content_is_canonical(store, tmp_path):
    record = {"b_key": 2, "a_key": 1}
    digest = store.store(record)

    found = list(tmp_path.rglob(f"{digest}.json"))
    assert len(found) == 1
    stored_bytes = found[0].read_bytes()
    re_canonical = canonicalize(json.loads(stored_bytes))
    assert stored_bytes == re_canonical


# ---------------------------------------------------------------------------
# Ambiguous prefix
# ---------------------------------------------------------------------------


def test_ambiguous_prefix_raises(store):
    r1 = {"type": "a", "v": 1}
    r2 = {"type": "a", "v": 2}
    d1 = store.store(r1)
    d2 = store.store(r2)
    if d1[:8] == d2[:8]:
        with pytest.raises(ValueError, match="Ambiguous"):
            store.retrieve(d1[:8])


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_search_finds_stored_records(store):
    r1 = {"type": "x", "v": 1}
    r2 = {"type": "x", "v": 2}
    d1 = store.store(r1)
    d2 = store.store(r2)
    all_results = store.search("1220")
    assert d1 in all_results
    assert d2 in all_results


def test_search_empty_store(store):
    assert store.search("1220") == []


# ---------------------------------------------------------------------------
# Concurrent writes (same content → idempotent)
# ---------------------------------------------------------------------------


def test_concurrent_writes_same_content(store):
    record = {"type": "concurrent", "v": 42}
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(store.store, record) for _ in range(4)]
        digests = [f.result() for f in futures]
    assert len(set(digests)) == 1
    assert store.retrieve(digests[0]) == record
