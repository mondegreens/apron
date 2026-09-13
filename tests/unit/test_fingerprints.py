"""Step 1 tests: migration infrastructure, fingerprint markers, canonicalization."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict

from apron.domain.canonical import canonicalize, record_digest_hex
from apron.domain.fingerprints import (
    DISPLAY,
    IDENTITY,
    FingerprintHex,
    assert_fully_classified,
    fingerprint_hex,
    identity_field_names,
)
from apron.domain.schemas.migrations import clear_registry, migrate, register


class ToyModelV1(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema_version: Annotated[int, DISPLAY] = 1
    name: Annotated[str, IDENTITY] = ""
    created_at: Annotated[datetime, DISPLAY] = datetime(2026, 1, 1, tzinfo=UTC)


class ToyModelV2(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema_version: Annotated[int, DISPLAY] = 2
    name: Annotated[str, IDENTITY] = ""
    created_at: Annotated[datetime, DISPLAY] = datetime(2026, 1, 1, tzinfo=UTC)
    description: Annotated[str | None, DISPLAY] = None


def setup_function() -> None:
    clear_registry()


# --- canonicalization with datetime ---


def test_model_dump_json_mode_with_datetime():
    m = ToyModelV1(name="hello", created_at=datetime(2026, 9, 12, 10, 30, tzinfo=UTC))
    dumped = m.model_dump(mode="json")
    assert isinstance(dumped["created_at"], str)
    canonical = canonicalize(dumped)
    assert isinstance(canonical, bytes)


def test_round_trip_toy_model():
    m = ToyModelV1(name="alpha", created_at=datetime(2026, 6, 1, tzinfo=UTC))
    canonical = canonicalize(m.model_dump(mode="json"))
    restored = ToyModelV1.model_validate_json(canonical)
    assert canonicalize(restored.model_dump(mode="json")) == canonical


# --- IDENTITY / DISPLAY markers ---


def test_identity_field_names():
    names = identity_field_names(ToyModelV1)
    assert names == frozenset({"name"})


def test_assert_fully_classified_passes():
    assert_fully_classified(ToyModelV1)
    assert_fully_classified(ToyModelV2)


def test_assert_fully_classified_fails_on_unclassified():
    class Bad(BaseModel):
        model_config = ConfigDict(frozen=True)
        name: str = ""

    try:
        assert_fully_classified(Bad)
        raise RuntimeError("should have raised")
    except AssertionError as e:
        assert "name" in str(e)


# --- fingerprint vs digest ---


def test_fingerprint_stable_on_display_change():
    a = ToyModelV1(name="x", created_at=datetime(2026, 1, 1, tzinfo=UTC))
    b = ToyModelV1(name="x", created_at=datetime(2026, 12, 31, tzinfo=UTC))
    assert fingerprint_hex(a) == fingerprint_hex(b)


def test_fingerprint_changes_on_identity_change():
    a = ToyModelV1(name="x")
    b = ToyModelV1(name="y")
    assert fingerprint_hex(a) != fingerprint_hex(b)


def test_fingerprint_differs_from_full_digest():
    m = ToyModelV1(name="x", created_at=datetime(2026, 6, 1, tzinfo=UTC))
    fp = fingerprint_hex(m)
    full = record_digest_hex(m.model_dump(mode="json"))
    assert fp != full


def test_fingerprint_hex_pattern():
    m = ToyModelV1(name="test")
    fp = fingerprint_hex(m)
    assert fp.startswith("1220")
    assert len(fp) == 68


def test_fingerprint_hex_validates():
    import re

    from apron.domain.fingerprints import MULTIHASH_SHA256_PATTERN

    m = ToyModelV1(name="test")
    fp = fingerprint_hex(m)
    assert re.match(MULTIHASH_SHA256_PATTERN, fp)


# --- migration ---


def test_migration_produces_new_digest():
    @register("ToyModel", 1)
    def _migrate_v1_to_v2(data: dict) -> dict:
        return {**data, "schema_version": 2, "description": None}

    v1 = ToyModelV1(name="original")
    v1_data = v1.model_dump(mode="json")
    v1_digest = record_digest_hex(v1_data)

    v2_data = migrate("ToyModel", v1_data, target_version=2)
    v2 = ToyModelV2.model_validate(v2_data)
    v2_digest = record_digest_hex(v2.model_dump(mode="json"))

    assert v2.schema_version == 2
    assert v2.description is None
    assert v2.name == "original"
    assert v1_digest != v2_digest


def test_migration_preserves_fingerprint():
    @register("ToyModel", 1)
    def _migrate_v1_to_v2(data: dict) -> dict:
        return {**data, "schema_version": 2, "description": None}

    v1 = ToyModelV1(name="original")
    v1_fp = fingerprint_hex(v1)

    v2_data = migrate("ToyModel", v1.model_dump(mode="json"), target_version=2)
    v2 = ToyModelV2.model_validate(v2_data)
    v2_fp = fingerprint_hex(v2)

    assert v1_fp == v2_fp


def test_old_reference_remains_valid():
    @register("ToyModel", 1)
    def _migrate_v1_to_v2(data: dict) -> dict:
        return {**data, "schema_version": 2, "description": None}

    v1 = ToyModelV1(name="keep-me")
    v1_data = v1.model_dump(mode="json")
    v1_digest = record_digest_hex(v1_data)

    _ = migrate("ToyModel", v1_data, target_version=2)

    assert record_digest_hex(v1_data) == v1_digest


def test_migration_missing_version_raises():
    try:
        migrate("ToyModel", {"name": "bad"}, target_version=2)
        raise RuntimeError("should have raised")
    except ValueError as e:
        assert "schema_version" in str(e)


def test_migration_unregistered_raises():
    try:
        migrate("ToyModel", {"schema_version": 1, "name": "x"}, target_version=2)
        raise RuntimeError("should have raised")
    except KeyError as e:
        assert "ToyModel" in str(e)


def test_migration_duplicate_registration_raises():
    @register("Dup", 1)
    def _first(data: dict) -> dict:
        return data

    try:

        @register("Dup", 1)
        def _second(data: dict) -> dict:
            return data

        raise RuntimeError("should have raised")
    except ValueError as e:
        assert "already registered" in str(e)
