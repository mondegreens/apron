"""Field classification markers and identity-subset fingerprinting.

Every field on every frozen model carries either IDENTITY or DISPLAY via
Annotated markers.  IDENTITY fields define what the record IS; DISPLAY
fields are presentation metadata that may change without altering the
record's identity.  ``fingerprint_hex`` hashes only the IDENTITY subset.
``record_digest_hex`` (in canonical.py) hashes the full record.

Do NOT use PEP 695 type aliases (``type X = Annotated[...]``) — pydantic 2.x
silently drops metadata from them.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, StringConstraints

from apron.domain.canonical import canonicalize, digest_hex

MULTIHASH_SHA256_PATTERN = r"^1220[0-9a-f]{64}$"

FingerprintHex = Annotated[
    str, StringConstraints(pattern=MULTIHASH_SHA256_PATTERN)
]


class _Marker:
    __slots__ = ("label",)

    def __init__(self, label: str) -> None:
        self.label = label

    def __repr__(self) -> str:
        return self.label


IDENTITY = _Marker("IDENTITY")
DISPLAY = _Marker("DISPLAY")


def identity_field_names(cls: type[BaseModel]) -> frozenset[str]:
    """Return the names of all fields marked IDENTITY on *cls*."""
    return frozenset(
        name
        for name, info in cls.model_fields.items()
        if any(m is IDENTITY for m in info.metadata)
    )


def assert_fully_classified(cls: type[BaseModel]) -> None:
    """Raise if any field on *cls* lacks an IDENTITY or DISPLAY marker."""
    unclassified = [
        name
        for name, info in cls.model_fields.items()
        if not any(m is IDENTITY or m is DISPLAY for m in info.metadata)
    ]
    if unclassified:
        raise AssertionError(
            f"{cls.__name__}: unclassified fields {unclassified}"
        )


def fingerprint_hex(record: BaseModel) -> str:
    """Hash the IDENTITY-field subset of *record* and return a multihash hex string."""
    full = record.model_dump(mode="json")
    id_names = identity_field_names(type(record))
    subset = {k: full[k] for k in sorted(id_names)}
    return digest_hex(canonicalize(subset))


def _has_display_fields(cls: type[BaseModel]) -> bool:
    """True if *cls* has at least one DISPLAY field."""
    return any(
        any(m is DISPLAY for m in info.metadata)
        for info in cls.model_fields.values()
    )
