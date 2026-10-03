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
from pydantic_core import to_jsonable_python

from apron.domain.canonical import canonicalize, digest_hex

MULTIHASH_SHA256_PATTERN = r"^1220[0-9a-f]{64}$"

FingerprintHex = Annotated[str, StringConstraints(pattern=MULTIHASH_SHA256_PATTERN)]


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
        raise AssertionError(f"{cls.__name__}: unclassified fields {unclassified}")


def _is_classified(cls: type[BaseModel]) -> bool:
    """True if *cls* marks at least one field IDENTITY or DISPLAY."""
    return any(
        any(m is IDENTITY or m is DISPLAY for m in info.metadata)
        for info in cls.model_fields.values()
    )


def identity_projection(value: Any) -> Any:
    """Project *value* onto its IDENTITY fields at every depth, in JSON mode.

    A classified ``BaseModel`` keeps only its IDENTITY fields, and each kept
    value is projected in turn — so a DISPLAY field on a nested model never
    reaches the hash.  Tuples, lists and dict values are projected element by
    element.  A model with no markers at all is plain data: every field is
    kept, so nothing is silently dropped from identity.  Leaves use pydantic's
    JSON-mode conversion, the same as ``model_dump(mode="json")``.
    """
    if isinstance(value, BaseModel):
        cls = type(value)
        names = identity_field_names(cls) if _is_classified(cls) else frozenset(cls.model_fields)
        return {name: identity_projection(getattr(value, name)) for name in sorted(names)}
    if isinstance(value, tuple | list):
        return [identity_projection(item) for item in value]
    if isinstance(value, dict):
        return {str(to_jsonable_python(k)): identity_projection(v) for k, v in value.items()}
    return to_jsonable_python(value)


def fingerprint_hex(record: BaseModel) -> str:
    """Hash the IDENTITY projection of *record* and return a multihash hex string.

    INV-43: the projection is serialized to RFC 8785 canonical form and hashed
    with SHA-256 under the multihash prefix.
    """
    return digest_hex(canonicalize(identity_projection(record)))


def _has_display_fields(cls: type[BaseModel]) -> bool:
    """True if *cls* has at least one DISPLAY field."""
    return any(any(m is DISPLAY for m in info.metadata) for info in cls.model_fields.values())
