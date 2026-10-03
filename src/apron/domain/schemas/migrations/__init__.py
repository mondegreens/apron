"""Forward-only schema migration registry (INV-5).

Migration is type-level schema evolution.  A function in the registry
transforms all v(N) instances of a type to v(N+1).  Old versions remain
retrievable by their old digest; the fingerprint is preserved when only
DISPLAY or optional fields change.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

MigrationFn = Callable[[dict[str, Any]], dict[str, Any]]

_REGISTRY: dict[tuple[str, int], MigrationFn] = {}


def register(schema_name: str, from_version: int) -> Callable[[MigrationFn], MigrationFn]:
    """Decorator that registers a migration from *from_version* to *from_version + 1*."""

    def decorator(fn: MigrationFn) -> MigrationFn:
        key = (schema_name, from_version)
        if key in _REGISTRY:
            raise ValueError(f"migration already registered: {schema_name} v{from_version}")
        _REGISTRY[key] = fn
        return fn

    return decorator


def migrate(schema_name: str, data: dict[str, Any], target_version: int) -> dict[str, Any]:
    """Apply successive migrations from ``data["schema_version"]`` up to *target_version*."""
    current = data.get("schema_version")
    if current is None:
        raise ValueError(f"missing schema_version in {schema_name} record")
    result = dict(data)
    while current < target_version:
        key = (schema_name, current)
        fn = _REGISTRY.get(key)
        if fn is None:
            raise KeyError(f"no migration registered: {schema_name} v{current} → v{current + 1}")
        result = fn(result)
        current = result.get("schema_version", current + 1)
    return result


def clear_registry() -> None:
    """Remove all registered migrations (for tests only)."""
    _REGISTRY.clear()


def load_record[M: BaseModel](cls: type[M], data: dict[str, Any]) -> M:
    """Load a stored record through the migration path, then validate strictly.

    Every stored record is migrated forward to the current schema version and
    validated with the model's ``extra="forbid"`` config.  A record with keys
    the schema does not know fails loudly with a ``ValidationError`` naming
    the field and its path; nothing is silently stripped.
    """
    field = cls.model_fields.get("schema_version")
    if field is None or not isinstance(field.default, int):
        return cls.model_validate(data)
    migrated = migrate(cls.__name__, data, field.default)
    return cls.model_validate(migrated)
