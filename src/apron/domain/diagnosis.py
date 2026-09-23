"""Diagnosis domain — failure taxonomy and rule matching.

The failure taxonomy is engine-independent and derived from loaded
rules, not hardcoded. Each engine adapter's source scanner produces
rule files; this module reads them and builds the taxonomy, extraction
schemas, and matching logic at load time.
"""

from __future__ import annotations

from typing import Any

_TYPE_MAP = {"string": str, "integer": int, "number": float, "float": float, "int": int}


def build_extraction_schemas(
    rules: list[dict[str, Any]],
) -> dict[str, list[tuple[str, type]]]:
    """Build extraction schemas from loaded rule files.

    Each rule carries an extraction_schema dict mapping field names
    to type strings. This converts them to the typed tuples the
    pipeline and LLM classifier consume.
    """
    schemas: dict[str, list[tuple[str, type]]] = {}
    for rule in rules:
        family = rule.get("error_family", "")
        raw_schema = rule.get("extraction_schema", {})
        if not family or not raw_schema:
            continue
        fields = [
            (name, _TYPE_MAP.get(type_str, str))
            for name, type_str in raw_schema.items()
            if name and not name.startswith("_")
        ]
        schemas[family] = fields
    return schemas


def build_failure_classes(rules: list[dict[str, Any]]) -> list[str]:
    """Build the list of known failure classes from loaded rules."""
    seen: set[str] = set()
    classes: list[str] = []
    for rule in rules:
        family = rule.get("error_family", "")
        if family and family not in seen:
            seen.add(family)
            classes.append(family)
    classes.append("unknown")
    return classes


def match_rule(failure_class: str, rules: list[dict[str, Any]]) -> dict[str, Any] | None:
    """First rule whose error_family matches failure_class."""
    for rule in rules:
        if rule.get("error_family") == failure_class:
            return rule
    return None


def extraction_confidence(
    failure_class: str,
    extracted: dict[str, Any],
    schemas: dict[str, list[tuple[str, type]]] | None = None,
) -> float:
    """Fraction of expected fields that were actually extracted (0.0-1.0)."""
    if schemas is None:
        return 1.0
    schema = schemas.get(failure_class)
    if not schema:
        return 1.0
    expected = len(schema)
    found = sum(1 for key, _ in schema if key in extracted)
    return found / expected
