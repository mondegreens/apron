"""Diagnosis rule matching — pure domain logic, no I/O.

Given a failure class and a loaded rule set, find the matching rule
by error_family. Returns None for unmatched classes (trace capture only).
"""

from __future__ import annotations

from typing import Any


def match_rule(failure_class: str, rules: list[dict[str, Any]]) -> dict[str, Any] | None:
    """First rule whose error_family matches failure_class."""
    for rule in rules:
        if rule.get("error_family") == failure_class:
            return rule
    return None
