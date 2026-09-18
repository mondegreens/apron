"""Load diagnosis rules from JSON files.

Rules live in rules/<engine>-v<major.minor>/*.json, version-pinned to
the engine whose error messages they match. This module handles I/O
only; matching logic lives in domain/diagnosis.py.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "engine",
        "engine_version",
        "error_family",
        "correction_strategy",
        "status",
    }
)


def _find_version_dir(rules_dir: Path, engine: str, version: str) -> Path | None:
    v = version if version.startswith("v") else f"v{version}"
    candidate = rules_dir / f"{engine}-{v}"
    if candidate.is_dir():
        return candidate
    parts = v.split(".")
    if len(parts) > 2:
        major_minor = ".".join(parts[:2])
        candidate = rules_dir / f"{engine}-{major_minor}"
        if candidate.is_dir():
            return candidate
    return None


def load_rules(rules_dir: Path, engine: str, version: str) -> list[dict[str, Any]]:
    """Load rules for an engine version from JSON files."""
    version_dir = _find_version_dir(rules_dir, engine, version)
    if version_dir is None:
        logger.debug("No rules directory for %s %s in %s", engine, version, rules_dir)
        return []

    rules: list[dict[str, Any]] = []
    for path in sorted(version_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            msg = f"Malformed JSON in {path}: {exc}"
            raise ValueError(msg) from exc

        missing = _REQUIRED_FIELDS - set(data.keys())
        if missing:
            msg = f"Rule {path.name} missing required fields: {sorted(missing)}"
            raise ValueError(msg)

        rules.append(data)

    return rules
