"""Local filesystem record store — implements RecordStore Protocol.

Stores records as canonical JSON files keyed by content-addressed
SHA-256 multihash digest. Idempotent: two stores of the same record
produce the same file at the same path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from apron.domain.canonical import canonicalize, record_digest_hex

_RECORD_TYPE_SUBDIRS: dict[str, str] = {
    "verification_report": "verification-reports",
    "task_attempt_record": "task-attempt-records",
    "remediation_record": "remediation-records",
    "deployment_plan": "deployment-plans",
    "decision_report": "decision-reports",
    "planning_claim": "planning-claims",
    "diagnosis_rule": "diagnosis-rules",
}


def _subdir_for(record: dict[str, Any]) -> str:
    """Determine the storage subdirectory from record content."""
    for key, subdir in _RECORD_TYPE_SUBDIRS.items():
        if key in str(record.get("claim_scope", "")) or key in str(record.get("__type__", "")):
            return subdir

    if "claim_scope" in record:
        scope = record["claim_scope"]
        mapping = {
            "boot": "verification-reports",
            "memory": "verification-reports",
            "config": "planning-claims",
            "remediation": "remediation-records",
            "serving_performance": "verification-reports",
            "task_outcome": "task-attempt-records",
            "outcome_economics": "task-attempt-records",
        }
        if scope in mapping:
            return mapping[scope]

    return "records"


class LocalRecordStore:
    """Filesystem-backed content-addressed record store."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base = base_dir or Path.home() / ".apron" / "records"

    def store(self, record: dict[str, Any]) -> str:
        canonical = canonicalize(record)
        digest = record_digest_hex(record)
        subdir = _subdir_for(record)
        path = self._base / subdir / f"{digest}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical)
        return digest

    def retrieve(self, digest: str) -> dict[str, Any] | None:
        if len(digest) < 8:
            return None

        for subdir in self._base.iterdir():
            if not subdir.is_dir():
                continue

            if len(digest) == 68:
                path = subdir / f"{digest}.json"
                if path.exists():
                    return json.loads(path.read_bytes())
            else:
                matches = list(subdir.glob(f"{digest}*.json"))
                if len(matches) == 1:
                    return json.loads(matches[0].read_bytes())
                if len(matches) > 1:
                    raise ValueError(f"Ambiguous prefix '{digest}' matches {len(matches)} records")

        return None

    def search(self, prefix: str) -> list[str]:
        results: list[str] = []
        if not self._base.exists():
            return results
        for subdir in self._base.iterdir():
            if not subdir.is_dir():
                continue
            for path in subdir.glob(f"{prefix}*.json"):
                results.append(path.stem)
        return sorted(results)
