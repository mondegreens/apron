"""Append-only JSONL budget ledger (implements budget.LedgerLog).

Each entry is written, flushed and fsynced before ``append`` returns, and
the budget tracker appends before the call it guards.  A crash can therefore
tear only the last line, and a torn last line means the guarded call never
started: it is reported and skipped, never parsed into a partial entry.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class JsonlLedger:
    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def append(self, entry: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, sort_keys=True) + "\n"
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())

    def read_all(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        lines = self._path.read_text(encoding="utf-8").splitlines()
        entries: list[dict[str, Any]] = []
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                if number == len(lines):
                    logger.error(
                        "ledger %s: torn final line %d skipped: %r", self._path, number, line
                    )
                    continue
                raise ValueError(f"ledger {self._path} line {number} is corrupt") from None
        return entries
