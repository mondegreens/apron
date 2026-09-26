"""File-backed rule repository (implements remediation.RuleRepository).

The current version of each rule is ``rules/<engine>-v<major.minor>/<family>.json``
(what ``load_rules`` reads).  Writing a new version moves the previous file to
``history/<family>.v<N>.json`` in the same directory — never overwritten, never
deleted — so a promotion can always be traced to the rule it superseded.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from apron.domain.canonical import record_digest_hex
from apron.domain.schemas.migrations import load_record
from apron.domain.schemas.records import DiagnosisRule

if TYPE_CHECKING:
    from pathlib import Path


class FileRuleRepository:
    def __init__(self, version_dir: Path) -> None:
        self._dir = version_dir

    def current(self, family: str) -> DiagnosisRule:
        path = self._dir / f"{family}.json"
        return load_record(DiagnosisRule, json.loads(path.read_text(encoding="utf-8")))

    def write_version(self, rule: DiagnosisRule) -> str:
        path = self._dir / f"{rule.error_family}.json"
        if path.exists():
            previous = self.current(rule.error_family)
            if rule.rule_version <= previous.rule_version:
                raise ValueError(
                    f"{rule.error_family}: version {rule.rule_version} does not supersede "
                    f"{previous.rule_version}"
                )
            history = self._dir / "history" / f"{rule.error_family}.v{previous.rule_version}.json"
            history.parent.mkdir(parents=True, exist_ok=True)
            if history.exists():
                raise FileExistsError(history)
            history.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        data = rule.model_dump(mode="json")
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return record_digest_hex(data)
