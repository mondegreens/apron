"""Promotion writes a new rule version and keeps the superseded one in history."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from apron.adapters.backends.rule_loader import load_rules
from apron.adapters.backends.rule_repository import FileRuleRepository
from apron.application.orchestration.remediation import promoted_rule

RULES = Path(__file__).parents[2] / "rules" / "vllm-v0.29"


@pytest.fixture()
def repo_dir(tmp_path: Path) -> Path:
    target = tmp_path / "vllm-v0.29"
    shutil.copytree(RULES, target)
    return target


def test_promotion_writes_new_version_and_history(repo_dir: Path) -> None:
    repo = FileRuleRepository(repo_dir)
    old = repo.current("max_model_len")
    repo.write_version(promoted_rule(old, "1220" + "ab" * 32))
    new = repo.current("max_model_len")
    assert new.status == "mechanism_verified" and new.rule_version == old.rule_version + 1
    history = json.loads((repo_dir / "history" / "max_model_len.v1.json").read_text())
    assert history["status"] == "hypothesis"
    # the loader reads only the current versions
    families = [r["error_family"] for r in load_rules(repo_dir.parent, "vllm", "v0.29")]
    assert families.count("max_model_len") == 1


def test_stale_version_is_refused(repo_dir: Path) -> None:
    repo = FileRuleRepository(repo_dir)
    with pytest.raises(ValueError, match="does not supersede"):
        repo.write_version(repo.current("max_model_len"))
