"""The committed calculator recheck is what the current calculator computes.

``_dev_notes/cohort-run/calculator-recheck.json`` (scripts/calculator_recheck.py)
feeds the findings; if the calculator changes, this fails until it is rerun.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
COMMITTED = REPO / "_dev_notes" / "cohort-run" / "calculator-recheck.json"


@pytest.mark.skipif(not COMMITTED.exists(), reason="no recheck on record")
def test_committed_recheck_equals_the_current_calculator(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "calculator_recheck", REPO / "scripts" / "calculator_recheck.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    setattr(module, "OUT", tmp_path / "recheck.json")  # noqa: B010 — a module global
    assert module.main() == 0
    assert json.loads(module.OUT.read_text()) == json.loads(COMMITTED.read_text()), (
        "run scripts/calculator_recheck.py"
    )
