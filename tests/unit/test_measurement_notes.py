"""L0-A3's activation spread becomes a note on every prediction delta (§6.1)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from apron.interfaces.cohort_root import measurement_notes

if TYPE_CHECKING:
    from pathlib import Path


def _stability(tmp_path: Path, spread: float) -> None:
    (tmp_path / "l0a3-stability.json").write_text(
        json.dumps(
            {
                "weight_identical": True,
                "kv_relative_difference": 0.002,
                "activation_relative_difference": spread,
                "activation_uncertainty_note_required": spread > 0.05,
            }
        )
    )


def test_no_note_before_l0a3_or_within_threshold(tmp_path: Path) -> None:
    assert measurement_notes(tmp_path) == ()
    _stability(tmp_path, 0.03)
    assert measurement_notes(tmp_path) == ()


def test_note_names_the_measured_spread(tmp_path: Path) -> None:
    _stability(tmp_path, 0.083)
    (note,) = measurement_notes(tmp_path)
    assert "8.3%" in note and "L0-A3" in note
