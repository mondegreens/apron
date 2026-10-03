"""F10: records are built through their Pydantic model and validated before storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from apron.adapters.backends.local_store import LocalRecordStore
from apron.application.orchestration.evidence import InvalidRecordError, store_validated
from apron.domain.schemas.records import VerificationReport

if TYPE_CHECKING:
    from pathlib import Path

_FP = "1220" + "ab" * 32


def _report() -> VerificationReport:
    return VerificationReport(
        target_kind="rented-provider",
        operator="apron",
        execution_fingerprint=_FP,
        claim_scope="memory",
        production_mode=False,
        reason="verification",
        lifecycle="observed",
    )


def test_valid_record_is_stored_as_json(tmp_path: Path) -> None:
    store = LocalRecordStore(tmp_path)
    digest = store_validated(store, _report())
    stored = store.retrieve(digest)
    assert stored is not None
    assert stored["operator"] == "apron"


def test_record_that_bypassed_validation_is_not_stored(tmp_path: Path) -> None:
    store = LocalRecordStore(tmp_path)
    broken = _report().model_copy(update={"solution_fingerprint": "not-a-fingerprint"})
    with pytest.raises(InvalidRecordError, match="solution_fingerprint"):
        store_validated(store, broken)
    assert store.search("") == []
