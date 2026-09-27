"""Re-scoring recorded outputs under a new protocol (application/orchestration/rescore.py)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from _cohort_fakes import accepted_inputs
from _synthetic_run import build_synthetic_run

from apron.adapters.backends.local_store import LocalRecordStore
from apron.adapters.evaluations.deterministic_scorer import DeterministicScorer
from apron.application.orchestration.rescore import RESCORE_REASON, rescore_attempts
from apron.domain.ports import UuidIdGenerator
from apron.domain.schemas.migrations import load_record
from apron.domain.schemas.records import TaskAttemptRecord
from apron.interfaces.cohort_root import load_cohort_run


def _stricter_inputs():  # the Phase 1b rule over the synthetic run's inputs
    inputs = accepted_inputs()
    template = dict(inputs.protocol_template)
    template["deterministic_checks"] = [
        "whitespace_normalized_exact_match",
        "strip_terminal_punctuation",
    ]
    return replace(inputs, protocol_template=template)


def test_every_recorded_attempt_gets_one_linked_free_rescore(tmp_path: Path) -> None:
    rules = build_synthetic_run(tmp_path)
    run = load_cohort_run(tmp_path, rules)
    store = LocalRecordStore(tmp_path / "records")
    result = rescore_attempts(
        run.records, _stricter_inputs(), store, DeterministicScorer().rescore,
        UuidIdGenerator(), "2026-09-27T05:00:00+00:00",
    )
    assert len(result.attempts) == len(run.records.attempts)
    for digest in result.attempts:
        new = load_record(TaskAttemptRecord, store.retrieve(digest) or {})
        (source,) = new.trace_references
        old = run.records.attempts[source]
        assert new.reason == RESCORE_REASON
        assert new.raw_observation_provenance == f"rescored from {source}"
        assert new.output == old.output and new.case_id == old.case_id
        assert new.infrastructure_cost == 0.0
        assert new.evaluation_protocol_fingerprint != old.evaluation_protocol_fingerprint
    assert {m["solution_fingerprint"] for m in result.manifest} == {
        a.solution_fingerprint for a in run.records.attempts.values()
    }


def test_rescoring_twice_adds_nothing(tmp_path: Path) -> None:
    rules = build_synthetic_run(tmp_path)
    store = LocalRecordStore(tmp_path / "records")
    first = rescore_attempts(
        load_cohort_run(tmp_path, rules).records, _stricter_inputs(), store,
        DeterministicScorer().rescore, UuidIdGenerator(), "t",
    )
    with (tmp_path / "solutions.jsonl").open("a") as fh:
        for entry in first.manifest:
            fh.write(json.dumps(entry) + "\n")
    second = rescore_attempts(
        load_cohort_run(tmp_path, rules).records, _stricter_inputs(), store,
        DeterministicScorer().rescore, UuidIdGenerator(), "t",
    )
    assert second.attempts == [] and second.manifest == []
    assert set(second.skipped.values()) == {"already scored under this protocol"}


def test_scorer_rescore_applies_the_protocol_checks() -> None:
    protocol = {
        "cases": [{"id": "fact-1", "expected": "Paris"}, {"id": "arith-1", "expected": "4"}],
        "deterministic_checks": ["whitespace_normalized_exact_match", "strip_terminal_punctuation"],
    }
    recorded = [
        {"case_id": "fact-1", "output": "Paris.", "time_seconds": 0.1},
        {"case_id": "arith-1", "output": "2", "time_seconds": 0.1},
    ]
    scored = DeterministicScorer().rescore(protocol, recorded)
    assert [s["accepted"] for s in scored] == [True, False]
    assert scored[0]["time_seconds"] == 0.1  # kept as observed
    strict = DeterministicScorer().rescore(
        {**protocol, "deterministic_checks": ["whitespace_normalized_exact_match"]}, recorded
    )
    assert strict[0]["accepted"] is False
