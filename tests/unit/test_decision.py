"""Tests for decision report generation."""

from apron.application.orchestration.decision import build_report
from apron.domain.canonical import canonicalize
from apron.domain.schemas.authority import DecisionRequest
from apron.domain.schemas.reports import CandidateEntry, DecisionReport


class _FixedClock:
    def now(self):
        from datetime import UTC, datetime

        return datetime(2026, 1, 1, tzinfo=UTC)


class _FixedIdGen:
    def generate(self):
        return "fixed-id"


_FP = "1220" + "ab" * 32
_FP2 = "1220" + "cd" * 32
_REQUEST = DecisionRequest(objective="Deploy a model for chat")


def test_single_candidate_report():
    entry = CandidateEntry(
        solution_fingerprint=_FP,
        qualification_status="qualified",
    )
    report = build_report(
        _REQUEST,
        [entry],
        clock=_FixedClock(),
        id_gen=_FixedIdGen(),
    )
    assert isinstance(report, DecisionReport)
    assert len(report.candidates) == 1
    assert report.disclosed_comparable_set == ()


def test_rejected_candidate_preserved():
    qualified = CandidateEntry(
        solution_fingerprint=_FP,
        qualification_status="qualified",
    )
    rejected = CandidateEntry(
        solution_fingerprint=_FP2,
        qualification_status="candidate",
        rejection_reason="Legacy entry",
    )
    report = build_report(
        _REQUEST,
        [qualified, rejected],
        clock=_FixedClock(),
        id_gen=_FixedIdGen(),
    )
    assert len(report.candidates) == 2
    assert report.candidates[1].rejection_reason == "Legacy entry"


def test_two_qualified_has_comparable_set():
    c1 = CandidateEntry(solution_fingerprint=_FP, qualification_status="qualified")
    c2 = CandidateEntry(solution_fingerprint=_FP2, qualification_status="qualified")
    report = build_report(
        _REQUEST,
        [c1, c2],
        clock=_FixedClock(),
        id_gen=_FixedIdGen(),
    )
    assert len(report.disclosed_comparable_set) == 2


def test_zero_candidates():
    report = build_report(
        _REQUEST,
        [],
        clock=_FixedClock(),
        id_gen=_FixedIdGen(),
    )
    assert len(report.candidates) == 0
    assert report.disclosed_comparable_set == ()


def test_round_trips_through_canonicalize():
    entry = CandidateEntry(solution_fingerprint=_FP, qualification_status="qualified")
    report = build_report(
        _REQUEST,
        [entry],
        clock=_FixedClock(),
        id_gen=_FixedIdGen(),
    )
    b1 = canonicalize(report.model_dump(mode="json"))
    b2 = canonicalize(report.model_dump(mode="json"))
    assert b1 == b2
