"""Tests for the qualification graph — verdicts come from stored evidence (F3)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from apron.adapters.backends.local_store import LocalRecordStore
from apron.application.orchestration.candidates import ResolvedCandidate
from apron.application.orchestration.qualification import (
    AcceptedRequest,
    EvidenceDigests,
    QualificationGraph,
)
from apron.domain.artifacts.identity import ArtifactIdentity
from apron.domain.canonical import canonicalize, digest_hex
from apron.domain.capabilities import CapabilitySignature
from apron.domain.schemas.authority import DecisionRequest
from apron.domain.schemas.models import ArtifactSpec, ExecutionSpec
from apron.domain.schemas.primitives import ArtifactLocator
from apron.domain.schemas.records import TaskAttemptRecord, VerificationReport
from apron.domain.schemas.tasks import ServingWorkloadSpec, TaskSuiteSpec

if TYPE_CHECKING:
    from pathlib import Path

_EXEC_SPEC = ExecutionSpec(
    engine_image_digest="sha256:abc123",
    resolved_checkpoint_method="safetensors",
)
_SOL = "1220" + "5a" * 32
_OTHER = "1220" + "6b" * 32
_FP = "1220" + "ab" * 32

_SUITE = TaskSuiteSpec(
    name="s",
    version="1",
    required_capabilities=(
        CapabilitySignature(
            operation="text_generation",
            required_inputs=("text",),
            output_representation="generated_text",
        ),
    ),
    cases=tuple({"id": f"c{i}", "prompt": "p", "expected": "e"} for i in range(5)),
)
_WORKLOAD = ServingWorkloadSpec(p99_ttft_ms=2000, p99_tpot_ms=100)
_ACCEPTED = AcceptedRequest(
    request=DecisionRequest(objective="o", quality_floor=0.6),
    task_suite=_SUITE,
    serving_workload=_WORKLOAD,
)


class _FixedClock:
    def now(self):  # type: ignore[no-untyped-def]
        from datetime import UTC, datetime

        return datetime(2026, 1, 1, tzinfo=UTC)


def _make_candidate(
    *,
    source: str = "resolved",
    execution_spec: ExecutionSpec | None = _EXEC_SPEC,
) -> ResolvedCandidate:
    identity = ArtifactIdentity(content_digest=digest_hex(canonicalize({"test": "data"})))
    spec = ArtifactSpec(
        identity=identity,
        locators=(ArtifactLocator(source_kind="huggingface", uri="test/model"),),
    )
    return ResolvedCandidate(
        artifact_spec=spec,
        execution_spec=execution_spec,
        source=source,  # type: ignore[arg-type]
    )


def _attempt(case: str, accepted: bool, *, solution: str = _SOL, run: str = "r1") -> dict:
    return TaskAttemptRecord(
        decision_fingerprint=_FP,
        task_suite_fingerprint=_FP,
        application_fingerprint=_FP,
        evaluation_protocol_fingerprint=_FP,
        solution_fingerprint=solution,
        case_id=case,
        attempt_id=f"{run}-{case}",
        accepted=accepted,
        infrastructure_cost=0.01,
        claim_scope="task_outcome",
        production_mode=False,
        reason="evaluation_run",
        lifecycle="observed",
    ).model_dump(mode="json")


def _serving(p99_ttft: float, *, completed: int = 50, failed: int = 0) -> dict:
    return VerificationReport(
        target_kind="rented-provider",
        operator="apron",
        execution_fingerprint=_FP,
        solution_fingerprint=_SOL,
        serving_completed=completed,
        serving_failed=failed,
        serving_latency_ms={"p99_ttft_ms": p99_ttft, "p99_tpot_ms": 40.0},
        market_equivalent_price=0.02,
        claim_scope="serving_performance",
        production_mode=False,
        reason="serving_measurement",
        lifecycle="observed",
    ).model_dump(mode="json")


def _store_all(store: LocalRecordStore, records: list[dict]) -> tuple[str, ...]:
    return tuple(store.store(r) for r in records)


def _graph(tmp_path: Path) -> tuple[QualificationGraph, LocalRecordStore]:
    store = LocalRecordStore(tmp_path)
    return QualificationGraph(clock=_FixedClock(), store=store), store


def _passing_run(store: LocalRecordStore, run: str = "r1") -> tuple[str, ...]:
    return _store_all(store, [_attempt(f"c{i}", i < 4, run=run) for i in range(5)])


def test_full_qualification(tmp_path: Path) -> None:
    graph, store = _graph(tmp_path)
    evidence = EvidenceDigests(
        task_attempts=_passing_run(store),
        serving_report=store.store(_serving(900.0)),
        reproduction_attempts=_passing_run(store, "r2"),
    )
    entry = graph.advance(
        _make_candidate(), _ACCEPTED, solution_fingerprint=_SOL, evidence=evidence
    )
    assert entry.qualification_status == "qualified"
    assert entry.rejection_reason is None
    assert entry.solution_fingerprint == _SOL


def test_caller_cannot_assert_a_verdict(tmp_path: Path) -> None:
    """The graph no longer accepts {"passed": bool}; only record digests."""
    import inspect

    params = inspect.signature(QualificationGraph.advance).parameters
    assert "task_evidence" not in params
    assert "serving_evidence" not in params
    assert "reproduction_evidence" not in params


def test_legacy_candidate_stays_at_candidate(tmp_path: Path) -> None:
    graph, _ = _graph(tmp_path)
    candidate = _make_candidate(source="legacy", execution_spec=None)
    entry = graph.advance(candidate, _ACCEPTED, solution_fingerprint=_SOL)
    assert entry.qualification_status == "candidate"
    assert "legacy" in (entry.rejection_reason or "").lower()


def test_capability_mismatch_pruned(tmp_path: Path) -> None:
    graph, _ = _graph(tmp_path)
    entry = graph.advance(
        _make_candidate(), _ACCEPTED, solution_fingerprint=_SOL, capabilities_match=False
    )
    assert entry.qualification_status == "candidate"
    state = json.loads(entry.qualification_graph_state or "{}")
    assert state["capability_eligible"]["passed"] is False


def test_task_failure_below_quality_floor(tmp_path: Path) -> None:
    graph, store = _graph(tmp_path)
    digests = _store_all(store, [_attempt(f"c{i}", i < 2) for i in range(5)])
    entry = graph.advance(
        _make_candidate(),
        _ACCEPTED,
        solution_fingerprint=_SOL,
        evidence=EvidenceDigests(task_attempts=digests),
    )
    assert entry.qualification_status == "identity_resolved"
    state = json.loads(entry.qualification_graph_state or "{}")
    assert state["task_evaluated"]["passed"] is False
    assert "0.40 below quality floor 0.60" in state["task_evaluated"]["reason"]


def test_missing_digest_fails_with_reason(tmp_path: Path) -> None:
    graph, _ = _graph(tmp_path)
    entry = graph.advance(
        _make_candidate(),
        _ACCEPTED,
        solution_fingerprint=_SOL,
        evidence=EvidenceDigests(task_attempts=("1220" + "99" * 32,)),
    )
    assert entry.qualification_status == "identity_resolved"
    assert "not found" in (entry.rejection_reason or "")


def test_unloadable_record_fails_with_reason(tmp_path: Path) -> None:
    graph, store = _graph(tmp_path)
    bad = {**_attempt("c0", True), "unknown_key": 1}
    digest = store.store(bad)
    entry = graph.advance(
        _make_candidate(),
        _ACCEPTED,
        solution_fingerprint=_SOL,
        evidence=EvidenceDigests(task_attempts=(digest,)),
    )
    assert entry.qualification_status == "identity_resolved"
    assert "not a valid TaskAttemptRecord" in (entry.rejection_reason or "")


def test_attempts_for_another_solution_fail(tmp_path: Path) -> None:
    graph, store = _graph(tmp_path)
    digests = _store_all(store, [_attempt(f"c{i}", True, solution=_OTHER) for i in range(5)])
    entry = graph.advance(
        _make_candidate(),
        _ACCEPTED,
        solution_fingerprint=_SOL,
        evidence=EvidenceDigests(task_attempts=digests),
    )
    assert entry.qualification_status == "identity_resolved"
    assert "different solution" in (entry.rejection_reason or "")


def test_serving_slo_failure(tmp_path: Path) -> None:
    graph, store = _graph(tmp_path)
    evidence = EvidenceDigests(
        task_attempts=_passing_run(store), serving_report=store.store(_serving(2500.0))
    )
    entry = graph.advance(
        _make_candidate(), _ACCEPTED, solution_fingerprint=_SOL, evidence=evidence
    )
    assert entry.qualification_status == "task_evaluated"
    assert "p99_ttft_ms 2500.0 exceeds SLO 2000.0" in (entry.rejection_reason or "")


def test_task_and_serving_verdicts_are_independent(tmp_path: Path) -> None:
    """A passing serving report cannot rescue a failing task run."""
    graph, store = _graph(tmp_path)
    failing = _store_all(store, [_attempt(f"c{i}", False) for i in range(5)])
    evidence = EvidenceDigests(task_attempts=failing, serving_report=store.store(_serving(10.0)))
    entry = graph.advance(
        _make_candidate(), _ACCEPTED, solution_fingerprint=_SOL, evidence=evidence
    )
    assert entry.qualification_status == "identity_resolved"


def test_reproduction_mismatch_fails(tmp_path: Path) -> None:
    graph, store = _graph(tmp_path)
    second = _store_all(store, [_attempt(f"c{i}", i != 0, run="r2") for i in range(5)])
    evidence = EvidenceDigests(
        task_attempts=_passing_run(store),
        serving_report=store.store(_serving(900.0)),
        reproduction_attempts=second,
    )
    entry = graph.advance(
        _make_candidate(), _ACCEPTED, solution_fingerprint=_SOL, evidence=evidence
    )
    assert entry.qualification_status == "serving_verified"
    assert "accepted sets differ" in (entry.rejection_reason or "")


def test_no_task_evidence_stops_at_identity(tmp_path: Path) -> None:
    graph, _ = _graph(tmp_path)
    entry = graph.advance(_make_candidate(), _ACCEPTED, solution_fingerprint=_SOL)
    assert entry.qualification_status == "identity_resolved"


def test_graph_state_records_evidence_digests(tmp_path: Path) -> None:
    graph, store = _graph(tmp_path)
    digests = _passing_run(store)
    entry = graph.advance(
        _make_candidate(),
        _ACCEPTED,
        solution_fingerprint=_SOL,
        evidence=EvidenceDigests(task_attempts=digests),
    )
    state = json.loads(entry.qualification_graph_state or "{}")
    assert state["stopped_at"] == "task_evaluated"
    assert state["task_evaluated"]["evidence"] == list(digests)
