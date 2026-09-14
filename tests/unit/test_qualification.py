"""Tests for the qualification graph."""

import json

from apron.application.orchestration.candidates import ResolvedCandidate
from apron.application.orchestration.qualification import QualificationGraph
from apron.domain.artifacts.identity import ArtifactIdentity
from apron.domain.canonical import canonicalize, digest_hex
from apron.domain.schemas.models import ArtifactSpec, ExecutionSpec
from apron.domain.schemas.primitives import ArtifactLocator

_EXEC_SPEC = ExecutionSpec(
    engine_image_digest="sha256:abc123",
    resolved_checkpoint_method="safetensors",
)


class _FixedClock:
    def now(self):
        from datetime import UTC, datetime

        return datetime(2026, 1, 1, tzinfo=UTC)


def _make_candidate(
    *,
    source: str = "resolved",
    execution_spec: ExecutionSpec | None = _EXEC_SPEC,
):
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


def test_full_qualification():
    graph = QualificationGraph(clock=_FixedClock())
    candidate = _make_candidate()
    entry = graph.advance(
        candidate,
        {},
        task_evidence={"passed": True, "scores": [1, 1, 1]},
        serving_evidence={"passed": True},
        reproduction_evidence={"passed": True},
    )
    assert entry.qualification_status == "qualified"
    assert entry.rejection_reason is None


def test_legacy_candidate_stays_at_candidate():
    graph = QualificationGraph(clock=_FixedClock())
    candidate = _make_candidate(source="legacy", execution_spec=None)
    entry = graph.advance(candidate, {})
    assert entry.qualification_status == "candidate"
    assert "legacy" in (entry.rejection_reason or "").lower()


def test_capability_mismatch_pruned():
    graph = QualificationGraph(clock=_FixedClock())
    candidate = _make_candidate()
    entry = graph.advance(candidate, {}, capabilities_match=False)
    assert entry.qualification_status == "candidate"
    state = json.loads(entry.qualification_graph_state or "{}")
    assert state["capability_eligible"]["passed"] is False


def test_task_failure_stops_at_identity_resolved():
    graph = QualificationGraph(clock=_FixedClock())
    candidate = _make_candidate()
    entry = graph.advance(
        candidate,
        {},
        task_evidence={"passed": False, "reason": "3/5 cases failed", "scores": [1, 0, 1, 0, 0]},
    )
    assert entry.qualification_status == "identity_resolved"
    state = json.loads(entry.qualification_graph_state or "{}")
    assert state["task_evaluated"]["passed"] is False


def test_serving_slo_failure():
    graph = QualificationGraph(clock=_FixedClock())
    candidate = _make_candidate()
    entry = graph.advance(
        candidate,
        {},
        task_evidence={"passed": True},
        serving_evidence={"passed": False, "reason": "P99 > 200ms"},
    )
    assert entry.qualification_status == "task_evaluated"


def test_no_task_evidence_stops_at_identity():
    graph = QualificationGraph(clock=_FixedClock())
    candidate = _make_candidate()
    entry = graph.advance(candidate, {})
    assert entry.qualification_status == "identity_resolved"


def test_graph_state_records_reasons():
    graph = QualificationGraph(clock=_FixedClock())
    candidate = _make_candidate()
    entry = graph.advance(
        candidate,
        {},
        task_evidence={"passed": False, "reason": "timeout", "scores": []},
    )
    state = json.loads(entry.qualification_graph_state or "{}")
    assert "stopped_at" in state
    assert state["task_evaluated"]["reason"] == "timeout"
