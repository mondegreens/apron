"""Phase 1b's evaluation protocol template binds strictly (extra fields fail)."""

from __future__ import annotations

from apron.application.orchestration.evidence import EvidenceContext
from apron.interfaces.cohort_root import load_inputs


def test_phase1b_protocol_binds_and_ignores_a_trailing_full_stop() -> None:
    inputs = load_inputs()
    ctx = EvidenceContext.bind(
        request=inputs.request,
        task_suite=inputs.task_suite,
        application=inputs.application,
        protocol_template=inputs.protocol_template,
        solution_fp="1220" + "ab" * 32,
    )
    assert ctx.protocol.deterministic_checks == (
        "whitespace_normalized_exact_match",
        "strip_terminal_punctuation",
    )
    assert ctx.protocol.repetitions == 1 and ctx.protocol.concurrency == 1
