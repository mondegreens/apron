"""Findings (PLAN §12): the article's numbers equal the records they cite.

Over a synthetic run (``_synthetic_run``): every row's numbers are the stored
records' values and every cited digest resolves in the record store.  Over
the committed files: the post's tables and the findings JSON equal what
``scripts/cohort_findings.py`` generates from ``_dev_notes/cohort-run`` —
the test that fails if the post's tables differ from the records.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from unit._synthetic_run import build_synthetic_run

from apron.adapters.backends.local_store import LocalRecordStore
from apron.application.orchestration.findings import (
    build_findings,
    read_blocks,
    render_tables,
    replace_blocks,
)
from apron.application.orchestration.remediation import SIX_CLASSES
from apron.application.sanitization import contains_secret
from apron.interfaces.cohort_root import load_cohort_run

if TYPE_CHECKING:
    from apron.application.orchestration.cohort_records import CohortRun

REPO = Path(__file__).resolve().parents[2]
POST = REPO / "docs" / "blog" / "posts" / "phase-1b-cohort-findings.md"


def _script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "cohort_findings", REPO / "scripts" / "cohort_findings.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, CohortRun]:
    run_dir = tmp_path_factory.mktemp("cohort-run")
    rules_dir = build_synthetic_run(run_dir)
    return run_dir, load_cohort_run(run_dir, rules_dir)


@pytest.fixture(scope="module")
def findings(synthetic: tuple[Path, CohortRun]) -> dict[str, Any]:
    return build_findings(synthetic[1], authorized=100.0)


# ---------------------------------------------------------------------------
# Numbers equal records
# ---------------------------------------------------------------------------


def test_memory_rows_equal_the_reports(synthetic: tuple[Path, CohortRun], findings: dict) -> None:
    rec = synthetic[1].records
    assert len(findings["memory"]) == len(rec.measured())
    for row in findings["memory"]:
        report = rec.reports[row["record"]]
        claim = rec.claims[row["claim"]]
        entry = rec.solutions[report.solution_fingerprint or ""]
        assert (row["model"], row["gpu"]) == (entry.model_id, entry.requested_execution.gpu_sku)
        assert row["measured_weight_bytes"] == report.model_weight_memory
        assert row["measured_persistent_bytes"] == report.persistent_consumption
        predicted = claim.proposed_configuration
        assert row["predicted_weight_bytes"] == predicted["weight_memory_bytes"]
        assert row["predicted_total_bytes"] == predicted["total_required_bytes"]
        delta = report.predicted_minus_measured or {}
        assert row["weight_delta_bytes"] == delta.get("weight_memory")
        assert row["total_delta_bytes"] == delta.get("total")


def test_serving_rows_equal_the_reports(synthetic: tuple[Path, CohortRun], findings: dict) -> None:
    rec = synthetic[1].records
    assert len(findings["serving"]) >= 3
    for row in findings["serving"]:
        report = rec.reports[row["record"]]
        latency = report.serving_latency_ms or {}
        assert row["p99_ttft_ms"] == latency["p99_ttft_ms"]
        assert row["p99_tpot_ms"] == latency["p99_tpot_ms"]
        assert row["verdict"] == report.serving_slo_verdict == row["recomputed_verdict"]
        assert (row["completed"], row["failed"]) == (
            report.serving_completed,
            report.serving_failed,
        )


def test_fix_rows_cover_all_six_classes(synthetic: tuple[Path, CohortRun], findings: dict) -> None:
    rec = synthetic[1].records
    rows = findings["fixes"]
    assert [r["class"] for r in rows] == [c.failure_class for c in SIX_CLASSES]
    for row in rows:
        record = rec.remediations[row["remediation"]]
        assert row["diagnosed"] == record.diagnosed_failure_class == row["family"]
        assert row["mechanism_outcome"] == record.mechanism_outcome == "verified"
        assert row["label"] == "Fixed"
        assert row["proving_boot"] == record.proving_record_fingerprints[0]
        assert row["promoted_rule"] is not None
        assert row["change"], row
    by_class = {r["class"]: r["change"] for r in rows}
    assert by_class[5]["tensor_parallel"] == [3, 2]
    assert by_class[1]["resource_allocation.gpu_sku"][1] == "NVIDIA RTX A6000"
    assert by_class[6]["resource_allocation.model_id"] == [
        "ISTA-DASLab/Qwen3-8B-FPQuant-RTN-MXFP4",
        "Qwen/Qwen3-8B-AWQ",
    ]
    assert rows[5]["fixed_model"] == "Qwen/Qwen3-8B-AWQ"


def test_prediction_errors_are_reported(findings: dict) -> None:
    kinds = {(r["model"], r["status"]) for r in findings["prediction_errors"]}
    assert ("state-spaces/mamba-2.8b-hf", "unknown") in kinds
    assert ("Qwen/Qwen3-32B", "infeasible") in kinds


def test_costs_equal_the_ledger(synthetic: tuple[Path, CohortRun], findings: dict) -> None:
    cost = findings["cost"]
    assert cost["records_total"] == pytest.approx(cost["ledger_settled"], abs=1e-3)
    # spent = per-solution pod time + classifier calls + pooled pods between solutions
    assert cost["ledger_spent"] == pytest.approx(
        cost["ledger_settled"] + cost["ledger_classifier"] + cost["ledger_pod_idle"]
    )
    assert cost["ledger_pod_idle"] > 0  # the synthetic run reuses pods
    assert cost["ledger_classifier"] == pytest.approx(6 * 0.02)
    assert cost["failed_boot_total"] > 0  # the six broken boots are paid for and shown
    rec = synthetic[1].records
    cited = {d for row in cost["per_solution"] for d in row["records"]}
    assert cited == set(rec.reports) | set(rec.attempts)


def test_task_rows_count_every_attempt(synthetic: tuple[Path, CohortRun], findings: dict) -> None:
    rec = synthetic[1].records
    assert sum(r["attempts"] for r in findings["tasks"]) == len(rec.attempts)
    assert all(r["passed"] for r in findings["tasks"])


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

_DIGEST = re.compile(r"`(1220[0-9a-f]{12})`")


def test_every_digest_in_the_tables_resolves(
    synthetic: tuple[Path, CohortRun], findings: dict
) -> None:
    store = LocalRecordStore(synthetic[0] / "records")
    blocks = render_tables(findings)
    cited = {m for text in blocks.values() for m in _DIGEST.findall(text)}
    cited -= {findings["setup"]["task_suite"]["fingerprint"][:16]}  # an object, not a record
    assert len(cited) > 20
    for prefix in cited:
        assert store.retrieve(prefix) is not None, prefix


def test_post_blocks_round_trip(findings: dict) -> None:
    blocks = render_tables(findings)
    text = replace_blocks(POST.read_text("utf-8"), blocks)
    in_post = read_blocks(text)
    assert set(in_post) <= set(blocks)
    assert {"headline", "memory", "fixes", "serving", "cost"} <= set(in_post)
    assert all(in_post[name] == blocks[name] for name in in_post)
    assert replace_blocks(text, blocks) == text


def test_unknown_block_is_an_error() -> None:
    with pytest.raises(KeyError, match="nonsense"):
        replace_blocks("<!-- findings:nonsense -->\nx\n<!-- /findings:nonsense -->", {})


def test_generated_outputs_hold_no_secrets(findings: dict) -> None:
    assert not contains_secret(json.dumps(findings))
    assert not any(contains_secret(text) for text in render_tables(findings).values())


# ---------------------------------------------------------------------------
# The committed post equals the committed records
# ---------------------------------------------------------------------------


def test_committed_findings_equal_the_records() -> None:
    script = _script()
    data, blocks, _ = script.generate(script.RUN_DIR, script.RULES_DIR / "vllm-v0.29")
    assert script.DATA.read_text("utf-8") == data, "run scripts/cohort_findings.py"
    for path in script.DOCUMENTS:
        in_doc = read_blocks(path.read_text("utf-8"))
        assert in_doc, path
        for name, body in in_doc.items():
            assert body == blocks[name], f"{path.name}: block {name} differs from the records"


def test_failed_spend_accounts_for_every_failed_boot(findings: dict) -> None:
    """Every failed boot's cost lands in exactly one cause, and the six broken
    plans are shown as failing on purpose, not as waste."""
    groups = findings["failed_spend"]
    assert sum(g["cost"] for g in groups) == pytest.approx(findings["cost"]["failed_boot_total"])
    assert sum(len(g["records"]) for g in groups) == sum(g["boots"] for g in groups)
    assert any(g["cause"].startswith("a broken plan, failing as its class names") for g in groups)


def test_an_answer_cut_at_the_token_limit_is_counted() -> None:
    from types import SimpleNamespace

    from apron.application.orchestration.findings import _truncated

    limits = {"arith-2": 8}
    cut = SimpleNamespace(case_id="arith-2", output_tokens=8, failures=())
    tagged = SimpleNamespace(
        case_id="arith-2", output_tokens=None, failures=("evaluation:truncated at max_tokens",)
    )
    short = SimpleNamespace(case_id="arith-2", output_tokens=2, failures=())
    assert _truncated(cut, limits) and _truncated(tagged, limits)
    assert not _truncated(short, limits)


# ---------------------------------------------------------------------------
# Modern models (groups A-D): every approved model is a row, run or not
# ---------------------------------------------------------------------------


def test_modern_models_list_every_approved_model(synthetic: tuple[Path, CohortRun]) -> None:
    plan = {
        "models": [
            {
                "group": "A",
                "model_id": "Qwen/Qwen3-32B",
                "params_b": 32.8,
                "gpu": "x",
                "gpu_count": 1,
            },
            {
                "group": "D",
                "model_id": "zai-org/GLM-5.3",
                "params_b": 753.3,
                "gpu": "NVIDIA H200",
                "gpu_count": 8,
            },
        ]
    }
    found = build_findings(synthetic[1], authorized=100.0, modern=plan)
    rows = {r["model"]: r for r in found["modern_models"]}
    measured = rows["Qwen/Qwen3-32B"]
    assert measured["status"] == "booted"
    assert measured["gpu"] == "NVIDIA H100 80GB HBM3"  # what ran, not the proposal
    assert measured["measured_weight_bytes"] and measured["records"]
    waiting = rows["zai-org/GLM-5.3"]
    assert waiting["status"] == "not run yet" and waiting["records"] == []
    assert waiting["gpu"] == "NVIDIA H200" and waiting["gpu_count"] == 8
    table = render_tables(found)["modern_models"]
    assert "not run yet" in table and "GLM-5.3" in table


def test_a_model_run_twice_shows_its_latest_solution(synthetic: tuple[Path, CohortRun]) -> None:
    """The card shows the latest booted solution (its last ``executed``
    event) and that solution's answers, whichever digest sorts first."""
    from dataclasses import replace

    run = synthetic[1]
    model = "Qwen/Qwen3-1.7B"
    plan = {"models": [{"group": "A", "model_id": model, "gpu": "x", "gpu_count": 1}]}
    solutions = sorted(
        {
            r.solution_fingerprint or ""
            for r in run.records.measured().values()
            if run.records.solutions[r.solution_fingerprint or ""].model_id == model
        }
    )
    assert len(solutions) > 1
    for latest in solutions:
        events = [
            *run.events,
            {"event": "executed", "solution_fingerprint": latest, "at": "2099-01-01T00:00:00"},
        ]
        found = build_findings(replace(run, events=events), authorized=100.0, modern=plan)
        card = found["modern_models"][0]
        memory = next(r for r in found["memory"] if r["solution"] == latest)
        tasks = [t for t in found["tasks"] if t["solution"] == latest]
        assert memory["record"] in card["records"]
        assert card["measured_weight_bytes"] == memory["measured_weight_bytes"]
        if tasks:
            assert card["accepted"] == max(tasks, key=lambda t: len(t["scoring"]))["accepted"]


def test_weights_time_is_empty_without_staged_boots(findings: dict) -> None:
    assert findings["weights_time"] == []
    assert "No boot from staged weights yet" in render_tables(findings)["weights_time"]


def test_diagnostic_probe_spend_is_its_own_line(synthetic: tuple[Path, CohortRun]) -> None:
    """A peak-probe boot (scripts/peak_probe.py) is paid for but is no solution:
    it is counted in the total, never in the per-solution settled spend."""
    from dataclasses import replace

    from apron.application.orchestration.findings import PROBE_PREFIX

    run = synthetic[1]
    before = build_findings(run, authorized=100.0)["cost"]
    label = f"{PROBE_PREFIX}zai-org/GLM-4.7-Flash"
    probed = replace(
        run,
        ledger=[
            *run.ledger,
            {"op": "hold", "label": label, "amount": 3.0},
            {"op": "settle", "label": label, "amount": 1.25, "estimate": 3.0},
        ],
    )
    cost = build_findings(probed, authorized=100.0)["cost"]
    assert cost["ledger_probe"] == pytest.approx(1.25)
    assert cost["ledger_settled"] == pytest.approx(before["ledger_settled"])
    assert cost["ledger_spent"] == pytest.approx(before["ledger_spent"] + 1.25)
