"""README's measured-records table equals the records (the repository's living numbers).

Over the committed files: the section between ``<!-- records:start -->`` and
``<!-- records:end -->`` in README.md equals what ``scripts/cohort_findings.py``
generates from ``_dev_notes/cohort-run``, and every record it links exists.
Over a synthetic run: each row's numbers are its records' values, every
boot lands in exactly one row or in the broken-on-purpose count, and a
record without a solution is listed as not rendered, never dropped.
"""

from __future__ import annotations

import importlib.util
import re
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from unit._synthetic_run import build_synthetic_run

from apron.application.orchestration.findings import (
    README_END,
    README_START,
    build_findings,
    engine_version,
    read_readme,
    render_readme,
    replace_readme,
    root_error,
)
from apron.interfaces.cohort_root import load_cohort_run

if TYPE_CHECKING:
    from apron.application.orchestration.cohort_records import CohortRun

REPO = Path(__file__).resolve().parents[2]
README = REPO / "README.md"
SYNTHETIC_IMAGE = "sha256:" + "c" * 64
ENGINES = {SYNTHETIC_IMAGE: "v9.9.9"}
_LINK = re.compile(r"\]\(([^)]+\.json)\)")


def _script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "cohort_findings", REPO / "scripts" / "cohort_findings.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> CohortRun:
    run_dir = tmp_path_factory.mktemp("cohort-run")
    return load_cohort_run(run_dir, build_synthetic_run(run_dir))


# ---------------------------------------------------------------------------
# The committed README equals the committed records
# ---------------------------------------------------------------------------


def test_readme_section_equals_the_records() -> None:
    script = _script()
    _, _, section = script.generate(script.RUN_DIR, script.RULES_DIR / "vllm-v0.29")
    assert read_readme(README.read_text("utf-8")) == section, "run scripts/cohort_findings.py"


def test_every_record_the_readme_links_exists() -> None:
    section = read_readme(README.read_text("utf-8"))
    assert section
    links = _LINK.findall(section)
    assert links
    for href in links:
        path = REPO / href
        assert path.is_file(), href
        assert path.parent.name == "verification-reports", href


# ---------------------------------------------------------------------------
# Rows equal records (synthetic run)
# ---------------------------------------------------------------------------


def test_rows_equal_the_reports(synthetic: CohortRun) -> None:
    rec = synthetic.records
    table = build_findings(synthetic, authorized=100.0, engines=ENGINES)["records_table"]
    assert table["rows"]
    for row in table["rows"]:
        report = rec.reports[row["record"]]
        entry = rec.solutions[report.solution_fingerprint or ""]
        assert (row["model"], row["gpu"], row["gpu_count"]) == (
            entry.model_id,
            entry.requested_execution.gpu_sku,
            entry.requested_execution.gpu_count,
        )
        assert row["vllm"] == "v9.9.9"
        assert row["boot"] == "booted"
        assert report.claim_scope == "memory" and report.boot_outcome == "healthy"
        assert row["measured_weight_bytes"] == report.model_weight_memory
        assert row["measured_peak_bytes"] == report.transient_peak_headroom
        assert row["kv_cache_tokens"] == report.kv_cache_tokens
        claim = next(
            c for c in rec.claims.values() if c.solution_fingerprint == entry.solution_fingerprint
        )
        assert row["predicted_weight_bytes"] == claim.proposed_configuration["weight_memory_bytes"]
        assert row["predicted_peak_bytes"] == claim.proposed_configuration.get(
            "activation_estimate_bytes"
        )
        cost = sum(
            (rec.reports[d].market_equivalent_price or 0.0)
            if d in rec.reports
            else (rec.attempts[d].infrastructure_cost or 0.0)
            for d in row["records"]
        )
        assert row["cost"] == pytest.approx(cost)
        assert row["p99_ttft_ms"] is not None and row["slo_verdict"] == "pass"
        assert row["cases"] and row["tasks_passed"]


def test_every_boot_lands_in_one_row_or_the_broken_count(synthetic: CohortRun) -> None:
    rec = synthetic.records
    table = build_findings(synthetic, authorized=100.0, engines=ENGINES)["records_table"]
    in_rows = [d for row in table["rows"] for d in row["records"] if d in rec.reports]
    assert len(in_rows) == len(set(in_rows))
    broken = table["broken_on_purpose"]["records"]
    assert broken and table["broken_on_purpose"]["boots"] == len(broken) == 6
    assert set(in_rows) | set(broken) == set(rec.reports)
    in_rows_attempts = {d for row in table["rows"] for d in row["records"] if d in rec.attempts}
    assert in_rows_attempts == set(rec.attempts)
    assert table["not_rendered"] == []


def test_a_failed_boot_is_a_row_with_its_error(synthetic: CohortRun) -> None:
    rec = synthetic.records
    digest, healthy = next(iter(rec.measured().items()))
    entry = rec.solutions[healthy.solution_fingerprint or ""]
    sfp = "1220" + "f" * 64
    failing = entry.model_copy(update={"model_id": "acme/Failing-7B", "solution_fingerprint": sfp})
    log = (
        "(EngineCore pid=7) ValueError: max_num_seqs (1024) exceeds available cache blocks\n"
        "(APIServer pid=3) RuntimeError: Engine core initialization failed. "
        "See root cause above. Failed core proc(s): {}\n"
    )
    failed = healthy.model_copy(
        update={
            "solution_fingerprint": sfp,
            "deployment_plan_digest": "1220" + "d" * 64,  # no fix's plan
            "boot_outcome": "failed",
            "claim_scope": "boot",
            "failures": ("boot:model_failure",),
            "log_tail": log,
        }
    )
    records = replace(
        rec,
        reports={**rec.reports, "1220" + "e" * 64: failed},
        solutions={**rec.solutions, sfp: failing},
    )
    table = build_findings(replace(synthetic, records=records), authorized=100.0)["records_table"]
    row = next(r for r in table["rows"] if r["model"] == "acme/Failing-7B")
    assert row["boot"] == "failed" and row["failure"] == "the model"
    assert row["error"] == "ValueError: max_num_seqs (1024) exceeds available cache blocks"
    assert row["measured_weight_bytes"] is None and row["kv_cache_tokens"] is None
    assert row["vllm"] == engine_version(entry.requested_execution.image_digest, None)
    text = render_readme({"records_table": table}, reports_href="records", fixes_href="post.md")
    assert "failed: model error" in text
    assert "`ValueError: max_num_seqs (1024) exceeds available cache blocks`" in text
    assert digest  # the healthy boot it was copied from is still its own row


def test_a_record_without_a_solution_is_listed_not_dropped(synthetic: CohortRun) -> None:
    rec = synthetic.records
    _, healthy = next(iter(rec.measured().items()))
    orphan = healthy.model_copy(update={"solution_fingerprint": "1220" + "a" * 64})
    orphan_digest = "1220" + "b" * 64
    records = replace(rec, reports={**rec.reports, orphan_digest: orphan})
    table = build_findings(replace(synthetic, records=records), authorized=100.0)["records_table"]
    assert table["not_rendered"] == [orphan_digest]
    text = render_readme({"records_table": table}, reports_href="r", fixes_href="p.md")
    assert f"`{orphan_digest[:16]}`" in text and "Not rendered" in text


def test_planned_models_are_labelled_predicted(synthetic: CohortRun) -> None:
    plan = {
        "models": [
            {"group": "A", "model_id": "Qwen/Qwen3-32B", "gpu": "x", "gpu_count": 1},
            {
                "group": "D",
                "model_id": "zai-org/GLM-5.3",
                "gpu": "NVIDIA H200",
                "gpu_count": 8,
                "engine": "v0.30.0",
                "prediction": {
                    "status": "planned",
                    "tensor_parallel": 8,
                    "weight_gib_per_gpu": 88.2,
                    "total_gib_per_gpu": 94.15,
                },
            },
        ]
    }
    table = build_findings(synthetic, authorized=100.0, modern=plan)["records_table"]
    assert [p["model"] for p in table["planned"]] == ["zai-org/GLM-5.3"]
    text = render_readme({"records_table": table}, reports_href="r", fixes_href="p.md")
    assert "Planned, not run yet (predicted before any GPU; nothing measured)" in text
    assert "zai-org/GLM-5.3 on H200 x8, vLLM v0.30.0: predicted 88.2 GiB weights" in text


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------


def test_unknown_image_digest_is_shown_not_guessed() -> None:
    assert engine_version(SYNTHETIC_IMAGE, ENGINES) == "v9.9.9"
    assert engine_version("sha256:" + "d" * 64, ENGINES) == "image sha256:dddddddddddd"


def test_root_error_skips_wrappers_and_shortens() -> None:
    assert root_error(None) is None
    assert root_error("(APIServer pid=1) RuntimeError: Engine core initialization failed.") is None
    long = "(EngineCore pid=2) ValueError: " + "x" * 300
    line = root_error(long)
    assert line and line.startswith("ValueError: x") and line.endswith("…")
    assert len(line) <= 110


def test_readme_markers_round_trip_and_are_required() -> None:
    text = f"# t\n\n{README_START}\n{README_END}\n\nrest\n"
    once = replace_readme(text, "| a |\n| b |")
    assert read_readme(once) == "| a |\n| b |"
    assert replace_readme(once, "| a |\n| b |") == once
    assert once.endswith("\n\nrest\n")
    with pytest.raises(KeyError, match="records:start"):
        replace_readme("# no markers\n", "x")
