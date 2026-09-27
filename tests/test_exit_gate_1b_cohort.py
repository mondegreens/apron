"""Exit gate 1b, evidence-cohort part (PLAN §1, §11) — asserted over stored records.

Every check reads a run directory the way the cohort left it:

- ``records/``       content-addressed records (LocalRecordStore);
- ``solutions.jsonl`` the identity manifest (objects behind every digest);
- ``ledger.jsonl``   the budget ledger; ``events.jsonl`` the machine events;
- the rule directory with its ``history/`` (promoted rule versions);
- every other file (logs, ``notebook.md``) for the secret scan.

Each check returns the problems it found; a passing item returns none.

Hermetic (CI): the checks run over a synthetic run — the real orchestrator,
fix-proof protocol, ledger, stores and rule repository driven by the mock
provider and engine in ``tests/unit/_cohort_fakes.py`` — and each check is
shown to fail on a record doctored to break it.

Real records (opt-in, ``@pytest.mark.cohort``): ``APRON_COHORT_GATE=1``
runs the same checks over ``_dev_notes/cohort-run`` and ``rules/vllm-v0.29``.

Item 6 (managed vs self-hosted under a recorded cost boundary, §9.3) is
pending Part 3: the boundary fields do not exist in the schema yet.  Its test
is a strict xfail on exactly that, so it fails the day Part 3 adds them.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from unit._synthetic_run import RULE_VERSION_DIR, build_synthetic_run

from apron.adapters.backends.local_store import LocalRecordStore
from apron.application.orchestration.budget import BudgetTracker
from apron.application.orchestration.evidence import (
    decision_request_digest,
    solution_fingerprint,
)
from apron.application.orchestration.remediation import SIX_CLASSES
from apron.application.orchestration.scheduler import rerank
from apron.application.orchestration.serving import evaluate_serving_slos
from apron.application.sanitization import contains_secret
from apron.domain.fingerprints import fingerprint_hex
from apron.domain.schemas.reports import CandidateEconomics, DecisionReport
from apron.domain.verdicts import task_verdict
from apron.interfaces.cohort_root import load_cohort_run

if TYPE_CHECKING:
    from pydantic import BaseModel

    from apron.application.orchestration.cohort_records import CohortRecords, SolutionEntry
    from apron.domain.schemas.records import DiagnosisRule

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# The run, as the checks see it
# ---------------------------------------------------------------------------


@dataclass
class GateRun:
    records: CohortRecords
    rules: list[DiagnosisRule]  # every version: current and history
    rule_errors: list[str]
    ledger: list[dict[str, Any]]
    events: list[dict[str, Any]]
    scanned: dict[str, str]  # path -> text, for the secret scan
    authorized: float
    ranking: dict[str, Any] | None = None  # cohort-ranking.json (§1 item 7)
    # Later passes (cohort-ranking-<tag>.json), e.g. the re-scoring pass.
    later_rankings: tuple[dict[str, Any], ...] = ()


def load_run(run_dir: Path, rules_dir: Path, authorized: float) -> GateRun:
    loaded = load_cohort_run(run_dir, rules_dir)
    ranking_path = run_dir / "cohort-ranking.json"
    scanned = {
        str(p.relative_to(root)): p.read_text("utf-8", errors="replace")
        for root in (run_dir, rules_dir)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    return GateRun(
        records=loaded.records,
        rules=loaded.rules,
        rule_errors=loaded.rule_errors,
        ledger=loaded.ledger,
        events=loaded.events,
        scanned=scanned,
        authorized=authorized,
        ranking=json.loads(ranking_path.read_text("utf-8")) if ranking_path.exists() else None,
        later_rankings=tuple(
            json.loads(p.read_text("utf-8")) for p in sorted(run_dir.glob("cohort-ranking-*.json"))
        ),
    )


def _measured_entries(run: GateRun) -> tuple[list[SolutionEntry], list[str]]:
    rec, problems = run.records, []
    entries: dict[str, SolutionEntry] = {}
    for digest, report in rec.measured().items():
        entry = rec.solutions.get(report.solution_fingerprint or "")
        if entry is None:
            problems.append(f"measured report {digest[:16]} has no manifest entry")
        else:
            entries[entry.solution_fingerprint] = entry
    return list(entries.values()), problems


# ---------------------------------------------------------------------------
# Item 1 — coverage (§1 items 1-3, §11 item 1)
# ---------------------------------------------------------------------------

REQUIRED_MECHANISMS = {"autoregressive_decode", "mla_decode"}
REQUIRED_SIZES = {"small", "mid", "large"}
REQUIRED_FEATURES = {"gqa", "mla", "moe", "sliding_window"}
REQUIRED_PREDICTION_ERRORS = {"unknown", "infeasible"}


def check_coverage(run: GateRun) -> list[str]:
    """Counted over *measured* solutions (a healthy memory report), not plans."""
    entries, problems = _measured_entries(run)
    models = {e.model_id for e in entries}
    skus = {e.requested_execution.gpu_sku for e in entries}
    mechanisms = {e.mechanism for e in entries}
    if len(models) < 6:
        problems.append(f"{len(models)} measured models < 6: {sorted(models)}")
    if len(skus) < 4:
        problems.append(f"{len(skus)} measured GPU SKUs < 4: {sorted(skus)}")
    if missing := REQUIRED_MECHANISMS - mechanisms:
        problems.append(f"no measured {sorted(missing)} model")
    if not any(e.quantization_method for e in entries):
        problems.append("no measured quantized model (config.json quantization_config)")
    for e in entries:
        if e.coverage.get("quantized") and not e.quantization_method:
            problems.append(f"{e.label}: declared quantized, config.json has no quant_method")
    if not any(e.requested_execution.gpu_count > 1 for e in entries):
        problems.append("no measured multi-GPU execution")

    sizes = {e.coverage.get("size_class") for e in entries}
    if missing := REQUIRED_SIZES - sizes:
        problems.append(f"no measured {sorted(missing)} model")
    hardware = {e.coverage.get("hardware_class") for e in entries}
    if "consumer" not in hardware:
        problems.append("no measured consumer GPU")
    if not hardware & {"professional", "datacenter"}:
        problems.append("no measured professional/datacenter GPU")
    features = {f for e in entries for f in e.coverage.get("features", ())}
    if missing := REQUIRED_FEATURES - features:
        problems.append(f"no measured architecture with {sorted(missing)}")

    statuses = {
        c.proposed_configuration.get("status") for c in run.records.claims.values()
    } & REQUIRED_PREDICTION_ERRORS
    if missing := REQUIRED_PREDICTION_ERRORS - statuses:
        problems.append(f"no prediction-error record of kind {sorted(missing)}")
    return problems


# ---------------------------------------------------------------------------
# Item 2 — all six failure classes proven (§10.2 gates A and B)
# ---------------------------------------------------------------------------


def _request_outcome(run: GateRun, entry: SolutionEntry, proving: tuple[str, ...]) -> str:
    """Recompute satisfied/violated from the stored task and serving records."""
    rec = run.records
    attempts = [rec.attempts[d] for d in proving if d in rec.attempts]
    serving = [
        rec.reports[d]
        for d in proving
        if d in rec.reports and rec.reports[d].claim_scope == "serving_performance"
    ]
    task = task_verdict(
        attempts,
        solution_fingerprint=entry.solution_fingerprint,
        case_ids=[c.get("id", "") for c in entry.task_suite.cases],
        quality_floor=entry.decision_request.quality_floor,
    )
    served = bool(serving) and evaluate_serving_slos(serving[0], entry.serving_workload).passed
    return "satisfied" if task.passed and served else "violated"


def _proof_problems(run: GateRun, case: Any, digest: str) -> list[str]:
    rec = run.records
    r = rec.remediations[digest]
    problems: list[str] = []
    if r.corrects != fingerprint_hex(case.broken_plan):
        problems.append("corrects is not the class's broken plan")
    if r.corrected_plan_digest in ("none", r.corrects):
        problems.append("gate B: no corrected plan, or it equals the broken plan")
    if r.mechanism_outcome != "verified":
        problems.append(f"mechanism_outcome {r.mechanism_outcome}")
        return problems
    boot = (
        rec.reports.get(r.proving_record_fingerprints[0])
        if r.proving_record_fingerprints
        else None
    )
    if boot is None or boot.claim_scope != "memory" or boot.boot_outcome != "healthy":
        problems.append("first proving record is not a stored healthy boot report")
        return problems
    if boot.deployment_plan_digest != r.corrected_plan_digest:
        problems.append("proving boot's plan digest != corrected_plan_digest")
    if boot.solution_fingerprint != r.corrected_solution_digest:
        problems.append("proving boot belongs to another solution")
    entry = rec.solutions.get(r.corrected_solution_digest or "")
    if entry is None:
        problems.append("corrected solution has no manifest entry")
    elif r.request_outcome not in ("satisfied", "violated"):
        problems.append(f"request_outcome {r.request_outcome} not recorded")
    elif r.request_outcome != _request_outcome(run, entry, r.proving_record_fingerprints):
        problems.append("request_outcome differs from the stored task and serving records")
    elif r.request_outcome == "violated":
        # PLAN §10.2 "Honest outcomes": a class ending `violated` is stored
        # and reported, "and it does not satisfy gate item 4".
        problems.append(
            "request_outcome violated: " + "; ".join(r.violated_constraints or ("recorded",))
        )
    boot_digest = r.proving_record_fingerprints[0]
    if not any(
        rule.error_family == case.expected_family
        and rule.status == "mechanism_verified"
        and rule.promoting_verification_fingerprint == boot_digest
        for rule in run.rules
    ):
        problems.append("no promoted rule version cites the proving boot")
    return problems


def check_six_classes(run: GateRun) -> list[str]:
    problems: list[str] = []
    for case in SIX_CLASSES:
        tried = {
            d: _proof_problems(run, case, d)
            for d, r in run.records.remediations.items()
            if r.reason.startswith(f"class {case.failure_class} ")
        }
        if not tried:
            problems.append(f"class {case.failure_class}: no remediation record")
            continue
        if any(not p for p in tried.values()):
            continue
        for digest, found in tried.items():
            r = run.records.remediations[digest]
            if r.diagnosed_failure_class != case.expected_family:
                found.insert(0, f"gate A: diagnosed {r.diagnosed_failure_class}")
            problems.extend(f"class {case.failure_class} {digest[:12]}: {p}" for p in found)
    for digest, r in run.records.remediations.items():
        family = next(
            (
                c.expected_family
                for c in SIX_CLASSES
                if r.reason.startswith(f"class {c.failure_class} ")
            ),
            None,
        )
        if family and r.mechanism_outcome == "verified" and r.diagnosed_failure_class != family:
            problems.append(f"{digest[:12]}: verified but gate A failed")
    return problems


# ---------------------------------------------------------------------------
# Item 3 — fingerprints reproduce from their validated objects
# ---------------------------------------------------------------------------

_PLACEHOLDER = re.compile(r"^1220(..)\1{31}$")


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list | tuple):
        return [s for v in value for s in _strings(v)]
    return []


def _context_problems(entry: SolutionEntry, where: str, **digests: str) -> list[str]:
    expected = {
        "decision": fingerprint_hex(entry.decision_request),
        "task_suite": fingerprint_hex(entry.task_suite),
        "application": fingerprint_hex(entry.application),
        "evaluation": fingerprint_hex(entry.evaluation_protocol),
        "request": decision_request_digest(entry.decision_request),
    }
    return [
        f"{where}: {name} fingerprint does not reproduce"
        for name, value in digests.items()
        if value != expected[name]
    ]


def check_fingerprints(run: GateRun) -> list[str]:
    rec = run.records
    problems: list[str] = []
    everything: dict[str, BaseModel] = {
        **rec.claims,
        **rec.reports,
        **rec.attempts,
        **rec.remediations,
        **rec.decisions,
    }
    for digest, record in everything.items():
        if any(_PLACEHOLDER.match(s) for s in _strings(record.model_dump(mode="json"))):
            problems.append(f"{digest[:12]}: placeholder fingerprint")

    identities: dict[tuple[str, str, str], str] = {}
    for sfp, e in rec.solutions.items():
        if solution_fingerprint(e.model_spec, e.deployment_plan, e.requested_execution) != sfp:
            problems.append(f"manifest {e.label}: solution fingerprint does not reproduce")
        protocol = e.evaluation_protocol
        if protocol.solution_fingerprint != sfp:
            problems.append(f"manifest {e.label}: protocol names another solution")
        if protocol.decision_request_digest != decision_request_digest(e.decision_request):
            problems.append(f"manifest {e.label}: protocol request digest does not reproduce")
        if protocol.task_suite_fingerprint != fingerprint_hex(e.task_suite):
            problems.append(f"manifest {e.label}: protocol task suite does not reproduce")
        if protocol.application_fingerprint != fingerprint_hex(e.application):
            problems.append(f"manifest {e.label}: protocol application does not reproduce")
        triple = (
            fingerprint_hex(e.model_spec),
            fingerprint_hex(e.deployment_plan),
            fingerprint_hex(e.requested_execution),
        )
        if identities.setdefault(triple, sfp) != sfp:
            problems.append(f"manifest {e.label}: one (model, plan, execution), two fingerprints")
    if len(set(identities.values())) != len(identities):
        problems.append("two (model, plan, execution) share a solution fingerprint")

    # A solution may be scored under more than one protocol (a changed scoring
    # rule re-scores it): every protocol the manifest records must bind to its
    # solution, and every attempt must name one of them.
    for fp, protocol in rec.protocols.items():
        entry = rec.solutions.get(protocol.solution_fingerprint)
        if entry is None:
            problems.append(f"protocol {fp[:12]}: names a solution not in the manifest")
            continue
        if protocol.decision_request_digest != decision_request_digest(entry.decision_request):
            problems.append(f"protocol {fp[:12]}: request digest does not reproduce")
        if protocol.task_suite_fingerprint != fingerprint_hex(entry.task_suite):
            problems.append(f"protocol {fp[:12]}: task suite does not reproduce")
        if protocol.application_fingerprint != fingerprint_hex(entry.application):
            problems.append(f"protocol {fp[:12]}: application does not reproduce")
    for digest, a in rec.attempts.items():
        entry = rec.solutions.get(a.solution_fingerprint)
        if entry is None:
            problems.append(f"attempt {digest[:12]}: solution not in the manifest")
            continue
        protocol = rec.protocols.get(a.evaluation_protocol_fingerprint)
        if protocol is None or protocol.solution_fingerprint != a.solution_fingerprint:
            problems.append(f"attempt {digest[:12]}: evaluation protocol not recorded for it")
        problems += _context_problems(
            entry,
            f"attempt {digest[:12]}",
            decision=a.decision_fingerprint,
            task_suite=a.task_suite_fingerprint,
            application=a.application_fingerprint,
        )
    for digest, r in rec.reports.items():
        entry = rec.solutions.get(r.solution_fingerprint or "")
        if entry is None:
            problems.append(f"report {digest[:12]}: solution not in the manifest")
        elif r.deployment_plan_digest != fingerprint_hex(entry.deployment_plan):
            problems.append(f"report {digest[:12]}: plan digest does not reproduce")
    for digest, c in rec.claims.items():
        if (c.solution_fingerprint or "") not in rec.solutions:
            problems.append(f"claim {digest[:12]}: solution not in the manifest")
    claimed = {c.solution_fingerprint for c in rec.claims.values()}
    for digest, r in rec.reports.items():
        if r.predicted_minus_measured and r.solution_fingerprint not in claimed:
            problems.append(f"report {digest[:12]}: prediction delta without its PlanningClaim")

    plans = {fingerprint_hex(e.deployment_plan): e for e in rec.solutions.values()}
    for digest, m in rec.remediations.items():
        bad = plans.get(m.corrects or "")
        if bad is None:
            problems.append(f"remediation {digest[:12]}: corrects no manifest plan")
            continue
        entry = bad
        if m.corrected_solution_digest is not None:
            fixed = rec.solutions.get(m.corrected_solution_digest)
            if fixed is None:
                problems.append(f"remediation {digest[:12]}: corrected solution not in manifest")
                continue
            if m.corrected_plan_digest != fingerprint_hex(fixed.deployment_plan):
                problems.append(f"remediation {digest[:12]}: corrected plan does not reproduce")
            entry = fixed
        problems += _context_problems(
            entry,
            f"remediation {digest[:12]}",
            request=m.accepted_request_digest,
            task_suite=m.task_fingerprint,
            application=m.application_fingerprint,
            evaluation=m.evaluation_fingerprint,
        )
    return problems


# ---------------------------------------------------------------------------
# Item 4 — cost in each schema's own fields, failures and retries included
# ---------------------------------------------------------------------------


def check_costs(run: GateRun) -> list[str]:
    rec = run.records
    problems = [
        f"attempt {d[:12]}: no infrastructure_cost"
        for d, a in rec.attempts.items()
        if a.infrastructure_cost is None or a.infrastructure_cost < 0
    ]
    problems += [
        f"report {d[:12]}: no market_equivalent_price"
        for d, r in rec.reports.items()
        if r.market_equivalent_price is None or r.market_equivalent_price < 0
    ]
    if not any(r.boot_outcome == "failed" for r in rec.reports.values()):
        problems.append("no failed boot on record (failed attempts must be first-class)")
    return problems


# ---------------------------------------------------------------------------
# Item 5 — serving vs SLO, separate from task verdicts
# ---------------------------------------------------------------------------


def check_serving(run: GateRun) -> list[str]:
    rec = run.records
    problems: list[str] = []
    models: set[str] = set()
    for digest, r in rec.reports.items():
        if r.claim_scope != "serving_performance":
            continue
        entry = rec.solutions.get(r.solution_fingerprint or "")
        if entry is None:
            problems.append(f"serving {digest[:12]}: solution not in the manifest")
            continue
        verdict = evaluate_serving_slos(r, entry.serving_workload)
        if r.serving_slo_verdict != verdict.verdict:
            problems.append(
                f"serving {digest[:12]}: stored verdict {r.serving_slo_verdict} "
                f"!= recomputed {verdict.verdict}"
            )
        if r.serving_slo_verdict in ("pass", "fail"):
            models.add(entry.model_id)
        if not rec.attempts_for(entry.solution_fingerprint):
            problems.append(f"serving {digest[:12]}: no separate task attempts for the solution")
    if len(models) < 3:
        problems.append(f"{len(models)} models with a computed SLO verdict < 3")
    return problems


# ---------------------------------------------------------------------------
# Item 6 — managed vs self-hosted under a recorded boundary (pending Part 3)
# ---------------------------------------------------------------------------

BOUNDARY_FIELDS = {"included_cost_components", "serving_horizon", "availability_target"}


def boundary_fields_exist() -> bool:
    fields = set(CandidateEconomics.model_fields) | set(DecisionReport.model_fields)
    return fields >= BOUNDARY_FIELDS


# ---------------------------------------------------------------------------
# Item 7 — every record validates strictly
# ---------------------------------------------------------------------------


def check_strict(run: GateRun) -> list[str]:
    problems = [f"{where}: {why}" for where, why in run.records.invalid]
    problems += run.rule_errors
    if not run.records.solutions:
        problems.append("no identity manifest")
    return problems


# ---------------------------------------------------------------------------
# Item 8 — no secrets anywhere; the F7 environment check is on record
# ---------------------------------------------------------------------------


def check_secrets(run: GateRun) -> list[str]:
    # Name the file only; never echo the matched text.
    problems = [
        f"{path}: matches a secret pattern"
        for path, text in run.scanned.items()
        if contains_secret(text)
    ]
    healthy = [e for e in run.events if e.get("event") == "executed" and e.get("healthy")]
    if not healthy:
        problems.append("no healthy boot, so no F7 environment check on record")
    for e in healthy:
        check = e.get("token_check") or {}
        if not check.get("processes") or check.get("token_free") is not True:
            problems.append(f"{e.get('label')}: F7 check missing or not token-free")
    return problems


# ---------------------------------------------------------------------------
# Item 9 — budget: spent <= authorized; ledger replay equals stored costs
# ---------------------------------------------------------------------------


class _ReadOnlyLedger:
    """Replay without touching the run's ledger (replay settles open holds)."""

    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self._entries = [dict(e) for e in entries]

    def append(self, entry: dict[str, Any]) -> None:
        self._entries.append(dict(entry))

    def read_all(self) -> list[dict[str, Any]]:
        return [dict(e) for e in self._entries]


class _FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 1, 1, tzinfo=UTC)


def check_budget(run: GateRun) -> list[str]:
    problems: list[str] = []
    tracker = BudgetTracker.replay(
        authorized=run.authorized, ledger=_ReadOnlyLedger(run.ledger), clock=_FixedClock()
    )
    if tracker.spent > run.authorized + 1e-9:
        problems.append(f"spent ${tracker.spent:.2f} > authorized ${run.authorized:.2f}")

    # One label can be held again for another solution (class 6 re-run with a
    # replacement plan): pair the n-th settle of a label with its n-th hold event.
    holds_by_label: dict[str, list[str]] = defaultdict(list)
    for e in run.events:
        if e.get("event") == "hold":
            holds_by_label[e["label"]].append(e["solution_fingerprint"])
    seen: dict[str, int] = defaultdict(int)
    settled: dict[str, float] = defaultdict(float)
    for entry in run.ledger:
        if entry["op"] != "settle":
            continue
        label = str(entry["label"])
        index = seen[label]
        seen[label] += 1
        if str(entry.get("flag", "")).startswith("replayed"):
            continue  # a replayed hold is a crashed pod: its records were never written
        if str(entry["label"]).startswith(("classifier:", "pod-idle:")):
            continue  # classifier calls and pooled pods' idle time: ledger-only cost
        fps = holds_by_label.get(label, [])
        if index >= len(fps):
            problems.append(f"settle {label!r} #{index + 1} has no hold event")
            continue
        settled[fps[index]] += float(entry["amount"])

    stored: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for r in run.records.reports.values():
        stored[r.solution_fingerprint or ""] += r.market_equivalent_price or 0.0
        counts[r.solution_fingerprint or ""] += 1
    for a in run.records.attempts.values():
        stored[a.solution_fingerprint] += a.infrastructure_cost or 0.0
        counts[a.solution_fingerprint] += 1
    for sfp in set(settled) | set(stored):
        tolerance = 1e-4 + 1e-6 * counts[sfp]
        if abs(settled[sfp] - stored[sfp]) > tolerance:
            problems.append(
                f"{sfp[:16]}: ledger ${settled[sfp]:.6f} != records ${stored[sfp]:.6f}"
            )
    return problems


# ---------------------------------------------------------------------------
# §1 item 7 — candidates chosen by the scheduler, inside authorization
# ---------------------------------------------------------------------------


def check_selection(run: GateRun) -> list[str]:
    """The recorded ranking re-derives from its inputs; every measured seed
    solution was ranked; every execution sits inside the authorization."""
    record = run.ranking
    if record is None:
        return ["no cohort ranking on record (cohort-ranking.json)"]
    problems: list[str] = []
    auth = record.get("authorization") or {}
    providers = set(auth.get("permitted_providers") or ())
    cloud = (auth.get("hard_target_constraints") or {}).get("cloud_type")
    if not providers or not cloud:
        problems.append("ranking carries no authorization (providers, cloud type)")
    for e in run.records.solutions.values():
        req = e.requested_execution
        if providers and req.provider not in providers:
            problems.append(f"{e.label}: provider {req.provider} outside authorization")
        if cloud and req.cloud_type != cloud:
            problems.append(f"{e.label}: cloud {req.cloud_type} outside authorization")
    chosen: set[str] = set()
    for n, ranking in enumerate((record, *run.later_rankings)):
        recorded = [r["key"] for r in ranking["ranked"]]
        chosen.update(recorded)
        if [r.seed.key for r in rerank(ranking).ranked] != recorded:
            where = "recorded ranking" if n == 0 else f"later ranking {n}"
            problems.append(f"{where} is not what the scheduler derives from its inputs")
        if n and not (ranking.get("authorization") or {}).get("permitted_providers"):
            problems.append(f"later ranking {n} carries no authorization")
    entries, _ = _measured_entries(run)
    for e in entries:
        key = f"{e.model_id}@{e.requested_execution.gpu_sku}x{e.requested_execution.gpu_count}"
        if e.coverage and key not in chosen:
            problems.append(f"{key}: measured but never chosen by the scheduler")
    return problems


def test_scheduler_names_no_model_provider_or_gpu() -> None:
    """No name is privileged or forbidden (§1 item 7): ranking reads seed data only."""
    source = (REPO / "src/apron/application/orchestration/scheduler.py").read_text()
    names = (
        r"Qwen|[Ll]lama|[Mm]istral|[Gg]emma|[Dd]eep[Ss]eek|[Mm]amba"
        r"|[Rr]un[Pp]od|NVIDIA|RTX|H100|A100|B200"
    )
    assert not re.findall(names, source)


CHECKS = {
    "1-coverage": check_coverage,
    "2-six-classes": check_six_classes,
    "3-fingerprints": check_fingerprints,
    "4-costs": check_costs,
    "5-serving": check_serving,
    "7a-selection": check_selection,
    "7-strict": check_strict,
    "8-secrets": check_secrets,
    "9-budget": check_budget,
}


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    run_dir = tmp_path_factory.mktemp("cohort-run")
    return run_dir, build_synthetic_run(run_dir)


@pytest.fixture
def run(synthetic: tuple[Path, Path]) -> GateRun:
    run_dir, rules_dir = synthetic
    return load_run(run_dir, rules_dir, authorized=100.0)


# ---------------------------------------------------------------------------
# Hermetic: every check passes on the synthetic run ...
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("item", sorted(CHECKS))
def test_synthetic_run_passes(run: GateRun, item: str) -> None:
    assert CHECKS[item](run) == []


def test_synthetic_run_is_not_trivial(run: GateRun) -> None:
    rec = run.records
    assert len(rec.remediations) == 6
    assert len(rec.measured()) >= 12  # 7 cohort boots + 6 fixed boots, minus shared ones
    assert {c.proposed_configuration.get("status") for c in rec.claims.values()} >= {
        "unknown",
        "infeasible",
    }


# ---------------------------------------------------------------------------
# ... and fails on a record doctored to break it
# ---------------------------------------------------------------------------


def _with(run: GateRun, **records: Any) -> GateRun:
    return replace(run, records=replace(run.records, **records))


def _drop(bucket: dict[str, Any], predicate: Any) -> dict[str, Any]:
    return {d: r for d, r in bucket.items() if not predicate(r)}


def _entry_of(run: GateRun, model_id: str) -> set[str]:
    return {s for s, e in run.records.solutions.items() if e.model_id == model_id}


def test_coverage_needs_mla(run: GateRun) -> None:
    mla = _entry_of(run, "deepseek-ai/DeepSeek-V2-Lite")
    doctored = _with(
        run, reports=_drop(run.records.reports, lambda r: r.solution_fingerprint in mla)
    )
    assert any("mla_decode" in p for p in check_coverage(doctored))


def test_coverage_needs_both_prediction_errors(run: GateRun) -> None:
    doctored = _with(
        run,
        claims=_drop(
            run.records.claims, lambda c: c.proposed_configuration.get("status") == "unknown"
        ),
    )
    assert any("unknown" in p for p in check_coverage(doctored))


def test_coverage_needs_multi_gpu(run: GateRun) -> None:
    multi = {s for s, e in run.records.solutions.items() if e.requested_execution.gpu_count > 1}
    doctored = _with(
        run, reports=_drop(run.records.reports, lambda r: r.solution_fingerprint in multi)
    )
    assert "no measured multi-GPU execution" in check_coverage(doctored)


def test_six_classes_need_a_promoted_rule(run: GateRun) -> None:
    doctored = replace(run, rules=[r for r in run.rules if r.error_family != "tp_divisibility"])
    assert any(
        p.startswith("class 5") and "promoted rule" in p for p in check_six_classes(doctored)
    )


def test_six_classes_need_a_verified_mechanism(run: GateRun) -> None:
    rems = {
        d: (
            r.model_copy(update={"mechanism_outcome": "failed"})
            if r.reason.startswith("class 3 ")
            else r
        )
        for d, r in run.records.remediations.items()
    }
    assert any(p.startswith("class 3") for p in check_six_classes(_with(run, remediations=rems)))


def test_six_classes_recompute_the_request_outcome(run: GateRun) -> None:
    rems = {
        d: (
            r.model_copy(update={"request_outcome": "violated"})
            if r.reason.startswith("class 2 ")
            else r
        )
        for d, r in run.records.remediations.items()
    }
    problems = check_six_classes(_with(run, remediations=rems))
    assert any(p.startswith("class 2") and "request_outcome" in p for p in problems)


def test_six_classes_gate_a(run: GateRun) -> None:
    rems = {
        d: (
            r.model_copy(update={"diagnosed_failure_class": "unknown"})
            if r.reason.startswith("class 4 ")
            else r
        )
        for d, r in run.records.remediations.items()
    }
    assert any("gate A" in p for p in check_six_classes(_with(run, remediations=rems)))


def test_fingerprints_reject_placeholders(run: GateRun) -> None:
    digest, attempt = next(iter(run.records.attempts.items()))
    doctored = {
        **run.records.attempts,
        digest: attempt.model_copy(update={"decision_fingerprint": "1220" + "00" * 32}),
    }
    problems = check_fingerprints(_with(run, attempts=doctored))
    assert any("placeholder" in p for p in problems)
    assert any("decision fingerprint does not reproduce" in p for p in problems)


def test_an_attempt_must_name_a_recorded_protocol(run: GateRun) -> None:
    digest, attempt = next(iter(run.records.attempts.items()))
    doctored = {
        **run.records.attempts,
        digest: attempt.model_copy(update={"evaluation_protocol_fingerprint": "1220" + "ab" * 32}),
    }
    problems = check_fingerprints(_with(run, attempts=doctored))
    assert any("evaluation protocol not recorded for it" in p for p in problems)


def test_fingerprints_need_the_manifest(run: GateRun) -> None:
    sfp = next(iter(_entry_of(run, "google/gemma-2-2b-it")))
    solutions = {s: e for s, e in run.records.solutions.items() if s != sfp}
    assert any(
        "not in the manifest" in p for p in check_fingerprints(_with(run, solutions=solutions))
    )


def test_fingerprints_catch_a_tampered_plan(run: GateRun) -> None:
    sfp, entry = next(iter(run.records.solutions.items()))
    tampered = entry.model_copy(
        update={"deployment_plan": entry.deployment_plan.model_copy(update={"dtype": "float32"})}
    )
    problems = check_fingerprints(_with(run, solutions={**run.records.solutions, sfp: tampered}))
    assert any("solution fingerprint does not reproduce" in p for p in problems)


def test_fingerprints_need_the_prediction_on_record(run: GateRun) -> None:
    doctored = _with(run, claims={})
    assert any("without its PlanningClaim" in p for p in check_fingerprints(doctored))


def test_costs_required_on_every_record(run: GateRun) -> None:
    digest, attempt = next(iter(run.records.attempts.items()))
    doctored = {
        **run.records.attempts,
        digest: attempt.model_copy(update={"infrastructure_cost": None}),
    }
    assert any("infrastructure_cost" in p for p in check_costs(_with(run, attempts=doctored)))
    reports = _drop(run.records.reports, lambda r: r.boot_outcome == "failed")
    assert any("failed boot" in p for p in check_costs(_with(run, reports=reports)))


def test_serving_verdict_is_recomputed(run: GateRun) -> None:
    digest, report = next(
        (d, r) for d, r in run.records.reports.items() if r.claim_scope == "serving_performance"
    )
    flipped = "fail" if report.serving_slo_verdict == "pass" else "pass"
    doctored = {
        **run.records.reports,
        digest: report.model_copy(update={"serving_slo_verdict": flipped}),
    }
    assert any("recomputed" in p for p in check_serving(_with(run, reports=doctored)))


def test_serving_needs_three_models(run: GateRun) -> None:
    keep = {next(iter(_entry_of(run, "Qwen/Qwen3-1.7B")))}
    reports = _drop(
        run.records.reports,
        lambda r: r.claim_scope == "serving_performance" and r.solution_fingerprint not in keep,
    )
    assert any("< 3" in p for p in check_serving(_with(run, reports=reports)))


def test_strict_catches_an_unknown_key_on_disk(
    synthetic: tuple[Path, Path], tmp_path: Path
) -> None:
    run_dir, rules_dir = synthetic
    copy = tmp_path / "run"
    shutil.copytree(run_dir, copy)
    store = LocalRecordStore(copy / "records")
    digest = next(iter(store.search("")))
    raw = store.retrieve(digest) or {}
    raw["silently_dropped"] = True
    new_digest = store.store(raw)
    problems = check_strict(load_run(copy, rules_dir, 100.0))
    assert any(new_digest in p and "silently_dropped" in p for p in problems)


def test_strict_catches_a_tampered_file(synthetic: tuple[Path, Path], tmp_path: Path) -> None:
    run_dir, rules_dir = synthetic
    copy = tmp_path / "run"
    shutil.copytree(run_dir, copy)
    path = next((copy / "records").rglob("*.json"))
    raw = json.loads(path.read_text())
    raw["reason"] = "edited after the fact"
    path.write_text(json.dumps(raw))
    problems = check_strict(load_run(copy, rules_dir, 100.0))
    assert any("does not hash to its storage key" in p for p in problems)


def test_secrets_scan_names_the_file_only(run: GateRun) -> None:
    key = "sk-ant-" + "a1B2" * 10
    doctored = replace(run, scanned={**run.scanned, "notebook.md": f"oops {key}"})
    problems = check_secrets(doctored)
    assert problems == ["notebook.md: matches a secret pattern"]


def test_secrets_need_the_f7_check(run: GateRun) -> None:
    events = [
        {**e, "token_check": None} if e.get("event") == "executed" and e.get("healthy") else e
        for e in run.events
    ]
    assert any("F7" in p for p in check_secrets(replace(run, events=events)))


def test_budget_ledger_must_equal_records(run: GateRun) -> None:
    ledger = [dict(e) for e in run.ledger]
    settle = next(e for e in ledger if e["op"] == "settle")
    settle["amount"] = float(settle["amount"]) + 0.5
    assert any("ledger $" in p for p in check_budget(replace(run, ledger=ledger)))


def test_budget_spent_within_authorized(run: GateRun) -> None:
    assert any("authorized" in p for p in check_budget(replace(run, authorized=0.01)))


def test_selection_needs_the_ranking(run: GateRun) -> None:
    assert check_selection(replace(run, ranking=None)) == [
        "no cohort ranking on record (cohort-ranking.json)"
    ]


def test_selection_rederives_the_order(run: GateRun) -> None:
    assert run.ranking is not None
    swapped = json.loads(json.dumps(run.ranking))
    swapped["ranked"][0], swapped["ranked"][1] = swapped["ranked"][1], swapped["ranked"][0]
    assert any("derives" in p for p in check_selection(replace(run, ranking=swapped)))


def test_selection_flags_an_unranked_measurement(run: GateRun) -> None:
    assert run.ranking is not None
    trimmed = json.loads(json.dumps(run.ranking))
    trimmed["ranked"] = [r for r in trimmed["ranked"] if not r["key"].startswith("mistralai/")]
    assert any("never chosen" in p for p in check_selection(replace(run, ranking=trimmed)))


def test_selection_flags_an_execution_outside_authorization(run: GateRun) -> None:
    sfp, entry = next(iter(run.records.solutions.items()))
    outside = entry.model_copy(
        update={
            "requested_execution": entry.requested_execution.model_copy(
                update={"cloud_type": "COMMUNITY"}
            )
        }
    )
    doctored = _with(run, solutions={**run.records.solutions, sfp: outside})
    assert any("outside authorization" in p for p in check_selection(doctored))


# ---------------------------------------------------------------------------
# Item 6 — pending Part 3
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="pending Part 3 (PLAN §9.3): the cost-boundary fields do not exist yet; "
    "when they land this XPASSes and item 6's check must be written",
)
def test_item_6_managed_vs_self_hosted_boundary() -> None:
    assert boundary_fields_exist()


# ---------------------------------------------------------------------------
# Real records (opt-in)
# ---------------------------------------------------------------------------

REAL_RUN = REPO / "_dev_notes" / "cohort-run"


@pytest.mark.cohort
@pytest.mark.skipif(
    os.environ.get("APRON_COHORT_GATE") != "1",
    reason="set APRON_COHORT_GATE=1 to assert the exit gate over the real cohort records",
)
@pytest.mark.parametrize("item", sorted(CHECKS))
def test_real_cohort_records(item: str) -> None:
    from apron.interfaces.cohort_root import AUTHORIZED_USD

    run = load_run(REAL_RUN, RULE_VERSION_DIR, AUTHORIZED_USD)
    assert CHECKS[item](run) == []
