"""Cohort findings — the article's numbers, generated from stored records (PLAN §12).

``build_findings`` turns a run (records, identity manifest, ledger, rule
versions) into plain data; every row carries the digests of the records its
numbers came from.  ``render_tables`` turns that data into Markdown tables,
and ``replace_blocks`` writes them between markers in a post:

    <!-- findings:memory -->
    ...generated table...
    <!-- /findings:memory -->

Nothing here is typed by hand, so the post can be checked against the
records (the findings test and ``scripts/cohort_findings.py --check``).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from apron.application.orchestration.billing import RECONCILE_PREFIX
from apron.application.orchestration.cohort import classify_harness_error
from apron.application.orchestration.remediation import SIX_CLASSES, failed_as_named
from apron.application.orchestration.serving import evaluate_serving_slos
from apron.domain.fingerprints import fingerprint_hex
from apron.domain.schemas.records import derive_remediation_result
from apron.domain.verdicts import task_verdict

if TYPE_CHECKING:
    from collections.abc import Mapping

    from apron.application.orchestration.cohort_records import CohortRun, SolutionEntry
    from apron.domain.schemas.records import VerificationReport

GIB = 1 << 30
SHORT = 16  # digest prefix shown in tables; the record store resolves prefixes


def build_findings(run: CohortRun, *, authorized: float) -> dict[str, Any]:
    rec = run.records
    return {
        "setup": _setup(run, authorized),
        "memory": _memory(run),
        "repeat_measurements": _repeats(run),
        "calculator_recheck": list((run.recheck or {}).get("rows", [])),
        "fixes": _fixes(run),
        "serving": _serving(run),
        "tasks": _tasks(run),
        "prediction_errors": _prediction_errors(run),
        "cost": _cost(run, authorized),
        "failed_spend": _failed_spend(run),
        "record_count": len(rec.claims)
        + len(rec.reports)
        + len(rec.attempts)
        + len(rec.remediations),
    }


# ---------------------------------------------------------------------------


def _entry(run: CohortRun, sfp: str | None) -> SolutionEntry | None:
    return run.records.solutions.get(sfp or "")


def _setup(run: CohortRun, authorized: float) -> dict[str, Any]:
    entries = list(run.records.solutions.values())
    measured = [_entry(run, r.solution_fingerprint) for r in run.records.measured().values()]
    at = sorted(e["at"] for e in run.events if "at" in e)
    first = entries[0] if entries else None
    workload = first.serving_workload if first else None
    return {
        "providers": sorted({e.requested_execution.provider for e in entries}),
        "cloud_types": sorted({e.requested_execution.cloud_type for e in entries}),
        "image_digests": sorted({e.requested_execution.image_digest for e in entries}),
        "models_measured": sorted({e.model_id for e in measured if e}),
        "gpus_measured": sorted({e.requested_execution.gpu_sku for e in measured if e}),
        "first_event": at[0] if at else None,
        "last_event": at[-1] if at else None,
        "task_suite": {
            "fingerprint": fingerprint_hex(first.task_suite),
            "cases": len(first.task_suite.cases),
        }
        if first
        else None,
        "serving_workload": {
            "fingerprint": fingerprint_hex(workload),
            "input_sequence_length": workload.input_sequence_length,
            "output_sequence_length": workload.output_sequence_length,
            "concurrency": workload.concurrency,
            "p99_ttft_ms": workload.p99_ttft_ms,
            "p99_tpot_ms": workload.p99_tpot_ms,
        }
        if workload
        else None,
        "authorized": authorized,
    }


def _claim_for(run: CohortRun, sfp: str | None) -> tuple[str | None, dict[str, Any]]:
    for digest, claim in run.records.claims.items():
        if claim.solution_fingerprint == sfp:
            return digest, dict(claim.proposed_configuration)
    return None, {}


def _memory(run: CohortRun) -> list[dict[str, Any]]:
    rows = []
    for digest, r in sorted(run.records.measured().items(), key=lambda kv: _row_key(run, kv[1])):
        entry = _entry(run, r.solution_fingerprint)
        claim_digest, predicted = _claim_for(run, r.solution_fingerprint)
        delta = r.predicted_minus_measured or {}
        rows.append(
            {
                "model": entry.model_id if entry else None,
                "gpu": entry.requested_execution.gpu_sku if entry else None,
                "gpu_count": entry.requested_execution.gpu_count if entry else None,
                "predicted_weight_bytes": predicted.get("weight_memory_bytes"),
                "measured_weight_bytes": r.model_weight_memory,
                "weight_delta_bytes": delta.get("weight_memory"),
                "predicted_total_bytes": predicted.get("total_required_bytes"),
                "measured_persistent_bytes": r.persistent_consumption,
                "total_delta_bytes": delta.get("total"),
                "kv_consistency_bytes": delta.get("kv_cache_consistency"),
                "notes": list(r.prediction_notes),
                "record": digest,
                "claim": claim_digest,
            }
        )
    return rows


def _row_key(run: CohortRun, r: VerificationReport) -> tuple[str, str, str]:
    entry = _entry(run, r.solution_fingerprint)
    if entry is None:
        return ("", "", r.solution_fingerprint or "")
    return (entry.model_id, entry.requested_execution.gpu_sku, entry.solution_fingerprint)


def _spread(values: list[int | None]) -> float | None:
    """Max minus min over mean, in percent (L0-A3 thresholds)."""
    xs = [v for v in values if v is not None]
    if len(xs) < 2 or not sum(xs):
        return None
    return round((max(xs) - min(xs)) / (sum(xs) / len(xs)) * 100, 3)


def _repeats(run: CohortRun) -> list[dict[str, Any]]:
    """Solutions measured more than once (L0-A3): weight, KV and activation spread."""
    by_solution: dict[str, list[tuple[str, VerificationReport]]] = defaultdict(list)
    for digest, r in run.records.measured().items():
        by_solution[r.solution_fingerprint or ""].append((digest, r))
    rows = []
    for sfp, reports in sorted(by_solution.items()):
        if len(reports) < 2:
            continue
        entry = _entry(run, sfp)
        rows.append(
            {
                "model": entry.model_id if entry else None,
                "gpu": entry.requested_execution.gpu_sku if entry else None,
                "boots": len(reports),
                "weight_spread_pct": _spread([r.model_weight_memory for _, r in reports]),
                "kv_spread_pct": _spread([r.available_kv_cache_memory for _, r in reports]),
                # activation = the profiling run's transient peak above persistent memory
                "activation_spread_pct": _spread([r.transient_peak_headroom for _, r in reports]),
                "records": sorted(d for d, _ in reports),
            }
        )
    return rows


def _plan_change(before: SolutionEntry | None, after: SolutionEntry | None) -> dict[str, Any]:
    if before is None or after is None:
        return {}
    a = before.deployment_plan.model_dump(mode="json", exclude={"serve_command", "docker_compose"})
    b = after.deployment_plan.model_dump(mode="json", exclude={"serve_command", "docker_compose"})
    change: dict[str, Any] = {}
    for key in sorted(set(a) | set(b)):
        if isinstance(a.get(key), dict) or isinstance(b.get(key), dict):
            old, new = a.get(key) or {}, b.get(key) or {}
            for sub in sorted(set(old) | set(new)):
                if old.get(sub) != new.get(sub):
                    change[f"{key}.{sub}"] = [old.get(sub), new.get(sub)]
        elif a.get(key) != b.get(key):
            change[key] = [a.get(key), b.get(key)]
    return change


def _fixes(run: CohortRun) -> list[dict[str, Any]]:
    rec = run.records
    plans = {fingerprint_hex(e.deployment_plan): e for e in rec.solutions.values()}
    rows = []
    for case in SIX_CLASSES:
        found = [
            (d, r)
            for d, r in sorted(rec.remediations.items())
            if r.reason.startswith(f"class {case.failure_class} ")
        ]
        if not found:
            rows.append(
                {
                    "class": case.failure_class,
                    "family": case.expected_family,
                    "call_site": case.call_site,
                    "status": "not run",
                }
            )
            continue
        # The record that went furthest: verified first, then any.
        digest, r = sorted(found, key=lambda dr: dr[1].mechanism_outcome != "verified")[0]
        broken = plans.get(r.corrects or "")
        fixed = _entry(run, r.corrected_solution_digest)
        boot = r.proving_record_fingerprints[0] if r.proving_record_fingerprints else None
        promoted = next(
            (
                fingerprint_hex(rule)
                for rule in run.rules
                if rule.error_family == case.expected_family
                and rule.status == "mechanism_verified"
                and rule.promoting_verification_fingerprint == boot
            ),
            None,
        )
        rows.append(
            {
                "class": case.failure_class,
                "family": case.expected_family,
                "call_site": case.call_site,
                "status": "run",
                "broken_model": broken.model_id if broken else None,
                "broken_gpu": broken.requested_execution.gpu_sku if broken else None,
                "error_line": r.classifier_evidence_span,
                "diagnosed": r.diagnosed_failure_class,
                "classifier_model_id": r.classifier_model_id,
                "strategy": r.correction_strategy,
                "change": _plan_change(broken, fixed),
                "fixed_model": fixed.model_id if fixed else None,
                "fixed_gpu": fixed.requested_execution.gpu_sku if fixed else None,
                "mechanism_outcome": r.mechanism_outcome,
                "request_outcome": r.request_outcome,
                "violated_constraints": list(r.violated_constraints),
                "label": derive_remediation_result(r.mechanism_outcome, r.request_outcome),
                "remediation": digest,
                "proving_boot": boot,
                "promoted_rule": promoted,
            }
        )
    return rows


def _serving(run: CohortRun) -> list[dict[str, Any]]:
    rows = []
    reports = [
        (d, r) for d, r in run.records.reports.items() if r.claim_scope == "serving_performance"
    ]
    for digest, r in sorted(reports, key=lambda kv: _row_key(run, kv[1])):
        entry = _entry(run, r.solution_fingerprint)
        latency = r.serving_latency_ms or {}
        workload = entry.serving_workload if entry else None
        rows.append(
            {
                "model": entry.model_id if entry else None,
                "gpu": entry.requested_execution.gpu_sku if entry else None,
                "p99_ttft_ms": latency.get("p99_ttft_ms"),
                "p99_tpot_ms": latency.get("p99_tpot_ms"),
                "slo_p99_ttft_ms": workload.p99_ttft_ms if workload else None,
                "slo_p99_tpot_ms": workload.p99_tpot_ms if workload else None,
                "completed": r.serving_completed,
                "failed": r.serving_failed,
                "output_token_throughput": r.serving_output_token_throughput,
                "verdict": r.serving_slo_verdict,
                "recomputed_verdict": evaluate_serving_slos(r, workload).verdict
                if workload
                else None,
                "record": digest,
            }
        )
    return rows


def _tasks(run: CohortRun) -> list[dict[str, Any]]:
    """One row per solution and scoring rule: attempts under different
    evaluation protocols are never pooled into one verdict."""
    by_solution: dict[tuple[str, str], list[str]] = defaultdict(list)
    for digest, a in run.records.attempts.items():
        by_solution[(a.solution_fingerprint, a.evaluation_protocol_fingerprint)].append(digest)
    rows = []

    def order(key: tuple[str, str]) -> tuple[Any, ...]:
        protocol = run.records.protocols.get(key[1])
        checks = protocol.deterministic_checks if protocol else ()
        return (*_entry_key(run, key[0]), len(checks), key[1])  # older, shorter rule first

    for (sfp, protocol_fp), digests in sorted(by_solution.items(), key=lambda kv: order(kv[0])):
        entry = _entry(run, sfp)
        protocol = run.records.protocols.get(protocol_fp)
        attempts = [run.records.attempts[d] for d in digests]
        verdict = (
            task_verdict(
                attempts,
                solution_fingerprint=sfp,
                case_ids=[c.get("id", "") for c in entry.task_suite.cases],
                quality_floor=entry.decision_request.quality_floor,
            )
            if entry
            else None
        )
        rows.append(
            {
                "model": entry.model_id if entry else None,
                "gpu": entry.requested_execution.gpu_sku if entry else None,
                "cases": len(entry.task_suite.cases) if entry else None,
                "attempts": len(attempts),
                "retries": sum(1 for a in attempts if a.retries),
                "accepted": len({a.case_id for a in attempts if a.accepted}),
                "passed": verdict.passed if verdict else None,
                "scoring": list(protocol.deterministic_checks) if protocol else None,
                "records": sorted(digests),
            }
        )
    return rows


def _entry_key(run: CohortRun, sfp: str) -> tuple[str, str, str]:
    entry = _entry(run, sfp)
    return (entry.model_id, entry.requested_execution.gpu_sku, sfp) if entry else ("", "", sfp)


def _prediction_errors(run: CohortRun) -> list[dict[str, Any]]:
    rows = []
    for digest, claim in sorted(run.records.claims.items()):
        status = claim.proposed_configuration.get("status")
        if status not in ("unknown", "infeasible"):
            continue
        entry = _entry(run, claim.solution_fingerprint)
        rows.append(
            {
                "model": entry.model_id if entry else None,
                "gpu": entry.requested_execution.gpu_sku if entry else None,
                "status": status,
                "mechanism": entry.mechanism if entry else None,
                "predicted_total_bytes": claim.proposed_configuration.get("total_required_bytes"),
                "record": digest,
            }
        )
    rows.sort(key=lambda row: (row["model"] or "", row["gpu"] or ""))
    return rows


def _cost(run: CohortRun, authorized: float) -> dict[str, Any]:
    rec = run.records
    per: dict[str, dict[str, Any]] = {}
    for digest, r in rec.reports.items():
        row = per.setdefault(r.solution_fingerprint or "", _cost_row(run, r.solution_fingerprint))
        row["cost"] += r.market_equivalent_price or 0.0
        row["records"].append(digest)
        if r.boot_outcome == "failed":
            row["failed_boots"] += 1
            row["failed_boot_cost"] += r.market_equivalent_price or 0.0
    for digest, a in rec.attempts.items():
        row = per.setdefault(a.solution_fingerprint, _cost_row(run, a.solution_fingerprint))
        row["cost"] += a.infrastructure_cost or 0.0
        row["records"].append(digest)
    rows = sorted(per.values(), key=lambda row: (row["model"] or "", row["gpu"] or ""))
    for row in rows:
        row["cost"] = round(row["cost"], 6)
        row["failed_boot_cost"] = round(row["failed_boot_cost"], 6)
        row["records"].sort()

    def is_classifier(e: dict[str, Any]) -> bool:
        return str(e.get("label", "")).startswith("classifier:")

    def is_pod_idle(e: dict[str, Any]) -> bool:
        return str(e.get("label", "")).startswith("pod-idle:")

    settled = sum(
        float(e["amount"])
        for e in run.ledger
        if e["op"] == "settle" and not is_classifier(e) and not is_pod_idle(e)
    )
    pod_idle = sum(
        float(e["amount"]) for e in run.ledger if e["op"] == "settle" and is_pod_idle(e)
    )
    classifier = sum(
        float(e["amount"])
        for e in run.ledger
        if e["op"] in ("settle", "spend") and is_classifier(e)
    )
    reconciled = sum(
        float(e["amount"])
        for e in run.ledger
        if e["op"] == "spend" and str(e.get("label", "")).startswith(RECONCILE_PREFIX)
    )
    billing = run.billing or {}
    return {
        "authorized": authorized,
        "ledger_settled": round(settled, 6),
        "ledger_classifier": round(classifier, 6),
        "ledger_pod_idle": round(pod_idle, 6),
        "ledger_reconciled": round(reconciled, 6),
        "ledger_spent": round(settled + classifier + pod_idle + reconciled, 6),
        "provider_billed": billing.get("billed_total"),
        "provider_not_yet_billed": list(billing.get("not_yet_billed", [])),
        "provider_mismatched": [
            {k: m[k] for k in ("pod", "ledger", "billed")} for m in billing.get("mismatched", [])
        ],
        "records_total": round(sum(row["cost"] for row in rows), 6),
        "failed_boot_total": round(sum(row["failed_boot_cost"] for row in rows), 6),
        "per_solution": rows,
    }


def _failed_spend(run: CohortRun) -> list[dict[str, Any]]:
    """Every failed boot, grouped by why it failed.

    A broken plan boots on purpose (the six class proofs); one can fail at an
    earlier check than its class names.  A fix can fail.  Everything else is
    the harness or the model.  Harness causes come from the recorded tag or,
    for boots recorded before a pattern existed, from the stored log with
    today's patterns (marked "from the log").
    """
    rec = run.records
    cases = {fingerprint_hex(c.broken_plan): c for c in SIX_CLASSES}
    fixes = {r.corrected_plan_digest for r in rec.remediations.values()}
    groups: dict[str, dict[str, Any]] = {}
    for digest, r in sorted(rec.reports.items()):
        if r.boot_outcome != "failed":
            continue
        log = r.log_tail or ""
        recorded = sorted({f for f in r.failures if f.startswith("harness:")})
        case = cases.get(r.deployment_plan_digest or "")
        if case is not None and failed_as_named(case, log):
            cause = "a broken plan, failing as its class names (on purpose)"
        elif case is not None or r.reason == "fix_proof_broken_boot":
            # Booted as a broken plan, but not the class's current one or not
            # at its named check: an earlier draft, replaced (class 6's first
            # attempt stopped at the deprecation check).
            cause = "a broken plan that failed at an earlier check (replaced)"
        elif recorded:
            cause = f"the harness: {', '.join(recorded)}"
        elif (kind := classify_harness_error(log)) is not None:
            cause = f"the harness: {kind} (from the log)"
        elif r.deployment_plan_digest in fixes:
            cause = "a fix that did not work"
        else:
            cause = "the model"
        group = groups.setdefault(cause, {"cause": cause, "boots": 0, "cost": 0.0, "records": []})
        group["boots"] += 1
        group["cost"] = round(group["cost"] + (r.market_equivalent_price or 0.0), 6)
        group["records"].append(digest)
    return sorted(groups.values(), key=lambda g: -g["cost"])


def _cost_row(run: CohortRun, sfp: str | None) -> dict[str, Any]:
    entry = _entry(run, sfp)
    return {
        "label": entry.label if entry else None,
        "model": entry.model_id if entry else None,
        "gpu": entry.requested_execution.gpu_sku if entry else None,
        "gpu_count": entry.requested_execution.gpu_count if entry else None,
        "cost": 0.0,
        "failed_boots": 0,
        "failed_boot_cost": 0.0,
        "records": [],
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _d(digest: str | None) -> str:
    return f"`{digest[:SHORT]}`" if digest else "—"


def _gib(value: int | None) -> str:
    return "—" if value is None else f"{value / GIB:.2f}"


def _signed_gib(value: int | None) -> str:
    return "—" if value is None else f"{value / GIB:+.2f}"


def _v(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value).replace("|", "\\|")


def _table(header: list[str], rows: list[list[str]], empty: str) -> str:
    if not rows:
        return f"_{empty}_"
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


NO_RECORDS = "No records yet."


def render_tables(findings: Mapping[str, Any]) -> dict[str, str]:
    """One Markdown block per findings section, keyed by block name."""
    setup, cost = findings["setup"], findings["cost"]
    fixes_done = [f for f in findings["fixes"] if f.get("label") == "Fixed"]
    blocks: dict[str, str] = {}
    blocks["headline"] = "\n".join(
        [
            f"- Models measured: {len(setup['models_measured'])} "
            f"on {len(setup['gpus_measured'])} GPU types",
            f"- Failure classes fixed and re-verified: {len(fixes_done)} "
            f"of {len(findings['fixes'])}",
            f"- Total cost: ${cost['ledger_spent']:.2f} of the ${cost['authorized']:.0f} cap "
            f"(failed boots included: ${cost['failed_boot_total']:.2f})",
        ]
    )
    workload = setup["serving_workload"] or {}
    blocks["setup"] = _table(
        ["Item", "Value"],
        [
            [
                "Provider / cloud",
                f"{', '.join(setup['providers'])} / {', '.join(setup['cloud_types'])}",
            ],
            ["Runner image", ", ".join(f"`{d}`" for d in setup["image_digests"]) or "—"],
            ["Models measured", ", ".join(setup["models_measured"]) or "—"],
            ["GPUs measured", ", ".join(setup["gpus_measured"]) or "—"],
            ["Run window (UTC)", f"{_v(setup['first_event'])} → {_v(setup['last_event'])}"],
            [
                "Task suite",
                f"{setup['task_suite']['cases']} cases, {_d(setup['task_suite']['fingerprint'])}"
                if setup["task_suite"]
                else "—",
            ],
            [
                "Serving workload",
                f"ISL {_v(workload.get('input_sequence_length'))}, "
                f"OSL {_v(workload.get('output_sequence_length'))}, "
                f"concurrency {_v(workload.get('concurrency'))}, "
                f"SLO p99 TTFT {_v(workload.get('p99_ttft_ms'))} ms, "
                f"p99 TPOT {_v(workload.get('p99_tpot_ms'))} ms"
                if workload
                else "—",
            ],
        ]
        if setup["providers"]
        else [],
        NO_RECORDS,
    )
    blocks["memory"] = _table(
        [
            "Model",
            "GPU",
            "Weights pred. GiB",
            "Weights meas. GiB",
            "Δ weights",
            "Total pred. GiB",
            "Persistent meas. GiB",
            "Δ total",
            "Record",
        ],
        [
            [
                _v(r["model"]),
                f"{_v(r['gpu'])} x{r['gpu_count']}",
                _gib(r["predicted_weight_bytes"]),
                _gib(r["measured_weight_bytes"]),
                _signed_gib(r["weight_delta_bytes"]),
                _gib(r["predicted_total_bytes"]),
                _gib(r["measured_persistent_bytes"]),
                _signed_gib(r["total_delta_bytes"]),
                _d(r["record"]),
            ]
            for r in findings["memory"]
        ],
        NO_RECORDS,
    )
    blocks["repeats"] = _table(
        ["Model", "GPU", "Boots", "Weights spread %", "KV spread %", "Activation spread %"],
        [
            [
                _v(r["model"]),
                _v(r["gpu"]),
                str(r["boots"]),
                _v(r["weight_spread_pct"]),
                _v(r["kv_spread_pct"]),
                _v(r["activation_spread_pct"]),
            ]
            for r in findings["repeat_measurements"]
        ],
        "No solution was measured twice yet.",
    )
    blocks["failed_spend"] = _table(
        ["Why the boot failed", "Boots", "Cost $", "Records"],
        [
            [
                g["cause"],
                str(g["boots"]),
                f"{g['cost']:.4f}",
                ", ".join(_d(d) for d in g["records"]),
            ]
            for g in findings.get("failed_spend", [])
        ],
        "No boot failed.",
    )
    blocks["recheck"] = _table(
        [
            "Model",
            "GPU",
            "TP",
            "Weights GiB: at run / now / measured",
            "Activation GiB: at run / now / measured",
            "Record",
        ],
        [
            [
                _v(r["model"]),
                _v(r["gpu"]),
                str(r["tensor_parallel"]),
                " / ".join(
                    _gib(r[k]) for k in ("weights_at_run", "weights_now", "weights_measured")
                ),
                " / ".join(
                    _gib(r[k])
                    for k in ("activation_at_run", "activation_now", "activation_measured")
                ),
                _d(r["record"]),
            ]
            for r in findings.get("calculator_recheck", [])
        ],
        "Run scripts/calculator_recheck.py.",
    )
    blocks["fixes"] = _table(
        [
            "Class",
            "Broken plan",
            "vLLM source",
            "Diagnosis",
            "Change",
            "Restart",
            "Request",
            "Label",
            "Record",
        ],
        [
            [
                f"{f['class']} {f['family']}",
                f"{_v(f.get('broken_model'))} on {_v(f.get('broken_gpu'))}",
                f"`{f['call_site']}`",
                _v(f.get("diagnosed")),
                "; ".join(f"{k}: {_v(a)} → {_v(b)}" for k, (a, b) in f.get("change", {}).items())
                or "—",
                _v(f.get("mechanism_outcome")),
                _v(f.get("request_outcome")),
                _v(f.get("label")),
                _d(f.get("remediation")),
            ]
            if f["status"] == "run"
            else [
                f"{f['class']} {f['family']}",
                "—",
                f"`{f['call_site']}`",
                "not run",
                "—",
                "—",
                "—",
                "—",
                "—",
            ]
            for f in findings["fixes"]
        ],
        NO_RECORDS,
    )
    blocks["serving"] = _table(
        [
            "Model",
            "GPU",
            "p99 TTFT ms",
            "p99 TPOT ms",
            "SLO TTFT / TPOT ms",
            "Completed / failed",
            "Verdict",
            "Record",
        ],
        [
            [
                _v(r["model"]),
                _v(r["gpu"]),
                _v(r["p99_ttft_ms"]),
                _v(r["p99_tpot_ms"]),
                f"{_v(r['slo_p99_ttft_ms'])} / {_v(r['slo_p99_tpot_ms'])}",
                f"{_v(r['completed'])} / {_v(r['failed'])}",
                _v(r["verdict"]),
                _d(r["record"]),
            ]
            for r in findings["serving"]
        ],
        NO_RECORDS,
    )
    blocks["tasks"] = _table(
        ["Model", "GPU", "Scoring", "Accepted / cases", "Attempts (retries)", "Passed"],
        [
            [
                _v(r["model"]),
                _v(r["gpu"]),
                ", ".join(r.get("scoring") or []) or "—",
                f"{r['accepted']} / {_v(r['cases'])}",
                f"{r['attempts']} ({r['retries']})",
                _v(r["passed"]),
            ]
            for r in findings["tasks"]
        ],
        NO_RECORDS,
    )
    blocks["prediction_errors"] = _table(
        ["Model", "GPU", "Outcome", "Mechanism", "Predicted total GiB", "Record"],
        [
            [
                _v(r["model"]),
                _v(r["gpu"]),
                _v(r["status"]),
                _v(r["mechanism"]),
                _gib(r["predicted_total_bytes"]),
                _d(r["record"]),
            ]
            for r in findings["prediction_errors"]
        ],
        NO_RECORDS,
    )
    blocks["cost"] = _table(
        ["Solution", "GPU", "Cost $", "Failed boots ($)", "Records"],
        [
            [
                _v(r["model"]),
                f"{_v(r['gpu'])} x{_v(r['gpu_count'])}",
                f"{r['cost']:.4f}",
                f"{r['failed_boots']} ({r['failed_boot_cost']:.4f})",
                str(len(r["records"])),
            ]
            for r in cost["per_solution"]
        ]
        + (
            [
                ["**Records total**", "", f"**{cost['records_total']:.4f}**", "", ""],
                ["Classifier calls (ledger)", "", f"{cost['ledger_classifier']:.4f}", "", ""],
                [
                    "Pooled pods between solutions (ledger)",
                    "",
                    f"{cost.get('ledger_pod_idle', 0):.4f}",
                    "",
                    "",
                ],
                [
                    "Pods billed but missing from the ledger (reconciled)",
                    "",
                    f"{cost.get('ledger_reconciled', 0):.4f}",
                    "",
                    "",
                ],
                [
                    "**Ledger spent**",
                    "",
                    f"**{cost['ledger_spent']:.4f}**",
                    "",
                    f"cap ${cost['authorized']:.0f}",
                ],
            ]
            + (
                [
                    [
                        "RunPod billed, same window",
                        "",
                        f"{cost['provider_billed']:.4f}",
                        "",
                        f"not billed yet: {len(cost['provider_not_yet_billed'])} pod(s)",
                    ]
                ]
                if cost.get("provider_billed") is not None
                else []
            )
            if cost["per_solution"]
            else []
        ),
        NO_RECORDS,
    )
    return blocks


_BLOCK = re.compile(
    r"(<!-- findings:(?P<name>[a-z_]+) -->\n)(?P<body>.*?)(\n<!-- /findings:(?P=name) -->)",
    re.DOTALL,
)


def read_blocks(text: str) -> dict[str, str]:
    return {m["name"]: m["body"] for m in _BLOCK.finditer(text)}


def replace_blocks(text: str, blocks: Mapping[str, str]) -> str:
    """Rewrite every marked block in *text*; an unknown block name is an error."""

    def sub(m: re.Match[str]) -> str:
        name = m["name"]
        if name not in blocks:
            raise KeyError(f"no findings block named {name!r}")
        return m.group(1) + blocks[name] + m.group(4)

    return _BLOCK.sub(sub, text)
