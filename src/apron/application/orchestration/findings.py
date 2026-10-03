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

import hashlib
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from apron.application.orchestration.billing import RECONCILE_PREFIX
from apron.application.orchestration.cohort import classify_harness_error
from apron.application.orchestration.remediation import SIX_CLASSES, failed_as_named
from apron.application.orchestration.serving import evaluate_serving_slos
from apron.application.orchestration.staging import STAGE_PREFIX, STORAGE_PREFIX
from apron.domain.fingerprints import fingerprint_hex
from apron.domain.schemas.records import derive_remediation_result
from apron.domain.verdicts import task_verdict

# Ledger label of a diagnostic boot (scripts/peak_probe.py LABEL_PREFIX).
PROBE_PREFIX = "probe:"

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from apron.application.orchestration.cohort_records import CohortRun, SolutionEntry
    from apron.domain.schemas.records import VerificationReport

GIB = 1 << 30
SHORT = 16  # digest prefix shown in tables; the record store resolves prefixes


def build_findings(
    run: CohortRun,
    *,
    authorized: float,
    modern: Mapping[str, Any] | None = None,
    engines: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Every number the article shows, with its records.

    *modern* is the owner-approved list of current models (groups A-D,
    ``cohort/modern-models.json``): each is a row, measured or not yet.
    *engines* maps a runner image digest to the vLLM version inside it (the
    pinned runner images); a digest it lacks is shown as the digest itself.
    """
    rec = run.records
    versions = {
        sfp: engine_version(e.requested_execution.image_digest, engines)
        for sfp, e in rec.solutions.items()
    }
    memory, tasks = _memory(run, versions), _tasks(run, versions)
    cost = _cost(run, authorized, versions)
    return {
        "modern_models": _modern(modern, memory, tasks, cost, run, versions),
        "records_table": _records_table(run, modern, versions),
        "weights_time": _weights_time(run, versions),
        "setup": _setup(run, authorized),
        "memory": memory,
        "repeat_measurements": _repeats(run, versions),
        "calculator_recheck": _recheck(run, versions),
        "fixes": _fixes(run, versions),
        "serving": _serving(run, versions),
        "tasks": tasks,
        "prediction_errors": _prediction_errors(run, versions),
        "cost": cost,
        "failed_spend": _failed_spend(run),
        "record_count": len(rec.claims)
        + len(rec.reports)
        + len(rec.attempts)
        + len(rec.remediations),
    }


# ---------------------------------------------------------------------------


def _entry(run: CohortRun, sfp: str | None) -> SolutionEntry | None:
    return run.records.solutions.get(sfp or "")


def engine_version(image_digest: str, engines: Mapping[str, str] | None) -> str:
    """The vLLM version in a runner image, or the digest when it is not a pinned one."""
    return (engines or {}).get(image_digest) or f"image {image_digest[:19]}"


def _version(versions: Mapping[str, str], sfp: str | None) -> str | None:
    return versions.get(sfp or "")


def _recheck(run: CohortRun, versions: Mapping[str, str]) -> list[dict[str, Any]]:
    rows = []
    for row in (run.recheck or {}).get("rows", []):
        report = run.records.reports.get(row.get("record") or "")
        rows.append(
            {**row, "vllm": _version(versions, report.solution_fingerprint if report else None)}
        )
    return rows


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


def _memory(run: CohortRun, versions: Mapping[str, str]) -> list[dict[str, Any]]:
    """One row per healthy memory report, the pending ones too (``pending``:
    the calculator cannot explain them yet; they stay visible, PLAN §18.2)."""
    rows = []
    pending = {
        d: r
        for d, r in run.pending.items()
        if r.claim_scope == "memory" and r.boot_outcome == "healthy"
    }
    reports = {**run.records.measured(), **pending}
    for digest, r in sorted(reports.items(), key=lambda kv: _row_key(run, kv[1])):
        entry = _entry(run, r.solution_fingerprint)
        claim_digest, predicted = _claim_for(run, r.solution_fingerprint)
        delta = r.predicted_minus_measured or {}
        rows.append(
            {
                "model": entry.model_id if entry else None,
                "gpu": entry.requested_execution.gpu_sku if entry else None,
                "gpu_count": entry.requested_execution.gpu_count if entry else None,
                "vllm": _version(versions, r.solution_fingerprint),
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
                "solution": r.solution_fingerprint,
                "pending": digest in pending,
            }
        )
    return rows


def _modern(
    plan: Mapping[str, Any] | None,
    memory: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    cost: Mapping[str, Any],
    run: CohortRun,
    versions: Mapping[str, str],
) -> list[dict[str, Any]]:
    """One row per approved modern model: not run yet, failed, or measured.

    A model run more than once (another plan, GPU or engine setting) shows
    its latest booted solution, by the solution's last ``executed`` event,
    with that solution's memory and task rows: Muse-Glimmer-30B was scored
    0/3 without its reasoning parser and 3/3 with it, and the card showed
    whichever solution sorted first by digest."""
    executed: dict[str, str] = {}
    for e in run.events:
        if e.get("event") == "executed" and e.get("solution_fingerprint"):
            sfp = e["solution_fingerprint"]
            executed[sfp] = max(executed.get(sfp, ""), str(e.get("at", "")))
    rows = []
    failed = defaultdict(list)
    failed_on: dict[str, set[str]] = defaultdict(set)
    for digest, r in run.records.reports.items():
        entry = _entry(run, r.solution_fingerprint)
        if entry is not None and r.boot_outcome == "failed":
            failed[entry.model_id].append(digest)
            failed_on[entry.model_id].add(versions.get(entry.solution_fingerprint) or "")
    for m in (plan or {}).get("models", []):
        model = m["model_id"]
        booted = [r for r in memory if r["model"] == model]
        best = max(booted, key=lambda r: executed.get(r["solution"], ""), default=None)
        scored = [
            t
            for t in tasks
            if t["model"] == model and (best is None or t["solution"] == best["solution"])
        ]
        latest_rule = max(scored, key=lambda t: len(t.get("scoring") or []), default=None)
        spent = sum(c["cost"] for c in cost.get("per_solution", []) if c.get("model") == model)
        status = "booted" if booted else ("failed" if failed.get(model) else "not run yet")
        if best is not None and best.get("pending"):
            status = "booted, calculator pending"
        if best:
            vllm = best["vllm"]
        elif failed.get(model):
            vllm = ", ".join(sorted(v for v in failed_on[model] if v)) or None
        else:
            vllm = m.get("engine")  # the version the plan names; nothing ran yet
        rows.append(
            {
                "group": m["group"],
                "model": model,
                "params_b": m.get("params_b"),
                "downloads_30d": m.get("downloads_30d"),
                "gpu": best["gpu"] if best else m.get("gpu"),
                "gpu_count": best["gpu_count"] if best else m.get("gpu_count"),
                "vllm": vllm,
                "status": status,
                "before_gpu": _before_gpu(m.get("prediction")),
                "predicted_weight_bytes": best["predicted_weight_bytes"] if best else None,
                "measured_weight_bytes": best["measured_weight_bytes"] if best else None,
                "accepted": latest_rule["accepted"] if latest_rule else None,
                "cases": latest_rule["cases"] if latest_rule else None,
                "cost": round(spent, 4) if spent else None,
                "records": sorted(([best["record"]] if best else []) + failed.get(model, [])),
            }
        )
    return rows


def _before_gpu(prediction: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """What the plan pipeline said with no GPU: its verdict and bytes per GPU."""
    if not prediction or prediction.get("error"):
        return {"status": "error", "detail": prediction.get("error")} if prediction else None
    return {
        "status": prediction.get("status"),
        "mechanism": prediction.get("mechanism"),
        "tensor_parallel": prediction.get("tensor_parallel"),
        "weight_gib_per_gpu": prediction.get("weight_gib_per_gpu"),
        "total_gib_per_gpu": prediction.get("total_gib_per_gpu"),
    }


def _before_gpu_text(before: Mapping[str, Any] | None) -> str:
    if not before:
        return "—"
    if before["status"] == "unknown":
        return "unknown (no memory model yet)"
    if before["status"] != "planned":
        return str(before["status"])
    return f"fits: {before['total_gib_per_gpu']} GiB per GPU"


def _records_table(
    run: CohortRun, plan: Mapping[str, Any] | None, versions: Mapping[str, str]
) -> dict[str, Any]:
    """The README's living table: one row per (model, GPU x count, vLLM version).

    A row shows the newest healthy boot (identity-manifest time, then digest)
    and, when none booted, the newest failed boot.  Serving is the serving
    report of that boot (same solution and execution), tasks the solution's
    attempts under its newest scoring rule, cost every record in the row.
    Boots of plans broken on purpose (the six fix proofs) are counted apart;
    a record without a solution in the identity manifest is listed as not
    rendered, never dropped.
    """
    rec = run.records
    groups: dict[tuple[str, str, int, str], list[tuple[str, VerificationReport]]] = defaultdict(
        list
    )
    broken: list[str] = []
    unrendered: list[str] = []
    broken_cost = 0.0
    for digest, r in sorted(rec.reports.items()):
        entry = _entry(run, r.solution_fingerprint)
        if entry is None:
            unrendered.append(digest)
            continue
        if r.boot_outcome == "failed" and _failure_cause(run, r)[0] == BROKEN:
            broken.append(digest)
            broken_cost += r.market_equivalent_price or 0.0
            continue
        ex = entry.requested_execution
        key = (entry.model_id, ex.gpu_sku, ex.gpu_count, versions[entry.solution_fingerprint])
        groups[key].append((digest, r))
    attempts: dict[str, list[str]] = defaultdict(list)
    for digest, a in rec.attempts.items():
        if _entry(run, a.solution_fingerprint) is None:
            unrendered.append(digest)
        else:
            attempts[a.solution_fingerprint].append(digest)

    def newest(pairs: list[tuple[str, VerificationReport]]) -> tuple[str, VerificationReport]:
        def when(dr: tuple[str, VerificationReport]) -> tuple[str, str]:
            entry = _entry(run, dr[1].solution_fingerprint)
            return (entry.at if entry else "", dr[0])

        return max(pairs, key=when)

    rows = []
    for (model, gpu, count, vllm), reports in sorted(
        groups.items(), key=lambda kv: (kv[0][0].lower(), kv[0][1], kv[0][2], kv[0][3])
    ):
        healthy = [
            (d, r) for d, r in reports if r.claim_scope == "memory" and r.boot_outcome == "healthy"
        ]
        failed = [(d, r) for d, r in reports if r.boot_outcome == "failed"]
        if not healthy and not failed:
            unrendered.extend(d for d, _ in reports)  # serving without any boot record
            continue
        digest, shown = newest(healthy or failed)
        sfp = shown.solution_fingerprint or ""
        _, predicted = _claim_for(run, sfp)
        solutions = {r.solution_fingerprint or "" for _, r in reports}
        row_attempts = sorted(d for s in solutions for d in attempts.get(s, []))
        cost = sum(r.market_equivalent_price or 0.0 for _, r in reports) + sum(
            rec.attempts[d].infrastructure_cost or 0.0 for d in row_attempts
        )
        row: dict[str, Any] = {
            "model": model,
            "gpu": gpu,
            "gpu_count": count,
            "vllm": vllm,
            "boot": "booted" if healthy else "failed",
            "failed_boots": len(failed),
            "failure": None,
            "error": None,
            "predicted_weight_bytes": predicted.get("weight_memory_bytes"),
            "measured_weight_bytes": None,
            # The startup peak: the activation peak of vLLM's profile run.
            "predicted_peak_bytes": predicted.get("activation_estimate_bytes"),
            "measured_peak_bytes": None,
            "kv_cache_tokens": None,
            "p99_ttft_ms": None,
            "slo_p99_ttft_ms": None,
            "slo_verdict": None,
            "accepted": None,
            "cases": None,
            "tasks_passed": None,
            "cold_start_seconds": None,
            "cost": round(cost, 6),
            "record": digest,
            "records": sorted([d for d, _ in reports] + row_attempts),
        }
        if not healthy:
            row["failure"] = _failure_cause(run, shown)[1]
            row["error"] = root_error(shown.log_tail)
            rows.append(row)
            continue
        phases = shown.phase_seconds or {}
        row.update(
            {
                "measured_weight_bytes": shown.model_weight_memory,
                "measured_peak_bytes": shown.transient_peak_headroom,
                "kv_cache_tokens": shown.kv_cache_tokens,
                # Pod up to engine ready: the weights on the pod, then the engine.
                "cold_start_seconds": round(phases["weights"] + phases["engine_start"], 1)
                if "weights" in phases and "engine_start" in phases
                else None,
            }
        )
        serving = sorted(
            (d, r)
            for d, r in reports
            if r.claim_scope == "serving_performance" and r.solution_fingerprint == sfp
        )
        same_boot = [
            (d, r) for d, r in serving if r.execution_fingerprint == shown.execution_fingerprint
        ]
        if same_boot or serving:
            _, s = (same_boot or serving)[-1]
            entry = _entry(run, sfp)
            row["p99_ttft_ms"] = (s.serving_latency_ms or {}).get("p99_ttft_ms")
            row["slo_p99_ttft_ms"] = entry.serving_workload.p99_ttft_ms if entry else None
            row["slo_verdict"] = s.serving_slo_verdict
        by_protocol: dict[str, list[str]] = defaultdict(list)
        for d in attempts.get(sfp, []):
            by_protocol[rec.attempts[d].evaluation_protocol_fingerprint].append(d)
        if by_protocol:
            # The newest scoring rule: the one with the most checks (as in _tasks).
            protocol_fp = max(
                by_protocol,
                key=lambda fp: (
                    len(rec.protocols[fp].deterministic_checks) if fp in rec.protocols else 0,
                    fp,
                ),
            )
            task = _task_row(run, sfp, protocol_fp, by_protocol[protocol_fp])
            row.update(
                {
                    "accepted": task["accepted"],
                    "cases": task["cases"],
                    "tasks_passed": task["passed"],
                }
            )
        rows.append(row)

    planned = []
    for m in (plan or {}).get("models", []):
        ran = any(r["model"] == m["model_id"] for r in rows)
        if not ran:
            planned.append(
                {
                    "model": m["model_id"],
                    "gpu": m.get("gpu"),
                    "gpu_count": m.get("gpu_count"),
                    "vllm": m.get("engine"),
                    "predicted": _before_gpu(m.get("prediction")),
                }
            )
    return {
        "rows": rows,
        "planned": planned,
        "broken_on_purpose": {
            "boots": len(broken),
            "cost": round(broken_cost, 6),
            "records": broken,
        },
        "not_rendered": sorted(unrendered),
        "record_count": len(rec.claims)
        + len(rec.reports)
        + len(rec.attempts)
        + len(rec.remediations),
    }


def _weights_time(run: CohortRun, versions: Mapping[str, str]) -> list[dict[str, Any]]:
    """Pod time per boot of every model that was ever booted from staged weights,
    beside its boots that downloaded on the pod."""
    staged_models = set()
    for r in run.records.reports.values():
        entry = _entry(run, r.solution_fingerprint)
        if entry is not None and r.weights_source and "network volume" in r.weights_source:
            staged_models.add(entry.model_id)
    rows = []
    for digest, r in sorted(run.records.reports.items(), key=lambda kv: _row_key(run, kv[1])):
        entry = _entry(run, r.solution_fingerprint)
        if entry is None or entry.model_id not in staged_models or r.claim_scope != "memory":
            continue
        phases = r.phase_seconds or {}
        rows.append(
            {
                "model": entry.model_id,
                "gpu": entry.requested_execution.gpu_sku,
                "vllm": _version(versions, entry.solution_fingerprint),
                "weights_source": r.weights_source or "downloaded to the pod volume",
                "weights_seconds": phases.get("weights"),
                "engine_start_seconds": phases.get("engine_start"),
                "pod_seconds": phases.get("total"),
                "hourly_rate": r.hourly_rate,
                "record": digest,
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


def _repeats(run: CohortRun, versions: Mapping[str, str]) -> list[dict[str, Any]]:
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
                "vllm": _version(versions, sfp),
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


def _fixes(run: CohortRun, versions: Mapping[str, str]) -> list[dict[str, Any]]:
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
        # The record that went furthest: Fixed first, then verified, then any.
        # Earlier attempts stay in the records and are counted in the row.
        digest, r = sorted(
            found,
            key=lambda dr: (
                derive_remediation_result(dr[1].mechanism_outcome, dr[1].request_outcome)
                != "Fixed",
                dr[1].mechanism_outcome != "verified",
            ),
        )[0]
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
                "vllm": _version(versions, r.corrected_solution_digest),
                "mechanism_outcome": r.mechanism_outcome,
                "request_outcome": r.request_outcome,
                "violated_constraints": list(r.violated_constraints),
                "label": derive_remediation_result(r.mechanism_outcome, r.request_outcome),
                "remediation": digest,
                "proving_boot": boot,
                "promoted_rule": promoted,
                "attempts": len(found),
            }
        )
    return rows


def _serving(run: CohortRun, versions: Mapping[str, str]) -> list[dict[str, Any]]:
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
                "vllm": _version(versions, r.solution_fingerprint),
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


def _tasks(run: CohortRun, versions: Mapping[str, str]) -> list[dict[str, Any]]:
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
        rows.append({**_task_row(run, sfp, protocol_fp, digests), "vllm": _version(versions, sfp)})
    return rows


def _task_row(run: CohortRun, sfp: str, protocol_fp: str, digests: list[str]) -> dict[str, Any]:
    """The attempts of one solution under one scoring rule, and their verdict."""
    entry = _entry(run, sfp)
    protocol = run.records.protocols.get(protocol_fp)
    limits = (
        {c.get("id", ""): int(c.get("max_tokens", 0) or 0) for c in entry.task_suite.cases}
        if entry
        else {}
    )
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
    return {
        "model": entry.model_id if entry else None,
        "gpu": entry.requested_execution.gpu_sku if entry else None,
        "cases": len(entry.task_suite.cases) if entry else None,
        "attempts": len(attempts),
        "retries": sum(1 for a in attempts if a.retries),
        "accepted": len({a.case_id for a in attempts if a.accepted}),
        # Suite v3: checks the plan does not serve (reason in the output).
        "skipped": len({a.case_id for a in attempts if a.accepted is None}),
        # Cut at the case's max_tokens: tagged since the harness reads
        # finish_reason; before that, the output used every allowed token.
        "truncated": sum(1 for a in attempts if _truncated(a, limits)),
        "passed": verdict.passed if verdict else None,
        "scoring": list(protocol.deterministic_checks) if protocol else None,
        "records": sorted(digests),
        "solution": sfp,
    }


def _truncated(attempt: Any, limits: Mapping[str, int]) -> bool:
    if "evaluation:truncated at max_tokens" in attempt.failures:
        return True
    limit = limits.get(attempt.case_id, 0)
    return bool(limit and attempt.output_tokens is not None and attempt.output_tokens >= limit)


def _entry_key(run: CohortRun, sfp: str) -> tuple[str, str, str]:
    entry = _entry(run, sfp)
    return (entry.model_id, entry.requested_execution.gpu_sku, sfp) if entry else ("", "", sfp)


def _prediction_errors(run: CohortRun, versions: Mapping[str, str]) -> list[dict[str, Any]]:
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
                "vllm": _version(versions, claim.solution_fingerprint),
                "status": status,
                "mechanism": entry.mechanism if entry else None,
                "predicted_total_bytes": claim.proposed_configuration.get("total_required_bytes"),
                "record": digest,
            }
        )
    rows.sort(key=lambda row: (row["model"] or "", row["gpu"] or ""))
    return rows


def _cost(run: CohortRun, authorized: float, versions: Mapping[str, str]) -> dict[str, Any]:
    rec = run.records
    per: dict[str, dict[str, Any]] = {}
    # Pending reports are stored evidence and were paid for (pending-records/).
    for digest, r in {**rec.reports, **run.pending}.items():
        row = per.setdefault(
            r.solution_fingerprint or "", _cost_row(run, r.solution_fingerprint, versions)
        )
        row["cost"] += r.market_equivalent_price or 0.0
        row["records"].append(digest)
        if r.boot_outcome == "failed":
            row["failed_boots"] += 1
            row["failed_boot_cost"] += r.market_equivalent_price or 0.0
    for digest, a in rec.attempts.items():
        row = per.setdefault(
            a.solution_fingerprint, _cost_row(run, a.solution_fingerprint, versions)
        )
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

    def is_staging(e: dict[str, Any]) -> bool:
        return str(e.get("label", "")).startswith(STAGE_PREFIX)

    def is_probe(e: dict[str, Any]) -> bool:
        # Diagnostic boots (scripts/peak_probe.py): paid, but not a solution.
        return str(e.get("label", "")).startswith(PROBE_PREFIX)

    # A run that starts while another holds settles the live hold
    # provisionally ("replayed"); the owner's later settle replaces it
    # (BudgetTracker.replay, "provisional"), so only the owner's counts.
    replaced: set[int] = set()
    provisional: dict[str, int] = {}
    for i, e in enumerate(run.ledger):
        label = str(e.get("label", ""))
        if e["op"] == "hold":
            provisional.pop(label, None)
        elif e["op"] == "settle" and str(e.get("flag") or "").startswith("replayed"):
            provisional[label] = i
        elif e["op"] == "settle" and label in provisional:
            replaced.add(provisional.pop(label))
    settles = [e for i, e in enumerate(run.ledger) if e["op"] == "settle" and i not in replaced]
    settled = sum(
        float(e["amount"])
        for e in settles
        if not is_classifier(e) and not is_pod_idle(e) and not is_staging(e) and not is_probe(e)
    )
    # Pods that never started, spent under their own label (cohort.py).
    abandoned = sum(
        float(e["amount"])
        for e in run.ledger
        if e["op"] == "spend" and str(e.get("label", "")).startswith("pod-abandoned:")
    )
    probe = sum(float(e["amount"]) for e in settles if is_probe(e))
    staging = sum(float(e["amount"]) for e in settles if is_staging(e))
    storage = sum(
        float(e["amount"])
        for e in run.ledger
        if e["op"] == "spend" and str(e.get("label", "")).startswith(STORAGE_PREFIX)
    )
    pod_idle = sum(float(e["amount"]) for e in settles if is_pod_idle(e))
    classifier = sum(
        float(e["amount"])
        for e in [*settles, *(e for e in run.ledger if e["op"] == "spend")]
        if is_classifier(e)
    )
    reconciled = sum(
        float(e["amount"])
        for e in run.ledger
        if e["op"] == "spend" and str(e.get("label", "")).startswith(RECONCILE_PREFIX)
    )
    corrected = sum(float(e["amount"]) for e in run.ledger if e["op"] == "correct")
    billing = run.billing or {}
    return {
        "authorized": authorized,
        "ledger_settled": round(settled, 6),
        "ledger_classifier": round(classifier, 6),
        "ledger_pod_idle": round(pod_idle, 6),
        "ledger_reconciled": round(reconciled, 6),
        "ledger_corrected": round(corrected, 6),
        "ledger_staging": round(staging, 6),
        "ledger_storage": round(storage, 6),
        "ledger_probe": round(probe, 6),
        "ledger_abandoned": round(abandoned, 6),
        "ledger_spent": round(
            settled
            + classifier
            + pod_idle
            + staging
            + storage
            + probe
            + reconciled
            + corrected
            + abandoned,
            6,
        ),
        "provider_billed": billing.get("billed_total"),
        "provider_not_yet_billed": list(billing.get("not_yet_billed", [])),
        "provider_mismatched": [
            {k: m[k] for k in ("pod", "ledger", "billed")} for m in billing.get("mismatched", [])
        ],
        "records_total": round(sum(row["cost"] for row in rows), 6),
        "failed_boot_total": round(sum(row["failed_boot_cost"] for row in rows), 6),
        "per_solution": rows,
    }


# What each harness failure tag means, for the reader (the JSON keeps the tag).
HARNESS_NAMES: dict[str, str] = {
    "harness:tokenizer_files_missing": "a full disk cut the download short (a tokenizer error)",
    "harness:boot_deadline": "engine still starting when the harness gave up",
    "harness:boot_stalled": "vLLM stopped writing to its log while starting (hung)",
    "harness:toolchain": "compiler missing from the engine's environment",
    "harness:download_incomplete": "download check flagged a complete file",
    "harness:exception:QueryError": "GPU provider API error",
    "harness:exception:RemoteCommandTimeout": "a command on the pod did not answer in time",
    "harness:host_nvls": "the host's NVSwitch could not bind NVLink SHARP (NVLS)",
    "harness:nccl_init": "NCCL failed on the host before the model loaded",
}


def _harness(tag: str) -> str:
    return HARNESS_NAMES.get(tag, tag)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _boot_evidence(run: CohortRun) -> dict[tuple[str, str], str]:
    """Log lines of a failed boot its stored tail lost, keyed by the boot's
    solution and a hash of that tail (``boot_evidence`` events).  The record
    stays as stored; the event adds the evidence captured from the same pod
    before it was torn down."""
    return {
        (str(e.get("solution_fingerprint")), str(e.get("log_tail_sha256"))): str(e["evidence"])
        for e in run.events
        if e.get("event") == "boot_evidence" and e.get("evidence")
    }


# Kinds of failed boot (``_failure_cause``); a broken plan fails on purpose.
BROKEN, HARNESS, FIX, MODEL = "broken on purpose", "harness", "fix", "model"


def _failure_cause(run: CohortRun, r: VerificationReport) -> tuple[str, str]:
    """(kind, cause) of one failed boot, *cause* in words for the reader.

    A broken plan boots on purpose (the six class proofs); one can fail at an
    earlier check than its class names.  A fix can fail.  Everything else is
    the harness or the model.  Harness causes come from the recorded tag or,
    for boots recorded before a pattern existed, from the stored log with
    today's patterns (marked "from the log").
    """
    cases = {fingerprint_hex(c.broken_plan): c for c in SIX_CLASSES}
    fixes = {m.corrected_plan_digest for m in run.records.remediations.values()}
    log = r.log_tail or ""
    log += "\n" + _boot_evidence(run).get((r.solution_fingerprint or "", _sha(log)), "")
    recorded = sorted({f for f in r.failures if f.startswith("harness:")})
    case = cases.get(r.deployment_plan_digest or "")
    if case is not None and failed_as_named(case, log):
        return BROKEN, "a broken plan, failing as its class names (on purpose)"
    if case is not None or r.reason == "fix_proof_broken_boot":
        # Booted as a broken plan that is not the class's current one:
        # class 6's first plan stopped at the deprecation check, and its
        # Qwen3-0.6B plan was replaced by the Qwen3-8B one.
        return BROKEN, "an earlier broken plan, since replaced"
    if recorded:
        return HARNESS, f"the harness: {'; '.join(_harness(t) for t in recorded)}"
    if (kind := classify_harness_error(log)) is not None:
        return HARNESS, f"the harness: {_harness(kind)} (classified from the log)"
    if r.deployment_plan_digest in fixes:
        return FIX, "a fix that did not work"
    return MODEL, "the model"


# The engine's own error line in a failed boot's log: "ValueError: ...".
_ERROR_LINE = re.compile(
    r"^(?:\([A-Za-z]+ pid=\d+\)\s*)?(?P<line>[A-Za-z_.]*(?:Error|Exception): .+)$"
)
# Wrappers that only point at the root cause printed above them.
_WRAPPER = re.compile(r"Engine core initialization failed|See root cause above")
ERROR_CHARS = 110


def root_error(log: str | None) -> str | None:
    """The last error line of a failed boot's log that is not a wrapper, shortened."""
    found = None
    for raw in (log or "").splitlines():
        m = _ERROR_LINE.match(raw.strip())
        if m and not _WRAPPER.search(m["line"]):
            found = m["line"].strip()
    if found is None:
        return None
    return found if len(found) <= ERROR_CHARS else found[: ERROR_CHARS - 1].rstrip() + "…"


def _failed_spend(run: CohortRun) -> list[dict[str, Any]]:
    """Every failed boot, grouped by why it failed (``_failure_cause``)."""
    rec = run.records
    groups: dict[str, dict[str, Any]] = {}
    for digest, r in sorted(rec.reports.items()):
        if r.boot_outcome != "failed":
            continue
        cause = _failure_cause(run, r)[1]
        group = groups.setdefault(cause, {"cause": cause, "boots": 0, "cost": 0.0, "records": []})
        group["boots"] += 1
        group["cost"] = round(group["cost"] + (r.market_equivalent_price or 0.0), 6)
        group["records"].append(digest)
    return sorted(groups.values(), key=lambda g: -g["cost"])


def _cost_row(run: CohortRun, sfp: str | None, versions: Mapping[str, str]) -> dict[str, Any]:
    entry = _entry(run, sfp)
    return {
        "label": entry.label if entry else None,
        "model": entry.model_id if entry else None,
        "gpu": entry.requested_execution.gpu_sku if entry else None,
        "gpu_count": entry.requested_execution.gpu_count if entry else None,
        "vllm": _version(versions, sfp),
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


# Plain names for the reader (the JSON keeps the schema values).
CLASS_NAMES = {
    1: "Weights do not fit the GPU",
    2: "No room for the KV cache",
    3: "Context length above the model's limit",
    4: "float16 not supported by the model",
    5: "Tensor parallelism does not divide the heads",
    6: "Quantization needs a newer GPU",
}
CHANGE_NAMES = {
    "resource_allocation.gpu_sku": "GPU",
    "resource_allocation.model_id": "model",
    "engine_configuration.max_model_len": "max_model_len",
    "tensor_parallel": "tensor parallel",
    "dtype": "dtype",
}
LABEL_NAMES = {
    "Fixed": "Fixed",
    "Alternative with trade-offs": "Runs, misses the request",
    "Unverified suggestion": "Not proven",
}
CHECK_NAMES: dict[str, str] = {
    "whitespace_normalized_exact_match": "exact match",
    "strip_terminal_punctuation": "final full stop ignored",
}


def _records(digests: Sequence[str]) -> str:
    """The first record's digest, and how many more (all are in the JSON)."""
    ordered = sorted(digests)
    if not ordered:
        return "—"
    more = f" +{len(ordered) - 1}" if len(ordered) > 1 else ""
    return f"{_d(ordered[0])}{more}"


def _passed_part(fix: Mapping[str, Any], part: str) -> str:
    if fix.get("request_outcome") not in ("satisfied", "violated"):
        return "—"
    failed = any(str(v).startswith(f"{part}:") for v in fix.get("violated_constraints", []))
    return "no" if failed else "yes"


def _staged_label(source: str) -> str:
    if "network volume" in source:
        return "staged on a network volume" + (
            f" ({source.rsplit(' in ', 1)[-1]})" if " in " in source else ""
        )
    return "downloaded on the GPU pod"


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
            "vLLM",
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
                _v(r.get("vllm")),
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
        [
            "Model",
            "GPU",
            "vLLM",
            "Boots",
            "Weights spread %",
            "KV spread %",
            "Activation spread %",
            "Records",
        ],
        [
            [
                _v(r["model"]),
                _v(r["gpu"]),
                _v(r.get("vllm")),
                str(r["boots"]),
                _v(r["weight_spread_pct"]),
                _v(r["kv_spread_pct"]),
                _v(r["activation_spread_pct"]),
                _records(r.get("records", [])),
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
    blocks["modern_models"] = _table(
        [
            "Group",
            "Model",
            "Size (B params)",
            "Downloads / 30 days",
            "GPU",
            "vLLM",
            "Before any GPU",
            "Status",
            "Weights GiB: predicted / measured",
            "Questions answered",
            "Cost $",
            "Records",
        ],
        [
            [
                r["group"],
                _v(r["model"]),
                _v(r["params_b"]),
                f"{r['downloads_30d']:,}" if r.get("downloads_30d") else "—",
                f"{_v(r['gpu'])} x{_v(r['gpu_count'])}",
                _v(r.get("vllm")),
                _before_gpu_text(r.get("before_gpu")),
                r["status"],
                f"{_gib(r['predicted_weight_bytes'])} / {_gib(r['measured_weight_bytes'])}"
                if r["status"].startswith("booted")
                else "—",
                f"{r['accepted']} / {r['cases']}" if r.get("cases") else "—",
                f"{r['cost']:.4f}" if r.get("cost") else "—",
                _records(r["records"]),
            ]
            for r in findings.get("modern_models", [])
        ],
        "No modern models approved yet.",
    )
    blocks["weights_time"] = _table(
        [
            "Model",
            "GPU",
            "vLLM",
            "Weights",
            "Weights s",
            "Engine start s",
            "Pod s",
            "GPU $/h",
            "Record",
        ],
        [
            [
                _v(r["model"]),
                _v(r["gpu"]),
                _v(r.get("vllm")),
                _staged_label(r["weights_source"]),
                _v(r["weights_seconds"]),
                _v(r["engine_start_seconds"]),
                _v(round(r["pod_seconds"]) if r.get("pod_seconds") else None),
                _v(r["hourly_rate"]),
                _d(r["record"]),
            ]
            for r in findings.get("weights_time", [])
        ],
        "No boot from staged weights yet.",
    )
    blocks["recheck"] = _table(
        [
            "Model",
            "GPU",
            "vLLM",
            "TP",
            "Weights GiB: at run / now / measured",
            "Activation GiB: at run / now / measured",
            "Record",
        ],
        [
            [
                _v(r["model"]),
                _v(r["gpu"]),
                _v(r.get("vllm")),
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
            "Failure",
            "Broken plan",
            "vLLM error from",
            "What Apron changed",
            "Fixed plan booted",
            "vLLM",
            "Tasks passed",
            "SLO passed",
            "Result",
            "Record",
        ],
        [
            [
                f"{f['class']}. {CLASS_NAMES.get(f['class'], f['family'])}",
                f"{_v(f.get('broken_model'))} on {_v(f.get('broken_gpu'))}",
                f"`{f['call_site']}`",
                "; ".join(
                    f"{CHANGE_NAMES.get(k, k)}: {_v(a)} → {_v(b)}"
                    for k, (a, b) in f.get("change", {}).items()
                )
                or "—",
                "yes" if f.get("mechanism_outcome") == "verified" else "no",
                _v(f.get("vllm")),
                _passed_part(f, "task"),
                _passed_part(f, "serving"),
                LABEL_NAMES.get(str(f.get("label")), _v(f.get("label"))),
                _d(f.get("remediation")),
            ]
            if f["status"] == "run"
            else [
                f"{f['class']}. {CLASS_NAMES.get(f['class'], f['family'])}",
                "—",
                f"`{f['call_site']}`",
                "not run",
                "—",
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
            "vLLM",
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
                _v(r.get("vllm")),
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
        [
            "Model",
            "GPU",
            "vLLM",
            "Scoring",
            "Accepted / cases",
            "Attempts (retries)",
            "Cut at the token limit",
            "Passed",
            "Records",
        ],
        [
            [
                _v(r["model"]),
                _v(r["gpu"]),
                _v(r.get("vllm")),
                ", ".join(CHECK_NAMES.get(c) or c for c in r.get("scoring") or []) or "—",
                f"{r['accepted']} / {_v(r['cases'])}",
                f"{r['attempts']} ({r['retries']})",
                str(r.get("truncated", 0)),
                _v(r["passed"]),
                _records(r.get("records", [])),
            ]
            for r in findings["tasks"]
        ],
        NO_RECORDS,
    )
    blocks["prediction_errors"] = _table(
        ["Model", "GPU", "vLLM", "Outcome", "Mechanism", "Predicted total GiB", "Record"],
        [
            [
                _v(r["model"]),
                _v(r["gpu"]),
                _v(r.get("vllm")),
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
        ["Solution", "GPU", "vLLM", "Cost $", "Failed boots ($)", "Records"],
        [
            [
                _v(r["model"]),
                f"{_v(r['gpu'])} x{_v(r['gpu_count'])}",
                _v(r.get("vllm")),
                f"{r['cost']:.4f}",
                f"{r['failed_boots']} ({r['failed_boot_cost']:.4f})",
                _records(r["records"]),
            ]
            for r in cost["per_solution"]
        ]
        + _blank_column(
            [
                ["**Records total**", "", f"**{cost['records_total']:.4f}**", "", ""],
                [
                    "Diagnosis model calls (Claude Haiku)",
                    "",
                    f"{cost['ledger_classifier']:.4f}",
                    "",
                    "",
                ],
                [
                    "Idle time of reused pods",
                    "",
                    f"{cost.get('ledger_pod_idle', 0):.4f}",
                    "",
                    "",
                ],
                [
                    "CPU pods that staged weights",
                    "",
                    f"{cost.get('ledger_staging', 0):.4f}",
                    "",
                    "",
                ],
                [
                    "Network volume storage for staged weights",
                    "",
                    f"{cost.get('ledger_storage', 0):.4f}",
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
                    "Corrections to RunPod's bill (estimates, clock differences)",
                    "",
                    f"{cost.get('ledger_corrected', 0):+.4f}",
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
            else [],
            at=2,  # the summary lines have no vLLM version
        ),
        NO_RECORDS,
    )
    return blocks


def _blank_column(rows: list[list[str]], *, at: int) -> list[list[str]]:
    return [[*row[:at], "", *row[at:]] for row in rows]


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


# ---------------------------------------------------------------------------
# README: the living table of measured records (the posts tell the story)
# ---------------------------------------------------------------------------

README_START = "<!-- records:start -->"
README_END = "<!-- records:end -->"
_README = re.compile(
    re.escape(README_START) + r"\n(?P<body>.*?)\n?" + re.escape(README_END), re.DOTALL
)


# Short failure names for a table cell (the JSON keeps the cause in words).
def _gpu_cell(gpu: str | None, count: int | None) -> str:
    name = (gpu or "—").removeprefix("NVIDIA ").removeprefix("GeForce ")
    return f"{name} x{_v(count)}"


def _pair_gib(predicted: int | None, measured: int | None) -> str:
    return f"{_gib(predicted)} / {_gib(measured)}"


def _prediction_text(predicted: Mapping[str, Any] | None) -> str:
    if not predicted:
        return "no prediction yet"
    if predicted["status"] == "planned":
        return (
            f"predicted {predicted['weight_gib_per_gpu']} GiB weights, "
            f"{predicted['total_gib_per_gpu']} GiB in all per GPU "
            f"(tensor parallel {predicted['tensor_parallel']})"
        )
    if predicted["status"] == "unknown":
        return "predicted: unknown (no memory model yet)"
    return f"prediction: {predicted['status']}"


def render_readme(
    findings: Mapping[str, Any],
    *,
    reports_href: str,
    fixes_href: str,
    omit_models: frozenset[str] = frozenset(),
) -> str:
    """The README section between the records markers.

    *reports_href* is the directory of the verification reports relative to
    the README; *fixes_href* the post whose fixes table shows the boots of
    plans broken on purpose.  Only rows that booted are shown; failed rows
    and models in *omit_models* stay in the stored records and the findings
    data, not on the front page.
    """
    table = findings["records_table"]
    shown = [r for r in table["rows"] if r["boot"] == "booted" and r["model"] not in omit_models]
    planned = [p for p in table["planned"] if p["model"] not in omit_models]

    def link(digest: str) -> str:
        return f"[`{digest[:SHORT]}`]({reports_href}/{digest}.json)"

    rows = [
        [
            _v(r["model"]),
            _gpu_cell(r["gpu"], r["gpu_count"]),
            _v(r["vllm"]),
            _pair_gib(r["predicted_weight_bytes"], r["measured_weight_bytes"]),
            _pair_gib(r["predicted_peak_bytes"], r["measured_peak_bytes"]),
            f"{r['kv_cache_tokens']:,}" if r["kv_cache_tokens"] is not None else "—",
            f"{r['p99_ttft_ms']:,.0f} ({r['slo_verdict']})"
            if r["p99_ttft_ms"] is not None
            else "—",
            f"{r['accepted']}/{r['cases']} {'pass' if r['tasks_passed'] else 'fail'}"
            if r["cases"]
            else "—",
            f"{r['cold_start_seconds']:.0f}" if r["cold_start_seconds"] is not None else "—",
            link(r["record"]) + (f" +{len(r['records']) - 1}" if len(r["records"]) > 1 else ""),
        ]
        for r in shown
    ]
    parts = [
        f"_Generated by `scripts/cohort_findings.py` from the {table['record_count']} stored "
        "records; not typed by hand. One row per model, GPU and vLLM version that booted, "
        "showing its newest boot._",
        "",
        "Weights and startup peak (the activation peak of vLLM's startup profile run), GiB: "
        "predicted by Apron's calculator when the plan was made / measured per GPU from "
        "vLLM's log. KV pool: tokens vLLM's KV cache holds. Tasks: questions answered "
        "correctly / asked. Cold start: weights ready on the pod plus engine start, in "
        "seconds. Record: the newest boot's report; +N more records in the row (serving, "
        "task attempts, other boots).",
        "",
        _table(
            [
                "Model",
                "GPU",
                "vLLM",
                "Weights GiB",
                "Startup peak GiB",
                "KV pool tokens",
                "p99 TTFT ms (SLO)",
                "Tasks",
                "Cold start s",
                "Record",
            ],
            rows,
            NO_RECORDS,
        ),
    ]
    broken = table["broken_on_purpose"]
    if broken["boots"]:
        parts += [
            "",
            f"Not in the table: {broken['boots']} boots of plans broken on purpose to prove the "
            "six fixes; they are in the fixes table of the "
            f"[cohort findings]({fixes_href}).",
        ]
    if planned:
        parts += ["", "Planned, not run yet (predicted before any GPU; nothing measured):", ""]
        parts += [
            f"- {p['model']} on {_gpu_cell(p['gpu'], p['gpu_count'])}"
            + (f", vLLM {p['vllm']}" if p.get("vllm") else "")
            + f": {_prediction_text(p['predicted'])}"
            for p in planned
        ]
    if table["not_rendered"]:
        parts += [
            "",
            "Not rendered (no solution in the identity manifest): "
            + ", ".join(_d(d) for d in table["not_rendered"]),
        ]
    return "\n".join(parts)


def read_readme(text: str) -> str | None:
    m = _README.search(text)
    return m["body"] if m else None


def replace_readme(text: str, body: str) -> str:
    """Rewrite the README's records section; missing markers are an error."""
    if read_readme(text) is None:
        raise KeyError(f"no {README_START} ... {README_END} section")
    return _README.sub(lambda m: f"{README_START}\n{body}\n{README_END}", text, count=1)
