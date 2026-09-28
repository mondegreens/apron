"""Predicted minus measured KV budget, per term, for every healthy memory boot.

vLLM's KV budget is ``requested - weights - torch peak - non-torch - CUDA-graph
estimate`` (calculator.py, "The KV budget"); every stored verification report
carries the measured terms.  For each healthy memory record this prints what the
calculator predicts for the plan as it booted (``cohort_root.recorded_prediction``:
recorded configs and weight bytes, no network) against the measurement, term by
term, and the compile segment vLLM may hold on top (``compile_segment_bytes``).
The table in ``_dev_notes/cohort-run/kv-budget-residuals.md`` is this output.

    uv run python scripts/kv_budget_residuals.py            # the table
    uv run python scripts/kv_budget_residuals.py --configs  # re-record the configs

``--configs`` downloads each checkpoint's config.json at the recorded revision
into ``tests/fixtures/cohort/configs.json`` and, for a multimodal wrapper (a
``vision_config`` / ``audio_config``), the processor files the planner reads
(``plan_pipeline.PROCESSOR_FILES``, null where the revision has none) into
``tests/fixtures/cohort/processors.json`` (the Hub; no weights).
"""

from __future__ import annotations

import json
import sys

from apron.interfaces.cohort_root import REPO, load_cohort_run, recorded_prediction

FIXTURES = REPO / "tests" / "fixtures" / "cohort"
GIB = 1 << 30


def record_configs() -> int:
    from apron.adapters.evidence.hf_hub import HFHubResolver
    from apron.application.orchestration.plan_pipeline import PROCESSOR_FILES
    from apron.domain.mechanisms.calculator import declares_towers

    weights = json.loads((FIXTURES / "weight-bytes.json").read_text())
    resolver = HFHubResolver()
    configs, processors = {}, {}
    for model_id, row in sorted(weights.items()):
        raw = resolver._download_file(model_id, "config.json", row["revision"])
        if raw is None:
            print(f"{model_id}: no config.json at {row['revision']}", file=sys.stderr)
            return 1
        configs[model_id] = json.loads(raw)
        if declares_towers(configs[model_id]):
            files = {
                name: resolver._download_file(model_id, name, row["revision"])
                for name, _ in PROCESSOR_FILES
            }
            processors[model_id] = {
                name: None if data is None else json.loads(data) for name, data in files.items()
            }
    for name, data in (("configs.json", configs), ("processors.json", processors)):
        (FIXTURES / name).write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
        print(f"{len(data)} models -> {FIXTURES / name}")
    return 0


def rows() -> list[dict[str, object]]:
    records = load_cohort_run().records
    configs = json.loads((FIXTURES / "configs.json").read_text())
    weights = json.loads((FIXTURES / "weight-bytes.json").read_text())
    out = []
    for digest, report in sorted(records.reports.items()):
        entry = records.solutions.get(report.solution_fingerprint or "")
        if entry is None or report.claim_scope != "memory" or report.boot_outcome != "healthy":
            continue
        p = recorded_prediction(entry, configs[entry.model_id], weights[entry.model_id])
        measured = {
            "requested": report.requested_memory or 0,
            "weights": report.model_weight_memory or 0,
            "peak": report.transient_peak_headroom or 0,
            "non_torch": report.non_pytorch_increase or 0,
            "cuda_graphs": report.cuda_graph_estimate or 0,
        }
        predicted = {
            "requested": p["gpu_available_bytes"],
            "weights": p["weight_memory_bytes"],
            "peak": p["activation_estimate_bytes"],
            "non_torch": p["non_pytorch_overhead_bytes"],
            "cuda_graphs": p["cuda_graph_estimate_bytes"],
        }
        # The KV budget's error, term by term (positive: more KV predicted).
        terms = {
            k: (predicted[k] - measured[k]) * (1 if k == "requested" else -1) for k in predicted
        }
        residual = p["available_kv_cache_bytes"] - (report.available_kv_cache_memory or 0)
        out.append(
            {
                "record": digest,
                "model": entry.model_id,
                "gpu": entry.requested_execution.gpu_sku,
                "tp": entry.deployment_plan.tensor_parallel,
                "dtype": entry.deployment_plan.dtype,
                "stored": (report.predicted_minus_measured or {}).get("kv_cache_consistency"),
                "residual": residual,
                "terms": terms,
                "compile_segment": p.get("compile_segment_bytes", 0),
                "rounding": residual - sum(terms.values()),
            }
        )
    return out


def main() -> int:
    if "--configs" in sys.argv[1:]:
        return record_configs()
    table = rows()
    head = (
        "| record | model | GPU | TP | stored | now | req | weights | peak | non-torch"
        " | graphs | segment |"
    )
    print(head)
    print("|" + "---|" * (head.count("|") - 1))
    for r in table:
        t = r["terms"]
        print(
            f"| {str(r['record'])[4:12]} | {r['model']} | {str(r['gpu']).replace('NVIDIA ', '')} "
            f"| {r['tp']} | {(r['stored'] or 0) / GIB:+.2f} | {r['residual'] / GIB:+.2f} "  # type: ignore[operator]
            + " ".join(f"| {t[k] / GIB:+.2f}" for k in t)  # type: ignore[index]
            + f" | {r['compile_segment'] / GIB:.2f} |"  # type: ignore[operator]
        )
    over = [
        r["residual"] - (r["compile_segment"] if r["residual"] > r["compile_segment"] / 2 else 0)  # type: ignore[operator]
        for r in table
    ]
    print(
        f"\n{len(table)} records; residual {min(r['residual'] for r in table) / GIB:+.3f} "  # type: ignore[type-var]
        f"to {max(r['residual'] for r in table) / GIB:+.3f} GiB; with a held compile "  # type: ignore[type-var]
        f"segment taken out: {min(over) / GIB:+.3f} to {max(over) / GIB:+.3f} GiB "  # type: ignore[type-var]
        f"(max {max(over):,} B)"  # type: ignore[type-var]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
