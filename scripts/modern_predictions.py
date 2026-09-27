"""GPU-free predictions for the owner-approved modern models (groups A-D).

For every model in ``cohort/modern-models.json`` on its proposed GPU and
count: the production plan pipeline (config and safetensors headers from the
Hub, the mechanism, the calculator, the load check), exactly as a cohort run
plans it — no pod, no paid call.  Writes
``_dev_notes/cohort-run/modern-predictions.json``: what Apron can say before
paying, and where it says "unknown".

    uv run python scripts/modern_predictions.py
"""

from __future__ import annotations

import json
import sys

from apron.application.orchestration.scheduler import CandidateSeed
from apron.interfaces.cohort_root import REPO, RUN_DIR, CohortPlanner

MODERN = REPO / "cohort" / "modern-models.json"
OUT = RUN_DIR / "modern-predictions.json"
GIB = 1 << 30


def main() -> int:
    planner = CohortPlanner(rates={})
    rows = []
    for m in json.loads(MODERN.read_text())["models"]:
        seed = CandidateSeed(
            model_id=m["model_id"],
            gpu_sku=m["gpu"],
            size_class="large",
            hardware_class="datacenter",
            mechanism="",
            weight_gb=0.0,
            gpu_count=int(m["gpu_count"]),
        )
        row = {
            "group": m["group"],
            "model": m["model_id"],
            "gpu": m["gpu"],
            "gpu_count": seed.gpu_count,
        }
        try:
            sp = planner.plan_seed(seed)
        except Exception as exc:  # reported per model, never hidden
            row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            rows.append(row)
            print(f"{m['model_id']}: ERROR {row['error'][:120]}")
            continue
        claim = dict(sp.claim.proposed_configuration) if sp.claim else {}
        row.update(
            status=sp.status,
            mechanism=sp.model_spec.components[0].mechanism,
            tensor_parallel=sp.plan.tensor_parallel,
            dtype=sp.plan.dtype,
            weight_gib_per_gpu=round((claim.get("weight_memory_bytes") or 0) / GIB, 2),
            activation_gib=round((claim.get("activation_estimate_bytes") or 0) / GIB, 2),
            total_gib_per_gpu=round((claim.get("total_required_bytes") or 0) / GIB, 2),
            load_problems=list(sp.load_problems or []),
        )
        rows.append(row)
        print(
            f"{m['group']} {m['model_id']:<48} {sp.status:<11} {row['mechanism']!s:<22} "
            f"TP{row['tensor_parallel']} w/GPU {row['weight_gib_per_gpu']} GiB "
            f"total {row['total_gib_per_gpu']} GiB"
        )
    OUT.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} models -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
