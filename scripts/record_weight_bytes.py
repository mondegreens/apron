"""Record the stored weight bytes of every model the cohort measured.

For each model with a healthy memory record, reads the safetensors headers
at the solution's resolved revision (range reads, no weights downloaded) and
writes the total, the ``lm_head.weight`` bytes and the tied-embeddings flag
to ``tests/fixtures/cohort/weight-bytes.json``.
``tests/unit/test_calculator_vs_cohort.py`` then checks the calculator's
weight prediction against every measured record with no network.

    uv run python scripts/record_weight_bytes.py
"""

from __future__ import annotations

import json
import sys

from apron.adapters.evidence.hf_hub import HFHubResolver
from apron.application.orchestration.plan_pipeline import _tied_embeddings
from apron.interfaces.cohort_root import REPO, load_cohort_run

OUT = REPO / "tests" / "fixtures" / "cohort" / "weight-bytes.json"


def main() -> int:
    run = load_cohort_run()
    rec = run.records
    resolver = HFHubResolver()
    models: dict[str, dict[str, object]] = {}
    for report in rec.reports.values():
        entry = rec.solutions.get(report.solution_fingerprint or "")
        if entry is None or report.claim_scope != "memory" or report.boot_outcome != "healthy":
            continue
        model_id = entry.model_id
        if model_id in models:
            continue
        revision = entry.model_spec.immutable_revision or "main"
        tensors = resolver._tensor_bytes(model_id, revision)
        config_raw = resolver._download_file(model_id, "config.json", revision)
        if not tensors or config_raw is None:
            print(f"{model_id}: no safetensors headers or config", file=sys.stderr)
            return 1
        models[model_id] = {
            "revision": revision,
            "total_bytes": sum(tensors.values()),
            "lm_head_bytes": tensors.get("lm_head.weight", 0),
            "tie_word_embeddings": _tied_embeddings(json.loads(config_raw)),
        }
        print(model_id, models[model_id])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(models, indent=2, sort_keys=True) + "\n")
    print(f"{len(models)} models -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
