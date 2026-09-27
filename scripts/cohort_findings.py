"""Generate the cohort findings from stored records (PLAN §12).

Reads the run directory (records, identity manifest, ledger, events) and the
rule versions, then writes:

- ``docs/blog/posts/_data/cohort-findings.json`` — every number with the
  digests of the records it came from;
- the tables between ``<!-- findings:NAME -->`` markers in the post and in
  ``docs/blog/.linkedin-draft.md`` (a dotfile, so the site build skips it).

Nothing in those tables is typed by hand.  ``--check`` changes nothing and
exits 1 when the committed JSON or any table differs from the records, or
when a generated output matches a secret pattern (F6).

    uv run python scripts/cohort_findings.py [--check] [--run-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from apron.application.orchestration.findings import (
    build_findings,
    read_blocks,
    render_tables,
    replace_blocks,
)
from apron.application.sanitization import contains_secret
from apron.interfaces.cohort_root import AUTHORIZED_USD, REPO, RULES_DIR, RUN_DIR, load_cohort_run

DATA = REPO / "docs" / "blog" / "posts" / "_data" / "cohort-findings.json"
MODERN = REPO / "cohort" / "modern-models.json"
DOCUMENTS = (
    REPO / "docs" / "blog" / "posts" / "phase-1b-cohort-findings.md",
    REPO / "docs" / "blog" / "posts" / "modern-models-findings.md",
    REPO / "docs" / "blog" / ".linkedin-draft.md",
)


def generate(run_dir: Path, rules_dir: Path) -> tuple[str, dict[str, str]]:
    run = load_cohort_run(run_dir, rules_dir)
    if run.records.invalid or run.rule_errors:
        problems = [*(f"{w}: {why}" for w, why in run.records.invalid), *run.rule_errors]
        raise SystemExit("invalid records — fix them first:\n" + "\n".join(problems))
    modern = json.loads(MODERN.read_text()) if MODERN.exists() else None
    predictions = run_dir / "modern-predictions.json"
    if modern is not None and predictions.exists():
        # The GPU-free plan of each model (scripts/modern_predictions.py).
        by_model = {r["model"]: r for r in json.loads(predictions.read_text())["rows"]}
        modern = {
            **modern,
            "models": [{**m, "prediction": by_model.get(m["model_id"])} for m in modern["models"]],
        }
    findings = build_findings(run, authorized=AUTHORIZED_USD, modern=modern)
    data = json.dumps(findings, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return data, render_tables(findings)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--rules-dir", type=Path, default=RULES_DIR / "vllm-v0.29")
    args = parser.parse_args()

    data, blocks = generate(args.run_dir, args.rules_dir)
    outputs = {DATA: data}
    for path in DOCUMENTS:
        text = path.read_text("utf-8")
        if not read_blocks(text):
            print(f"{path.relative_to(REPO)}: no findings blocks", file=sys.stderr)
            return 1
        outputs[path] = replace_blocks(text, blocks)

    stale = [p for p, text in outputs.items() if not p.exists() or p.read_text("utf-8") != text]
    leaks = [p for p, text in outputs.items() if contains_secret(text)]
    for path in leaks:
        print(f"{path.relative_to(REPO)}: matches a secret pattern", file=sys.stderr)
    if args.check:
        for path in stale:
            print(f"{path.relative_to(REPO)}: differs from the records", file=sys.stderr)
        return 1 if stale or leaks else 0
    if leaks:
        return 1
    for path in stale:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(outputs[path], "utf-8")
        print(f"wrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
