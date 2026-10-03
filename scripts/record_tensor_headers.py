"""Record config.json and the safetensors headers of models whose weight
prediction reads tensor names (MTP layers the loader drops, tables kept in
host memory, weights every tensor-parallel rank holds whole).

For each model, at its resolved revision: the config and every tensor's stored
dtype and bytes (range reads, no weights downloaded), with expert and n-gram
shard indices folded (``experts.*.``, ``shard_*``: bytes summed) to keep the
file small; nothing the planner matches on is folded.  Writes
``tests/fixtures/cohort/headers/<org>__<name>.json``, which
``tests/unit/test_modern_weight_prediction.py`` plans from with no network.

    uv run python scripts/record_tensor_headers.py [MODEL_ID ...]

Without arguments: the group C models of ``cohort/modern-models.json``.
"""

from __future__ import annotations

import json
import re
import sys

from apron.adapters.evidence.hf_hub import HFHubResolver
from apron.interfaces.cohort_root import REPO

OUT = REPO / "tests" / "fixtures" / "cohort" / "headers"
_FOLD = ((re.compile(r"\.experts\.\d+\."), ".experts.*."), (re.compile(r"shard_\d+"), "shard_*"))


def fold(meta: dict[str, tuple[str, int]]) -> dict[str, list[object]]:
    """Tensor name -> [dtype, bytes, tensors folded into it]."""
    out: dict[str, list[object]] = {}
    for name, (dtype, size) in sorted(meta.items()):
        key = name
        for pattern, repl in _FOLD:
            key = pattern.sub(repl, key)
        entry = out.setdefault(key, [dtype, 0, 0])
        if entry[0] != dtype:
            raise ValueError(f"{key}: folds {entry[0]} and {dtype}")
        entry[1] = int(entry[1]) + size  # type: ignore[call-overload]
        entry[2] = int(entry[2]) + 1  # type: ignore[call-overload]
    return out


def main(argv: list[str]) -> int:
    models = argv or [
        m["model_id"]
        for m in json.loads((REPO / "cohort" / "modern-models.json").read_text())["models"]
        if m["group"] == "C"
    ]
    resolver = HFHubResolver()
    OUT.mkdir(parents=True, exist_ok=True)
    for model_id in models:
        revision = resolver._get_model_info(model_id, None)["sha"]
        raw = resolver._download_file(model_id, "config.json", revision)
        meta = resolver._tensor_meta(model_id, revision)
        if raw is None or not meta:
            print(f"{model_id}: no config or safetensors headers", file=sys.stderr)
            return 1
        record = {
            "model_id": model_id,
            "revision": revision,
            "config": json.loads(raw),
            "tensors": fold(meta),
        }
        path = OUT / (model_id.replace("/", "__") + ".json")
        path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        total = sum(size for _, size in meta.values())
        print(f"{model_id}@{revision[:12]}: {len(meta)} tensors, {total:,} bytes -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
