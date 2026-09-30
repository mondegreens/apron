"""Record the stored weight bytes of every model the cohort measured.

For each model with a healthy memory record, reads the safetensors headers
at the solution's resolved revision (range reads, no weights downloaded) and
writes the total, the ``lm_head.weight`` bytes, the tied-embeddings flag and
the config fields the activation estimate reads to
``tests/fixtures/cohort/weight-bytes.json``.
``tests/unit/test_calculator_vs_cohort.py`` then checks the calculator's
weight prediction against every measured record with no network.  With
``--model`` only that model's row is recorded, the others kept as they are.

    uv run python scripts/record_weight_bytes.py
    uv run python scripts/record_weight_bytes.py --model zai-org/GLM-5.3-Flash
"""

from __future__ import annotations

import json
import sys

from apron.adapters.evidence.hf_hub import HFHubResolver
from apron.application.orchestration.plan_pipeline import (
    _tied_embeddings,
    download_processor_config,
    host_tensor_bytes,
    loaded_tensor_bytes,
    mtp_tensor_bytes,
    replicated_tensor_bytes,
    widened_tensor_bytes,
)
from apron.domain.mechanisms.calculator import activation_config
from apron.interfaces.cohort_root import REPO, load_cohort_run

OUT = REPO / "tests" / "fixtures" / "cohort" / "weight-bytes.json"


def main(argv: list[str]) -> int:
    only = argv[argv.index("--model") + 1] if "--model" in argv else None
    run = load_cohort_run()
    rec = run.records
    resolver = HFHubResolver()
    models: dict[str, dict[str, object]] = {}
    for report in rec.reports.values():
        entry = rec.solutions.get(report.solution_fingerprint or "")
        if entry is None or report.claim_scope != "memory" or report.boot_outcome != "healthy":
            continue
        model_id = entry.model_id
        if model_id in models or (only is not None and model_id != only):
            continue
        revision = entry.model_spec.immutable_revision or "main"
        meta = resolver._tensor_meta(model_id, revision)
        tensors = None if meta is None else {n: size for n, (_, size) in meta.items()}
        stored = dict(meta or {})
        config_raw = resolver._download_file(model_id, "config.json", revision)
        if not tensors or config_raw is None:
            print(f"{model_id}: no safetensors headers or config", file=sys.stderr)
            return 1
        config = json.loads(config_raw)
        # The planner's own reading of the processor files (run_plan_pipeline).
        processor = download_processor_config(resolver, model_id, revision)
        text = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
        models[model_id] = {
            "revision": revision,
            "total_bytes": sum(tensors.values()),
            "f32_bytes": sum(size for dtype, size in (meta or {}).values() if dtype == "F32"),
            "quantized": bool(config.get("quantization_config")),
            "lm_head_bytes": tensors.get("lm_head.weight", 0),
            "mtp_bytes": mtp_tensor_bytes(tensors, config),
            # Tables vLLM keeps in pinned host memory, not on a GPU
            # (plan_pipeline.host_tensor_bytes, at their loaded width).
            "host_bytes": host_tensor_bytes(loaded_tensor_bytes(stored, config), config),
            "tie_word_embeddings": _tied_embeddings(config),
            "vocab_size": config.get("vocab_size", text.get("vocab_size")),
            "hidden_size": config.get("hidden_size", text.get("hidden_size")),
            "torch_dtype": config.get("torch_dtype")
            or config.get("dtype")
            or text.get("torch_dtype")
            or text.get("dtype")
            or "bfloat16",
            # The config fields the startup-peak (activation) estimate reads.
            "activation": activation_config(config, processor),
            # Mamba-1 state shape fields (None for attention models).
            "ssm": {
                k: config[k]
                for k in ("num_hidden_layers", "intermediate_size", "state_size", "conv_kernel")
            }
            if all(config.get(k) for k in ("intermediate_size", "state_size", "conv_kernel"))
            and not config.get("num_attention_heads")
            else None,
            # What the totals above cannot carry: the model-specific load widths
            # and the weights every tensor-parallel rank holds whole, as the
            # planner resolves them (0 for the families not traced).
            "widened_bytes": widened_tensor_bytes(stored, config),
            "replicated_bytes": replicated_tensor_bytes(
                loaded_tensor_bytes(stored, config), config
            ),
        }
        print(model_id, models[model_id])
    if only is not None:
        if only not in models:
            print(f"{only}: no healthy memory record", file=sys.stderr)
            return 1
        models = {**json.loads(OUT.read_text()), **models}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(models, indent=2, sort_keys=True) + "\n")
    print(f"{len(models)} models -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
