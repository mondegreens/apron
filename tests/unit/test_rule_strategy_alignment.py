"""§10.3: every key a strategy reads is declared in its rule, or comes from F5 config.

The six class rules declare typed fields only (INV-6: int / float / enum), and
every field cites the vLLM v0.29.0 source line that prints it.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pytest

from apron.adapters.backends.rule_loader import load_rules
from apron.application.orchestration.correction import (
    _STRATEGIES,
    STRATEGY_CONFIG_KEYS,
    STRATEGY_EXTRACTED_KEYS,
)
from apron.application.orchestration.diagnosis_pipeline import DIAGNOSIS_CONFIG_FIELDS

RULES = load_rules(Path(__file__).parents[2] / "rules", "vllm", "v0.29")
BY_FAMILY = {r["error_family"]: r for r in RULES}
# The pinned vLLM checkout (.sources/ is not committed); override with APRON_VLLM_SOURCE.
VLLM = Path(
    os.environ.get("APRON_VLLM_SOURCE", Path(__file__).parents[2] / ".sources" / "vllm" / "vllm")
)

SIX = {
    "oom_weight_load": "retarget_memory",
    "oom_kv_cache": "reduce_memory_pressure",
    "max_model_len": "clamp_max_model_len",
    "dtype_incompatible": "fallback_dtype",
    "tp_divisibility": "reduce_tensor_parallel",
    "quant_compute_capability": "retarget_capability",
}


@pytest.mark.parametrize(("family", "strategy"), sorted(SIX.items()))
def test_class_rule_uses_the_expected_strategy(family: str, strategy: str) -> None:
    assert BY_FAMILY[family]["correction_strategy"] == strategy
    assert strategy in _STRATEGIES


class _Recording(dict):  # type: ignore[type-arg]
    """A mapping that records every key a strategy looks up."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.read: set[str] = set()

    def __contains__(self, key: object) -> bool:
        self.read.add(str(key))
        return super().__contains__(key)

    def __getitem__(self, key: str) -> Any:
        self.read.add(key)
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        self.read.add(key)
        return super().get(key, default)


def _observed_reads(strategy: str, rule: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Run the real strategy twice and union what it looked up.

    Empty inputs run every "is this key present" check in order; seeded inputs
    (every declared field, and every F5 config field, filled) get past the
    early returns so the later lookups run too.
    """
    from apron.application.orchestration.correction import CatalogEntry, CorrectionContext
    from apron.domain.schemas.primitives import HardwareSpec
    from apron.domain.schemas.solutions import DeploymentPlan

    hw = HardwareSpec(gpu_sku="g", total_memory_bytes=24 << 30, compute_capability="9.0")
    target = HardwareSpec(gpu_sku="t", total_memory_bytes=179 << 30, compute_capability="10.0")
    context = CorrectionContext(
        catalog=(CatalogEntry(hw, 1.0), CatalogEntry(target, 2.0)),
        predicted_total_bytes=1 << 30,
    )
    plan = DeploymentPlan(
        tensor_parallel=4, resource_allocation={"gpu_sku": "g", "gpu_count": "4"}
    )
    seeded_extraction = {
        name: (spec.get("enum") or ["x"])[0] if spec["type"] in ("enum", "string") else 100
        for name, spec in rule["extraction_schema"].items()
    }
    seeded_config = {
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 4096,
        "model_type": "gemma2",
        "architectures": ["X"],
        "quantization_config": {"quant_method": "fp_quant"},
    }
    extraction_reads: set[str] = set()
    config_reads: set[str] = set()
    for extracted, config in (
        (_Recording(), _Recording()),
        (_Recording(seeded_extraction), _Recording(seeded_config)),
    ):
        _STRATEGIES[strategy](extracted, plan, config, hw, None, rule=rule, context=context)
        extraction_reads |= extracted.read
        config_reads |= config.read
    return extraction_reads, config_reads


def test_seeded_run_reaches_the_class6_config_lookup() -> None:
    """The early return on a missing minimum no longer hides later lookups."""
    _, config_reads = _observed_reads("retarget_capability", BY_FAMILY["quant_compute_capability"])
    assert "quantization_config" in config_reads


@pytest.mark.parametrize(("family", "strategy"), sorted(SIX.items()))
def test_every_key_a_strategy_reads_is_declared_or_supplied_by_config(
    family: str, strategy: str
) -> None:
    """Derived from execution, not from a hand-kept list: the extraction keys the
    strategy actually looks up are declared by the rule; its config lookups are
    F5 fields."""
    rule = BY_FAMILY[family]
    extracted_reads, config_reads = _observed_reads(strategy, rule)
    declared = set(rule["extraction_schema"])
    assert extracted_reads <= declared, (
        f"{strategy} reads {sorted(extracted_reads - declared)} not declared by {family}"
    )
    assert config_reads <= set(DIAGNOSIS_CONFIG_FIELDS) | {"num_kv_heads"}
    # the documented constants must match what the code does
    assert extracted_reads == set(STRATEGY_EXTRACTED_KEYS[strategy])
    assert config_reads - {"num_kv_heads"} <= set(STRATEGY_CONFIG_KEYS.get(strategy, set()))


@pytest.mark.parametrize("family", sorted(SIX))
def test_class_rule_fields_are_typed_and_cite_a_source_line(family: str) -> None:
    schema = BY_FAMILY[family]["extraction_schema"]
    for name, spec in schema.items():
        assert spec["type"] in ("int", "float", "enum"), f"{family}.{name} is {spec['type']}"
        if spec["type"] == "enum":
            assert spec["enum"], f"{family}.{name} enum without values"
        assert re.match(r"^[\w/]+\.py:\d+", spec["source"] or ""), f"{family}.{name} cites no line"


@pytest.mark.parametrize("family", sorted(SIX))
def test_declared_fields_are_read_by_the_strategy_or_are_evidence(family: str) -> None:
    """No field is declared that nothing reads — except the class 1 byte counts,
    which are kept as evidence of the allocator failure (§10.1)."""
    declared = set(BY_FAMILY[family]["extraction_schema"])
    reads = STRATEGY_EXTRACTED_KEYS[SIX[family]]
    evidence_only = {"tried_to_allocate_bytes", "total_capacity_bytes", "free_bytes"}
    evidence_only |= {"current_capability"}
    assert declared - reads <= evidence_only


# A distinctive fragment of the message that prints each field (PLAN §10.3).
FRAGMENTS = {
    ("oom_weight_load", "tried_to_allocate_bytes"): "original error",
    ("oom_weight_load", "total_capacity_bytes"): "original error",
    ("oom_weight_load", "free_bytes"): "original error",
    ("oom_kv_cache", "estimated_max_model_len"): "estimated maximum model length is",
    ("oom_kv_cache", "max_num_seqs_attempted"): "dummy requests",
    ("max_model_len", "derived_max"): "derived max_model_len",
    ("dtype_incompatible", "model_type"): "does not support float16",
    ("dtype_incompatible", "unsupported_dtype"): "does not support float16",
    ("tp_divisibility", "num_heads"): "Total number of attention heads",
    (
        "quant_compute_capability",
        "min_capability",
    ): "capability: {quant_config.get_min_capability()}",
    ("quant_compute_capability", "current_capability"): "Current capability",
}


@pytest.mark.skipif(not VLLM.is_dir(), reason="pinned vLLM source not available")
@pytest.mark.parametrize("family", sorted(SIX))
def test_cited_source_lines_print_the_field(family: str) -> None:
    """Each cited file:line (or the two lines after it, the rest of the message)
    contains that field's distinctive message fragment in the pinned source."""
    for name, spec in BY_FAMILY[family]["extraction_schema"].items():
        path, line = re.match(r"^([\w/]+\.py):(\d+)", spec["source"]).groups()  # type: ignore[union-attr]
        lines = (VLLM / path).read_text().splitlines()
        window = "\n".join(lines[int(line) - 2 : int(line) + 2])
        assert FRAGMENTS[(family, name)] in window, f"{family}.{name} @ {path}:{line}"
