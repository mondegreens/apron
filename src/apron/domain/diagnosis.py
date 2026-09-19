"""Diagnosis domain — failure taxonomy and rule matching.

The failure taxonomy is engine-independent. OOM is OOM regardless of
whether vLLM or SGLang produced the error. Engine adapters classify
errors into this taxonomy; correction strategies consume it.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class FailureClass(Enum):
    OOM = "oom"
    MAX_MODEL_LEN = "max_model_len"
    DTYPE_INCOMPATIBLE = "dtype_incompatible"
    TP_DIVISIBILITY = "tp_divisibility"
    QUANT_COMPUTE_CAPABILITY = "quant_compute_capability"
    ENGINE_INIT = "engine_init"
    UNKNOWN = "unknown"


EXTRACTION_SCHEMAS: dict[str, list[tuple[str, type]]] = {
    "oom": [
        ("estimated_max_model_len", int),
        ("max_model_len", int),
        ("needed_gib", float),
        ("available_gib", float),
        ("max_num_seqs_attempted", int),
    ],
    "max_model_len": [
        ("requested", int),
        ("derived_max", int),
        ("max_len_key", str),
    ],
    "dtype_incompatible": [
        ("model_type", str),
        ("unsupported_dtype", str),
        ("supported_list", str),
    ],
    "tp_divisibility": [
        ("num_heads", int),
        ("tp_size", int),
    ],
    "quant_compute_capability": [
        ("method", str),
        ("min_cap", int),
        ("cur_cap", int),
    ],
    "engine_init": [
        ("suggested_fix", str),
    ],
}


def match_rule(failure_class: str, rules: list[dict[str, Any]]) -> dict[str, Any] | None:
    """First rule whose error_family matches failure_class."""
    for rule in rules:
        if rule.get("error_family") == failure_class:
            return rule
    return None


def extraction_confidence(failure_class: str, extracted: dict[str, Any]) -> float:
    """Fraction of expected fields that were actually extracted (0.0-1.0)."""
    schema = EXTRACTION_SCHEMAS.get(failure_class)
    if not schema:
        return 1.0
    expected = len(schema)
    found = sum(1 for key, _ in schema if key in extracted)
    return found / expected
