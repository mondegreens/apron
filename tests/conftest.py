"""Shared test fixtures for diagnosis pipeline tests.

FakeDiagnosisEngine provides deterministic classify/extract results
without LLM calls, for testing the pipeline logic in isolation.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

# Fakes and suite inputs for the conformance suites (F9); see conformance/plugin.py.
pytest_plugins = ["conformance.plugin"]

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@pytest.fixture(autouse=True)
def _no_test_writes_real_run_artifacts(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point the real leaked-pods list at a temp file for every test.

    ``_dev_notes/cohort-run/leaked_pods.json`` is a safety artifact; a test
    that exercises a failed teardown must never append fake pods to it.
    """
    from apron.adapters.backends import runpod

    monkeypatch.setattr(
        runpod, "DEFAULT_LEAK_LOG", tmp_path_factory.mktemp("run") / "leaked_pods.json"
    )


@pytest.fixture(autouse=True)
def _isolate_migration_registry() -> Iterator[None]:
    """Restore the schema-migration registry; some tests clear it."""
    import apron.domain.schemas.records  # noqa: F401 — registers DiagnosisRule v3→v4
    from apron.domain.schemas import migrations

    saved = dict(migrations._REGISTRY)
    yield
    migrations._REGISTRY.clear()
    migrations._REGISTRY.update(saved)


@pytest.fixture(autouse=True)
def _isolate_calculator_registry() -> Iterator[None]:
    """Restore the global calculator registry after every test.

    Some tests clear it or register fakes; without this, a later test that
    runs the real plan pipeline would dispatch to a leftover fake.
    """
    import apron.domain.mechanisms.calculator  # noqa: F401 — registers real calculators
    from apron.domain import mechanisms

    saved = dict(mechanisms._CALCULATOR_REGISTRY)
    yield
    mechanisms._CALCULATOR_REGISTRY.clear()
    mechanisms._CALCULATOR_REGISTRY.update(saved)


def _bytes(text: str) -> int:
    value, unit = text.split()
    return int(float(value) * (1 << 30 if unit == "GiB" else 1 << 20))


class FakeDiagnosisEngine:
    """Deterministic engine for pipeline tests — no LLM calls.

    Uses simple keyword matching for classification and basic regex
    for extraction. NOT the production classifier — this exists only
    to make pipeline tests fast and deterministic.
    """

    _engine_name = "vllm"
    _engine_version = "v0.29.0"

    @property
    def engine_name(self) -> str:
        return self._engine_name

    @property
    def engine_version(self) -> str:
        return self._engine_version

    _KEYWORDS: ClassVar[list[tuple[str, str]]] = [
        ("when warming up sampler", "oom_kv_cache"),
        ("when warming up pooler", "oom_kv_cache"),
        ("Failed to load model - not enough GPU memory", "oom_weight_load"),
        ("OutOfMemoryError", "oom_weight_load"),
        ("CUDA out of memory", "oom_weight_load"),
        ("available KV cache memory", "oom_kv_cache"),
        ("No available memory for the cache blocks", "oom_kv_cache"),
        ("max_model_len", "max_model_len"),
        ("does not support float16", "dtype_incompatible"),
        ("not supported for quantization", "dtype_incompatible"),
        ("must be divisible by tensor parallel", "tp_divisibility"),
        ("not supported for the current GPU", "quant_compute_capability"),
        ("Minimum capability", "quant_compute_capability"),
        ("Unsupported task", "config_incompatible"),
        ("LoRA is not enabled", "lora_config"),
        ("no draft model is configured", "speculative_config"),
        ("draft checkpoint", "speculative_config"),
        ("num_speculative_tokens", "speculative_config"),
        ("Failed to infer device type", "config_incompatible"),
        ("cudagraph_capture_sizes", "config_incompatible"),
        ("compile_sizes", "config_incompatible"),
        ("load_format", "config_incompatible"),
        ("not compatible with", "config_incompatible"),
        ("must be one of", "config_incompatible"),
        ("context parallelism", "parallelism_config"),
        ("data_parallel", "parallelism_config"),
        ("tensor-parallel-size", "parallelism_config"),
        ("inductor_graph_partition", "compilation_config"),
        ("cuda graphs", "compilation_config"),
        ("custom_ops", "compilation_config"),
        ("nsight profiling", "profiler_config"),
        ("profiler", "profiler_config"),
        ("encoder_cudagraph_max_vision", "multimodal_config"),
        ("mm_processor", "multimodal_config"),
        ("kv_role", "kv_transfer_config"),
        ("kv_connector", "kv_transfer_config"),
        ("FLASH_ATTN requires", "platform_unsupported"),
        ("not supported on this platform", "platform_unsupported"),
        ("max_num_batched_tokens", "scheduler_config"),
        ("activation_sparsity", "model_runtime"),
        ("GeluAndMul", "model_runtime"),
        ("All options must be of the same type", "other_correctable"),
    ]

    _EXTRACTORS: ClassVar[dict[str, list[tuple[str, re.Pattern[str], Callable[[str], Any]]]]] = {
        "oom_weight_load": [
            ("tried_to_allocate_bytes", re.compile(r"Tried to allocate ([\d.]+ [GM]iB)"), _bytes),
            ("total_capacity_bytes", re.compile(r"total capacity of ([\d.]+ [GM]iB)"), _bytes),
            ("free_bytes", re.compile(r"of which ([\d.]+ [GM]iB) is free"), _bytes),
        ],
        "oom_kv_cache": [
            (
                "estimated_max_model_len",
                re.compile(r"estimated maximum model length is (\d+)"),
                int,
            ),
            (
                "max_num_seqs_attempted",
                re.compile(r"warming up (?:sampler|pooler)[^\d]*with\s*(\d+)"),
                int,
            ),
        ],
        "max_model_len": [
            ("derived_max", re.compile(r"derived max_model_len \([^=]+=(\d+)"), int),
        ],
        "dtype_incompatible": [
            ("model_type", re.compile(r"model type '(\w+)'"), str),
            ("unsupported_dtype", re.compile(r"does not support (\w+)"), str),
        ],
        "tp_divisibility": [
            ("num_heads", re.compile(r"attention heads \((\d+)\)"), int),
        ],
        "quant_compute_capability": [
            ("min_capability", re.compile(r"Minimum capability:\s*(\d+)"), int),
            ("current_capability", re.compile(r"Current capability:\s*(\d+)"), int),
        ],
        "lora_config": [
            (
                "suggested_fix",
                re.compile(r"(?:Use |please set )(.+?)(?:\.|$)", re.IGNORECASE),
                str,
            ),
        ],
    }

    def classify(self, error: str) -> dict[str, Any]:
        for keyword, failure_class in self._KEYWORDS:
            if keyword.lower() in error.lower():
                return {
                    "failure_class": failure_class,
                    "confidence": 0.95,
                    "evidence_span": keyword,
                }
        return {"failure_class": "unknown", "confidence": 0.0, "evidence_span": ""}

    def extract(self, error: str, failure_class: str) -> dict[str, int | float | str]:
        extractors = self._EXTRACTORS.get(failure_class)
        if extractors is None:
            return {}
        result: dict[str, int | float | str] = {}
        for key, pattern, converter in extractors:
            m = pattern.search(error)
            if m:
                result[key] = converter(m.group(1))
        return result

    def detect_engine_version(self, output: str) -> str | None:
        m = re.search(r"vLLM\s+v?(\d+\.\d+\.\d+)", output)
        return m.group(1) if m else None

    def resolve_support(self, model_spec: Any, image_tag: str) -> dict[str, Any]:
        return {
            "supported": True,
            "architecture": "",
            "tasks": [],
            "engine": "vllm",
            "version": "v0.29.0",
        }

    def validate(self, plan: Any, target: Any) -> list[str]:
        return []

    def render(self, plan: Any) -> dict[str, Any]:
        return {"args": [], "engine": "vllm"}

    def verify(self, plan: Any, target: Any, health_timeout: int = 600) -> dict[str, Any]:
        return {"initial_total_memory": 0}

    def extract_schema(self, image_tag: str) -> dict[str, Any]:
        return {"architectures": [], "tasks": [], "image_tag": image_tag}
