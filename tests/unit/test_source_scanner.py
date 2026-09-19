"""Tests for the vLLM source scanner."""

from __future__ import annotations

from pathlib import Path

import pytest

from apron.adapters.backends.source_scanner import (
    classify_error,
    derive_correction_spec,
    scan_source,
)

VLLM_SOURCE = Path(__file__).parents[2].parent / ".sources" / "vllm" / "vllm"


@pytest.fixture()
def scan_results() -> list[dict]:
    if not VLLM_SOURCE.is_dir():
        pytest.skip("Pinned vLLM source not available")
    return scan_source(VLLM_SOURCE)


class TestScanSource:
    def test_finds_known_tp_divisibility_site(self, scan_results: list[dict]) -> None:
        tp_sites = [
            e for e in scan_results if e["file"] == "config/model.py" and e["line"] == 1414
        ]
        assert len(tp_sites) == 1
        assert "divisible" in tp_sites[0]["message"].lower()

    def test_finds_known_max_model_len_site(self, scan_results: list[dict]) -> None:
        sites = [e for e in scan_results if e["file"] == "config/model.py" and e["line"] == 2517]
        assert len(sites) == 1
        assert "max_model_len" in sites[0]["message"]

    def test_finds_kv_cache_oom_site(self, scan_results: list[dict]) -> None:
        sites = [
            e
            for e in scan_results
            if e["file"] == "v1/core/kv_cache_utils.py" and e["line"] == 879
        ]
        assert len(sites) == 1

    def test_extracts_format_vars_from_fstrings(self, scan_results: list[dict]) -> None:
        tp_site = next(
            e for e in scan_results if e["file"] == "config/model.py" and e["line"] == 1414
        )
        assert "total_num_attention_heads" in tp_site["format_vars"]
        assert "tensor_parallel_size" in tp_site["format_vars"]

    def test_config_time_flag_for_config_dir(self, scan_results: list[dict]) -> None:
        config_entries = [e for e in scan_results if e["file"].startswith("config/")]
        assert all(e["is_config_time"] for e in config_entries)

    def test_total_count_is_reasonable(self, scan_results: list[dict]) -> None:
        assert len(scan_results) > 500


class TestClassifyError:
    def test_oom(self) -> None:
        entry = {"message": "CUDA out of memory during warmup", "file": "worker.py"}
        assert classify_error(entry) == "oom"

    def test_max_model_len(self) -> None:
        entry = {"message": "max_model_len (131072) is greater", "file": "config/model.py"}
        assert classify_error(entry) == "max_model_len"

    def test_tp_divisibility(self) -> None:
        entry = {
            "message": "must be divisible by tensor parallel size",
            "file": "config/model.py",
        }
        assert classify_error(entry) == "tp_divisibility"

    def test_dtype(self) -> None:
        entry = {"message": "dtype float16 is not supported", "file": "config/vllm.py"}
        assert classify_error(entry) == "dtype_incompatible"

    def test_speculative(self) -> None:
        entry = {"message": "Draft model not found", "file": "config/speculative.py"}
        assert classify_error(entry) == "speculative_config"

    def test_lora(self) -> None:
        entry = {"message": "LoRA adapter size exceeds limit", "file": "worker.py"}
        assert classify_error(entry) == "lora_config"

    def test_parallelism(self) -> None:
        entry = {
            "message": "pipeline_parallel_size must be 1",
            "file": "config/parallel.py",
        }
        assert classify_error(entry) == "parallelism_config"


class TestDeriveCorrectionSpec:
    def test_flag_enable(self) -> None:
        entry = {
            "message": "LoRA is not enabled. Use --enable-lora to enable LoRA.",
            "format_vars": [],
        }
        spec = derive_correction_spec(entry)
        assert spec["action"] == "set_field"
        assert spec["field"] == "enable_lora"
        assert spec["value"] == "true"

    def test_try_lowering(self) -> None:
        entry = {
            "message": "Try lowering max_num_seqs",
            "format_vars": ["max_num_seqs"],
        }
        spec = derive_correction_spec(entry)
        assert spec["action"] == "scale_field"
        assert spec["factor"] == 0.5

    def test_try_increasing(self) -> None:
        entry = {
            "message": "Try increasing gpu_memory_utilization",
            "format_vars": ["gpu_memory_utilization"],
        }
        spec = derive_correction_spec(entry)
        assert spec["action"] == "scale_field"
        assert spec["factor"] == 2.0

    def test_use_instead(self) -> None:
        entry = {"message": "Use bfloat16 instead of float16", "format_vars": []}
        spec = derive_correction_spec(entry)
        assert spec["action"] == "use_alternative"

    def test_fallback_for_unclear(self) -> None:
        entry = {
            "message": "Something must be configured properly",
            "format_vars": [],
        }
        spec = derive_correction_spec(entry)
        assert spec["action"] == "fallback"
