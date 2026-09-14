"""Tests for vLLM constraint extraction from pinned source files."""

from pathlib import Path

from apron.adapters.backends.vllm_constraints import (
    extract_constraints,
    extract_kv_cache_specs,
    extract_supported_architectures,
    extract_task_registry,
)

FIXTURE_DIR = Path("tests/fixtures/external-formats/vllm")


def test_extract_supported_architectures():
    archs = extract_supported_architectures(FIXTURE_DIR / "supported_models.md")
    assert isinstance(archs, list)
    assert len(archs) > 0
    assert "Qwen2ForCausalLM" in archs
    assert "LlamaForCausalLM" in archs


def test_extract_task_registry():
    tasks = extract_task_registry(FIXTURE_DIR / "tasks.py")
    assert isinstance(tasks, list)
    assert "generate" in tasks
    assert "embed" in tasks
    assert "classify" in tasks


def test_extract_kv_cache_specs():
    specs = extract_kv_cache_specs(FIXTURE_DIR / "kv_cache_interface.py")
    assert isinstance(specs, list)
    assert "FullAttentionSpec" in specs
    assert "MLAAttentionSpec" in specs
    assert "SlidingWindowSpec" in specs


def test_extract_constraints_combined():
    constraints = extract_constraints(FIXTURE_DIR)
    assert "supported_architectures" in constraints
    assert "task_registry" in constraints
    assert "kv_cache_specs" in constraints
    assert "Qwen2ForCausalLM" in constraints["supported_architectures"]
    assert "generate" in constraints["task_registry"]


def test_missing_files_produce_partial_result(tmp_path):
    result = extract_constraints(tmp_path)
    assert isinstance(result, dict)
    assert "supported_architectures" not in result
