"""Unit tests for vLLM EngineAdapter.

Collects the EngineAdapter conformance suite via pytest_plugins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from apron.adapters.backends.vllm_engine import VllmEngineAdapter
from apron.domain.protocols import EngineAdapter
from apron.domain.schemas.primitives import HardwareSpec
from apron.domain.schemas.solutions import DeploymentPlan

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine() -> VllmEngineAdapter:
    return VllmEngineAdapter()


class FakeTarget:
    @property
    def kind(self) -> str:
        return "local-container"

    @property
    def execution_fingerprint(self) -> str:
        return "1220" + "ee" * 32

    @property
    def hardware(self) -> HardwareSpec:
        return HardwareSpec(
            gpu_sku="RTX 4090",
            total_memory_bytes=25_769_803_776,
            compute_capability="8.9",
        )

    def execute(self, command: str) -> dict[str, Any]:
        return {"stdout": "", "stderr": "", "exit_code": 0}

    def collect(self, paths: list[str] | None = None) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_engine_adapter_protocol(engine: VllmEngineAdapter) -> None:
    assert isinstance(engine, EngineAdapter)


def test_has_engine_name(engine: VllmEngineAdapter) -> None:
    assert engine.engine_name == "vllm"


def test_has_engine_version(engine: VllmEngineAdapter) -> None:
    assert isinstance(engine.engine_version, str)
    assert engine.engine_version


# ---------------------------------------------------------------------------
# resolve_support
# ---------------------------------------------------------------------------


def test_resolve_support_dict_with_architecture(engine: VllmEngineAdapter) -> None:
    result = engine.resolve_support({"architecture": "Qwen2ForCausalLM"}, "vllm:0.29.0")
    assert isinstance(result, dict)
    assert "supported" in result
    assert isinstance(result["supported"], bool)
    assert "tasks" in result
    assert isinstance(result["tasks"], list)


def test_resolve_support_with_components(engine: VllmEngineAdapter) -> None:
    spec = {"components": [{"architecture": "LlamaForCausalLM"}]}
    result = engine.resolve_support(spec, "vllm:0.29.0")
    assert result["architecture"] == "LlamaForCausalLM"


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_validate_valid_plan(engine: VllmEngineAdapter) -> None:
    plan = DeploymentPlan(tensor_parallel=1)
    errors = engine.validate(plan, FakeTarget())
    assert errors == []


def test_validate_tp3_for_32_heads(engine: VllmEngineAdapter) -> None:
    plan = DeploymentPlan(
        tensor_parallel=3,
        engine_configuration={"num_attention_heads": "32", "num_kv_heads": "8"},
    )
    errors = engine.validate(plan, None)
    assert len(errors) > 0
    assert "TP=3" in errors[0]


def test_validate_tp2_for_32_heads(engine: VllmEngineAdapter) -> None:
    plan = DeploymentPlan(
        tensor_parallel=2,
        engine_configuration={"num_attention_heads": "32", "num_kv_heads": "8"},
    )
    errors = engine.validate(plan, None)
    assert errors == []


def test_validate_dtype_compute_capability(engine: VllmEngineAdapter) -> None:
    plan = DeploymentPlan(dtype="float16")
    low_cc_target = MagicMock()
    low_cc_target.hardware = HardwareSpec(
        gpu_sku="GTX 1080",
        total_memory_bytes=8_589_934_592,
        compute_capability="6.1",
    )
    errors = engine.validate(plan, low_cc_target)
    assert any("compute capability" in e for e in errors)


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def test_render_produces_args(engine: VllmEngineAdapter) -> None:
    plan = DeploymentPlan(
        tensor_parallel=4,
        dtype="bfloat16",
        engine_configuration={"gpu_memory_utilization": "0.90"},
    )
    result = engine.render(plan)
    assert isinstance(result, dict)
    assert "--tensor-parallel-size" in result["args"]
    assert "4" in result["args"]
    assert "--dtype" in result["args"]
    assert "bfloat16" in result["args"]


# ---------------------------------------------------------------------------
# extract_schema
# ---------------------------------------------------------------------------


def test_extract_schema_returns_dict(engine: VllmEngineAdapter) -> None:
    result = engine.extract_schema("vllm:0.29.0")
    assert isinstance(result, dict)
    assert "architectures" in result
    assert isinstance(result["architectures"], list)


# ---------------------------------------------------------------------------
# verify (with fake target)
# ---------------------------------------------------------------------------


def test_verify_returns_15_fields(engine: VllmEngineAdapter) -> None:
    target = MagicMock()
    target.kind = "local-container"
    target.execution_fingerprint = "1220" + "ff" * 32

    vllm_log = (
        "Memory profiling takes 2.50 seconds. "
        "Total non KV cache memory: 17.50GiB; "
        "torch peak memory increase: 1.20GiB; "
        "total consumed (from mem_get_info): 16.00GiB; "
        "weights memory: 15.30GiB.\n"
        "Available KV cache memory: 5.50 GiB\n"
        "CUDA graph pool memory: 0.50 GiB (actual), 0.80 GiB (estimated), "
        "difference: 0.30 GiB (37.5%).\n"
        "Free memory on device (22.50/24.00 GiB) on startup. "
        "Desired GPU memory utilization is (0.9, 20.25 GiB). "
        "Actual usage is 16.00 GiB for consumed memory (weights + non-torch), "
        "1.20 GiB for peak activation, and 0.50 GiB for CUDAGraph memory."
    )

    def fake_execute(cmd: str) -> dict[str, Any]:
        if "mem_get_info" in cmd:
            return {
                "stdout": '{"post_free":8000000000,"post_total":25769803776}',
                "stderr": "",
                "exit_code": 0,
            }
        if "/var/log/vllm.log" in cmd:
            return {"stdout": vllm_log, "stderr": "", "exit_code": 0}
        return {"stdout": "", "stderr": "", "exit_code": 0}

    target.execute.side_effect = fake_execute

    plan = DeploymentPlan(
        engine_configuration={"gpu_memory_utilization": "0.90", "max_model_len": "640"},
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(engine, "_wait_for_health", lambda *a, **kw: None)
        result = engine.verify(plan, target)

    expected_keys = {
        "model_weight_memory",
        "persistent_consumption",
        "transient_peak_headroom",
        "non_pytorch_increase",
        "cuda_graph_estimate",
        "cuda_graph_applied",
        "cuda_graph_actual",
        "available_kv_cache_memory",
        "safety_buffer",
        "initial_total_memory",
        "initial_free_memory",
        "requested_memory",
        "profiling_shape",
        "execution_fingerprint",
        "target_kind",
    }
    assert expected_keys.issubset(result.keys())
    assert result["model_weight_memory"] > 0
    assert result["initial_total_memory"] > 0
    assert result["cuda_graph_actual"] > 0
    assert result["available_kv_cache_memory"] > 0
    assert result["transient_peak_headroom"] > 0


def test_parse_profiling_logs_extracts_vllm_output(engine: VllmEngineAdapter) -> None:
    log_text = (
        "Memory profiling takes 2.50 seconds. "
        "Total non KV cache memory: 17.50GiB; "
        "torch peak memory increase: 1.20GiB; "
        "total consumed (from mem_get_info): 16.00GiB; "
        "weights memory: 15.30GiB.\n"
        "Available KV cache memory: 5.50 GiB\n"
        "CUDA graph pool memory: 0.50 GiB (actual), 0.80 GiB (estimated), "
        "difference: 0.30 GiB (37.5%).\n"
    )
    parsed = engine.parse_profiling_logs(log_text)

    gib = 1 << 30
    assert abs(parsed["weights_memory"] - 15.30 * gib) < 0.1 * gib
    assert abs(parsed["torch_peak_increase"] - 1.20 * gib) < 0.1 * gib
    assert abs(parsed["total_consumed"] - 16.00 * gib) < 0.1 * gib
    assert abs(parsed["available_kv_cache_gib"] - 5.50) < 0.01
    assert abs(parsed["cuda_graph_actual_gib"] - 0.50) < 0.01
    assert abs(parsed["cuda_graph_estimate_gib"] - 0.80) < 0.01


def test_parse_profiling_logs_empty(engine: VllmEngineAdapter) -> None:
    parsed = engine.parse_profiling_logs("")
    assert parsed == {}


# ---------------------------------------------------------------------------
# extraction_confidence
# ---------------------------------------------------------------------------


def test_extraction_confidence_full() -> None:
    from apron.adapters.backends.rule_loader import load_rules
    from apron.domain.diagnosis import build_extraction_schemas, extraction_confidence

    rules = load_rules(Path(__file__).parents[2] / "rules", "vllm", "v0.29")
    schemas = build_extraction_schemas(rules)
    tp_fields = {k for k, _ in schemas.get("tp_divisibility", [])}
    extracted = {k: 1 for k in tp_fields}
    assert extraction_confidence("tp_divisibility", extracted, schemas) == 1.0


def test_extraction_confidence_partial() -> None:
    from apron.adapters.backends.rule_loader import load_rules
    from apron.domain.diagnosis import build_extraction_schemas, extraction_confidence

    rules = load_rules(Path(__file__).parents[2] / "rules", "vllm", "v0.29")
    schemas = build_extraction_schemas(rules)
    tp_fields = [k for k, _ in schemas.get("tp_divisibility", [])]
    extracted = {tp_fields[0]: 1} if tp_fields else {}
    conf = extraction_confidence("tp_divisibility", extracted, schemas)
    assert 0.0 < conf < 1.0


def test_extraction_confidence_empty() -> None:
    from apron.adapters.backends.rule_loader import load_rules
    from apron.domain.diagnosis import build_extraction_schemas, extraction_confidence

    rules = load_rules(Path(__file__).parents[2] / "rules", "vllm", "v0.29")
    schemas = build_extraction_schemas(rules)
    assert extraction_confidence("tp_divisibility", {}, schemas) == 0.0


def test_extraction_confidence_unknown() -> None:
    from apron.domain.diagnosis import extraction_confidence

    assert extraction_confidence("unknown", {}) == 1.0


# ---------------------------------------------------------------------------
# detect_engine_version
# ---------------------------------------------------------------------------


def test_detect_version_from_output(engine: VllmEngineAdapter) -> None:
    output = "INFO 09-18 vLLM v0.29.0 starting on http://0.0.0.0:8000"
    assert engine.detect_engine_version(output) == "0.29.0"


def test_detect_version_missing(engine: VllmEngineAdapter) -> None:
    assert engine.detect_engine_version("no version info here") is None


def test_detect_version_different(engine: VllmEngineAdapter) -> None:
    output = "vLLM 0.30.1 loaded"
    assert engine.detect_engine_version(output) == "0.30.1"
