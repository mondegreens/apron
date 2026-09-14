"""Unit tests for vLLM EngineAdapter.

Collects the EngineAdapter conformance suite via pytest_plugins.
"""

from __future__ import annotations

import json
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
# classify
# ---------------------------------------------------------------------------


def test_classify_oom(engine: VllmEngineAdapter) -> None:
    result = engine.classify("torch.OutOfMemoryError: CUDA out of memory")
    assert result["failure_class"] == "oom"


def test_classify_engine_init(engine: VllmEngineAdapter) -> None:
    result = engine.classify("RuntimeError: Failed to initialize engine")
    assert result["failure_class"] == "engine_init"


def test_classify_max_model_len(engine: VllmEngineAdapter) -> None:
    result = engine.classify("ValueError: max_model_len exceeds maximum")
    assert result["failure_class"] == "max_model_len"


def test_classify_dtype(engine: VllmEngineAdapter) -> None:
    result = engine.classify("BFloat16 is not supported on this GPU")
    assert result["failure_class"] == "dtype_incompatible"


def test_classify_tp_divisibility(engine: VllmEngineAdapter) -> None:
    result = engine.classify("num_heads is not divisible by tensor_parallel")
    assert result["failure_class"] == "tp_divisibility"


def test_classify_unknown(engine: VllmEngineAdapter) -> None:
    result = engine.classify("something completely different happened")
    assert result["failure_class"] == "unknown"
    assert "raw_error" in result


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
    import json

    target = MagicMock()
    target.kind = "local-container"
    target.execution_fingerprint = "1220" + "ff" * 32

    pre_data = json.dumps({"pre_free": 23_000_000_000, "pre_total": 25_769_803_776})
    post_data = json.dumps({"post_free": 8_000_000_000, "post_total": 25_769_803_776})
    cg_data = json.dumps({"post_cg_free": 7_500_000_000, "post_cg_total": 25_769_803_776})

    call_count = [0]

    def fake_execute(cmd: str) -> dict[str, Any]:
        if "cat /workspace/apron_pre_load.json" in cmd:
            return {"stdout": pre_data, "stderr": "", "exit_code": 0}
        if "cat /workspace/apron_post_load.json" in cmd:
            return {"stdout": post_data, "stderr": "", "exit_code": 0}
        if "cat /workspace/apron_post_cuda_graph.json" in cmd:
            return {"stdout": cg_data, "stderr": "", "exit_code": 0}
        call_count[0] += 1
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
