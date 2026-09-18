"""Unit tests for vLLM EngineAdapter.

Collects the EngineAdapter conformance suite via pytest_plugins.
"""

from __future__ import annotations

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
    result = engine.classify("Unsupported task: 'summarize' for model 'Qwen3-8B'")
    assert result["failure_class"] == "engine_init"


def test_classify_max_model_len(engine: VllmEngineAdapter) -> None:
    result = engine.classify(
        "User-specified max_model_len (131072) is greater "
        "than the derived max_model_len (max_position_embeddings=32768)"
    )
    assert result["failure_class"] == "max_model_len"


def test_classify_dtype(engine: VllmEngineAdapter) -> None:
    result = engine.classify(
        "The model type 'Qwen3MoeForCausalLM' does not support float16. "
        "Reason: quantized models require bfloat16"
    )
    assert result["failure_class"] == "dtype_incompatible"


def test_classify_tp_divisibility(engine: VllmEngineAdapter) -> None:
    result = engine.classify(
        "Total number of attention heads (28) must be divisible by tensor parallel size (4)."
    )
    assert result["failure_class"] == "tp_divisibility"


def test_classify_oom_kv_cache(engine: VllmEngineAdapter) -> None:
    result = engine.classify(
        "cache is needed, which is larger than the available KV cache memory (6.20 GiB)."
    )
    assert result["failure_class"] == "oom"


def test_classify_quant_compute_capability(engine: VllmEngineAdapter) -> None:
    result = engine.classify(
        "The quantization method gptq "
        "is not supported for the current GPU. Minimum "
        "capability: 80. Current capability: 75."
    )
    assert result["failure_class"] == "quant_compute_capability"


def test_classify_returns_match_position(engine: VllmEngineAdapter) -> None:
    result = engine.classify("torch.cuda.OutOfMemoryError: CUDA error")
    assert "match_start" in result
    assert "match_end" in result
    assert result["match_start"] >= 0
    assert result["match_end"] > result["match_start"]


def test_classify_order_specificity(engine: VllmEngineAdapter) -> None:
    """OOM KV cache matches before dtype for messages containing 'not supported'."""
    error = (
        "cache is needed, which is larger than the available KV cache "
        "memory (6.20 GiB). dtype not supported"
    )
    assert engine.classify(error)["failure_class"] == "oom"


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
# extract
# ---------------------------------------------------------------------------


def test_extract_oom_kv_cache(engine: VllmEngineAdapter) -> None:
    error = (
        "To serve at least one request with the model's max seq len "
        "(32768), (14.50 GiB KV "
        "cache is needed, which is larger than the available KV cache "
        "memory (6.20 GiB). "
        "Based on the available memory, "
        "the estimated maximum model length is 8192."
    )
    extracted = engine.extract(error, "oom")
    assert extracted["estimated_max_model_len"] == 8192
    assert extracted["max_model_len"] == 32768
    assert abs(float(extracted["needed_gib"]) - 14.50) < 0.01
    assert abs(float(extracted["available_gib"]) - 6.20) < 0.01


def test_extract_oom_warmup(engine: VllmEngineAdapter) -> None:
    error = "CUDA out of memory occurred when warming up sampler with 256 dummy requests."
    extracted = engine.extract(error, "oom")
    assert extracted["max_num_seqs_attempted"] == 256


def test_extract_max_model_len(engine: VllmEngineAdapter) -> None:
    error = (
        "User-specified max_model_len (131072) is greater "
        "than the derived max_model_len (max_position_embeddings="
        "32768 or model_max_length=None in model's config.json)."
    )
    extracted = engine.extract(error, "max_model_len")
    assert extracted["requested"] == 131072
    assert extracted["derived_max"] == 32768
    assert extracted["max_len_key"] == "max_position_embeddings"


def test_extract_dtype_incompatible(engine: VllmEngineAdapter) -> None:
    error = (
        "The model type 'Qwen3MoeForCausalLM' does not support float16. "
        "Reason: quantized models require bfloat16"
    )
    extracted = engine.extract(error, "dtype_incompatible")
    assert extracted["model_type"] == "Qwen3MoeForCausalLM"
    assert extracted["unsupported_dtype"] == "float16"


def test_extract_tp_divisibility(engine: VllmEngineAdapter) -> None:
    error = "Total number of attention heads (28) must be divisible by tensor parallel size (4)."
    extracted = engine.extract(error, "tp_divisibility")
    assert extracted["num_heads"] == 28
    assert extracted["tp_size"] == 4


def test_extract_quant_compute_capability(engine: VllmEngineAdapter) -> None:
    error = (
        "The quantization method gptq "
        "is not supported for the current GPU. Minimum "
        "capability: 80. Current capability: 75."
    )
    extracted = engine.extract(error, "quant_compute_capability")
    assert extracted["method"] == "gptq"
    assert extracted["min_cap"] == 80
    assert extracted["cur_cap"] == 75


def test_extract_engine_init(engine: VllmEngineAdapter) -> None:
    error = "LoRA is not enabled. Use --enable-lora to enable LoRA."
    extracted = engine.extract(error, "engine_init")
    assert "suggested_fix" in extracted


def test_extract_unknown_returns_empty(engine: VllmEngineAdapter) -> None:
    extracted = engine.extract("something random", "unknown")
    assert extracted == {}
