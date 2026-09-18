"""Regex accuracy test — verify ERROR_PATTERNS match pinned vLLM v0.29.0 source.

Each test string is copied verbatim from the pinned source at
.sources/vllm/. The source file and line number are noted per test.
Zero-GPU, zero-cost verification. Layer 0 gate for Phase 1b diagnosis.
"""

from __future__ import annotations

import re

import pytest

from apron.adapters.backends.vllm_engine import ERROR_PATTERNS


def _classify(error: str) -> str:
    """Classify using the module-level patterns (not the adapter instance)."""
    for pattern, failure_class in ERROR_PATTERNS:
        if pattern.search(error):
            return failure_class
    return "unknown"


# -------------------------------------------------------------------
# OOM — torch CUDA OOM during weight loading
# Source: .sources/vllm/vllm/v1/worker/gpu_model_runner.py:5460
# -------------------------------------------------------------------


class TestOomWeightLoading:
    def test_torch_cuda_oom_exception(self) -> None:
        error = "torch.cuda.OutOfMemoryError: CUDA error: out of memory"
        assert _classify(error) == "oom"

    def test_weight_loading_message(self) -> None:
        error = (
            "Failed to load model - not enough GPU memory. "
            "Try lowering --gpu-memory-utilization to free memory for weights, "
            "increasing --tensor-parallel-size, or using --quantization. "
            "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            "for more tips. (original error: torch.cuda.OutOfMemoryError: "
            "CUDA error: out of memory)"
        )
        assert _classify(error) == "oom"


# -------------------------------------------------------------------
# OOM — CUDA OOM during sampler/pooler warmup
# Source: gpu_model_runner.py:6370-6376, 6473-6479
# -------------------------------------------------------------------


class TestOomWarmup:
    def test_sampler_warmup_oom(self) -> None:
        error = (
            "CUDA out of memory occurred when warming up sampler with "
            "256 dummy requests. Please try lowering "
            "`max_num_seqs` or `gpu_memory_utilization` when "
            "initializing the engine."
        )
        assert _classify(error) == "oom"

    def test_pooler_warmup_oom(self) -> None:
        error = (
            "CUDA out of memory occurred when warming up pooler "
            "(task='classify') with 256 dummy requests. Please try "
            "lowering `max_num_seqs` or `gpu_memory_utilization` when "
            "initializing the engine."
        )
        assert _classify(error) == "oom"


# -------------------------------------------------------------------
# OOM — KV cache memory insufficient (ValueError, not torch OOM)
# Source: .sources/vllm/vllm/v1/core/kv_cache_utils.py:858-866
# -------------------------------------------------------------------


class TestOomKvCacheNoMemory:
    def test_no_available_memory(self) -> None:
        error = (
            "No available memory for the cache blocks. "
            "Try increasing `gpu_memory_utilization` when initializing the engine "
            "(this flag also controls CPU memory reservation on the CPU "
            "backend, despite its name). "
            "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            "for more details."
        )
        assert _classify(error) == "oom"


# -------------------------------------------------------------------
# OOM — KV cache insufficient for max_model_len
# Source: kv_cache_utils.py:879-889
# -------------------------------------------------------------------


class TestOomKvCacheTooSmall:
    def test_kv_cache_larger_than_available(self) -> None:
        error = (
            "To serve at least one request with the model's max seq len "
            "(32768), (14.50 GiB KV "
            "cache is needed, which is larger than the available KV cache "
            "memory (6.20 GiB). "
            "Based on the available memory, "
            "the estimated maximum model length is 8192. "
            "Try increasing `gpu_memory_utilization` (which also controls "
            "CPU memory on the CPU backend) or decreasing `max_model_len` "
            "when initializing the engine. "
            "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            "for more details."
        )
        assert _classify(error) == "oom"

    def test_kv_cache_without_estimate(self) -> None:
        error = (
            "To serve at least one request with the model's max seq len "
            "(32768), (14.50 GiB KV "
            "cache is needed, which is larger than the available KV cache "
            "memory (6.20 GiB). "
            "Try increasing `gpu_memory_utilization` (which also controls "
            "CPU memory on the CPU backend) or decreasing `max_model_len` "
            "when initializing the engine. "
            "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            "for more details."
        )
        assert _classify(error) == "oom"


# -------------------------------------------------------------------
# max_model_len — user value exceeds model-derived maximum
# Source: .sources/vllm/vllm/config/model.py:2500-2519
# -------------------------------------------------------------------


class TestMaxModelLen:
    def test_user_specified_exceeds_derived(self) -> None:
        error = (
            "User-specified max_model_len (131072) is greater "
            "than the derived max_model_len (max_position_embeddings="
            "32768 or model_max_length="
            "None in model's config.json). "
            "To allow overriding this maximum, set "
            "the env var VLLM_ALLOW_LONG_MAX_MODEL_LEN=1. "
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN must be used with extreme "
            "caution."
        )
        assert _classify(error) == "max_model_len"


# -------------------------------------------------------------------
# dtype_incompatible — dtype not supported by model type
# Source: .sources/vllm/vllm/config/model.py:2258-2263
# -------------------------------------------------------------------


class TestDtypeIncompatible:
    def test_float16_not_supported(self) -> None:
        error = (
            "The model type 'Qwen3MoeForCausalLM' does not support float16. "
            "Reason: quantized models require bfloat16"
        )
        assert _classify(error) == "dtype_incompatible"

    def test_quant_dtype_not_supported(self) -> None:
        # Source: config/vllm.py:799-802
        error = (
            "torch.float16 is not supported for quantization "
            "method awq. Supported dtypes: "
            "{torch.bfloat16}"
        )
        assert _classify(error) == "dtype_incompatible"


# -------------------------------------------------------------------
# tp_divisibility — attention heads not divisible by TP size
# Source: .sources/vllm/vllm/config/model.py:1414-1418
# -------------------------------------------------------------------


class TestTpDivisibility:
    def test_attention_heads_not_divisible(self) -> None:
        error = (
            "Total number of attention heads (28)"
            " must be divisible by tensor parallel size "
            "(4)."
        )
        assert _classify(error) == "tp_divisibility"


# -------------------------------------------------------------------
# quant_compute_capability — GPU too old for quantization method
# Source: .sources/vllm/vllm/config/vllm.py:791-796
# -------------------------------------------------------------------


class TestQuantComputeCapability:
    def test_gpu_below_min_capability(self) -> None:
        error = (
            "The quantization method gptq "
            "is not supported for the current GPU. Minimum "
            "capability: 80. "
            "Current capability: 75."
        )
        assert _classify(error) == "quant_compute_capability"


# -------------------------------------------------------------------
# engine_init — correctable configuration RuntimeErrors
# -------------------------------------------------------------------


class TestEngineInit:
    def test_unsupported_task(self) -> None:
        # Source: config/model.py:1783
        error = "Unsupported task: 'summarize' for model 'Qwen3-8B'"
        assert _classify(error) == "engine_init"

    def test_lora_not_enabled(self) -> None:
        # Source: lora_model_runner_mixin.py:87
        error = "LoRA is not enabled. Use --enable-lora to enable LoRA."
        assert _classify(error) == "engine_init"

    def test_no_draft_model(self) -> None:
        # Source: gpu_worker.py:961
        error = (
            "Draft model weight update requested, but no draft model is configured."
        )
        assert _classify(error) == "engine_init"

    def test_device_type_infer_failure(self) -> None:
        # Source: config/device.py:56
        error = (
            "RuntimeError: Failed to infer device type, please set "
            "the environment variable `VLLM_LOGGING_LEVEL=DEBUG` "
            "to turn on verbose logging to help debug the issue."
        )
        assert _classify(error) == "engine_init"


# -------------------------------------------------------------------
# unknown — uncaught RuntimeErrors should NOT match engine_init
# -------------------------------------------------------------------


class TestUnknownDoesNotMatchEngineInit:
    def test_nccl_timeout_is_unknown(self) -> None:
        error = "RuntimeError: NCCL communicator was aborted on rank 0 due to timeout"
        assert _classify(error) == "unknown"

    def test_executor_crash_is_unknown(self) -> None:
        error = "RuntimeError: Worker process died unexpectedly"
        assert _classify(error) == "unknown"

    def test_socket_thread_death_is_unknown(self) -> None:
        error = "RuntimeError: Socket thread exited unexpectedly"
        assert _classify(error) == "unknown"


# -------------------------------------------------------------------
# Specificity — more-specific patterns match before less-specific
# -------------------------------------------------------------------


class TestClassifySpecificity:
    def test_kv_cache_oom_before_dtype(self) -> None:
        """KV cache OOM containing 'not supported' should NOT match dtype."""
        error = (
            "cache is needed, which is larger than the available KV cache "
            "memory (6.20 GiB). dtype not supported in this context."
        )
        assert _classify(error) == "oom"

    def test_oom_before_engine_init(self) -> None:
        """OOM during warmup (RuntimeError) should match oom, not engine_init."""
        error = (
            "CUDA out of memory occurred when warming up sampler with "
            "256 dummy requests."
        )
        assert _classify(error) == "oom"
