"""Classification accuracy test — verify FakeDiagnosisEngine classifies
actual vLLM v0.29.0 error strings correctly.

These tests verify the test fake against the same error strings the
production LLM classifier must handle. If the fake can't classify them,
neither can the pipeline tests that use it.
"""

from __future__ import annotations

from tests.conftest import FakeDiagnosisEngine


def _classify(error: str) -> str:
    engine = FakeDiagnosisEngine()
    return engine.classify(error)["failure_class"]


class TestOom:
    def test_torch_cuda_oom(self) -> None:
        assert _classify("torch.cuda.OutOfMemoryError: CUDA error: out of memory") == "oom"

    def test_warmup_oom(self) -> None:
        assert (
            _classify(
                "CUDA out of memory occurred when warming up sampler with 256 dummy requests."
            )
            == "oom"
        )

    def test_kv_cache_oom(self) -> None:
        error = "cache is needed, which is larger than the available KV cache memory (6.20 GiB)."
        assert _classify(error) == "oom"

    def test_no_available_memory(self) -> None:
        error = "No available memory for the cache blocks."
        assert _classify(error) == "oom"


class TestMaxModelLen:
    def test_user_specified_exceeds_derived(self) -> None:
        error = "User-specified max_model_len (131072) is greater than the derived max_model_len"
        assert _classify(error) == "max_model_len"


class TestDtypeIncompatible:
    def test_float16_not_supported(self) -> None:
        error = "The model type 'Qwen3MoeForCausalLM' does not support float16."
        assert _classify(error) == "dtype_incompatible"

    def test_quant_dtype_not_supported(self) -> None:
        error = "torch.float16 is not supported for quantization method awq."
        assert _classify(error) == "dtype_incompatible"


class TestTpDivisibility:
    def test_attention_heads_not_divisible(self) -> None:
        error = (
            "Total number of attention heads (28) must be divisible by tensor parallel size (4)."
        )
        assert _classify(error) == "tp_divisibility"


class TestQuantComputeCapability:
    def test_gpu_below_min_capability(self) -> None:
        error = (
            "The quantization method gptq is not supported for the "
            "current GPU. Minimum capability: 80."
        )
        assert _classify(error) == "quant_compute_capability"


class TestRefinedClasses:
    def test_lora_not_enabled(self) -> None:
        assert _classify("LoRA is not enabled. Use --enable-lora to enable LoRA.") == "lora_config"

    def test_no_draft_model(self) -> None:
        assert (
            _classify("Draft model weight update requested, but no draft model is configured.")
            == "speculative_config"
        )

    def test_device_type_infer_failure(self) -> None:
        assert _classify("Failed to infer device type") == "config_incompatible"


class TestUnknown:
    def test_nccl_timeout(self) -> None:
        assert _classify("RuntimeError: NCCL communicator was aborted") == "unknown"

    def test_random_text(self) -> None:
        assert _classify("something completely different") == "unknown"
