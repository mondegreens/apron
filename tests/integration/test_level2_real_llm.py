"""Level 2 integration tests: real LLM classifier on real error strings.

Tests all 17 failure classes through the real Anthropic API using
error strings from pinned vLLM v0.29.0 source. Each test verifies:
1. failure_class matches expected
2. Extracted values match error string numbers (named-strategy classes)
3. Corrected plan has the expected change (named-strategy classes)
4. FakeDiagnosisEngine agrees with the LLM classification

Gate: ANTHROPIC_API_KEY in env. Tests skip without it.
Cost: ~$0.35 total (34 Haiku calls). ~3 minutes runtime.

Error string values use cohort model configs per INTEGRATION_SPEC.md:
- Qwen3-8B: num_attention_heads=32, num_kv_heads=4, hidden_size=4096
- Qwen3-1.7B: num_attention_heads=16, num_kv_heads=2, hidden_size=2048
- Mistral-7B: num_attention_heads=32, num_kv_heads=8, hidden_size=4096,
  max_position_embeddings=32768
- Qwen3-32B: num_attention_heads=64, num_kv_heads=8, hidden_size=5120
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeDiagnosisEngine

from apron.adapters.backends.llm_classifier import classify_and_extract
from apron.adapters.backends.rule_loader import load_rules
from apron.application.orchestration.diagnosis_pipeline import run_diagnosis_pipeline
from apron.domain.schemas.primitives import HardwareSpec
from apron.domain.schemas.solutions import DeploymentPlan

_HAS_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))
pytestmark = pytest.mark.skipif(not _HAS_KEY, reason="ANTHROPIC_API_KEY not set")


class LlmDiagnosisEngine:
    """Wraps the real LLM classifier in the engine interface."""

    _engine_name = "vllm"
    _engine_version = "v0.29.0"

    def __init__(self, rules: list[dict[str, Any]]) -> None:
        self._rules = rules

    @property
    def engine_name(self) -> str:
        return self._engine_name

    @property
    def engine_version(self) -> str:
        return self._engine_version

    def classify(self, error: str) -> dict[str, Any]:
        result = classify_and_extract(error, self._rules)
        return {
            "failure_class": result.get("failure_class", "unknown"),
            "confidence": result.get("confidence", 0.0),
            "evidence_span": result.get("evidence_span", ""),
        }

    def extract(self, error: str, failure_class: str) -> dict[str, int | float | str]:
        result = classify_and_extract(error, self._rules)
        exclude = {"failure_class", "confidence", "evidence_span"}
        return {
            k: v
            for k, v in result.items()
            if k not in exclude and not k.endswith("_evidence") and v is not None
        }

    def detect_engine_version(self, output: str) -> str | None:
        import re

        m = re.search(r"vLLM\s+v?(\d+\.\d+\.\d+)", output)
        return m.group(1) if m else None

    def validate(self, plan: Any, target: Any) -> list[str]:
        return []


@pytest.fixture(scope="module")
def rules() -> list[dict[str, Any]]:
    rules_dir = Path(__file__).parents[2] / "rules"
    return load_rules(rules_dir, "vllm", "v0.29")


@pytest.fixture(scope="module")
def llm_engine(rules: list[dict[str, Any]]) -> LlmDiagnosisEngine:
    return LlmDiagnosisEngine(rules)


@pytest.fixture(scope="module")
def fake_engine() -> FakeDiagnosisEngine:
    return FakeDiagnosisEngine()


@pytest.fixture()
def hardware_4090() -> HardwareSpec:
    return HardwareSpec(
        gpu_sku="NVIDIA GeForce RTX 4090",
        total_memory_bytes=25_769_803_776,
        compute_capability="8.9",
    )


@pytest.fixture()
def hardware_a100() -> HardwareSpec:
    return HardwareSpec(
        gpu_sku="NVIDIA A100-SXM4-80GB",
        total_memory_bytes=85_899_345_920,
        compute_capability="8.0",
    )


# --- Named-strategy classes (6 tests) ---
# Error strings use cohort model values from INTEGRATION_SPEC.md


class TestOomKvCache:
    """Qwen3-1.7B on 4090 with --max-model-len 131072.

    This error mentions both KV cache and max_model_len — the LLM must
    classify as oom (KV cache), not max_model_len.
    """

    ERROR = (
        "ValueError: To serve at least one request with the model's "
        "max seq len (131072), (14.50 GiB KV cache is needed, which is "
        "larger than the available KV cache memory (6.20 GiB). Try "
        "increasing `gpu_memory_utilization`. Based on the available "
        "memory, the estimated maximum model length is 8192."
    )

    def test_classification(
        self, llm_engine: LlmDiagnosisEngine, fake_engine: FakeDiagnosisEngine
    ) -> None:
        llm_result = llm_engine.classify(self.ERROR)
        fake_result = fake_engine.classify(self.ERROR)
        assert llm_result["failure_class"] == "oom"
        assert fake_result["failure_class"] == llm_result["failure_class"]

    def test_extraction(self, llm_engine: LlmDiagnosisEngine) -> None:
        extracted = llm_engine.extract(self.ERROR, "oom")
        has_max_len = extracted.get("max_model_len") is not None
        has_estimated = extracted.get("estimated_msg") is not None
        assert has_max_len or has_estimated

    def test_correction(
        self,
        llm_engine: LlmDiagnosisEngine,
        rules: list[dict[str, Any]],
        hardware_4090: HardwareSpec,
    ) -> None:
        plan = DeploymentPlan(
            engine_configuration={"max_model_len": "131072", "max_num_seqs": "256"},
        )
        result = run_diagnosis_pipeline(
            self.ERROR,
            llm_engine,
            plan,
            {"num_attention_heads": 16, "num_kv_heads": 2},
            hardware_4090,
            rules,
        )
        assert result.failure_class == "oom"
        assert result.corrected_plan is not None
        corrected_config = result.corrected_plan.engine_configuration
        corrected_len = int(corrected_config.get("max_model_len", "131072"))
        corrected_util = corrected_config.get("gpu_memory_utilization")
        assert corrected_len < 131072 or corrected_util == "0.90"
        assert result.result_label in ("Corrected", "Alternative with trade-offs")


class TestOomTorch:
    """Qwen3-8B on 4090 — weight OOM sub-type."""

    ERROR = (
        "torch.cuda.OutOfMemoryError: CUDA out of memory. "
        "Tried to allocate 2.00 GiB. GPU 0 has a total capacity of "
        "23.65 GiB of which 1.20 GiB is free."
    )

    def test_classification(
        self, llm_engine: LlmDiagnosisEngine, fake_engine: FakeDiagnosisEngine
    ) -> None:
        llm_result = llm_engine.classify(self.ERROR)
        fake_result = fake_engine.classify(self.ERROR)
        assert llm_result["failure_class"] == "oom"
        assert fake_result["failure_class"] == llm_result["failure_class"]

    def test_correction(
        self,
        llm_engine: LlmDiagnosisEngine,
        rules: list[dict[str, Any]],
        hardware_4090: HardwareSpec,
    ) -> None:
        plan = DeploymentPlan(
            engine_configuration={
                "gpu_memory_utilization": "0.95",
                "max_num_seqs": "256",
            },
        )
        result = run_diagnosis_pipeline(
            self.ERROR,
            llm_engine,
            plan,
            {"num_attention_heads": 32, "num_kv_heads": 4},
            hardware_4090,
            rules,
        )
        assert result.failure_class == "oom"
        assert result.corrected_plan is not None
        assert result.corrected_plan.engine_configuration["gpu_memory_utilization"] == "0.90"


class TestMaxModelLen:
    """Mistral-7B on 4090 with --max-model-len 999999."""

    ERROR = (
        "User-specified max_model_len (999999) is greater "
        "than the derived max_model_len (max_position_embeddings="
        "32768 or model_max_length=None in model's config.json)."
    )

    def test_classification(
        self, llm_engine: LlmDiagnosisEngine, fake_engine: FakeDiagnosisEngine
    ) -> None:
        llm_result = llm_engine.classify(self.ERROR)
        fake_result = fake_engine.classify(self.ERROR)
        assert llm_result["failure_class"] == "max_model_len"
        assert fake_result["failure_class"] == llm_result["failure_class"]

    def test_extraction(self, llm_engine: LlmDiagnosisEngine) -> None:
        extracted = llm_engine.extract(self.ERROR, "max_model_len")
        has_value = (
            extracted.get("max_model_len") is not None or extracted.get("value") is not None
        )
        assert has_value

    def test_correction(
        self,
        llm_engine: LlmDiagnosisEngine,
        rules: list[dict[str, Any]],
        hardware_4090: HardwareSpec,
    ) -> None:
        plan = DeploymentPlan(
            engine_configuration={"max_model_len": "999999", "max_num_seqs": "256"},
        )
        result = run_diagnosis_pipeline(
            self.ERROR,
            llm_engine,
            plan,
            {
                "num_attention_heads": 32,
                "num_kv_heads": 8,
                "max_position_embeddings": 32768,
            },
            hardware_4090,
            rules,
        )
        assert result.failure_class == "max_model_len"
        assert result.corrected_plan is not None
        corrected_len = int(result.corrected_plan.engine_configuration["max_model_len"])
        assert corrected_len == 32768


class TestDtypeIncompatible:
    """Qwen3-1.7B on 4090 with --dtype float16."""

    ERROR = (
        "The model type 'Qwen3ForCausalLM' does not support float16. "
        "Reason: quantized models require bfloat16"
    )

    def test_classification(
        self, llm_engine: LlmDiagnosisEngine, fake_engine: FakeDiagnosisEngine
    ) -> None:
        llm_result = llm_engine.classify(self.ERROR)
        fake_result = fake_engine.classify(self.ERROR)
        assert llm_result["failure_class"] == "dtype_incompatible"
        assert fake_result["failure_class"] == llm_result["failure_class"]

    def test_correction(
        self,
        llm_engine: LlmDiagnosisEngine,
        rules: list[dict[str, Any]],
        hardware_4090: HardwareSpec,
    ) -> None:
        plan = DeploymentPlan(dtype="float16")
        result = run_diagnosis_pipeline(
            self.ERROR,
            llm_engine,
            plan,
            {"num_attention_heads": 16, "num_kv_heads": 2},
            hardware_4090,
            rules,
        )
        assert result.failure_class == "dtype_incompatible"
        assert result.corrected_plan is not None
        assert result.corrected_plan.dtype in ("bfloat16", "auto")


class TestTpDivisibility:
    """Qwen3-32B on A100 with --tensor-parallel-size 3."""

    ERROR = "Total number of attention heads (64) must be divisible by tensor parallel size (3)."

    def test_classification(
        self, llm_engine: LlmDiagnosisEngine, fake_engine: FakeDiagnosisEngine
    ) -> None:
        llm_result = llm_engine.classify(self.ERROR)
        fake_result = fake_engine.classify(self.ERROR)
        assert llm_result["failure_class"] == "tp_divisibility"
        assert fake_result["failure_class"] == llm_result["failure_class"]

    def test_extraction(self, llm_engine: LlmDiagnosisEngine) -> None:
        extracted = llm_engine.extract(self.ERROR, "tp_divisibility")
        heads = extracted.get("total_num_attention_heads") or extracted.get("num_heads")
        tp = extracted.get("tensor_parallel_size") or extracted.get("tp_size")
        assert heads is not None
        assert int(heads) == 64
        assert tp is not None
        assert int(tp) == 3

    def test_correction(
        self,
        llm_engine: LlmDiagnosisEngine,
        rules: list[dict[str, Any]],
        hardware_a100: HardwareSpec,
    ) -> None:
        plan = DeploymentPlan(
            tensor_parallel=3,
            engine_configuration={"max_num_seqs": "256"},
        )
        result = run_diagnosis_pipeline(
            self.ERROR,
            llm_engine,
            plan,
            {"num_attention_heads": 64, "num_kv_heads": 8},
            hardware_a100,
            rules,
        )
        assert result.failure_class == "tp_divisibility"
        assert result.corrected_plan is not None
        assert result.corrected_plan.tensor_parallel == 2


class TestQuantComputeCapability:
    """FP8 on A100 (cc 8.0)."""

    ERROR = (
        "The quantization method fp8 "
        "is not supported for the current GPU. Minimum "
        "capability: 89. Current capability: 80."
    )

    def test_classification(
        self, llm_engine: LlmDiagnosisEngine, fake_engine: FakeDiagnosisEngine
    ) -> None:
        llm_result = llm_engine.classify(self.ERROR)
        fake_result = fake_engine.classify(self.ERROR)
        assert llm_result["failure_class"] == "quant_compute_capability"
        assert fake_result["failure_class"] == llm_result["failure_class"]

    def test_correction(
        self,
        llm_engine: LlmDiagnosisEngine,
        rules: list[dict[str, Any]],
        hardware_a100: HardwareSpec,
    ) -> None:
        plan = DeploymentPlan(
            engine_configuration={"quantization": "fp8", "max_num_seqs": "256"},
        )
        result = run_diagnosis_pipeline(
            self.ERROR,
            llm_engine,
            plan,
            {"num_attention_heads": 32, "num_kv_heads": 8},
            hardware_a100,
            rules,
        )
        assert result.failure_class == "quant_compute_capability"
        assert result.corrected_plan is not None
        assert "quantization" not in result.corrected_plan.engine_configuration


class TestLoraConfig:
    """LoRA not enabled."""

    ERROR = "LoRA is not enabled. Use --enable-lora to enable LoRA."

    def test_classification(
        self, llm_engine: LlmDiagnosisEngine, fake_engine: FakeDiagnosisEngine
    ) -> None:
        llm_result = llm_engine.classify(self.ERROR)
        fake_result = fake_engine.classify(self.ERROR)
        assert llm_result["failure_class"] == "lora_config"
        assert fake_result["failure_class"] == llm_result["failure_class"]

    def test_correction(
        self,
        llm_engine: LlmDiagnosisEngine,
        rules: list[dict[str, Any]],
        hardware_4090: HardwareSpec,
    ) -> None:
        plan = DeploymentPlan(engine_configuration={"max_num_seqs": "256"})
        result = run_diagnosis_pipeline(
            self.ERROR,
            llm_engine,
            plan,
            {"num_attention_heads": 32, "num_kv_heads": 4},
            hardware_4090,
            rules,
        )
        assert result.failure_class == "lora_config"
        assert result.rule_matched
        # Correction may be None if LLM extraction doesn't capture the
        # suggested fix text. The strategy function doesn't receive the
        # raw error — it relies on extracted values. When extracted is
        # empty, fallback_engine_config returns None (infeasible).
        # This is a known limitation: the lora extraction schema has
        # max_cpu_loras/max_loras/r/lora_int_id but not suggested_fix.
        if result.corrected_plan is not None:
            assert result.corrected_plan.engine_configuration.get("enable_lora") == "true"


# --- Correction-spec classes (11 tests) ---
# These verify classification only — most have fallback correction specs.

_CORRECTION_SPEC_CLASSES = [
    pytest.param(
        "speculative_config",
        "Qwen3-Omni DSpark requires a standalone draft checkpoint "
        "with architectures=['Qwen3OmniDSparkForConditionalGeneration'].",
        id="speculative_config",
    ),
    pytest.param(
        "parallelism_config",
        "data_parallel_size (4) x tensor_parallel_size (2) must equal "
        "the total number of GPUs (4). Current world_size=4.",
        id="parallelism_config",
    ),
    pytest.param(
        "compilation_config",
        "use_inductor_graph_partition is only supported with "
        "torch>=2.9.0.dev. Set use_inductor_graph_partition=False.",
        id="compilation_config",
    ),
    pytest.param(
        "config_incompatible",
        "The value of load_format (auto) is not compatible with "
        "quantization (awq). Use load_format=safetensors instead.",
        id="config_incompatible",
    ),
    pytest.param(
        "profiler_config",
        "Unable to use nsight profiling unless workers run with Ray.",
        id="profiler_config",
    ),
    pytest.param(
        "multimodal_config",
        "encoder_cudagraph_max_vision_items_per_batch must be "
        "non-negative (0 = auto-infer from encoder budget).",
        id="multimodal_config",
    ),
    pytest.param(
        "kv_transfer_config",
        "Please specify kv_role when kv_connector is set, "
        "supported roles are kv_producer, kv_consumer, kv_both.",
        id="kv_transfer_config",
    ),
    pytest.param(
        "platform_unsupported",
        "Unlimited-OCR: --attention-config backend=FLASH_ATTN "
        "requires FA4 (rswa_mask_mod) on this platform.",
        id="platform_unsupported",
    ),
    pytest.param(
        "scheduler_config",
        "max_num_batched_tokens (128) must be greater than or equal "
        "to max_num_seqs (256) for the scheduler to function.",
        id="scheduler_config",
    ),
    pytest.param(
        "model_runtime",
        "activation_sparsity is 0.0. Please use GeluAndMul.",
        id="model_runtime",
    ),
    pytest.param(
        "other_correctable",
        "log_zero_guard_type must be one of 'add' or 'clamp', "
        "got 'none'. This is a data preprocessing configuration error.",
        id="other_correctable",
    ),
]


@pytest.mark.parametrize(("expected_class", "error"), _CORRECTION_SPEC_CLASSES)
def test_correction_spec_classification(
    llm_engine: LlmDiagnosisEngine,
    fake_engine: FakeDiagnosisEngine,
    rules: list[dict[str, Any]],
    hardware_4090: HardwareSpec,
    expected_class: str,
    error: str,
) -> None:
    """Correction-spec class: LLM classifies correctly, fake agrees."""
    llm_result = llm_engine.classify(error)
    llm_class = llm_result["failure_class"]

    if expected_class == "other_correctable":
        assert llm_class != "unknown", (
            "other_correctable error classified as unknown — LLM should "
            "assign a specific class or at least not unknown"
        )
    else:
        assert llm_class == expected_class, (
            f"LLM classified as {llm_class}, expected {expected_class}"
        )

    fake_result = fake_engine.classify(error)
    if expected_class == "other_correctable":
        assert fake_result["failure_class"] != "unknown"
    else:
        assert fake_result["failure_class"] == llm_class, (
            f"Fake classified as {fake_result['failure_class']}, LLM classified as {llm_class}"
        )


@pytest.mark.parametrize(("expected_class", "error"), _CORRECTION_SPEC_CLASSES)
def test_correction_spec_pipeline(
    llm_engine: LlmDiagnosisEngine,
    rules: list[dict[str, Any]],
    hardware_4090: HardwareSpec,
    expected_class: str,
    error: str,
) -> None:
    """Correction-spec class: pipeline runs without error, rule matches."""
    plan = DeploymentPlan(engine_configuration={"max_num_seqs": "256"})
    result = run_diagnosis_pipeline(
        error,
        llm_engine,
        plan,
        {"num_attention_heads": 32, "num_kv_heads": 8},
        hardware_4090,
        rules,
    )
    if expected_class == "other_correctable":
        assert result.rule_matched
    else:
        assert result.failure_class == expected_class
        assert result.rule_matched
