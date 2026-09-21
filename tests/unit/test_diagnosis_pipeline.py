"""Tests for the generic diagnosis pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import FakeDiagnosisEngine

from apron.adapters.backends.rule_loader import load_rules
from apron.application.orchestration.diagnosis_pipeline import run_diagnosis_pipeline
from apron.domain.schemas.primitives import HardwareSpec
from apron.domain.schemas.solutions import DeploymentPlan


@pytest.fixture()
def engine() -> FakeDiagnosisEngine:
    return FakeDiagnosisEngine()


@pytest.fixture()
def hardware() -> HardwareSpec:
    return HardwareSpec(
        gpu_sku="RTX 4090",
        total_memory_bytes=25_769_803_776,
        compute_capability="8.9",
    )


@pytest.fixture()
def base_plan() -> DeploymentPlan:
    return DeploymentPlan(
        tensor_parallel=4,
        dtype="bfloat16",
        engine_configuration={
            "gpu_memory_utilization": "0.90",
            "max_model_len": "32768",
            "max_num_seqs": "256",
        },
    )


@pytest.fixture()
def model_config() -> dict:
    return {"num_attention_heads": 32, "num_kv_heads": 8}


@pytest.fixture()
def rules() -> list[dict]:
    rules_dir = Path(__file__).parents[2] / "rules"
    return load_rules(rules_dir, "vllm", "v0.29.0")


# -------------------------------------------------------------------
# Pipeline per class
# -------------------------------------------------------------------


class TestPipelinePerClass:
    def test_oom_kv_cache(
        self,
        engine: FakeDiagnosisEngine,
        base_plan: DeploymentPlan,
        model_config: dict,
        hardware: HardwareSpec,
        rules: list[dict],
    ) -> None:
        error = (
            "To serve at least one request with the model's max seq len "
            "(32768), (14.50 GiB KV "
            "cache is needed, which is larger than the available KV cache "
            "memory (6.20 GiB). "
            "Based on the available memory, "
            "the estimated maximum model length is 8192."
        )
        result = run_diagnosis_pipeline(error, engine, base_plan, model_config, hardware, rules)
        assert result.failure_class == "oom"
        assert result.rule_matched
        assert result.corrected_plan is not None
        assert result.corrected_plan.engine_configuration["max_model_len"] == "8192"
        assert result.result_label in ("Corrected", "Alternative with trade-offs")

    def test_oom_warmup(
        self,
        engine: FakeDiagnosisEngine,
        base_plan: DeploymentPlan,
        model_config: dict,
        hardware: HardwareSpec,
        rules: list[dict],
    ) -> None:
        error = "CUDA out of memory occurred when warming up sampler with 256 dummy requests."
        result = run_diagnosis_pipeline(error, engine, base_plan, model_config, hardware, rules)
        assert result.failure_class == "oom"
        assert result.corrected_plan is not None
        assert result.corrected_plan.engine_configuration["max_num_seqs"] == "128"

    def test_max_model_len(
        self,
        engine: FakeDiagnosisEngine,
        base_plan: DeploymentPlan,
        model_config: dict,
        hardware: HardwareSpec,
        rules: list[dict],
    ) -> None:
        error = (
            "User-specified max_model_len (131072) is greater "
            "than the derived max_model_len (max_position_embeddings="
            "32768 or model_max_length=None in model's config.json)."
        )
        result = run_diagnosis_pipeline(error, engine, base_plan, model_config, hardware, rules)
        assert result.failure_class == "max_model_len"
        assert result.corrected_plan is not None
        assert result.corrected_plan.engine_configuration["max_model_len"] == "32768"

    def test_dtype_incompatible(
        self,
        engine: FakeDiagnosisEngine,
        base_plan: DeploymentPlan,
        model_config: dict,
        hardware: HardwareSpec,
        rules: list[dict],
    ) -> None:
        error = (
            "The model type 'Qwen3MoeForCausalLM' does not support float16. "
            "Reason: quantized models require bfloat16"
        )
        result = run_diagnosis_pipeline(error, engine, base_plan, model_config, hardware, rules)
        assert result.failure_class == "dtype_incompatible"
        assert result.corrected_plan is not None
        assert result.corrected_plan.dtype in ("bfloat16", "float16")

    def test_tp_divisibility(
        self,
        engine: FakeDiagnosisEngine,
        base_plan: DeploymentPlan,
        model_config: dict,
        hardware: HardwareSpec,
        rules: list[dict],
    ) -> None:
        error = (
            "Total number of attention heads (28) must be divisible by tensor parallel size (4)."
        )
        result = run_diagnosis_pipeline(error, engine, base_plan, model_config, hardware, rules)
        assert result.failure_class == "tp_divisibility"
        assert result.corrected_plan is not None
        assert 28 % result.corrected_plan.tensor_parallel == 0

    def test_quant_compute_capability(
        self,
        engine: FakeDiagnosisEngine,
        hardware: HardwareSpec,
        model_config: dict,
        rules: list[dict],
    ) -> None:
        plan = DeploymentPlan(
            engine_configuration={"quantization": "gptq", "max_model_len": "4096"},
        )
        error = (
            "The quantization method gptq "
            "is not supported for the current GPU. Minimum "
            "capability: 80. Current capability: 75."
        )
        result = run_diagnosis_pipeline(error, engine, plan, model_config, hardware, rules)
        assert result.failure_class == "quant_compute_capability"
        assert result.corrected_plan is not None
        assert "quantization" not in result.corrected_plan.engine_configuration


# -------------------------------------------------------------------
# Edge cases
# -------------------------------------------------------------------


class TestPipelineEdgeCases:
    def test_unknown_error_no_correction(
        self,
        engine: FakeDiagnosisEngine,
        base_plan: DeploymentPlan,
        model_config: dict,
        hardware: HardwareSpec,
        rules: list[dict],
    ) -> None:
        result = run_diagnosis_pipeline(
            "RuntimeError: NCCL timeout",
            engine,
            base_plan,
            model_config,
            hardware,
            rules,
        )
        assert result.failure_class == "unknown"
        assert not result.rule_matched
        assert result.corrected_plan is None
        assert result.error_trace

    def test_corrected_plan_is_valid_schema(
        self,
        engine: FakeDiagnosisEngine,
        base_plan: DeploymentPlan,
        model_config: dict,
        hardware: HardwareSpec,
        rules: list[dict],
    ) -> None:
        error = (
            "To serve at least one request with the model's max seq len "
            "(32768), (14.50 GiB KV cache is needed, which is larger "
            "than the available KV cache memory (6.20 GiB). "
            "Based on the available memory, "
            "the estimated maximum model length is 8192."
        )
        result = run_diagnosis_pipeline(error, engine, base_plan, model_config, hardware, rules)
        assert result.corrected_plan is not None
        assert isinstance(result.corrected_plan, DeploymentPlan)
        assert result.corrected_plan.schema_version == 1

    def test_original_plan_not_mutated(
        self,
        engine: FakeDiagnosisEngine,
        base_plan: DeploymentPlan,
        model_config: dict,
        hardware: HardwareSpec,
        rules: list[dict],
    ) -> None:
        original_config = dict(base_plan.engine_configuration)
        run_diagnosis_pipeline(
            "CUDA out of memory occurred when warming up sampler with 256 dummy requests.",
            engine,
            base_plan,
            model_config,
            hardware,
            rules,
        )
        assert base_plan.engine_configuration == original_config

    def test_infeasible_correction_returns_none(
        self,
        engine: FakeDiagnosisEngine,
        model_config: dict,
        rules: list[dict],
    ) -> None:
        """Weight OOM on a GPU too small for the model → infeasible."""
        small_gpu = HardwareSpec(
            gpu_sku="RTX 3060",
            total_memory_bytes=12_884_901_888,
            compute_capability="8.6",
        )
        plan = DeploymentPlan(engine_configuration={"max_model_len": "4096"})
        vr = {"model_weight_memory": 20_000_000_000}
        error = "torch.cuda.OutOfMemoryError: CUDA error: out of memory"
        result = run_diagnosis_pipeline(error, engine, plan, model_config, small_gpu, rules, vr)
        assert result.failure_class == "oom"
        assert result.rule_matched
        assert result.corrected_plan is None
        assert result.result_label == "Correction infeasible"

    def test_serving_degraded_labels_tradeoff(
        self,
        engine: FakeDiagnosisEngine,
        hardware: HardwareSpec,
        model_config: dict,
        rules: list[dict],
    ) -> None:
        """Correction that reduces max_model_len gets trade-off label."""
        plan = DeploymentPlan(
            engine_configuration={"max_model_len": "32768", "max_num_seqs": "256"},
        )
        error = (
            "To serve at least one request with the model's max seq len "
            "(32768), (14.50 GiB KV cache is needed, which is larger "
            "than the available KV cache memory (6.20 GiB). "
            "Based on the available memory, "
            "the estimated maximum model length is 8192."
        )
        result = run_diagnosis_pipeline(error, engine, plan, model_config, hardware, rules)
        assert result.result_label == "Alternative with trade-offs"
        assert result.corrected_plan is not None

    def test_corrects_carries_original_digest(
        self,
        engine: FakeDiagnosisEngine,
        base_plan: DeploymentPlan,
        model_config: dict,
        hardware: HardwareSpec,
        rules: list[dict],
    ) -> None:
        error = "CUDA out of memory occurred when warming up sampler with 256 dummy requests."
        result = run_diagnosis_pipeline(error, engine, base_plan, model_config, hardware, rules)
        assert result.corrects is not None
        assert len(result.corrects) == 68
        assert result.corrects.startswith("1220")


class TestIteration:
    def test_cycle_detection(
        self,
        engine: FakeDiagnosisEngine,
        base_plan: DeploymentPlan,
        model_config: dict,
        hardware: HardwareSpec,
        rules: list[dict],
    ) -> None:
        from apron.application.orchestration.diagnosis_pipeline import iterate_diagnosis

        same_error = "CUDA out of memory occurred when warming up sampler with 256 dummy requests."
        result = iterate_diagnosis(
            [same_error, same_error],
            engine,
            base_plan,
            model_config,
            hardware,
            rules,
        )
        assert result.result_label == "Cycle detected"
        assert result.corrected_plan is None

    def test_iteration_applies_corrected_plan(
        self,
        engine: FakeDiagnosisEngine,
        model_config: dict,
        hardware: HardwareSpec,
        rules: list[dict],
    ) -> None:
        from apron.application.orchestration.diagnosis_pipeline import iterate_diagnosis

        error1 = (
            "To serve at least one request with the model's max seq len "
            "(32768), (14.50 GiB KV cache is needed, which is larger "
            "than the available KV cache memory (6.20 GiB). "
            "Based on the available memory, "
            "the estimated maximum model length is 8192."
        )
        error2 = "The model type 'Qwen3MoeForCausalLM' does not support float16."
        plan = DeploymentPlan(
            dtype="float16",
            engine_configuration={"max_model_len": "32768"},
        )
        result = iterate_diagnosis(
            [error1, error2],
            engine,
            plan,
            model_config,
            hardware,
            rules,
        )
        assert result.corrected_plan is not None
        assert result.corrected_plan.dtype == "bfloat16"
