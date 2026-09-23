"""Tests for generic correction function."""

from __future__ import annotations

import pytest

from apron.application.orchestration.correction import (
    CORRECTION_BOUNDS,
    compute_correction,
)
from apron.domain.schemas.primitives import HardwareSpec
from apron.domain.schemas.solutions import DeploymentPlan


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
    return {
        "num_attention_heads": 32,
        "num_kv_heads": 8,
    }


# -------------------------------------------------------------------
# reduce_memory_pressure (OOM)
# -------------------------------------------------------------------


class TestReduceMemoryPressure:
    def test_kv_cache_oom_uses_estimated_max(
        self, base_plan: DeploymentPlan, hardware: HardwareSpec, model_config: dict
    ) -> None:
        extracted = {"estimated_max_model_len": 8192, "max_model_len": 32768}
        result = compute_correction(
            "reduce_memory_pressure", extracted, base_plan, model_config, hardware, None
        )
        assert result is not None
        assert result.engine_configuration["max_model_len"] == "8192"

    def test_warmup_oom_halves_max_num_seqs(
        self, base_plan: DeploymentPlan, hardware: HardwareSpec, model_config: dict
    ) -> None:
        extracted = {"max_num_seqs_attempted": 256}
        result = compute_correction(
            "reduce_memory_pressure", extracted, base_plan, model_config, hardware, None
        )
        assert result is not None
        assert result.engine_configuration["max_num_seqs"] == "128"

    def test_weight_oom_fallback_gpu_util(
        self, base_plan: DeploymentPlan, hardware: HardwareSpec, model_config: dict
    ) -> None:
        extracted: dict = {}
        result = compute_correction(
            "reduce_memory_pressure", extracted, base_plan, model_config, hardware, None
        )
        assert result is not None
        assert result.engine_configuration["gpu_memory_utilization"] == "0.90"

    def test_halved_max_num_seqs_never_zero(
        self, base_plan: DeploymentPlan, hardware: HardwareSpec, model_config: dict
    ) -> None:
        extracted = {"max_num_seqs_attempted": 1}
        result = compute_correction(
            "reduce_memory_pressure", extracted, base_plan, model_config, hardware, None
        )
        assert result is not None
        assert int(result.engine_configuration["max_num_seqs"]) >= 1


# -------------------------------------------------------------------
# clamp_max_model_len
# -------------------------------------------------------------------


class TestClampMaxModelLen:
    def test_clamps_to_derived_max(
        self, base_plan: DeploymentPlan, hardware: HardwareSpec, model_config: dict
    ) -> None:
        extracted = {"requested": 131072, "derived_max": 32768}
        result = compute_correction(
            "clamp_max_model_len", extracted, base_plan, model_config, hardware, None
        )
        assert result is not None
        assert result.engine_configuration["max_model_len"] == "32768"

    def test_fallback_when_no_derived(
        self, base_plan: DeploymentPlan, hardware: HardwareSpec, model_config: dict
    ) -> None:
        extracted: dict = {"requested": 131072}
        result = compute_correction(
            "clamp_max_model_len", extracted, base_plan, model_config, hardware, None
        )
        assert result is not None
        assert result.engine_configuration["max_model_len"] == "4096"


# -------------------------------------------------------------------
# fallback_dtype
# -------------------------------------------------------------------


class TestFallbackDtype:
    def test_low_cc_falls_to_float16(self, base_plan: DeploymentPlan, model_config: dict) -> None:
        low_cc = HardwareSpec(
            gpu_sku="GTX 1080",
            total_memory_bytes=8_589_934_592,
            compute_capability="6.1",
        )
        result = compute_correction("fallback_dtype", {}, base_plan, model_config, low_cc, None)
        assert result is not None
        assert result.dtype == "float16"

    def test_float16_unsupported_falls_to_bf16(
        self, base_plan: DeploymentPlan, hardware: HardwareSpec, model_config: dict
    ) -> None:
        extracted = {"unsupported_dtype": "float16"}
        result = compute_correction(
            "fallback_dtype", extracted, base_plan, model_config, hardware, None
        )
        assert result is not None
        assert result.dtype == "bfloat16"

    def test_high_cc_gpu_gets_bfloat16(
        self, base_plan: DeploymentPlan, model_config: dict
    ) -> None:
        blackwell = HardwareSpec(
            gpu_sku="B200",
            total_memory_bytes=196_608_000_000,
            compute_capability="10.0",
        )
        result = compute_correction("fallback_dtype", {}, base_plan, model_config, blackwell, None)
        assert result is not None
        assert result.dtype == "bfloat16"

    def test_blocklisted_model_gets_bfloat16(
        self, base_plan: DeploymentPlan, hardware: HardwareSpec, model_config: dict
    ) -> None:
        extracted = {"model_type": "gemma2", "unsupported_dtype": "float16"}
        result = compute_correction(
            "fallback_dtype", extracted, base_plan, model_config, hardware, None
        )
        assert result is not None
        assert result.dtype == "bfloat16"

    def test_quant_first_supported(
        self, base_plan: DeploymentPlan, hardware: HardwareSpec, model_config: dict
    ) -> None:
        extracted = {"supported_list": "{torch.bfloat16, torch.float32}"}
        result = compute_correction(
            "fallback_dtype", extracted, base_plan, model_config, hardware, None
        )
        assert result is not None
        assert result.dtype == "bfloat16"


# -------------------------------------------------------------------
# reduce_tensor_parallel
# -------------------------------------------------------------------


class TestReduceTensorParallel:
    def test_finds_largest_valid_tp(
        self, base_plan: DeploymentPlan, hardware: HardwareSpec, model_config: dict
    ) -> None:
        extracted = {"num_heads": 28, "tp_size": 4}
        result = compute_correction(
            "reduce_tensor_parallel", extracted, base_plan, model_config, hardware, None
        )
        assert result is not None
        assert 28 % result.tensor_parallel == 0
        assert result.tensor_parallel < 4

    def test_falls_to_1_when_no_divisor(self, hardware: HardwareSpec, model_config: dict) -> None:
        plan = DeploymentPlan(tensor_parallel=2)
        extracted = {"num_heads": 7, "tp_size": 2}
        result = compute_correction(
            "reduce_tensor_parallel",
            extracted,
            plan,
            {**model_config, "num_kv_heads": 7},
            hardware,
            None,
        )
        assert result is not None
        assert result.tensor_parallel == 1

    def test_missing_num_heads_uses_model_config(
        self, base_plan: DeploymentPlan, hardware: HardwareSpec
    ) -> None:
        extracted: dict = {}
        mc = {"num_attention_heads": 28, "num_kv_heads": 4}
        result = compute_correction(
            "reduce_tensor_parallel", extracted, base_plan, mc, hardware, None
        )
        assert result is not None
        assert 28 % result.tensor_parallel == 0

    def test_no_heads_anywhere_returns_infeasible(
        self, base_plan: DeploymentPlan, hardware: HardwareSpec
    ) -> None:
        result = compute_correction("reduce_tensor_parallel", {}, base_plan, {}, hardware, None)
        assert result is None

    def test_engine_init_uncorrectable_returns_infeasible(
        self, base_plan: DeploymentPlan, hardware: HardwareSpec, model_config: dict
    ) -> None:
        extracted = {"suggested_fix": "set VLLM_LOGGING_LEVEL=DEBUG"}
        result = compute_correction(
            "fallback_engine_config", extracted, base_plan, model_config, hardware, None
        )
        assert result is None


# -------------------------------------------------------------------
# remove_quantization
# -------------------------------------------------------------------


class TestRemoveQuantization:
    def test_removes_quantization_key(self, hardware: HardwareSpec, model_config: dict) -> None:
        plan = DeploymentPlan(
            engine_configuration={"quantization": "gptq", "max_model_len": "4096"},
        )
        result = compute_correction("remove_quantization", {}, plan, model_config, hardware, None)
        assert result is not None
        assert "quantization" not in result.engine_configuration
        assert result.engine_configuration["max_model_len"] == "4096"


# -------------------------------------------------------------------
# fallback_engine_config
# -------------------------------------------------------------------


class TestFallbackEngineConfig:
    def test_lora_fix_adds_enable_lora(
        self, base_plan: DeploymentPlan, hardware: HardwareSpec, model_config: dict
    ) -> None:
        extracted = {"suggested_fix": "--enable-lora to enable LoRA"}
        result = compute_correction(
            "fallback_engine_config", extracted, base_plan, model_config, hardware, None
        )
        assert result is not None
        assert result.engine_configuration.get("enable_lora") == "true"


# -------------------------------------------------------------------
# Cross-cutting
# -------------------------------------------------------------------


class TestCrossCutting:
    def test_unknown_strategy_raises(
        self, base_plan: DeploymentPlan, hardware: HardwareSpec, model_config: dict
    ) -> None:
        with pytest.raises(ValueError, match="Unknown correction strategy"):
            compute_correction("nonexistent", {}, base_plan, model_config, hardware, None)

    def test_correction_preserves_unrelated_fields(
        self, base_plan: DeploymentPlan, hardware: HardwareSpec, model_config: dict
    ) -> None:
        extracted = {"estimated_max_model_len": 8192}
        result = compute_correction(
            "reduce_memory_pressure", extracted, base_plan, model_config, hardware, None
        )
        assert result is not None
        assert result.dtype == base_plan.dtype
        assert result.tensor_parallel == base_plan.tensor_parallel
        assert result.engine_configuration["gpu_memory_utilization"] == "0.90"

    def test_determinism(
        self, base_plan: DeploymentPlan, hardware: HardwareSpec, model_config: dict
    ) -> None:
        extracted = {"estimated_max_model_len": 8192}
        r1 = compute_correction(
            "reduce_memory_pressure", extracted, base_plan, model_config, hardware, None
        )
        r2 = compute_correction(
            "reduce_memory_pressure", extracted, base_plan, model_config, hardware, None
        )
        assert r1 == r2

    def test_value_validation_clamps(
        self, base_plan: DeploymentPlan, hardware: HardwareSpec, model_config: dict
    ) -> None:
        extracted = {"estimated_max_model_len": 0}
        result = compute_correction(
            "reduce_memory_pressure", extracted, base_plan, model_config, hardware, None
        )
        assert result is not None
        lo = CORRECTION_BOUNDS["max_model_len"][0]
        assert int(result.engine_configuration["max_model_len"]) >= lo

    def test_original_plan_not_mutated(
        self, base_plan: DeploymentPlan, hardware: HardwareSpec, model_config: dict
    ) -> None:
        original_config = dict(base_plan.engine_configuration)
        compute_correction(
            "reduce_memory_pressure",
            {"estimated_max_model_len": 8192},
            base_plan,
            model_config,
            hardware,
            None,
        )
        assert base_plan.engine_configuration == original_config

    def test_feasibility_gate_rejects_oversized_model(
        self, base_plan: DeploymentPlan, model_config: dict
    ) -> None:
        small_gpu = HardwareSpec(
            gpu_sku="RTX 3060",
            total_memory_bytes=12_884_901_888,
            compute_capability="8.6",
        )
        vr = {"model_weight_memory": 15_000_000_000}
        result = compute_correction(
            "reduce_memory_pressure", {}, base_plan, model_config, small_gpu, vr
        )
        assert result is None

    def test_feasibility_gate_passes_when_model_fits(
        self, base_plan: DeploymentPlan, hardware: HardwareSpec, model_config: dict
    ) -> None:
        vr = {"model_weight_memory": 5_000_000_000}
        result = compute_correction(
            "reduce_memory_pressure", {}, base_plan, model_config, hardware, vr
        )
        assert result is not None

    def test_feasibility_gate_skipped_without_vr(
        self, base_plan: DeploymentPlan, hardware: HardwareSpec, model_config: dict
    ) -> None:
        result = compute_correction(
            "reduce_memory_pressure", {}, base_plan, model_config, hardware, None
        )
        assert result is not None

    def test_weight_oom_infeasible_with_vr(
        self, base_plan: DeploymentPlan, model_config: dict
    ) -> None:
        """Weight OOM where vr shows model can't fit returns None (infeasible)."""
        small_gpu = HardwareSpec(
            gpu_sku="RTX 3060",
            total_memory_bytes=12_884_901_888,
            compute_capability="8.6",
        )
        vr = {"model_weight_memory": 20_000_000_000}
        result = compute_correction(
            "reduce_memory_pressure", {}, base_plan, model_config, small_gpu, vr
        )
        assert result is None
