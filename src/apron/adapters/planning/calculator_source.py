"""PlanningSource adapter wrapping the mechanism-aware calculator.

Implements the PlanningSource Protocol by dispatching to the registered
calculator for the model's primary mechanism, then packaging the result
as a PlanningClaim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from apron.domain.canonical import canonicalize, digest_hex
from apron.domain.mechanisms import CalculatorInput, ComponentMechanism, TextWorkload, calculate
from apron.domain.schemas.primitives import HardwareSpec
from apron.domain.schemas.solutions import PlanningClaim

if TYPE_CHECKING:
    from apron.domain.ports import Clock


class CalculatorPlanningSource:
    """PlanningSource backed by mechanism-aware calculator dispatch."""

    def __init__(self, *, clock: Clock) -> None:
        self._clock = clock

    @property
    def producer_name(self) -> str:
        return "apron-calculator"

    @property
    def producer_version(self) -> str:
        return "0.1"

    def predict(
        self,
        model_spec: Any,
        hardware_spec: Any,
        workload_shape: Any,
    ) -> PlanningClaim:
        config = model_spec if isinstance(model_spec, dict) else {}
        hw = hardware_spec if isinstance(hardware_spec, HardwareSpec) else None

        components = config.get("components", ())
        primary = None
        for comp in components:
            if isinstance(comp, ComponentMechanism):
                if comp.role == "decoder":
                    primary = comp
                    break
            elif isinstance(comp, dict) and comp.get("role") == "decoder":
                primary = ComponentMechanism(**comp)
                break

        if primary is None:
            return self._unknown_claim(config, hardware_spec, workload_shape)

        workload = self._build_workload(workload_shape)
        execution_data = workload_shape if isinstance(workload_shape, dict) else {}

        if hw is None:
            hw = HardwareSpec(
                gpu_sku="unknown",
                total_memory_bytes=25_769_803_776,
                compute_capability="0.0",
            )

        calc_input = CalculatorInput(
            mechanism=primary,
            workload=workload,
            artifact_metadata=config.get("artifact_metadata", config),
            hardware=hw,
            execution_spec_data=execution_data,
        )

        result = calculate(calc_input)
        if result is None:
            return self._unknown_claim(config, hardware_spec, workload_shape)

        input_fp = digest_hex(
            canonicalize(
                {
                    "model_spec": str(config),
                    "hardware_spec": str(hardware_spec),
                    "workload_shape": str(workload_shape),
                }
            )
        )

        total = result.get("total_required_bytes", 0)
        return PlanningClaim(
            producer=self.producer_name,
            version=self.producer_version,
            input_fingerprint=input_fp,
            proposed_configuration={
                "weight_memory_bytes": result["weight_memory_bytes"],
                "kv_cache_bytes": result["kv_cache_bytes"],
                "activation_estimate_bytes": result["activation_estimate_bytes"],
                "non_pytorch_overhead_bytes": result["non_pytorch_overhead_bytes"],
                "cuda_graph_estimate_bytes": result["cuda_graph_estimate_bytes"],
                "available_kv_cache_bytes": result["available_kv_cache_bytes"],
                "total_required_bytes": result["total_required_bytes"],
            },
            claim_scope="memory",
            producer_epistemic_tier="MECHANISM",
            uncertainty={
                "lower": total * 0.85,
                "upper": total * 1.50,
            },
        )

    def _unknown_claim(
        self,
        model_spec: Any,
        hardware_spec: Any,
        workload_shape: Any,
    ) -> PlanningClaim:
        input_fp = digest_hex(
            canonicalize(
                {
                    "model_spec": str(model_spec),
                    "hardware_spec": str(hardware_spec),
                    "workload_shape": str(workload_shape),
                }
            )
        )
        return PlanningClaim(
            producer=self.producer_name,
            version=self.producer_version,
            input_fingerprint=input_fp,
            proposed_configuration={"status": "unknown"},
            claim_scope="memory",
            producer_epistemic_tier="UNKNOWN",
        )

    def _build_workload(self, workload_shape: Any) -> TextWorkload:
        if isinstance(workload_shape, dict):
            return TextWorkload(
                kind="text",
                input_length=workload_shape.get("isl", workload_shape.get("input_length", 512)),
                output_length=workload_shape.get("osl", workload_shape.get("output_length", 128)),
            )
        if hasattr(workload_shape, "kind"):
            return workload_shape
        return TextWorkload(kind="text", input_length=512, output_length=128)
