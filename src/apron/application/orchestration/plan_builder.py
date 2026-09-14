"""PlanBuilder — convert calculator output into a DeploymentPlan.

The calculator predicts memory; the PlanBuilder decides configuration
(tensor parallelism, batch size, dtype, engine parameters).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from apron.domain.schemas.solutions import DeploymentPlan, PlanningClaim

if TYPE_CHECKING:
    from apron.domain.ports import Clock, IdGenerator
    from apron.domain.schemas.models import ExecutionSpec, ModelSpec
    from apron.domain.schemas.primitives import HardwareSpec
    from apron.domain.schemas.tasks import ServingWorkloadSpec


def build_plan(
    claim: PlanningClaim,
    model_spec: ModelSpec,
    hardware_spec: HardwareSpec,
    execution_spec: ExecutionSpec | None,
    workload: ServingWorkloadSpec | None,
    *,
    clock: Clock,
    id_gen: IdGenerator,
) -> DeploymentPlan:
    """Build a DeploymentPlan from calculator output and model constraints."""
    config = claim.proposed_configuration

    if config.get("status") == "unknown":
        return DeploymentPlan()

    weight_bytes = config.get("weight_memory_bytes", 0)
    total_required = config.get("total_required_bytes", 0)
    available_kv = config.get("available_kv_cache_bytes", 0)

    dtype = _derive_dtype(model_spec)
    tp = _derive_tensor_parallel(total_required, hardware_spec, config)
    batch_size = _derive_batch_size(available_kv, config, tp)

    gpu_util = 0.90
    max_model_len = _derive_max_model_len(workload)

    engine_configuration: dict[str, str] = {
        "gpu_memory_utilization": str(gpu_util),
        "max_model_len": str(max_model_len),
    }

    resource_allocation: dict[str, str] = {
        "gpu_sku": hardware_spec.gpu_sku,
        "gpu_count": str(tp),
        "weight_bytes": str(weight_bytes),
    }

    return DeploymentPlan(
        tensor_parallel=tp,
        pipeline_parallel=1,
        dtype=dtype,
        batch_size=batch_size,
        engine_configuration=engine_configuration,
        resource_allocation=resource_allocation,
    )


def _derive_dtype(model_spec: ModelSpec) -> str:
    dtypes = model_spec.component_bytes_dtype
    if "decoder" in dtypes:
        return dtypes["decoder"]
    if dtypes:
        return next(iter(dtypes.values()))
    return "bfloat16"


def _derive_tensor_parallel(
    total_required: int,
    hardware: HardwareSpec,
    config: dict[str, Any],
) -> int:
    gpu_usable = int(hardware.total_memory_bytes * 0.90)
    if total_required <= gpu_usable:
        return 1

    num_heads = config.get("num_attention_heads", 1)
    num_kv_heads = config.get("num_kv_heads", num_heads)

    for tp in (2, 4, 8):
        if num_heads % tp == 0 and num_kv_heads % tp == 0 and total_required / tp <= gpu_usable:
            return tp

    return 1


def _derive_batch_size(
    available_kv: int,
    config: dict[str, Any],
    tp: int,
) -> int:
    kv_per_token = config.get("kv_per_token_bytes", 0)
    isl = config.get("isl", 512)
    osl = config.get("osl", 128)

    if kv_per_token <= 0:
        return 4

    per_sequence = kv_per_token * (isl + osl)
    if per_sequence <= 0:
        return 4

    adjusted_kv = available_kv * tp if tp > 1 else available_kv
    max_seqs = adjusted_kv // per_sequence
    return max(1, min(max_seqs, 256))


def _derive_max_model_len(workload: ServingWorkloadSpec | None) -> int:
    if workload is None:
        return 640
    isl = 512
    osl = 128
    return isl + osl
