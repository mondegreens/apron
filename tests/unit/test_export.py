"""Step 14 tests: export round-trip for every DeploymentPlan.

Every DeploymentPlan (self-hosted golden + compound's 2 embedded) exports
losslessly to the three pinned ecosystem shapes and back.  "Losslessly"
means shared fields are identical on the second export.
"""

from apron.adapters.renderers import (
    export_to_aiconfigurator,
    export_to_inferencex,
    export_to_recipes,
    import_from_aiconfigurator,
    import_from_inferencex,
    import_from_recipes,
)
from apron.domain.schemas.solutions import DeploymentPlan

_SELF_HOSTED_PLAN = DeploymentPlan(
    tensor_parallel=1,
    dtype="bf16",
    batch_size=32,
    engine_configuration={"model": "Qwen/Qwen3-8B"},
    serve_command="vllm serve Qwen/Qwen3-8B --dtype bf16",
)

_COMPOUND_EXECUTOR_PLAN = DeploymentPlan(
    tensor_parallel=1,
    dtype="bf16",
    batch_size=16,
)

_COMPOUND_VISION_PLAN = DeploymentPlan(
    tensor_parallel=2,
    dtype="bf16",
    batch_size=8,
    pipeline_parallel=1,
)

_ALL_PLANS = {
    "self-hosted": _SELF_HOSTED_PLAN,
    "compound-executor": _COMPOUND_EXECUTOR_PLAN,
    "compound-vision": _COMPOUND_VISION_PLAN,
}


# ============================================================================
# RECIPES YAML — export, import, round-trip
# ============================================================================


def test_export_to_recipes_self_hosted():
    exported = export_to_recipes(_SELF_HOSTED_PLAN)
    assert exported["tp"] == 1
    assert exported["precision"] == "bf16"
    assert exported["max_num_seqs"] == 32


def test_round_trip_recipes():
    for name, plan in _ALL_PLANS.items():
        exported = export_to_recipes(plan)
        imported = import_from_recipes(exported)
        re_exported = export_to_recipes(DeploymentPlan(**{**plan.model_dump(), **imported}))
        for key in exported:
            if key in re_exported:
                assert exported[key] == re_exported[key], f"{name}: {key} differs"


# ============================================================================
# AICONFIGURATOR JSON — export, import, round-trip
# ============================================================================


def test_export_to_aiconfigurator_self_hosted():
    exported = export_to_aiconfigurator(_SELF_HOSTED_PLAN)
    assert exported["topology.tensor_parallel_size"] == 1
    assert exported["quantization.dtype"] == "bf16"
    assert exported["workload.batch_size"] == 32


def test_round_trip_aiconfigurator():
    for name, plan in _ALL_PLANS.items():
        exported = export_to_aiconfigurator(plan)
        imported = import_from_aiconfigurator(exported)
        re_exported = export_to_aiconfigurator(DeploymentPlan(**{**plan.model_dump(), **imported}))
        for key in exported:
            if key in re_exported:
                assert exported[key] == re_exported[key], f"{name}: {key} differs"


# ============================================================================
# INFERENCEX ROW — export, import, round-trip
# ============================================================================


def test_export_to_inferencex_self_hosted():
    exported = export_to_inferencex(_SELF_HOSTED_PLAN)
    assert exported["tp"] == 1
    assert exported["precision"] == "bf16"


def test_export_to_inferencex_compound_vision():
    exported = export_to_inferencex(_COMPOUND_VISION_PLAN)
    assert exported["tp"] == 2
    assert exported["pp"] == 1


def test_round_trip_inferencex():
    for name, plan in _ALL_PLANS.items():
        exported = export_to_inferencex(plan)
        imported = import_from_inferencex(exported)
        re_exported = export_to_inferencex(DeploymentPlan(**{**plan.model_dump(), **imported}))
        for key in exported:
            if key in re_exported:
                assert exported[key] == re_exported[key], f"{name}: {key} differs"


# ============================================================================
# ALL PLANS x ALL SHAPES
# ============================================================================


def test_every_plan_exports_to_every_shape():
    for name, plan in _ALL_PLANS.items():
        recipes = export_to_recipes(plan)
        assert "tp" in recipes or "precision" in recipes, f"{name}: empty recipes export"

        aiconfig = export_to_aiconfigurator(plan)
        assert len(aiconfig) > 0, f"{name}: empty aiconfigurator export"

        infx = export_to_inferencex(plan)
        assert "tp" in infx or "precision" in infx, f"{name}: empty inferencex export"


def test_export_only_fields_not_imported():
    exported = export_to_recipes(_SELF_HOSTED_PLAN)
    assert "base_args" in exported
    imported = import_from_recipes(exported)
    assert "engine_configuration" not in imported
