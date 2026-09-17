"""Conformance suite: EngineAdapter Protocol."""

from apron.domain.protocols import EngineAdapter
from apron.domain.schemas.solutions import DeploymentPlan


def test_satisfies_protocol(engine_adapter):
    assert isinstance(engine_adapter, EngineAdapter)


def test_has_engine_name(engine_adapter):
    assert isinstance(engine_adapter.engine_name, str)
    assert engine_adapter.engine_name


def test_has_engine_version(engine_adapter):
    assert isinstance(engine_adapter.engine_version, str)
    assert engine_adapter.engine_version


def test_resolve_support_returns_typed_dict(engine_adapter):
    result = engine_adapter.resolve_support({"arch": "LlamaForCausalLM"}, "vllm:0.8.5")
    assert isinstance(result, dict)
    assert "supported" in result
    assert isinstance(result["supported"], bool)
    assert "tasks" in result
    assert isinstance(result["tasks"], list)


def test_validate_returns_empty_for_valid_config(engine_adapter):
    plan = DeploymentPlan(tensor_parallel=2)
    errors = engine_adapter.validate(plan, None)
    assert isinstance(errors, list)
    assert errors == []


def test_validate_returns_nonempty_for_incompatible_tp(engine_adapter):
    plan = DeploymentPlan(tensor_parallel=3)
    errors = engine_adapter.validate(plan, None)
    assert isinstance(errors, list)
    assert len(errors) > 0


def test_render_returns_nonempty_dict(engine_adapter):
    plan = DeploymentPlan(tensor_parallel=4)
    rendered = engine_adapter.render(plan)
    assert isinstance(rendered, dict)
    assert len(rendered) > 0
    assert rendered.get("tp") == 4


def test_verify_returns_dict_with_memory_keys(engine_adapter):
    plan = DeploymentPlan(tensor_parallel=1)
    result = engine_adapter.verify(plan, None)
    assert isinstance(result, dict)
    assert len(result) >= 15
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


def test_classify_returns_dict_with_failure_class(engine_adapter):
    result = engine_adapter.classify("torch.OutOfMemoryError: CUDA out of memory")
    assert isinstance(result, dict)
    assert "failure_class" in result


def test_extract_schema_returns_dict_with_architectures(engine_adapter):
    result = engine_adapter.extract_schema("vllm:0.8.5")
    assert isinstance(result, dict)
    assert "architectures" in result
    assert isinstance(result["architectures"], list)
    assert len(result["architectures"]) > 0
