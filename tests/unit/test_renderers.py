"""Tests for all RenderTarget implementations."""

from apron.adapters.renderers.aiconfigurator_export import AIConfiguratorExportRenderer
from apron.adapters.renderers.docker_compose import DockerComposeRenderer
from apron.adapters.renderers.inferencex_export import InferenceXExportRenderer
from apron.adapters.renderers.recipes_export import RecipesExportRenderer
from apron.adapters.renderers.vllm_serve import VllmServeRenderer
from apron.domain.protocols import RenderTarget
from apron.domain.schemas.primitives import ArtifactLocator, HardwareSpec
from apron.domain.schemas.solutions import DeploymentPlan, RenderContext


def _ctx(plan: DeploymentPlan | None = None) -> RenderContext:
    return RenderContext(
        plan=plan or DeploymentPlan(tensor_parallel=2, dtype="bfloat16"),
        locator=ArtifactLocator(source_kind="huggingface", uri="Qwen/Qwen3-8B"),
        hardware=HardwareSpec(
            gpu_sku="RTX 4090",
            total_memory_bytes=25_769_803_776,
            compute_capability="8.9",
        ),
    )


# ---- VllmServe ----


def test_vllm_serve_satisfies_protocol():
    assert isinstance(VllmServeRenderer(), RenderTarget)


def test_vllm_serve_command_contains_model():
    rendered = VllmServeRenderer().render(_ctx())
    assert "Qwen/Qwen3-8B" in rendered["command"]


def test_vllm_serve_round_trip():
    renderer = VllmServeRenderer()
    ctx = _ctx(DeploymentPlan(tensor_parallel=4, dtype="bfloat16"))
    rendered = renderer.render(ctx)
    parsed = renderer.parse(rendered)
    assert parsed["tensor_parallel"] == 4
    assert parsed["dtype"] == "bfloat16"


def test_vllm_serve_tp1_no_flag():
    plan = DeploymentPlan(tensor_parallel=1, dtype="bfloat16")
    rendered = VllmServeRenderer().render(_ctx(plan))
    assert "--tensor-parallel-size" not in rendered["command"]


def test_vllm_serve_engine_config():
    plan = DeploymentPlan(
        tensor_parallel=1,
        dtype="bfloat16",
        engine_configuration={"gpu_memory_utilization": "0.90", "max_model_len": "640"},
    )
    rendered = VllmServeRenderer().render(_ctx(plan))
    assert "--gpu-memory-utilization" in rendered["command"]
    assert "--max-model-len" in rendered["command"]


# ---- DockerCompose ----


def test_docker_compose_satisfies_protocol():
    assert isinstance(DockerComposeRenderer(), RenderTarget)


def test_docker_compose_gpu_reservation():
    rendered = DockerComposeRenderer().render(_ctx())
    compose = rendered["compose"]
    devices = compose["services"]["vllm"]["deploy"]["resources"]["reservations"]["devices"]
    assert len(devices) == 1
    assert devices[0]["driver"] == "nvidia"
    assert devices[0]["count"] == 2


def test_docker_compose_round_trip():
    renderer = DockerComposeRenderer()
    ctx = _ctx(DeploymentPlan(tensor_parallel=4, dtype="bfloat16"))
    rendered = renderer.render(ctx)
    parsed = renderer.parse(rendered)
    assert parsed["tensor_parallel"] == 4
    assert parsed["dtype"] == "bfloat16"


def test_docker_compose_has_vllm_service():
    rendered = DockerComposeRenderer().render(_ctx())
    compose = rendered["compose"]
    assert "vllm" in compose["services"]
    assert compose["services"]["vllm"]["image"].startswith("vllm/vllm-openai")
    assert "8000:8000" in compose["services"]["vllm"]["ports"]


# ---- Recipes export ----


def test_recipes_satisfies_protocol():
    assert isinstance(RecipesExportRenderer(), RenderTarget)


def test_recipes_format():
    assert RecipesExportRenderer().target_format == "recipes_yaml"


def test_recipes_render_has_model_id():
    r = RecipesExportRenderer().render(_ctx())
    assert r["model_id"] == "Qwen/Qwen3-8B"


def test_recipes_round_trip():
    renderer = RecipesExportRenderer()
    ctx = _ctx(DeploymentPlan(tensor_parallel=4, dtype="bfloat16"))
    rendered = renderer.render(ctx)
    parsed = renderer.parse(rendered)
    assert parsed["tensor_parallel"] == 4
    assert parsed["dtype"] == "bfloat16"


# ---- AIConfigurator export ----


def test_aiconfigurator_satisfies_protocol():
    assert isinstance(AIConfiguratorExportRenderer(), RenderTarget)


def test_aiconfigurator_format():
    assert AIConfiguratorExportRenderer().target_format == "aiconfigurator"


def test_aiconfigurator_render_has_model_path():
    r = AIConfiguratorExportRenderer().render(_ctx())
    assert r["model.path"] == "Qwen/Qwen3-8B"


def test_aiconfigurator_round_trip():
    renderer = AIConfiguratorExportRenderer()
    ctx = _ctx(DeploymentPlan(tensor_parallel=4, dtype="bfloat16"))
    rendered = renderer.render(ctx)
    parsed = renderer.parse(rendered)
    assert parsed["tensor_parallel"] == 4
    assert parsed["dtype"] == "bfloat16"


# ---- InferenceX export ----


def test_inferencex_satisfies_protocol():
    assert isinstance(InferenceXExportRenderer(), RenderTarget)


def test_inferencex_format():
    assert InferenceXExportRenderer().target_format == "inferencex"


def test_inferencex_render_has_model():
    r = InferenceXExportRenderer().render(_ctx())
    assert r["model"] == "Qwen/Qwen3-8B"


def test_inferencex_round_trip():
    renderer = InferenceXExportRenderer()
    ctx = _ctx(DeploymentPlan(tensor_parallel=4, dtype="bfloat16"))
    rendered = renderer.render(ctx)
    parsed = renderer.parse(rendered)
    assert parsed["tensor_parallel"] == 4
    assert parsed["dtype"] == "bfloat16"
