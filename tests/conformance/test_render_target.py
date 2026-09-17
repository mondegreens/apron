"""Conformance suite: RenderTarget Protocol."""

from apron.domain.protocols import RenderTarget
from apron.domain.schemas.primitives import ArtifactLocator, HardwareSpec
from apron.domain.schemas.solutions import DeploymentPlan, RenderContext


def _make_context(plan: DeploymentPlan) -> RenderContext:
    return RenderContext(
        plan=plan,
        locator=ArtifactLocator(source_kind="huggingface", uri="test/model"),
        hardware=HardwareSpec(
            gpu_sku="RTX 4090",
            total_memory_bytes=25_769_803_776,
            compute_capability="8.9",
        ),
    )


def test_satisfies_protocol(render_target):
    assert isinstance(render_target, RenderTarget)


def test_target_format_is_nonempty_string(render_target):
    assert isinstance(render_target.target_format, str)
    assert render_target.target_format


def test_render_returns_nonempty_dict(render_target):
    ctx = _make_context(DeploymentPlan(tensor_parallel=2))
    rendered = render_target.render(ctx)
    assert isinstance(rendered, dict)
    assert len(rendered) > 0


def test_round_trip_recovers_shared_fields(render_target):
    ctx = _make_context(DeploymentPlan(tensor_parallel=4, dtype="bfloat16"))
    rendered = render_target.render(ctx)
    parsed = render_target.parse(rendered)
    assert isinstance(parsed, dict)
    assert parsed["tensor_parallel"] == 4
    assert parsed.get("dtype") == "bfloat16"
