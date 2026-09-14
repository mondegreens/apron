"""Conformance suite: RenderTarget Protocol."""

from apron.domain.protocols import RenderTarget
from apron.domain.schemas.solutions import DeploymentPlan


def test_satisfies_protocol(render_target):
    assert isinstance(render_target, RenderTarget)


def test_target_format_is_nonempty_string(render_target):
    assert isinstance(render_target.target_format, str)
    assert render_target.target_format


def test_render_returns_nonempty_dict(render_target):
    plan = DeploymentPlan(tensor_parallel=2)
    rendered = render_target.render(plan)
    assert isinstance(rendered, dict)
    assert len(rendered) > 0


def test_round_trip_recovers_shared_fields(render_target):
    plan = DeploymentPlan(tensor_parallel=4, dtype="bfloat16")
    rendered = render_target.render(plan)
    parsed = render_target.parse(rendered)
    assert isinstance(parsed, dict)
    assert parsed["tensor_parallel"] == 4
