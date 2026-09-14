"""Conformance suite: ExecutionTarget Protocol."""

from apron.domain.schemas.primitives import ExecutionTarget, HardwareSpec


def test_satisfies_protocol(execution_target):
    assert isinstance(execution_target, ExecutionTarget)


def test_kind_is_nonempty(execution_target):
    assert isinstance(execution_target.kind, str)
    assert execution_target.kind


def test_operator_is_nonempty(execution_target):
    assert isinstance(execution_target.operator, str)
    assert execution_target.operator


def test_provider_is_nonempty(execution_target):
    provider = execution_target.provider
    assert provider is None or isinstance(provider, str)
    if provider is not None:
        assert provider


def test_hardware_returns_hardware_spec(execution_target):
    hw = execution_target.hardware
    assert isinstance(hw, HardwareSpec)


def test_execution_fingerprint_is_nonempty(execution_target):
    fp = execution_target.execution_fingerprint
    assert isinstance(fp, str)
    assert fp


def test_lifecycle_callable_in_order(execution_target):
    execution_target.prepare()
    execution_target.provision()
    result = execution_target.execute("echo hello")
    assert result is not None
    obs = execution_target.observe()
    assert isinstance(obs, dict)
    collected = execution_target.collect()
    assert isinstance(collected, dict)
    execution_target.teardown()


def test_teardown_is_idempotent(execution_target):
    execution_target.teardown()
    execution_target.teardown()
