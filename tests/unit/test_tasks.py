"""Step 5 tests: Layer 3 — task, application, serving independence and INV-24."""

from apron.domain.canonical import canonicalize
from apron.domain.capabilities import CapabilitySignature
from apron.domain.fingerprints import assert_fully_classified, fingerprint_hex
from apron.domain.schemas.tasks import (
    ApplicationSpec,
    ServingWorkloadSpec,
    TaskSuiteSpec,
    WorkloadSpec,
)


def _round_trip(model_cls, instance):
    canonical = canonicalize(instance.model_dump(mode="json"))
    restored = model_cls.model_validate_json(canonical)
    assert canonicalize(restored.model_dump(mode="json")) == canonical
    return restored


_TEXT_CAP = CapabilitySignature(
    operation="text_generation",
    required_inputs=("text",),
    output_representation="generated_text",
)


def _make_task_suite(**overrides):
    defaults = {
        "name": "simple-text",
        "version": "1.0",
        "required_capabilities": (_TEXT_CAP,),
        "cases": ({"input": "hello", "expected": "world"},),
        "success_criteria": ("exact_match",),
    }
    return TaskSuiteSpec(**(defaults | overrides))


def _make_serving(**overrides):
    defaults = {
        "concurrency": 4,
        "latency_p99_ms": 2000.0,
        "throughput_target_rps": 10.0,
    }
    return ServingWorkloadSpec(**(defaults | overrides))


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


def test_task_suite_spec_classified():
    assert_fully_classified(TaskSuiteSpec)


def test_application_spec_classified():
    assert_fully_classified(ApplicationSpec)


def test_serving_workload_spec_classified():
    assert_fully_classified(ServingWorkloadSpec)


def test_workload_spec_classified():
    assert_fully_classified(WorkloadSpec)


# ---------------------------------------------------------------------------
# round-trip
# ---------------------------------------------------------------------------


def test_task_suite_spec_round_trip():
    ts = _make_task_suite()
    _round_trip(TaskSuiteSpec, ts)


def test_application_spec_round_trip():
    app = ApplicationSpec(
        name="chat-app",
        version="2.1",
        logical_roles=("planner", "executor"),
        turn_limit=5,
    )
    _round_trip(ApplicationSpec, app)


def test_serving_workload_spec_round_trip():
    sw = _make_serving()
    _round_trip(ServingWorkloadSpec, sw)


def test_workload_spec_round_trip():
    ts = _make_task_suite()
    sw = _make_serving()
    ws = WorkloadSpec(
        task_suite_fingerprint=fingerprint_hex(ts),
        serving_workload_fingerprint=fingerprint_hex(sw),
    )
    _round_trip(WorkloadSpec, ws)


# ---------------------------------------------------------------------------
# independence (INV-24)
# ---------------------------------------------------------------------------


def test_task_and_serving_are_independent():
    ts = _make_task_suite()
    sw = _make_serving()

    ts_fp = fingerprint_hex(ts)
    sw_fp = fingerprint_hex(sw)
    assert ts_fp != sw_fp

    sw_modified = _make_serving(throughput_target_rps=1000.0)
    sw_modified_fp = fingerprint_hex(sw_modified)

    assert ts_fp == fingerprint_hex(ts)
    assert sw_fp != sw_modified_fp


def test_workload_spec_binds_both():
    ts = _make_task_suite()
    sw = _make_serving()
    ws = WorkloadSpec(
        task_suite_fingerprint=fingerprint_hex(ts),
        serving_workload_fingerprint=fingerprint_hex(sw),
    )
    assert ws.task_suite_fingerprint == fingerprint_hex(ts)
    assert ws.serving_workload_fingerprint == fingerprint_hex(sw)


def test_serving_throughput_cannot_establish_task_success():
    ts_failing = _make_task_suite(
        success_criteria=("exact_match",),
        cases=({"input": "hello", "expected": "wrong_answer"},),
    )
    sw_passing = _make_serving(throughput_target_rps=10000.0)

    ts_fp = fingerprint_hex(ts_failing)
    sw_fp = fingerprint_hex(sw_passing)
    assert ts_fp != sw_fp

    ws = WorkloadSpec(
        task_suite_fingerprint=ts_fp,
        serving_workload_fingerprint=sw_fp,
    )
    assert ws.task_suite_fingerprint == ts_fp
    assert ws.serving_workload_fingerprint == sw_fp


def test_task_suite_privacy_classification():
    public_ts = _make_task_suite(privacy_classification="public")
    private_ts = _make_task_suite(privacy_classification="private")
    assert fingerprint_hex(public_ts) != fingerprint_hex(private_ts)
