"""A plan is diagnosed by the rules of the vLLM version it runs on.

Group B's third attempt (2026-09-28) ran on v0.30.0 while the cohort's one
adapter carried v0.29.0's rules: a v0.30 error would have been refused as a
version mismatch.  The ports now hold one adapter per pinned version.
"""

from __future__ import annotations

from types import SimpleNamespace

from apron.adapters.backends.rule_loader import load_rules
from apron.adapters.backends.vllm_engine import VllmEngineAdapter
from apron.adapters.runner_image import RUNNER_IMAGES
from apron.application.orchestration.cohort import CohortPorts
from apron.interfaces.cohort_root import RULES_DIR


def test_the_plan_engine_version_picks_its_adapter() -> None:
    v29 = VllmEngineAdapter(engine_version="v0.29.0")
    v30 = VllmEngineAdapter(engine_version="v0.30.0")
    ports = SimpleNamespace(engine=v29, engines={"v0.29.0": v29, "v0.30.0": v30})
    on_v30 = SimpleNamespace(engine_version="v0.30.0")
    assert CohortPorts.engine_for(ports, on_v30) is v30  # type: ignore[arg-type]
    unknown = SimpleNamespace(engine_version="v9.9.9")
    assert CohortPorts.engine_for(ports, unknown) is v29  # type: ignore[arg-type]
    one = SimpleNamespace(engine=v29, engines=None)
    assert CohortPorts.engine_for(one, on_v30) is v29  # type: ignore[arg-type]


def test_every_pinned_version_has_its_own_rules() -> None:
    for version in RUNNER_IMAGES:
        rules = load_rules(RULES_DIR, "vllm", version)
        assert rules, f"no diagnosis rules for vLLM {version}"
        assert {r["engine_version"] for r in rules} == {version}
