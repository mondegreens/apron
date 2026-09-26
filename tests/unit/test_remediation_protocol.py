"""§10.2 fix proof with a mock engine for all six classes, plus the structural guard.

The boots are simulated (tests/unit/_cohort_fakes.py): a broken plan fails with
the class's vLLM error text, any other plan boots healthy.  Classification is
the deterministic test classifier; the pipeline, rules, strategies, gates,
records and promotion are the real code.  The per-class tests on the *real*
captured L0-F logs are in tests/unit/test_fix_proof_real_logs.py.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest
from _cohort_fakes import (
    CATALOG,
    MemoryRuleRepository,
    accepted_inputs,
    correction_context,
    fix_plan_solution,
    ports,
)
from conftest import FakeDiagnosisEngine

from apron.adapters.backends.rule_loader import load_rules
from apron.application.orchestration import remediation
from apron.application.orchestration.remediation import (
    SIX_CLASSES,
    FixProofPorts,
    promoted_rule,
    prove_all,
    prove_fix,
)
from apron.domain.fingerprints import fingerprint_hex
from apron.domain.schemas.migrations import load_record
from apron.domain.schemas.records import RemediationRecord, VerificationReport

RULES = load_rules(Path(__file__).parents[2] / "rules", "vllm", "v0.29")

# The class's error, in the pinned vLLM v0.29.0 format (call sites in SIX_CLASSES).
CLASS_LOGS = {
    1: "ERROR Failed to load model - not enough GPU memory. Try lowering "
    "--gpu-memory-utilization ... (original error: CUDA out of memory. Tried to allocate "
    "1.50 GiB. GPU 0 has a total capacity of 23.52 GiB of which 1.12 GiB is free.)",
    2: "ValueError: To serve at least one request with the model's max seq len (40960), "
    "(5.62 GiB KV cache is needed, which is larger than the available KV cache memory "
    "(2.40 GiB). Based on the available memory, the estimated maximum model length is 17472.\n"
    "RuntimeError: Engine core initialization failed. See root cause above.",
    3: "ValueError: User-specified max_model_len (999999) is greater than the derived "
    "max_model_len (max_position_embeddings=32768 or model_max_length=None in model's "
    "config.json).",
    4: "ValueError: The model type 'gemma2' does not support float16. Reason: Numerical "
    "instability. Please use bfloat16 or float32 instead.",
    5: "ValueError: Total number of attention heads (32) must be divisible by tensor "
    "parallel size (3).",
    6: "ValueError: The quantization method fp_quant is not supported for the current GPU. "
    "Minimum capability: 100. Current capability: 90.",
}
BROKEN = {id(case.broken_plan): case.failure_class for case in SIX_CLASSES}


def _scenario(plan: Any, target: Any) -> str:
    """A broken plan fails as its class names; every other plan boots."""
    cls = BROKEN.get(id(plan))
    return CLASS_LOGS[cls] if cls else "healthy"


def _fix_ports(repo: MemoryRuleRepository) -> FixProofPorts:
    return FixProofPorts(
        plan_solution=fix_plan_solution,
        diagnosis_engine=FakeDiagnosisEngine(),
        rules=RULES,
        rule_repository=repo,
        correction_context=correction_context,
        hardware_for=lambda sku: CATALOG[sku][0],
    )


EXPECTED_CHANGE = {
    1: ("resource_allocation", "gpu_sku", "NVIDIA RTX A6000"),
    2: ("engine_configuration", "max_model_len", "17472"),
    3: ("engine_configuration", "max_model_len", "32768"),
    4: ("plan", "dtype", "bfloat16"),
    5: ("plan", "tensor_parallel", 2),
    6: ("resource_allocation", "gpu_sku", "NVIDIA B200"),
}


def test_all_six_classes_fixed_on_the_broken_plan(tmp_path: Path) -> None:
    cohort_ports, _, _, _ = ports(tmp_path, _scenario)
    repo = MemoryRuleRepository(RULES)
    proofs = prove_all(accepted_inputs(), cohort_ports, _fix_ports(repo))

    assert [p.failure_class for p in proofs] == [1, 2, 3, 4, 5, 6]
    for proof in proofs:
        case = SIX_CLASSES[proof.failure_class - 1]
        assert proof.broken_failed, proof
        assert proof.gate_a, (proof.failure_class, proof.diagnosed_family)
        assert proof.gate_b
        where, key, value = EXPECTED_CHANGE[proof.failure_class]
        fixed = proof.corrected_plan
        assert fixed is not None
        assert (getattr(fixed, key) if where == "plan" else getattr(fixed, where)[key]) == value
        assert proof.mechanism_outcome == "verified"
        assert proof.request_outcome == "satisfied"
        assert proof.label == "Fixed"
        assert proof.fixed_solution_fp != proof.broken_solution_fp

        record = load_record(
            RemediationRecord, cohort_ports.store.retrieve(proof.remediation_digest or "") or {}
        )
        assert record.corrects == fingerprint_hex(case.broken_plan)
        assert record.corrected_plan_digest == fingerprint_hex(fixed)
        assert record.corrects != record.corrected_plan_digest
        assert record.diagnosed_failure_class == case.expected_family
        boot = load_record(
            VerificationReport,
            cohort_ports.store.retrieve(record.proving_record_fingerprints[0]) or {},
        )
        assert boot.deployment_plan_digest == record.corrected_plan_digest
        assert boot.boot_outcome == "healthy"
        assert len(record.proving_record_fingerprints) == 1 + 3 + 1  # boot, 3 attempts, serving

    assert [r.status for r in repo.written] == ["mechanism_verified"] * 6
    assert all(r.rule_version == 2 and r.supersedes for r in repo.written)


def test_retarget_moves_the_solution_to_another_gpu(tmp_path: Path) -> None:
    cohort_ports, _, targets, _ = ports(tmp_path, _scenario)
    proof = prove_fix(
        SIX_CLASSES[0], accepted_inputs(), cohort_ports, _fix_ports(MemoryRuleRepository(RULES))
    )
    assert [t.requested.gpu_sku for t in targets] == [
        "NVIDIA GeForce RTX 4090",
        "NVIDIA RTX A6000",
    ]
    assert proof.label == "Fixed"


def test_the_booted_plan_is_the_object_diagnosis_returned(tmp_path: Path) -> None:
    cohort_ports, engine, _, _ = ports(tmp_path, _scenario)
    for case in SIX_CLASSES:
        engine.booted.clear()
        proof = prove_fix(
            case, accepted_inputs(), cohort_ports, _fix_ports(MemoryRuleRepository(RULES))
        )
        assert engine.booted[0] is case.broken_plan
        assert engine.booted[-1] is proof.corrected_plan


def test_broken_boot_is_reused_not_repeated(tmp_path: Path) -> None:
    cohort_ports, engine, _, _ = ports(tmp_path, _scenario)
    case = SIX_CLASSES[2]
    prove_fix(case, accepted_inputs(), cohort_ports, _fix_ports(MemoryRuleRepository(RULES)))
    engine.booted.clear()
    prove_fix(case, accepted_inputs(), cohort_ports, _fix_ports(MemoryRuleRepository(RULES)))
    assert engine.booted == []  # both boots already on record


def test_broken_plan_that_boots_counts_for_nothing(tmp_path: Path) -> None:
    cohort_ports, *_ = ports(tmp_path, lambda plan, target: "healthy")
    proof = prove_fix(
        SIX_CLASSES[3], accepted_inputs(), cohort_ports, _fix_ports(MemoryRuleRepository(RULES))
    )
    assert "replace it" in proof.notes[0]
    assert proof.remediation_digest is None and proof.mechanism_outcome == "not_evaluated"


def test_gate_a_mismatch_is_stored_and_the_class_fails(tmp_path: Path) -> None:
    def wrong(plan: Any, target: Any) -> str:
        return CLASS_LOGS[3] if id(plan) == id(SIX_CLASSES[3].broken_plan) else "healthy"

    cohort_ports, *_ = ports(tmp_path, wrong)
    repo = MemoryRuleRepository(RULES)
    proof = prove_fix(SIX_CLASSES[3], accepted_inputs(), cohort_ports, _fix_ports(repo))
    assert not proof.gate_a
    assert proof.label == "Unverified suggestion"
    record = load_record(
        RemediationRecord, cohort_ports.store.retrieve(proof.remediation_digest or "") or {}
    )
    assert record.mechanism_outcome == "not_evaluated"
    assert "gate A fail" in record.reason
    assert repo.written == []


def test_fixed_boot_that_fails_is_not_promoted(tmp_path: Path) -> None:
    def never(plan: Any, target: Any) -> str:
        cls = BROKEN.get(id(plan))
        return CLASS_LOGS[cls] if cls else "RuntimeError: CUDA error: an illegal memory access"

    cohort_ports, *_ = ports(tmp_path, never)
    repo = MemoryRuleRepository(RULES)
    proof = prove_fix(SIX_CLASSES[5], accepted_inputs(), cohort_ports, _fix_ports(repo))
    assert proof.mechanism_outcome == "failed"
    assert proof.label == "Unverified suggestion"
    assert repo.written == []


def test_violated_request_is_an_alternative_with_trade_offs(tmp_path: Path) -> None:
    from _cohort_fakes import FakeEvaluator

    cohort_ports, *_ = ports(
        tmp_path, _scenario, evaluator=FakeEvaluator(wrong=frozenset({"arith-1", "arith-2"}))
    )
    proof = prove_fix(
        SIX_CLASSES[2], accepted_inputs(), cohort_ports, _fix_ports(MemoryRuleRepository(RULES))
    )
    assert proof.mechanism_outcome == "verified"
    assert proof.request_outcome == "violated"
    assert proof.label == "Alternative with trade-offs"
    assert any(v.startswith("task:") for v in proof.violated_constraints)


def test_classifier_call_needs_authorization_and_is_recorded(tmp_path: Path) -> None:
    cohort_ports, *_ = ports(tmp_path, _scenario)
    with pytest.raises(PermissionError):
        prove_fix(
            SIX_CLASSES[2],
            accepted_inputs(task_data_destinations=()),
            cohort_ports,
            _fix_ports(MemoryRuleRepository(RULES)),
        )
    cohort_ports, *_ = ports(tmp_path / "b", _scenario)
    prove_fix(
        SIX_CLASSES[2], accepted_inputs(), cohort_ports, _fix_ports(MemoryRuleRepository(RULES))
    )
    assert any(e["label"] == "classifier:class3" for e in cohort_ports.budget.ledger.read_all())


def test_promoted_rule_cites_the_proving_boot() -> None:
    from apron.domain.schemas.records import DiagnosisRule

    rule = DiagnosisRule.model_validate(
        next(r for r in RULES if r["error_family"] == "max_model_len")
    )
    proven = promoted_rule(rule, "1220" + "ab" * 32)
    assert proven.status == "mechanism_verified"
    assert proven.rule_version == rule.rule_version + 1
    assert proven.supersedes == fingerprint_hex(rule)
    assert proven.promoting_verification_fingerprint == "1220" + "ab" * 32


# ---------------------------------------------------------------------------
# Structural guard (§10.3)
# ---------------------------------------------------------------------------

SOURCE = Path(remediation.__file__).read_text()
TREE = ast.parse(SOURCE)


def test_no_good_flags_anywhere() -> None:
    assert "good_flags" not in SOURCE


def test_only_six_plan_literals_and_they_are_the_broken_plans() -> None:
    literals = [
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "DeploymentPlan"
    ]
    assert len(literals) == 6
    assert {case.expected_family for case in SIX_CLASSES} == {
        "oom_weight_load",
        "oom_kv_cache",
        "max_model_len",
        "dtype_incompatible",
        "tp_divisibility",
        "quant_compute_capability",
    }


def test_one_function_runs_all_six_without_per_class_branches() -> None:
    prove = next(
        n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef) and n.name == "prove_fix"
    )
    families = {case.expected_family for case in SIX_CLASSES}
    for node in ast.walk(prove):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert node.value not in families, f"per-class branch on {node.value!r}"
        if isinstance(node, ast.Compare):
            for comparator in node.comparators:
                assert not (
                    isinstance(comparator, ast.Constant) and comparator.value in range(1, 7)
                )
    assert "prove_fix(case" in inspect.getsource(prove_all)
