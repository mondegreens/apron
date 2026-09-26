"""F2: strict schemas — nothing a user wrote is silently dropped.

For every pydantic model under ``apron.domain`` (enumerated, not hand-picked):
a minimal valid input is synthesized from the field annotations, validated,
then an unknown key is added — at the top level and inside each nested model —
and validation must fail with ``extra_forbidden`` naming that key and its path.
Every JSON document under ``tests/fixtures/`` that is a domain record
validates strictly against its schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _domain_models import all_domain_models, minimal_input
from pydantic import BaseModel, ValidationError

from apron.domain.schemas.migrations import load_record

MODELS = all_domain_models()
_UNKNOWN = "field_nobody_declared"


def test_every_domain_model_forbids_extra() -> None:
    assert len(MODELS) >= 69
    lax = [m.__qualname__ for m in MODELS if m.model_config.get("extra") != "forbid"]
    assert lax == []


# ---------------------------------------------------------------------------
# Minimal valid input synthesis
# ---------------------------------------------------------------------------


def _nested_paths(model: type[BaseModel], data: dict[str, Any]) -> list[tuple[str, ...]]:
    """Paths (within *data*) of nested model dicts present in the minimal input."""
    paths: list[tuple[str, ...]] = []
    for name, value in data.items():
        if isinstance(value, dict) and value and name in model.model_fields:
            paths.append((name,))
    return paths


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.__qualname__)
def test_minimal_input_is_valid(model: type[BaseModel]) -> None:
    model.model_validate(minimal_input(model))


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.__qualname__)
def test_unknown_top_level_key_is_rejected(model: type[BaseModel]) -> None:
    data = {**minimal_input(model), _UNKNOWN: "user wrote this"}
    with pytest.raises(ValidationError) as exc:
        model.model_validate(data)
    errors = exc.value.errors()
    assert [(e["type"], e["loc"]) for e in errors] == [("extra_forbidden", (_UNKNOWN,))]


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.__qualname__)
def test_unknown_nested_key_is_rejected_with_path(model: type[BaseModel]) -> None:
    base = minimal_input(model)
    for path in _nested_paths(model, base):
        data = json.loads(json.dumps(base))
        target = data
        for key in path:
            target = target[key]
        target[_UNKNOWN] = 1
        with pytest.raises(ValidationError) as exc:
            model.model_validate(data)
        locs = [e["loc"] for e in exc.value.errors() if e["type"] == "extra_forbidden"]
        assert locs and locs[0][: len(path)] == path and locs[0][-1] == _UNKNOWN


def test_load_record_fails_loudly_on_unknown_key() -> None:
    from apron.domain.schemas.records import VerificationReport

    stored = {**minimal_input(VerificationReport), "schema_version": 1, "stale_key": 3}
    with pytest.raises(ValidationError) as exc:
        load_record(VerificationReport, stored)
    assert exc.value.errors()[0]["loc"] == ("stale_key",)


def test_load_record_validates_known_record() -> None:
    from apron.domain.schemas.records import VerificationReport

    stored = {**minimal_input(VerificationReport), "schema_version": 1}
    record = load_record(VerificationReport, stored)
    assert record.model_dump(mode="json")["execution_fingerprint"] == "x"


# ---------------------------------------------------------------------------
# Every fixture document validates strictly
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parents[1] / "fixtures"


def _fixture_schemas() -> dict[str, type[BaseModel]]:
    from apron.domain.schemas.authority import DecisionRequest
    from apron.domain.schemas.provenance import ExternalFormatProvenance
    from apron.domain.schemas.records import (
        DiagnosisRule,
        RemediationRecord,
        TaskAttemptRecord,
        VerificationReport,
    )
    from apron.domain.schemas.reports import DecisionReport
    from apron.domain.schemas.solutions import DeploymentPlan, EvaluationProtocol
    from apron.domain.schemas.tasks import (
        ApplicationSpec,
        ServingWorkloadSpec,
        TaskSuiteSpec,
        WorkloadSpec,
    )
    from apron.domain.solutions import InferenceSolution

    by_name = {
        "decision-request.json": DecisionRequest,
        "task-suite-spec.json": TaskSuiteSpec,
        "task-suite.json": TaskSuiteSpec,
        "application-spec.json": ApplicationSpec,
        "application.json": ApplicationSpec,
        "evaluation-protocol.json": EvaluationProtocol,
        "eval-protocol.json": EvaluationProtocol,
        "serving-workload-spec.json": ServingWorkloadSpec,
        "serving.json": ServingWorkloadSpec,
        "workload.json": WorkloadSpec,
        "solution.json": InferenceSolution,
        "report.json": DecisionReport,
        "attempt.json": TaskAttemptRecord,
        "verification.json": VerificationReport,
        "plan.json": DeploymentPlan,
        "executor-plan.json": DeploymentPlan,
        "vision-plan.json": DeploymentPlan,
        "original-plan.json": DeploymentPlan,
        "corrected-plan.json": DeploymentPlan,
        "original-verification.json": VerificationReport,
        "corrected-verification.json": VerificationReport,
        "diagnosis.json": DiagnosisRule,
        "fixed.json": RemediationRecord,
        "tradeoff.json": RemediationRecord,
        "unverified.json": RemediationRecord,
        "provenance.json": ExternalFormatProvenance,
        "high-throughput-task-failure.json": DecisionReport,
        "serving-slo-failure.json": DecisionReport,
        "incomparable-cost-boundary.json": DecisionReport,
    }
    return by_name


def _domain_fixture_files() -> list[Path]:
    roots = [FIXTURES / "phase-1a-run", FIXTURES / "golden"]
    files = [p for root in roots for p in sorted(root.rglob("*.json"))]
    files += sorted((FIXTURES / "external-formats").glob("*/provenance.json"))
    return files


@pytest.mark.parametrize(
    "path", _domain_fixture_files(), ids=lambda p: str(p.relative_to(FIXTURES))
)
def test_fixture_validates_strictly(path: Path) -> None:
    schemas = _fixture_schemas()
    schema = schemas[path.name]
    schema.model_validate(json.loads(path.read_text()))


# Test-generated binaries that are neither Apron documents nor pinned
# third-party files.  Each needs a reason.
_SYNTHETIC = {
    "external-formats/huggingface-hub/synthetic.safetensors": (
        "synthetic header written for the safetensors reader tests; not a third-party file"
    ),
}


def _pinned_third_party() -> dict[Path, dict[str, Any]]:
    """Every third-party file named by an external-format provenance.json."""
    pinned: dict[Path, dict[str, Any]] = {}
    for provenance in sorted((FIXTURES / "external-formats").glob("*/provenance.json")):
        for entry in json.loads(provenance.read_text())["pinned_files"]:
            pinned[provenance.parent / entry["local_path"]] = entry
    return pinned


def test_every_fixture_file_is_accounted_for() -> None:
    """Each file under tests/fixtures/ is exactly one of: an Apron document
    validated strictly (above), a pinned third-party file whose bytes match
    its provenance entry (below), or a listed synthetic artifact.  Third-party
    formats (vLLM source, recipes, schemas) are not Apron documents; strict
    for them means byte-identical to what was pinned."""
    documents = set(_domain_fixture_files())
    pinned = set(_pinned_third_party())
    synthetic = {FIXTURES / rel for rel in _SYNTHETIC}
    unaccounted = [
        str(p.relative_to(FIXTURES))
        for p in sorted(FIXTURES.rglob("*"))
        if p.is_file() and p not in documents | pinned | synthetic
    ]
    assert unaccounted == []
    for path in synthetic:
        assert path.exists(), path


@pytest.mark.parametrize(
    "path", sorted(_pinned_third_party()), ids=lambda p: str(p.relative_to(FIXTURES))
)
def test_pinned_third_party_file_matches_provenance(path: Path) -> None:
    import hashlib

    entry = _pinned_third_party()[path]
    data = path.read_bytes()
    assert len(data) == entry["size_bytes"]
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
