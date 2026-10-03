"""F1: fingerprints hash IDENTITY fields at every depth.

A DISPLAY change on a nested model leaves the digest unchanged; an IDENTITY
change on a nested model changes it.  Containers (tuples, lists, dict values)
are projected element by element.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict

from apron.domain.artifacts.identity import ArtifactIdentity
from apron.domain.capabilities import CapabilitySignature
from apron.domain.fingerprints import DISPLAY, IDENTITY, fingerprint_hex, identity_projection
from apron.domain.mechanisms import ComponentMechanism
from apron.domain.schemas.models import ArtifactSpec, ModelSpec
from apron.domain.schemas.reports import CandidateEntry, DecisionReport
from apron.domain.schemas.tasks import TaskSuiteSpec

_FP = "1220" + "ab" * 32


class _Leaf(BaseModel):
    model_config = ConfigDict(frozen=True)
    key: Annotated[str, IDENTITY]
    note: Annotated[str, DISPLAY] = ""


class _Holder(BaseModel):
    model_config = ConfigDict(frozen=True)
    one: Annotated[_Leaf, IDENTITY]
    many: Annotated[tuple[_Leaf, ...], IDENTITY] = ()
    listed: Annotated[list[_Leaf], IDENTITY] = []
    mapped: Annotated[dict[str, _Leaf], IDENTITY] = {}
    label: Annotated[str, DISPLAY] = ""


class _Plain(BaseModel):
    """A model with no markers is plain data: every field is identity."""

    a: int
    b: str


class _HoldsPlain(BaseModel):
    model_config = ConfigDict(frozen=True)
    plain: Annotated[_Plain, IDENTITY]


def _holder(note: str, key: str = "k") -> _Holder:
    leaf = _Leaf(key=key, note=note)
    return _Holder(one=leaf, many=(leaf,), listed=[leaf], mapped={"x": leaf}, label=note)


def test_nested_display_change_leaves_digest_unchanged() -> None:
    assert fingerprint_hex(_holder("first")) == fingerprint_hex(_holder("second"))


def test_nested_identity_change_changes_digest() -> None:
    assert fingerprint_hex(_holder("n", key="a")) != fingerprint_hex(_holder("n", key="b"))


def test_identity_change_inside_each_container_changes_digest() -> None:
    base = _holder("n")
    other = _Leaf(key="other")
    variants = [
        base.model_copy(update={"one": other}),
        base.model_copy(update={"many": (other,)}),
        base.model_copy(update={"listed": [other]}),
        base.model_copy(update={"mapped": {"x": other}}),
    ]
    digests = {fingerprint_hex(v) for v in variants}
    assert fingerprint_hex(base) not in digests
    assert len(digests) == len(variants)


def test_projection_drops_nested_display_fields() -> None:
    projected = identity_projection(_holder("hidden"))
    assert projected == {
        "listed": [{"key": "k"}],
        "many": [{"key": "k"}],
        "mapped": {"x": {"key": "k"}},
        "one": {"key": "k"},
    }


def test_unmarked_nested_model_keeps_every_field() -> None:
    a = _HoldsPlain(plain=_Plain(a=1, b="x"))
    b = _HoldsPlain(plain=_Plain(a=1, b="y"))
    assert identity_projection(a) == {"plain": {"a": 1, "b": "x"}}
    assert fingerprint_hex(a) != fingerprint_hex(b)


def test_flat_model_digest_matches_previous_definition() -> None:
    """A model without nested models hashes exactly as before F1."""
    from apron.domain.canonical import canonicalize, digest_hex
    from apron.domain.fingerprints import identity_field_names

    leaf = _Leaf(key="k", note="n")
    full = leaf.model_dump(mode="json")
    previous = digest_hex(canonicalize({k: full[k] for k in identity_field_names(_Leaf)}))
    assert fingerprint_hex(leaf) == previous


# --- real schemas whose digest F1 changes -----------------------------------


def _cap(schema_version: int) -> CapabilitySignature:
    return CapabilitySignature(
        schema_version=schema_version,
        operation="text_generation",
        required_inputs=("text",),
        output_representation="generated_text",
    )


def test_task_suite_ignores_nested_capability_display() -> None:
    a = TaskSuiteSpec(name="s", version="1", required_capabilities=(_cap(1),))
    b = TaskSuiteSpec(name="s", version="1", required_capabilities=(_cap(2),))
    assert fingerprint_hex(a) == fingerprint_hex(b)


def test_model_spec_ignores_nested_component_display() -> None:
    a = ModelSpec(components=(ComponentMechanism(mechanism="m", role="decoder"),))
    b = ModelSpec(
        components=(ComponentMechanism(schema_version=9, mechanism="m", role="decoder"),)
    )
    assert fingerprint_hex(a) == fingerprint_hex(b)
    c = ModelSpec(components=(ComponentMechanism(mechanism="other", role="decoder"),))
    assert fingerprint_hex(a) != fingerprint_hex(c)


def test_artifact_spec_ignores_nested_identity_display() -> None:
    a = ArtifactSpec(identity=ArtifactIdentity(content_digest="d"))
    b = ArtifactSpec(identity=ArtifactIdentity(schema_version=3, content_digest="d"))
    assert fingerprint_hex(a) == fingerprint_hex(b)


def test_decision_report_ignores_candidate_display() -> None:
    def report(reason: str | None) -> DecisionReport:
        return DecisionReport(
            decision_request_digest="r",
            candidates=(
                CandidateEntry(
                    solution_fingerprint=_FP,
                    qualification_status="candidate",
                    rejection_reason=reason,
                ),
            ),
        )

    assert fingerprint_hex(report("a")) == fingerprint_hex(report("b"))
