"""Strict shape of a recorded L0-F classifier fixture (tests/fixtures/l0f/classN.json).

Written by scripts/record_l0f_classifier.py; replayed by
test_fix_proof_real_logs.py.  Unknown keys fail (F2), like every fixture.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class RecordedClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    failure_class: str
    confidence: float
    evidence_span: str
    classifier_model_id: str
    classifier_input_digest: str
    classifier_cost_usd: float | None


class L0FRecording(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recorded_at: str
    failure_class: int
    expected_family: str
    broken_plan_digest: str
    source_record: str
    log: str
    gpu_sku: str
    model_config_: dict[str, Any] = {}
    predicted_total_bytes: int | None
    classification: RecordedClassification
    extraction: dict[str, int | float | str]
    classifier_usage: dict[str, int] | None

    @classmethod
    def load(cls, data: dict[str, Any]) -> L0FRecording:
        data = dict(data)
        data["model_config_"] = data.pop("model_config")
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# tests/fixtures/l0f/classN-lineage.json (scripts/record_class6_lineage.py)
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BaseProposal(_Strict):
    base_model_id: str
    reason: str
    classifier_model_id: str
    classifier_input_digest: str
    classifier_usage: dict[str, int]
    classifier_cost_usd: float | None


class BaseModelEvidence(_Strict):
    id: str | None
    source: str
    proposal: BaseProposal | None = None
    confirmed: bool | None = None


class Considered(_Strict):
    model_id: str
    method: str
    bits: int | None
    min_capability: int | None
    dropped: str | None = None


class LineageEvidence(_Strict):
    requested_model_id: str
    requested_weight_bits: int | None
    base_model: BaseModelEvidence
    dropped_by_format: dict[str, int] = {}
    considered: list[Considered] = []
    result: str


class Candidate(_Strict):
    model_id: str
    shape: dict[str, Any]
    weight_bits: int
    min_capability: int
    quant_method: str | None
    downloads: int


class LineageSearch(_Strict):
    requested_model_id: str
    base_model_id: str
    requested_weight_bits: int
    requested_shape: dict[str, Any]
    candidates: list[Candidate]


class LineageRecording(_Strict):
    recorded_at: str
    evidence: LineageEvidence
    search: LineageSearch | None


# ---------------------------------------------------------------------------
# tests/fixtures/cohort/weight-bytes.json (scripts/record_weight_bytes.py)
# ---------------------------------------------------------------------------


class RecordedWeightBytes(_Strict):
    revision: str
    total_bytes: int
    lm_head_bytes: int
    tie_word_embeddings: bool


def load_weight_bytes(data: dict[str, Any]) -> dict[str, RecordedWeightBytes]:
    return {model: RecordedWeightBytes.model_validate(v) for model, v in data.items()}
