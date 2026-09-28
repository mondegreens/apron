"""Strict shape of a recorded L0-F classifier fixture (tests/fixtures/l0f/classN.json).

Written by scripts/record_l0f_classifier.py; replayed by
test_fix_proof_real_logs.py.  Unknown keys fail (F2), like every fixture.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


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
    same_final_norm: bool | None = None


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
    f32_bytes: int
    quantized: bool
    lm_head_bytes: int
    mtp_bytes: (
        int  # multi-token-prediction layers vLLM does not load (plan_pipeline.mtp_tensor_bytes)
    )
    tie_word_embeddings: bool
    vocab_size: int
    hidden_size: int
    torch_dtype: str
    # The config fields the activation estimate reads (calculator.activation_config).
    activation: dict[str, int | float | bool | str]
    ssm: dict[str, int] | None = None
    # Loaded bytes beyond the generic width rule (plan_pipeline.widened_tensor_bytes)
    # and loaded bytes every tensor-parallel rank holds whole
    # (plan_pipeline.replicated_tensor_bytes); 0 for the families not traced.
    widened_bytes: int = 0
    replicated_bytes: int = 0

    @field_validator("activation")
    @classmethod
    def _known_activation_fields(
        cls, value: dict[str, int | float | bool | str]
    ) -> dict[str, int | float | bool | str]:
        from apron.domain.mechanisms.calculator import (
            ACTIVATION_FIELDS,
            DERIVED_ACTIVATION_FIELDS,
            ENCODER_FIELDS,
        )

        known = {
            *ACTIVATION_FIELDS,
            *DERIVED_ACTIVATION_FIELDS,
            *ENCODER_FIELDS,
            "vision_config",
            "audio_config",
        }
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"unknown activation fields: {sorted(unknown)}")
        return value


def load_weight_bytes(data: dict[str, Any]) -> dict[str, RecordedWeightBytes]:
    return {model: RecordedWeightBytes.model_validate(v) for model, v in data.items()}


# ---------------------------------------------------------------------------
# tests/fixtures/cohort/headers/*.json (scripts/record_tensor_headers.py)
# ---------------------------------------------------------------------------


class RecordedHeaders(_Strict):
    model_id: str
    revision: str
    config: dict[str, Any]  # the checkpoint's config.json, as published
    # tensor name (expert and n-gram shard indices folded) -> [dtype, bytes, tensors]
    tensors: dict[str, tuple[str, int, int]]
