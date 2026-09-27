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
