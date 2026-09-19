"""LLM-based error classification and extraction.

Replaces regex pattern matching with structured LLM output.
Classification uses the engine-independent failure taxonomy from
domain/diagnosis.py. Extraction uses per-class typed schemas with
evidence-substring verification to prevent hallucinated values.
"""

from __future__ import annotations

import logging
from typing import Any

from apron.domain.diagnosis import EXTRACTION_SCHEMAS, FailureClass

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a deployment failure classifier for GPU inference engines (vLLM, SGLang).

Given error output from a failed deployment, classify the failure and extract values.

Failure classes:
- oom: Out of memory — CUDA OOM during weight loading, warmup, or KV cache insufficient
- max_model_len: User-specified max_model_len exceeds derived maximum
- dtype_incompatible: Requested dtype not supported by model/GPU/quant
- tp_divisibility: Attention heads not evenly divisible by tensor parallel size
- quant_compute_capability: GPU compute capability too low for the quantization method
- engine_init: Correctable configuration error (LoRA not enabled, unsupported task, etc.)
- unknown: None of the above, or insufficient information to classify

For each extracted value, include the verbatim substring from the error where you found it.
If a value is not present in the error text, set it to null — never guess.\
"""


def _build_extraction_tool(failure_class: str) -> dict[str, Any]:
    """Build a tool schema for extracting values from a classified error."""
    schema = EXTRACTION_SCHEMAS.get(failure_class, [])

    properties: dict[str, Any] = {
        "failure_class": {
            "type": "string",
            "enum": [fc.value for fc in FailureClass],
        },
        "confidence": {
            "type": "number",
            "description": "Classification confidence 0.0-1.0",
        },
        "evidence_span": {
            "type": "string",
            "description": "Verbatim substring identifying this failure class",
        },
    }

    for field_name, field_type in schema:
        type_str = {int: "integer", float: "number", str: "string"}[field_type]
        properties[field_name] = {
            "type": [type_str, "null"],
            "description": "Extracted from error text. Null if not found.",
        }
        properties[f"{field_name}_evidence"] = {
            "type": ["string", "null"],
            "description": f"Verbatim substring where {field_name} was found. Null if not found.",
        }

    return {
        "name": "classify_and_extract",
        "description": "Classify the deployment failure and extract typed values",
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": ["failure_class", "confidence", "evidence_span"],
        },
    }


def _classify_tool() -> dict[str, Any]:
    """Initial classification tool — just the failure class."""
    return {
        "name": "classify_failure",
        "description": "Classify the deployment failure",
        "input_schema": {
            "type": "object",
            "properties": {
                "failure_class": {
                    "type": "string",
                    "enum": [fc.value for fc in FailureClass],
                },
                "confidence": {
                    "type": "number",
                    "description": "Classification confidence 0.0-1.0",
                },
                "evidence_span": {
                    "type": "string",
                    "description": "Verbatim substring identifying this failure class",
                },
            },
            "required": ["failure_class", "confidence", "evidence_span"],
        },
    }


def classify_and_extract(
    error: str,
    model: str = "claude-haiku-4-5-20251001",
) -> dict[str, Any]:
    """Classify an error and extract typed values via LLM.

    Two-step: first classify (cheap, schema-light), then extract
    with the class-specific schema (richer fields).

    Returns a dict with failure_class, confidence, evidence_span,
    and per-field extracted values with evidence substrings.
    """
    import anthropic

    client = anthropic.Anthropic()
    truncated = error[:8000]

    classify_tool: Any = _classify_tool()
    classification = client.messages.create(
        model=model,
        max_tokens=256,
        system=_SYSTEM_PROMPT,
        tools=[classify_tool],
        tool_choice={"type": "tool", "name": "classify_failure"},
        messages=[
            {"role": "user", "content": f"Classify this deployment failure:\n\n{truncated}"}
        ],
    )

    tool_use = next((b for b in classification.content if b.type == "tool_use"), None)
    if tool_use is None:
        return {"failure_class": "unknown", "confidence": 0.0, "evidence_span": ""}

    result = dict(tool_use.input)  # type: ignore[arg-type]
    failure_class = str(result.get("failure_class", "unknown"))

    if failure_class == "unknown" or failure_class not in EXTRACTION_SCHEMAS:
        return result

    extract_tool: Any = _build_extraction_tool(failure_class)
    extraction = client.messages.create(
        model=model,
        max_tokens=512,
        system=_SYSTEM_PROMPT,
        tools=[extract_tool],
        tool_choice={"type": "tool", "name": "classify_and_extract"},
        messages=[
            {
                "role": "user",
                "content": (
                    f"This is a {failure_class} failure. Extract the typed values.\n\n{truncated}"
                ),
            }
        ],
    )

    tool_use_2 = next((b for b in extraction.content if b.type == "tool_use"), None)
    if tool_use_2 is None:
        return result

    extracted = dict(tool_use_2.input)  # type: ignore[arg-type]
    extracted["failure_class"] = failure_class
    extracted["confidence"] = result.get("confidence", 0.0)
    extracted["evidence_span"] = result.get("evidence_span", "")

    verified = _verify_evidence(extracted, error)
    return verified


def _verify_evidence(extracted: dict[str, Any], original_error: str) -> dict[str, Any]:
    """Verify that evidence substrings actually appear in the original error.

    Any extracted value whose evidence substring is NOT a substring of the
    original error is removed (set to None). No regex — just string containment.
    """
    result = dict(extracted)
    schema = EXTRACTION_SCHEMAS.get(extracted.get("failure_class", ""), [])

    for field_name, _ in schema:
        evidence_key = f"{field_name}_evidence"
        evidence = result.get(evidence_key)
        value = result.get(field_name)

        if value is None:
            continue

        if evidence is None or evidence not in original_error:
            logger.warning(
                "Removing unverified extraction: %s=%r (evidence %r not found in error)",
                field_name,
                value,
                evidence,
            )
            result[field_name] = None
            result[evidence_key] = None

    return result
