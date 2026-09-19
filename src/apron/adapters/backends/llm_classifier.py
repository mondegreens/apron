"""LLM-based error classification and extraction.

Classification uses a taxonomy derived from loaded rules.
Extraction uses per-class typed schemas with evidence-substring
verification to prevent hallucinated values.
"""

from __future__ import annotations

import logging
from typing import Any

from apron.domain.diagnosis import build_extraction_schemas, build_failure_classes

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_TEMPLATE = """\
You are a deployment failure classifier for GPU inference engines (vLLM, SGLang).

Given error output from a failed deployment, classify the failure and extract values.

Failure classes:
{class_list}
- unknown: None of the above, or insufficient information to classify

For each extracted value, include the verbatim substring from the error where you found it.
If a value is not present in the error text, set it to null — never guess.\
"""


def _build_system_prompt(rules: list[dict[str, Any]]) -> str:
    lines = []
    for rule in rules:
        family = rule.get("error_family", "")
        examples = rule.get("examples", [])
        desc = examples[0][:80] if examples else family
        lines.append(f"- {family}: {desc}")
    return _SYSTEM_PROMPT_TEMPLATE.format(class_list="\n".join(lines))


def _build_extraction_tool(
    failure_class: str,
    classes: list[str],
    schemas: dict[str, list[tuple[str, type]]],
) -> dict[str, Any]:
    schema = schemas.get(failure_class, [])

    properties: dict[str, Any] = {
        "failure_class": {"type": "string", "enum": classes},
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
        type_str = {int: "integer", float: "number", str: "string"}.get(field_type, "string")
        properties[field_name] = {
            "type": [type_str, "null"],
            "description": "Extracted from error text. Null if not found.",
        }
        properties[f"{field_name}_evidence"] = {
            "type": ["string", "null"],
            "description": (
                f"Verbatim substring where {field_name} was found. Null if not found."
            ),
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


def _classify_tool(classes: list[str]) -> dict[str, Any]:
    return {
        "name": "classify_failure",
        "description": "Classify the deployment failure",
        "input_schema": {
            "type": "object",
            "properties": {
                "failure_class": {"type": "string", "enum": classes},
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


_classification_cache: dict[str, dict[str, Any]] = {}


def classify_and_extract(
    error: str,
    rules: list[dict[str, Any]] | None = None,
    model: str = "claude-haiku-4-5-20251001",
) -> dict[str, Any]:
    """Classify an error and extract typed values via LLM.

    Results are cached by error content hash within the process
    lifetime — same error always returns the same classification,
    preserving determinism for records and corrections.
    """
    import hashlib

    import anthropic

    if rules is None:
        rules = []

    cache_key = hashlib.sha256(error.encode()[:8000]).hexdigest()
    if cache_key in _classification_cache:
        return _classification_cache[cache_key]

    classes = build_failure_classes(rules)
    schemas = build_extraction_schemas(rules)
    system_prompt = _build_system_prompt(rules)

    client = anthropic.Anthropic()
    truncated = error[:8000]

    classify_tool: Any = _classify_tool(classes)
    classification = client.messages.create(
        model=model,
        max_tokens=256,
        system=system_prompt,
        tools=[classify_tool],
        tool_choice={"type": "tool", "name": "classify_failure"},
        messages=[
            {
                "role": "user",
                "content": f"Classify this deployment failure:\n\n{truncated}",
            }
        ],
    )

    tool_use = next((b for b in classification.content if b.type == "tool_use"), None)
    if tool_use is None:
        unknown = {"failure_class": "unknown", "confidence": 0.0, "evidence_span": ""}
        _classification_cache[cache_key] = unknown
        return unknown

    result = dict(tool_use.input)  # type: ignore[arg-type]
    failure_class = str(result.get("failure_class", "unknown"))

    if failure_class == "unknown" or failure_class not in schemas:
        _classification_cache[cache_key] = result
        return result

    extract_tool: Any = _build_extraction_tool(failure_class, classes, schemas)
    extraction = client.messages.create(
        model=model,
        max_tokens=512,
        system=system_prompt,
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
        _classification_cache[cache_key] = result
        return result

    extracted = dict(tool_use_2.input)  # type: ignore[arg-type]
    extracted["failure_class"] = failure_class
    extracted["confidence"] = result.get("confidence", 0.0)
    extracted["evidence_span"] = result.get("evidence_span", "")

    verified = _verify_evidence(extracted, error, schemas)
    _classification_cache[cache_key] = verified
    return verified


def _verify_evidence(
    extracted: dict[str, Any],
    original_error: str,
    schemas: dict[str, list[tuple[str, type]]],
) -> dict[str, Any]:
    """Verify that evidence substrings appear in the original error."""
    result = dict(extracted)
    schema = schemas.get(extracted.get("failure_class", ""), [])

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
