"""LLM-based error classification and extraction.

Classification uses a taxonomy derived from loaded rules.
Extraction uses per-class typed schemas with evidence-substring
verification to prevent hallucinated values.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from apron.domain.canonical import canonicalize, digest_hex
from apron.domain.diagnosis import (
    build_extraction_enums,
    build_extraction_schemas,
    build_failure_classes,
)

logger = logging.getLogger(__name__)

_NOISE_PATTERNS = re.compile(
    r"^\s*$"
    r"|Downloading.*\|"
    r"|downloading.*shards"
    r"|\.safetensors:"
    r"|^INFO.*HTTP Request:"
    r"|^INFO.*graphql"
    r"|^DEBUG.*contextlib"
    r"|^INFO.*Connecting to"
    r"|^INFO.*Starting download"
    r"|^INFO.*Download complete"
    r"|token_healing"
    r"|Special tokens"
    r"|^Fetching \d+ files",
    re.IGNORECASE,
)


# vLLM's engine-core wrapper: the real cause is printed above it
# (v1/engine/utils.py:1320-1322).  Classifying the wrapper line is wrong.
ENGINE_WRAPPER = "Engine core initialization failed. See root cause above."
_EXCEPTION_LINE = re.compile(r"^\S*(Error|Exception)\b.*:", re.MULTILINE)


def root_cause_window(error: str, context_lines: int = 40) -> str | None:
    """The root-cause exception above vLLM's engine-core wrapper, with context.

    Returns ``None`` when the log has no wrapper.  Otherwise returns the
    ``context_lines`` before the last exception line that precedes the
    wrapper, through that line — so the classifier sees the cause, never
    only the wrapper.
    """
    at = error.rfind(ENGINE_WRAPPER)
    if at < 0:
        return None
    wrapper_line_start = error.rfind("\n", 0, at) + 1
    before = error[:wrapper_line_start]
    causes = list(_EXCEPTION_LINE.finditer(before))
    if not causes:
        return None
    cause = causes[-1]
    line_end = before.find("\n", cause.end())
    line_end = len(before) if line_end < 0 else line_end
    lines = before[:line_end].splitlines()
    return "\n".join(lines[-context_lines:])


def _extract_relevant(error: str, limit: int = 8000) -> str:
    """Extract error-relevant lines, stripping download/init noise.

    When the log ends in the engine-core wrapper, the root-cause window is
    placed first so truncation can never drop it.
    """
    window = root_cause_window(error)
    if window is not None:
        tail = error[-min(2000, limit // 4) :]
        joined = f"ROOT CAUSE (above the engine-core wrapper):\n{window}\n...\n{tail}"
        return joined[:limit]
    if len(error) <= limit:
        return error
    lines = error.splitlines(keepends=True)
    kept = [ln for ln in lines if not _NOISE_PATTERNS.search(ln)]
    joined = "".join(kept)
    if len(joined) <= limit:
        return joined
    return joined[-limit:]


_SYSTEM_PROMPT_TEMPLATE = """\
You are a deployment failure classifier for GPU inference engines (vLLM, SGLang).

Given error output from a failed deployment, classify the failure and extract values.

Failure classes:
{class_list}
- unknown: None of the above, or insufficient information to classify

When the log ends with a wrapper such as "Engine core initialization failed. See root cause \
above.", classify the root-cause exception printed above the wrapper — never the wrapper itself.

For each extracted value, include the verbatim substring from the error where you found it.
If a value is not present in the error text, set it to null — never guess.
Byte fields (names ending in _bytes) are integers: convert "1.50 GiB" to 1610612736.\
"""


_CLASS_DESCRIPTIONS: dict[str, str] = {
    "oom_weight_load": (
        "Out of memory while loading model weights: 'Failed to load model - not enough "
        "GPU memory' with a torch.cuda.OutOfMemoryError (original error: CUDA out of "
        "memory. Tried to allocate ...). The weights do not fit on this GPU."
    ),
    "oom_kv_cache": (
        "Memory pressure after the weights loaded: the KV cache cannot hold one full "
        "sequence ('larger than the available KV cache memory', 'estimated maximum "
        "model length'), 'No available memory for the cache blocks', or CUDA OOM while "
        "warming up the sampler/pooler with N dummy requests. Use this even when "
        "max_model_len appears in the error, if the root cause is insufficient memory."
    ),
    "max_model_len": (
        "User-specified max_model_len exceeds the model's derived maximum from "
        "max_position_embeddings or model_max_length. NOT memory-related."
    ),
    "dtype_incompatible": "Model or quantization does not support the requested dtype.",
    "tp_divisibility": "Attention heads not divisible by tensor parallel size.",
    "quant_compute_capability": "GPU compute capability too low for the quantization method.",
    "lora_config": "LoRA misconfiguration (not enabled, parameter errors).",
    "speculative_config": "Draft model, MTP, Eagle, or speculative decoding misconfiguration.",
    "parallelism_config": "DP, PP, EP, context parallelism, or world_size misconfiguration.",
    "compilation_config": "CUDA graphs, custom ops, inductor, or compilation misconfiguration.",
    "config_incompatible": "General configuration incompatibility between flags.",
    "profiler_config": "Torch or nsight profiler misconfiguration.",
    "multimodal_config": "Vision encoder, mm processor, or multimodal budget misconfiguration.",
    "kv_transfer_config": "KV connector or EC connector role misconfiguration.",
    "platform_unsupported": "Feature not supported on this device or platform.",
    "scheduler_config": "Batched tokens, prefill threshold, or scheduler misconfiguration.",
    "model_runtime": "Model-specific runtime error (activation, weight loading, kernel).",
    "other_correctable": "Config conflict or flag incompatibility not in other classes.",
}


def _build_system_prompt(rules: list[dict[str, Any]]) -> str:
    lines = []
    families_seen: set[str] = set()
    for rule in rules:
        family = rule.get("error_family", "")
        if family in families_seen:
            continue
        families_seen.add(family)
        desc = _CLASS_DESCRIPTIONS.get(family, "")
        if not desc:
            examples = rule.get("examples", [])
            desc = examples[0][:80] if examples else family
        lines.append(f"- {family}: {desc}")
    return _SYSTEM_PROMPT_TEMPLATE.format(class_list="\n".join(lines))


def _build_extraction_tool(
    failure_class: str,
    classes: list[str],
    schemas: dict[str, list[tuple[str, type]]],
    enums: dict[str, dict[str, tuple[str, ...]]] | None = None,
) -> dict[str, Any]:
    schema = schemas.get(failure_class, [])
    allowed = (enums or {}).get(failure_class, {})

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
        if field_name in allowed:
            properties[field_name]["enum"] = [*allowed[field_name], None]
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

DEFAULT_CLASSIFIER_MODEL = "claude-haiku-4-5-20251001"


def classifier_input_digest(model: str, system_prompt: str, classifier_input: str) -> str:
    """Digest of exactly what the classifier sees (F6 provenance)."""
    return digest_hex(
        canonicalize({"model": model, "system": system_prompt, "input": classifier_input})
    )


def classify_and_extract(
    error: str,
    rules: list[dict[str, Any]] | None = None,
    model: str = DEFAULT_CLASSIFIER_MODEL,
) -> dict[str, Any]:
    """Classify an error and extract typed values via LLM.

    Results are cached by error content hash within the process
    lifetime — same error always returns the same classification,
    preserving determinism for records and corrections.

    Every result carries ``classifier_model_id`` and
    ``classifier_input_digest`` so a remediation record can state which
    classifier saw which input (F6).  The API key comes from the
    environment only (``anthropic.Anthropic()`` reads ``ANTHROPIC_API_KEY``).
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
    truncated = _extract_relevant(error, limit=8000)
    provenance = {
        "classifier_model_id": model,
        "classifier_input_digest": classifier_input_digest(model, system_prompt, truncated),
    }
    logger.debug(
        "Classifier input (%d→%d chars): ...%s",
        len(error),
        len(truncated),
        truncated[-500:],
    )

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
        unknown = {
            "failure_class": "unknown",
            "confidence": 0.0,
            "evidence_span": "",
            **provenance,
        }
        _classification_cache[cache_key] = unknown
        return unknown

    result = {**dict(tool_use.input), **provenance}  # type: ignore[arg-type]
    failure_class = str(result.get("failure_class", "unknown"))

    if failure_class == "unknown" or failure_class not in schemas:
        _classification_cache[cache_key] = result
        return result

    extract_tool: Any = _build_extraction_tool(
        failure_class, classes, schemas, build_extraction_enums(rules)
    )
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
    extracted.update(provenance)

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
