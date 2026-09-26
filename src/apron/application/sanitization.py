"""Sanitization — credential stripping and provenance validation.

Two tools: ``sanitize`` redacts whole string values in a record that carry a
secret (records never store half a credential), and ``mask_secrets`` masks
only the secret spans in free text such as logs and captured log tails, so
the surrounding diagnostic text survives.  ``SecretMaskingFilter`` applies
``mask_secrets`` to every log record it sees (F6).
"""

import copy
import logging
import re
from typing import Any

from apron.domain.protocols import RecordStore

# Secret *values*.  Order matters for masking: specific prefixes first.
SECRET_VALUE_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    re.compile(r"sk-proj-[A-Za-z0-9_-]{20,}"),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{30,}"),
    re.compile(r"rpa?_[a-zA-Z0-9]{20,}"),
    re.compile(r"Bearer\s+\S+"),
    re.compile(r"api_key=[^&\s\"']+"),
    re.compile(
        r"(?:RUNPOD_API_KEY|HF_TOKEN|HUGGING_FACE_HUB_TOKEN|ANTHROPIC_API_KEY)\s*[=:]\s*\S+"
    ),
]

SENSITIVE_PATTERNS = [
    re.compile(r"RUNPOD_API_KEY|HF_TOKEN|HUGGING_FACE_HUB_TOKEN|ANTHROPIC_API_KEY"),
    *SECRET_VALUE_PATTERNS,
]

REDACTED = "[REDACTED]"


def contains_secret(text: str) -> bool:
    """True if *text* holds a secret value (used by publication scans)."""
    return any(p.search(text) for p in SECRET_VALUE_PATTERNS)


def mask_secrets(text: str) -> str:
    """Replace each secret span in *text* with ``[REDACTED]``; keep the rest."""
    for pattern in SECRET_VALUE_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


class SecretMaskingFilter(logging.Filter):
    """Logging filter that masks secrets in the formatted message."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        masked = mask_secrets(message)
        if masked != message:
            record.msg = masked
            record.args = None
        return True


_FINGERPRINT_PATTERN = re.compile(r"^1220[0-9a-f]{64}$")


def _redact_value(value: str) -> str:
    if _FINGERPRINT_PATTERN.match(value):
        return value
    for pattern in SENSITIVE_PATTERNS:
        if pattern.search(value):
            return REDACTED
    return value


def _walk(obj: Any) -> Any:
    if isinstance(obj, str):
        return _redact_value(obj)
    if isinstance(obj, dict):
        return {k: _walk(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_walk(v) for v in obj)
    return obj


def sanitize(record: dict[str, Any]) -> dict[str, Any]:
    return _walk(copy.deepcopy(record))


def validate_provenance(
    record: dict[str, Any],
    store: RecordStore,
) -> list[str]:
    errors: list[str] = []
    _check_fields(record, store, errors, prefix="")
    return errors


def _check_fields(
    obj: Any,
    store: RecordStore,
    errors: list[str],
    prefix: str,
) -> None:
    if not isinstance(obj, dict):
        return
    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else key
        if value is None:
            continue
        if key.endswith("_fingerprint") and isinstance(value, str):
            if not _FINGERPRINT_PATTERN.match(value):
                errors.append(f"{path}: invalid fingerprint format")
        elif key.endswith("_digest") and isinstance(value, str):
            if store.retrieve(value) is None:
                errors.append(f"{path}: dangling digest reference '{value}'")
        elif key == "corrects" and isinstance(value, str):
            if store.retrieve(value) is None:
                errors.append(f"{path}: dangling corrects reference '{value}'")
        elif isinstance(value, dict):
            _check_fields(value, store, errors, path)
