"""Sanitization — credential stripping and provenance validation."""

import copy
import re
from typing import Any

from apron.domain.protocols import RecordStore

SENSITIVE_PATTERNS = [
    re.compile(r"RUNPOD_API_KEY|HF_TOKEN|HUGGING_FACE_HUB_TOKEN"),
    re.compile(r"Bearer\s+\S+"),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"rp_[a-zA-Z0-9]{20,}"),
]

_FINGERPRINT_PATTERN = re.compile(r"^1220[0-9a-f]{64}$")


def _redact_value(value: str) -> str:
    if _FINGERPRINT_PATTERN.match(value):
        return value
    for pattern in SENSITIVE_PATTERNS:
        if pattern.search(value):
            return "[REDACTED]"
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
