"""Tests for sanitization and provenance validation."""

from typing import Any

from apron.application.sanitization import sanitize, validate_provenance


class FakeStore:
    def __init__(self, records: dict[str, Any] | None = None) -> None:
        self._records = records or {}

    def store(self, record: dict[str, Any]) -> str:
        return "digest"

    def retrieve(self, digest: str) -> dict[str, Any] | None:
        return self._records.get(digest)

    def search(self, prefix: str) -> list[str]:
        return []


# ---- sanitize() ----


def test_redact_runpod_api_key():
    record = {"engine_configuration": {"key": "RUNPOD_API_KEY=rp_abc12345678901234567890"}}
    result = sanitize(record)
    assert result["engine_configuration"]["key"] == "[REDACTED]"


def test_redact_bearer_token():
    record = {"nested": {"auth": "Bearer sk-abcdefghijklmnopqrstuvwxyz"}}
    result = sanitize(record)
    assert result["nested"]["auth"] == "[REDACTED]"


def test_redact_rp_token():
    record = {"token": "rp_abcdefghijklmnopqrst1234"}
    result = sanitize(record)
    assert result["token"] == "[REDACTED]"


def test_preserve_fingerprint():
    fp = "1220" + "aa" * 32
    record = {"execution_fingerprint": fp}
    result = sanitize(record)
    assert result["execution_fingerprint"] == fp


def test_sanitize_returns_copy():
    record = {"key": "RUNPOD_API_KEY=secret"}
    result = sanitize(record)
    assert record["key"] == "RUNPOD_API_KEY=secret"
    assert result["key"] == "[REDACTED]"


def test_deeply_nested_redaction():
    record = {"a": {"b": {"c": [{"token": "sk-12345678901234567890abcdef"}]}}}
    result = sanitize(record)
    assert result["a"]["b"]["c"][0]["token"] == "[REDACTED]"


def test_clean_record_unchanged():
    record = {"model_id": "Qwen/Qwen3-8B", "tp": 1}
    result = sanitize(record)
    assert result == record


# ---- validate_provenance() ----


def test_clean_record_no_errors():
    record = {"model_id": "test", "tp": 1}
    errors = validate_provenance(record, FakeStore())
    assert errors == []


def test_valid_fingerprint_no_errors():
    fp = "1220" + "bb" * 32
    record = {"execution_fingerprint": fp}
    errors = validate_provenance(record, FakeStore())
    assert errors == []


def test_invalid_fingerprint_format():
    record = {"execution_fingerprint": "not-a-fingerprint"}
    errors = validate_provenance(record, FakeStore())
    assert len(errors) == 1
    assert "invalid fingerprint" in errors[0]


def test_dangling_digest():
    record = {"decision_request_digest": "1220" + "cc" * 32}
    errors = validate_provenance(record, FakeStore())
    assert len(errors) == 1
    assert "dangling digest" in errors[0]


def test_resolved_digest_no_error():
    digest = "1220" + "cc" * 32
    record = {"decision_request_digest": digest}
    store = FakeStore({digest: {"some": "record"}})
    errors = validate_provenance(record, store)
    assert errors == []


def test_dangling_corrects():
    record = {"corrects": "1220" + "dd" * 32}
    errors = validate_provenance(record, FakeStore())
    assert len(errors) == 1
    assert "dangling corrects" in errors[0]


def test_none_values_skipped():
    record = {"execution_fingerprint": None, "corrects": None}
    errors = validate_provenance(record, FakeStore())
    assert errors == []


# ---- F6: key formats and log masking ----

_ANT = "sk-ant-api03-" + "Ab1_" * 12
_PROJ = "sk-proj-" + "Zz9-" * 10
_HF = "hf_" + "aB3" * 12


def test_redact_anthropic_openai_project_and_hf_values_in_records():
    from apron.application.sanitization import sanitize

    record = {"a": _ANT, "b": [_PROJ], "c": {"d": f"token {_HF} here"}}
    result = sanitize(record)
    assert result == {"a": "[REDACTED]", "b": ["[REDACTED]"], "c": {"d": "[REDACTED]"}}


def test_mask_secrets_keeps_surrounding_text():
    from apron.application.sanitization import mask_secrets

    text = f"ERROR 401 for {_HF} while calling api?api_key=rpa_{'Q' * 30}&x=1"
    masked = mask_secrets(text)
    assert _HF not in masked
    assert "rpa_" not in masked
    assert masked.startswith("ERROR 401 for [REDACTED] while calling api?")


def test_mask_secrets_in_logs(caplog):
    import logging

    from apron.application.sanitization import SecretMaskingFilter

    logger = logging.getLogger("apron.test.sanitization")
    handler_filter = SecretMaskingFilter()
    logger.addFilter(handler_filter)
    try:
        with caplog.at_level(logging.WARNING, logger="apron.test.sanitization"):
            logger.warning("key %s and %s and %s", _ANT, _PROJ, _HF)
    finally:
        logger.removeFilter(handler_filter)
    assert caplog.records
    text = caplog.text
    for secret in (_ANT, _PROJ, _HF):
        assert secret not in text
    assert text.count("[REDACTED]") == 3


def test_contains_secret_detects_each_kind():
    from apron.application.sanitization import contains_secret

    for secret in (_ANT, _PROJ, _HF, "rp_" + "k" * 24):
        assert contains_secret(f"x {secret} y")
    assert not contains_secret("Qwen/Qwen3-8B on NVIDIA L4, fingerprint 1220" + "ab" * 32)
