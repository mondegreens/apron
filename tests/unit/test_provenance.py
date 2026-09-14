"""Provenance integrity: every pinned external format has valid provenance."""

import hashlib
import json
from pathlib import Path

import pytest

from apron.domain.schemas.provenance import ExternalFormatProvenance

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "external-formats"


def _source_dirs():
    return sorted(d for d in FIXTURES_DIR.iterdir() if d.is_dir())


@pytest.fixture(params=_source_dirs(), ids=lambda d: d.name)
def source_dir(request):
    return request.param


def test_provenance_valid(source_dir):
    provenance_path = source_dir / "provenance.json"
    assert provenance_path.exists(), f"missing provenance.json in {source_dir.name}"
    raw = json.loads(provenance_path.read_text())
    prov = ExternalFormatProvenance(**raw)
    assert prov.source_name
    assert prov.retrieval_date
    assert prov.license
    assert len(prov.locations) >= 1
    assert len(prov.pinned_files) >= 1


def test_pinned_files_exist(source_dir):
    raw = json.loads((source_dir / "provenance.json").read_text())
    for entry in raw["pinned_files"]:
        path = source_dir / entry["local_path"]
        assert path.exists(), f"missing pinned file: {entry['local_path']} in {source_dir.name}"


def test_pinned_sha256_matches(source_dir):
    raw = json.loads((source_dir / "provenance.json").read_text())
    for entry in raw["pinned_files"]:
        path = source_dir / entry["local_path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == entry["sha256"], (
            f"SHA-256 mismatch for {entry['local_path']} in {source_dir.name}: "
            f"expected {entry['sha256']}, got {actual}"
        )


def test_pinned_size_matches(source_dir):
    raw = json.loads((source_dir / "provenance.json").read_text())
    for entry in raw["pinned_files"]:
        path = source_dir / entry["local_path"]
        actual = path.stat().st_size
        assert actual == entry["size_bytes"], (
            f"size mismatch for {entry['local_path']} in {source_dir.name}: "
            f"expected {entry['size_bytes']}, got {actual}"
        )
