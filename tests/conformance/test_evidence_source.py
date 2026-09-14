"""Conformance suite: EvidenceSource Protocol."""

from apron.domain.protocols import EvidenceSource


def test_satisfies_protocol(evidence_source):
    assert isinstance(evidence_source, EvidenceSource)


def test_source_name_is_nonempty_string(evidence_source):
    assert isinstance(evidence_source.source_name, str)
    assert evidence_source.source_name


def test_collect_returns_list(evidence_source):
    results = evidence_source.collect()
    assert isinstance(results, list)


def test_each_entry_has_model_id_and_import_status(evidence_source):
    for entry in evidence_source.collect():
        assert "model_id" in entry
        assert "import_status" in entry
