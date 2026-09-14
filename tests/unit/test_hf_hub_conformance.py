"""Run artifact_resolver conformance assertions against FixtureHFHubResolver."""

from pathlib import Path

import pytest

from apron.adapters.evidence.hf_hub import FixtureHFHubResolver
from apron.domain.artifacts import ArtifactSourceObservation
from apron.domain.artifacts.identity import ArtifactIdentity
from apron.domain.protocols import ArtifactSourceResolver
from apron.domain.schemas.primitives import ArtifactLocator

FIXTURE_DIR = Path("tests/fixtures/external-formats/huggingface-hub")


@pytest.fixture()
def resolver():
    return FixtureHFHubResolver(FIXTURE_DIR)


def test_satisfies_protocol(resolver):
    assert isinstance(resolver, ArtifactSourceResolver)


def test_supports_huggingface(resolver):
    assert resolver.supports("huggingface") is True


def test_does_not_support_unknown(resolver):
    assert resolver.supports("unknown") is False


def test_resolve_returns_observation(resolver):
    locator = ArtifactLocator(source_kind="huggingface", uri="Qwen/Qwen3-8B")
    obs = resolver.resolve(locator)
    assert isinstance(obs, ArtifactSourceObservation)


def test_observation_has_nonempty_revision(resolver):
    locator = ArtifactLocator(source_kind="huggingface", uri="Qwen/Qwen3-8B")
    obs = resolver.resolve(locator)
    assert obs.resolved_revision
    assert isinstance(obs.resolved_revision, str)


def test_observation_has_manifest_digest(resolver):
    locator = ArtifactLocator(source_kind="huggingface", uri="Qwen/Qwen3-8B")
    obs = resolver.resolve(locator)
    assert obs.manifest_digest is not None


def test_observation_has_file_digests(resolver):
    locator = ArtifactLocator(source_kind="huggingface", uri="Qwen/Qwen3-8B")
    obs = resolver.resolve(locator)
    assert len(obs.file_digests) > 0
    for fd in obs.file_digests:
        assert fd.path
        assert fd.sha256
        assert fd.size_bytes > 0


def test_same_content_produces_same_identity(resolver):
    locator = ArtifactLocator(source_kind="huggingface", uri="Qwen/Qwen3-8B")
    obs1 = resolver.resolve(locator)
    obs2 = resolver.resolve(locator)
    id1 = ArtifactIdentity.from_observation(obs1)
    id2 = ArtifactIdentity.from_observation(obs2)
    assert id1.content_digest == id2.content_digest
