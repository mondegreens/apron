"""Conformance suite: ArtifactSourceResolver Protocol."""

from apron.domain.artifacts import ArtifactSourceObservation
from apron.domain.artifacts.identity import ArtifactIdentity
from apron.domain.protocols import ArtifactSourceResolver
from apron.domain.schemas.primitives import ArtifactLocator


def test_satisfies_protocol(artifact_resolver):
    assert isinstance(artifact_resolver, ArtifactSourceResolver)


def test_supports_huggingface(artifact_resolver):
    assert artifact_resolver.supports("huggingface") is True


def test_does_not_support_unknown(artifact_resolver):
    assert artifact_resolver.supports("unknown") is False


def test_resolve_returns_observation(artifact_resolver):
    locator = ArtifactLocator(source_kind="huggingface", uri="Qwen/Qwen3-8B")
    obs = artifact_resolver.resolve(locator)
    assert isinstance(obs, ArtifactSourceObservation)


def test_observation_has_nonempty_revision(artifact_resolver):
    locator = ArtifactLocator(source_kind="huggingface", uri="Qwen/Qwen3-8B")
    obs = artifact_resolver.resolve(locator)
    assert obs.resolved_revision
    assert isinstance(obs.resolved_revision, str)


def test_observation_has_manifest_digest(artifact_resolver):
    locator = ArtifactLocator(source_kind="huggingface", uri="Qwen/Qwen3-8B")
    obs = artifact_resolver.resolve(locator)
    assert obs.manifest_digest is not None


def test_observation_has_file_digests(artifact_resolver):
    locator = ArtifactLocator(source_kind="huggingface", uri="Qwen/Qwen3-8B")
    obs = artifact_resolver.resolve(locator)
    assert len(obs.file_digests) > 0
    for fd in obs.file_digests:
        assert fd.path
        assert fd.sha256
        assert fd.size_bytes > 0


def test_same_content_produces_same_identity(artifact_resolver):
    locator = ArtifactLocator(source_kind="huggingface", uri="Qwen/Qwen3-8B")
    obs1 = artifact_resolver.resolve(locator)
    obs2 = artifact_resolver.resolve(locator)
    id1 = ArtifactIdentity.from_observation(obs1)
    id2 = ArtifactIdentity.from_observation(obs2)
    assert id1.content_digest == id2.content_digest
