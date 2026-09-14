"""Tests for HF Hub resolver — all against pinned local fixtures, no network."""

from pathlib import Path

from apron.adapters.evidence.hf_hub import FixtureHFHubResolver
from apron.domain.artifacts import ArtifactSourceObservation
from apron.domain.artifacts.identity import ArtifactIdentity
from apron.domain.protocols import ArtifactSourceResolver
from apron.domain.schemas.primitives import ArtifactLocator

FIXTURE_DIR = Path("tests/fixtures/external-formats/huggingface-hub")
PINNED_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def _resolver() -> FixtureHFHubResolver:
    return FixtureHFHubResolver(FIXTURE_DIR)


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------


def test_satisfies_artifact_source_resolver_protocol():
    assert isinstance(_resolver(), ArtifactSourceResolver)


def test_supports_huggingface():
    assert _resolver().supports("huggingface") is True


def test_does_not_support_unknown():
    assert _resolver().supports("oci") is False


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_resolve_returns_observation():
    locator = ArtifactLocator(source_kind="huggingface", uri="Qwen/Qwen3-8B")
    obs = _resolver().resolve(locator)
    assert isinstance(obs, ArtifactSourceObservation)


def test_resolved_revision_matches_pinned():
    locator = ArtifactLocator(source_kind="huggingface", uri="Qwen/Qwen3-8B")
    obs = _resolver().resolve(locator)
    assert obs.resolved_revision == PINNED_REVISION


def test_file_digests_present():
    locator = ArtifactLocator(source_kind="huggingface", uri="Qwen/Qwen3-8B")
    obs = _resolver().resolve(locator)
    paths = {fd.path for fd in obs.file_digests}
    assert "config.json" in paths
    assert "model.safetensors.index.json" in paths


def test_file_digests_have_sha256():
    locator = ArtifactLocator(source_kind="huggingface", uri="Qwen/Qwen3-8B")
    obs = _resolver().resolve(locator)
    for fd in obs.file_digests:
        assert len(fd.sha256) == 64
        assert fd.size_bytes > 0


def test_manifest_digest_is_config_sha256():
    locator = ArtifactLocator(source_kind="huggingface", uri="Qwen/Qwen3-8B")
    obs = _resolver().resolve(locator)
    config_fd = next(fd for fd in obs.file_digests if fd.path == "config.json")
    assert obs.manifest_digest == config_fd.sha256


def test_license_observed():
    locator = ArtifactLocator(source_kind="huggingface", uri="Qwen/Qwen3-8B")
    obs = _resolver().resolve(locator)
    assert obs.license_observed == "apache-2.0"


def test_gating_not_set_for_open_model():
    locator = ArtifactLocator(source_kind="huggingface", uri="Qwen/Qwen3-8B")
    obs = _resolver().resolve(locator)
    assert obs.gating_observed is None


def test_publisher_metadata_has_parameters():
    locator = ArtifactLocator(source_kind="huggingface", uri="Qwen/Qwen3-8B")
    obs = _resolver().resolve(locator)
    assert obs.publisher_metadata is not None
    assert "parameters_BF16" in obs.publisher_metadata
    assert obs.publisher_metadata["parameters_BF16"] == "8190735360"


# ---------------------------------------------------------------------------
# Identity derivation from resolved observation
# ---------------------------------------------------------------------------


def test_identity_derivation_deterministic():
    locator = ArtifactLocator(source_kind="huggingface", uri="Qwen/Qwen3-8B")
    obs1 = _resolver().resolve(locator)
    obs2 = _resolver().resolve(locator)
    id1 = ArtifactIdentity.from_observation(obs1)
    id2 = ArtifactIdentity.from_observation(obs2)
    assert id1.content_digest == id2.content_digest
    assert id1.content_digest.startswith("1220")


def test_identity_changes_with_different_content():
    locator = ArtifactLocator(source_kind="huggingface", uri="Qwen/Qwen3-8B")
    obs = _resolver().resolve(locator)
    id_original = ArtifactIdentity.from_observation(obs)

    modified_obs = ArtifactSourceObservation(
        source_kind="huggingface",
        resolved_revision=obs.resolved_revision,
        file_digests=(obs.file_digests[0],),
    )
    id_modified = ArtifactIdentity.from_observation(modified_obs)
    assert id_original.content_digest != id_modified.content_digest


# ---------------------------------------------------------------------------
# Missing file handling
# ---------------------------------------------------------------------------


def test_missing_file_not_in_digests(tmp_path):
    """Resolver with a fixture dir missing README.md should still work."""
    import shutil

    shutil.copy(FIXTURE_DIR / "config.json", tmp_path / "config.json")
    shutil.copy(
        FIXTURE_DIR / "model.safetensors.index.json",
        tmp_path / "model.safetensors.index.json",
    )
    shutil.copy(FIXTURE_DIR / "model_info.json", tmp_path / "model_info.json")

    resolver = FixtureHFHubResolver(tmp_path)
    locator = ArtifactLocator(source_kind="huggingface", uri="Qwen/Qwen3-8B")
    obs = resolver.resolve(locator)
    paths = {fd.path for fd in obs.file_digests}
    assert "README.md" not in paths
    assert "config.json" in paths
