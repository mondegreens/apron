"""Step 3 tests: Layer 1 artifacts — round-trip, classification, identity derivation."""

from __future__ import annotations

from apron.domain.artifacts import (
    ArtifactSourceObservation,
    FileDigest,
    ModelLineage,
    OfflineTransformSpec,
    QuantizationSpec,
    RuntimeTransformSpec,
)
from apron.domain.artifacts.identity import ArtifactIdentity
from apron.domain.canonical import canonicalize
from apron.domain.fingerprints import assert_fully_classified


def _round_trip(model_cls, instance):
    canonical = canonicalize(instance.model_dump(mode="json"))
    restored = model_cls.model_validate_json(canonical)
    assert canonicalize(restored.model_dump(mode="json")) == canonical
    return restored


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


def test_file_digest_fully_classified():
    assert_fully_classified(FileDigest)


def test_artifact_source_observation_fully_classified():
    assert_fully_classified(ArtifactSourceObservation)


def test_artifact_identity_fully_classified():
    assert_fully_classified(ArtifactIdentity)


def test_model_lineage_fully_classified():
    assert_fully_classified(ModelLineage)


def test_quantization_spec_fully_classified():
    assert_fully_classified(QuantizationSpec)


def test_offline_transform_spec_fully_classified():
    assert_fully_classified(OfflineTransformSpec)


def test_runtime_transform_spec_fully_classified():
    assert_fully_classified(RuntimeTransformSpec)


# ---------------------------------------------------------------------------
# round-trip
# ---------------------------------------------------------------------------


def _make_observation(source_kind: str, revision: str, files: list[tuple[str, str, int]]):
    return ArtifactSourceObservation(
        source_kind=source_kind,
        resolved_revision=revision,
        file_digests=tuple(FileDigest(path=p, sha256=h, size_bytes=s) for p, h, s in files),
    )


_FILES_A = [
    ("config.json", "aabb" * 16, 1024),
    ("model.safetensors.index.json", "ccdd" * 16, 2048),
]


def test_artifact_source_observation_round_trip():
    obs = _make_observation("huggingface", "abc123", _FILES_A)
    _round_trip(ArtifactSourceObservation, obs)


def test_artifact_identity_round_trip():
    ident = ArtifactIdentity(content_digest="1220" + "ab" * 32)
    _round_trip(ArtifactIdentity, ident)


def test_model_lineage_round_trip():
    lin = ModelLineage(family="Qwen3", publishers=("Qwen",))
    _round_trip(ModelLineage, lin)


def test_quantization_spec_round_trip():
    qs = QuantizationSpec(
        weight_format="fp8_e4m3",
        scheme="per_tensor",
        bits=8,
    )
    _round_trip(QuantizationSpec, qs)


def test_offline_transform_spec_round_trip():
    ots = OfflineTransformSpec(
        producer="auto-gptq",
        version="0.7.1",
        recipe="default",
    )
    _round_trip(OfflineTransformSpec, ots)


def test_runtime_transform_spec_round_trip():
    rts = RuntimeTransformSpec(kind="none")
    _round_trip(RuntimeTransformSpec, rts)

    rts2 = RuntimeTransformSpec(
        kind="fp8_online",
        engine_version="vllm-0.8.5",
        target_layer_rules=("all_linear",),
        exclusions=("lm_head",),
        producer="vllm",
        version="0.8.5",
    )
    _round_trip(RuntimeTransformSpec, rts2)


# ---------------------------------------------------------------------------
# identity derivation: equal contents from two sources → same identity
# ---------------------------------------------------------------------------


def test_identity_derived_from_contents():
    obs = _make_observation("huggingface", "rev1", _FILES_A)
    ident = ArtifactIdentity.from_observation(obs)
    assert ident.content_digest.startswith("1220")
    assert len(ident.content_digest) == 68


def test_equal_contents_two_sources_share_identity():
    obs_hf = _make_observation("huggingface", "rev1", _FILES_A)
    obs_local = _make_observation("local", "rev999", _FILES_A)

    id_hf = ArtifactIdentity.from_observation(obs_hf)
    id_local = ArtifactIdentity.from_observation(obs_local)

    assert id_hf.content_digest == id_local.content_digest


def test_different_contents_different_identity():
    obs_a = _make_observation("huggingface", "rev1", _FILES_A)
    files_b = [("config.json", "eeff" * 16, 512)]
    obs_b = _make_observation("huggingface", "rev2", files_b)

    id_a = ArtifactIdentity.from_observation(obs_a)
    id_b = ArtifactIdentity.from_observation(obs_b)

    assert id_a.content_digest != id_b.content_digest


def test_identity_ignores_source_metadata():
    obs1 = _make_observation("huggingface", "rev1", _FILES_A)
    obs2 = ArtifactSourceObservation(
        source_kind="local",
        resolved_revision="different_rev",
        file_digests=tuple(FileDigest(path=p, sha256=h, size_bytes=s) for p, h, s in _FILES_A),
        license_observed="Apache-2.0",
        gating_observed="manual",
        publisher_metadata={"org": "Qwen"},
        observed_at="2026-09-12T00:00:00Z",
    )

    id1 = ArtifactIdentity.from_observation(obs1)
    id2 = ArtifactIdentity.from_observation(obs2)

    assert id1.content_digest == id2.content_digest
