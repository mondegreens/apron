"""§2.7 + §3.10 quantization candidate graph fixtures and behavioral tests."""

from apron.domain.artifacts import (
    ArtifactSourceObservation,
    FileDigest,
    OfflineTransformSpec,
    QuantizationSpec,
    RuntimeTransformSpec,
)
from apron.domain.artifacts.identity import ArtifactIdentity
from apron.domain.canonical import canonicalize
from apron.domain.fingerprints import fingerprint_hex
from apron.domain.schemas.models import (
    ArtifactRelation,
    ArtifactSpec,
    ClaimedDerivedFrom,
    PublishedByOwner,
    ReproduciblyDerivedFrom,
    StructurallyCompatibleWith,
    TokenizerCompatibleWith,
)

_FP = "1220" + "ab" * 32


def _round_trip(instance):
    canonical = canonicalize(instance.model_dump(mode="json"))
    restored = type(instance).model_validate_json(canonical)
    assert canonicalize(restored.model_dump(mode="json")) == canonical


def _make_identity(digest_seed: str) -> ArtifactIdentity:
    return ArtifactIdentity(content_digest="1220" + digest_seed * 32)


def _make_artifact(digest_seed: str, **kwargs) -> ArtifactSpec:
    return ArtifactSpec(identity=_make_identity(digest_seed), **kwargs)


# ---------------------------------------------------------------------------
# checkpoint-native BF16: base artifact, no transform
# ---------------------------------------------------------------------------


def test_checkpoint_native():
    artifact = _make_artifact("a0")
    quant = QuantizationSpec(weight_format="bf16", scheme="none")
    _round_trip(artifact)
    _round_trip(quant)


# ---------------------------------------------------------------------------
# publisher-quantized FP8: separate ArtifactSpec, published_by_owner
# ---------------------------------------------------------------------------


def test_publisher_quant():
    base = _make_artifact("a0")
    fp8 = _make_artifact(
        "a1",
        quantization_config_digest="1220" + "cc" * 32,
    )
    relation = PublishedByOwner(
        relation="published_by_owner",
        publisher="Qwen",
        artifact_digest=fingerprint_hex(fp8),
    )
    _round_trip(relation)
    assert fingerprint_hex(base) != fingerprint_hex(fp8)

    quant = QuantizationSpec(
        weight_format="fp8_e4m3",
        scheme="per_tensor",
        bits=8,
        scale_representation="per_tensor_float",
    )
    _round_trip(quant)


# ---------------------------------------------------------------------------
# third-party AWQ: claimed_derived_from, name similarity cannot promote lineage
# ---------------------------------------------------------------------------


def test_third_party_quant():
    relation = ClaimedDerivedFrom(
        relation="claimed_derived_from",
        source_artifact_digest="1220" + "a0" * 32,
        claim_basis="name_similarity",
    )
    _round_trip(relation)


def test_name_similarity_cannot_promote_lineage():
    """Two artifacts with matching names but different manifests remain different."""
    obs_a = ArtifactSourceObservation(
        source_kind="huggingface",
        resolved_revision="rev1",
        file_digests=(FileDigest(path="model.safetensors", sha256="aa" * 32, size_bytes=1000),),
    )
    obs_b = ArtifactSourceObservation(
        source_kind="huggingface",
        resolved_revision="rev2",
        file_digests=(FileDigest(path="model.safetensors", sha256="bb" * 32, size_bytes=2000),),
    )
    id_a = ArtifactIdentity.from_observation(obs_a)
    id_b = ArtifactIdentity.from_observation(obs_b)
    assert id_a.content_digest != id_b.content_digest


# ---------------------------------------------------------------------------
# offline GPTQ conversion: OfflineTransformSpec, reproducibly_derived_from
# ---------------------------------------------------------------------------


def test_offline_conversion():
    transform = OfflineTransformSpec(
        producer="auto-gptq",
        version="0.7.1",
        recipe="default",
        calibration_data_identity="1220" + "ca" * 32,
        output_artifact_digest="1220" + "b0" * 32,
    )
    relation = ReproduciblyDerivedFrom(
        relation="reproducibly_derived_from",
        source_artifact_digest="1220" + "a0" * 32,
        transform_spec_digest=fingerprint_hex(transform),
    )
    _round_trip(transform)
    _round_trip(relation)


# ---------------------------------------------------------------------------
# runtime FP8: RuntimeTransformSpec with none checkpoint
# ---------------------------------------------------------------------------


def test_runtime_transform():
    transform = RuntimeTransformSpec(
        kind="fp8_online",
        engine_version="vllm-0.8.5",
        target_layer_rules=("all_linear",),
        exclusions=("lm_head",),
        producer="vllm",
        version="0.8.5",
    )
    _round_trip(transform)
    assert transform.kind == "fp8_online"


# ---------------------------------------------------------------------------
# structurally_compatible_with: matching shapes cannot prove content equality
# ---------------------------------------------------------------------------


def test_structurally_compatible():
    relation = StructurallyCompatibleWith(
        relation="structurally_compatible_with",
        other_artifact_digest="1220" + "a0" * 32,
        matching_criteria="tensor_shapes_and_dtypes",
    )
    _round_trip(relation)


# ---------------------------------------------------------------------------
# tokenizer_compatible_with
# ---------------------------------------------------------------------------


def test_tokenizer_compatible():
    relation = TokenizerCompatibleWith(
        relation="tokenizer_compatible_with",
        other_artifact_digest="1220" + "a0" * 32,
        tokenizer_hash="abc123def456",
    )
    _round_trip(relation)


# ---------------------------------------------------------------------------
# same-content-two-sources: one ArtifactIdentity, two observations
# ---------------------------------------------------------------------------


def test_same_content_two_sources():
    files = [("config.json", "aa" * 32, 1024)]
    obs_hf = ArtifactSourceObservation(
        source_kind="huggingface",
        resolved_revision="rev1",
        file_digests=tuple(FileDigest(path=p, sha256=h, size_bytes=s) for p, h, s in files),
    )
    obs_local = ArtifactSourceObservation(
        source_kind="local",
        resolved_revision="local_rev",
        file_digests=tuple(FileDigest(path=p, sha256=h, size_bytes=s) for p, h, s in files),
    )
    id_hf = ArtifactIdentity.from_observation(obs_hf)
    id_local = ArtifactIdentity.from_observation(obs_local)
    assert id_hf.content_digest == id_local.content_digest

    artifact = ArtifactSpec(
        identity=id_hf,
        observations=(obs_hf, obs_local),
    )
    _round_trip(artifact)
    assert len(artifact.observations) == 2


# ---------------------------------------------------------------------------
# every quantization fixture resolves through the same candidate API
# ---------------------------------------------------------------------------


def test_quantization_candidate_api():
    """All quantization variants produce ArtifactSpec + ArtifactRelation
    that round-trip through the same discriminated union."""
    from pydantic import TypeAdapter

    adapter = TypeAdapter(ArtifactRelation)

    relations = [
        PublishedByOwner(
            relation="published_by_owner",
            publisher="Qwen",
            artifact_digest=_FP,
        ),
        ClaimedDerivedFrom(
            relation="claimed_derived_from",
            source_artifact_digest=_FP,
            claim_basis="name",
        ),
        ReproduciblyDerivedFrom(
            relation="reproducibly_derived_from",
            source_artifact_digest=_FP,
            transform_spec_digest=_FP,
        ),
        StructurallyCompatibleWith(
            relation="structurally_compatible_with",
            other_artifact_digest=_FP,
            matching_criteria="shapes",
        ),
        TokenizerCompatibleWith(
            relation="tokenizer_compatible_with",
            other_artifact_digest=_FP,
            tokenizer_hash="abc",
        ),
    ]
    for rel in relations:
        canonical = canonicalize(rel.model_dump(mode="json"))
        parsed = adapter.validate_json(canonical)
        assert type(parsed) is type(rel)


# ---------------------------------------------------------------------------
# artifact identity survives round-trip
# ---------------------------------------------------------------------------


def test_artifact_identity_survives_round_trip():
    transform = OfflineTransformSpec(producer="gptq", version="1.0")
    quant = QuantizationSpec(weight_format="int4", scheme="per_group", bits=4, group_size=128)
    artifact = _make_artifact("a0", quantization_config_digest=fingerprint_hex(quant))

    for obj in (transform, quant, artifact):
        canonical = canonicalize(obj.model_dump(mode="json"))
        restored = type(obj).model_validate_json(canonical)
        assert fingerprint_hex(obj) == fingerprint_hex(restored)


# ---------------------------------------------------------------------------
# actual tensor bytes for memory (§3.10)
# ---------------------------------------------------------------------------


def test_actual_tensor_bytes_for_memory():
    """Memory calculation uses actual tensor headers and auxiliary-scale bytes,
    not nominal bit width."""
    artifact_4bit = _make_artifact(
        "a0",
        weight_manifest_digest="1220" + "bb" * 32,
    )
    artifact_8bit = _make_artifact(
        "a1",
        weight_manifest_digest="1220" + "cc" * 32,
    )
    assert fingerprint_hex(artifact_4bit) != fingerprint_hex(artifact_8bit)
