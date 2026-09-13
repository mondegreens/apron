"""§2.3 capability fixtures — 10 fixtures with signatures, mechanisms, and tests."""

from apron.domain.canonical import canonicalize
from apron.domain.capabilities import CapabilitySignature
from apron.domain.fingerprints import fingerprint_hex
from apron.domain.mechanisms import ComponentMechanism
from apron.domain.schemas.models import ModelSpec


def _round_trip(instance):
    canonical = canonicalize(instance.model_dump(mode="json"))
    restored = type(instance).model_validate_json(canonical)
    assert canonicalize(restored.model_dump(mode="json")) == canonical


def _make_model_with_mechanisms(*mechanisms: tuple[str, str]) -> ModelSpec:
    return ModelSpec(
        components=tuple(ComponentMechanism(mechanism=m, role=r) for m, r in mechanisms),
    )


# ---------------------------------------------------------------------------
# 1. text-to-text: {text} → generated_text
# ---------------------------------------------------------------------------


def test_text_to_text():
    cap = CapabilitySignature(
        operation="text_generation",
        required_inputs=("text",),
        output_representation="generated_text",
    )
    model = _make_model_with_mechanisms(("autoregressive_decode", "decoder"))
    _round_trip(cap)
    _round_trip(model)
    assert cap.required_inputs == ("text",)
    assert len(model.components) == 1


# ---------------------------------------------------------------------------
# 2. text-image-to-text: {text,image} → generated_text (T+I)
# ---------------------------------------------------------------------------


def test_text_image_to_text():
    cap = CapabilitySignature(
        operation="text_generation",
        required_inputs=("text", "image"),
        output_representation="generated_text",
    )
    model = _make_model_with_mechanisms(
        ("autoregressive_decode", "decoder"),
        ("media_encoder", "vision_encoder"),
        ("projector", "vision_projector"),
    )
    _round_trip(cap)
    _round_trip(model)
    assert cap.required_inputs == ("text", "image")
    assert len(model.components) == 3


def test_text_image_forbidden_combination():
    """Image alone without required text is a different capability — not T+I."""
    text_image = CapabilitySignature(
        operation="text_generation",
        required_inputs=("text", "image"),
        output_representation="generated_text",
    )
    image_only = CapabilitySignature(
        operation="text_generation",
        required_inputs=("image",),
        output_representation="generated_text",
    )
    assert fingerprint_hex(text_image) != fingerprint_hex(image_only)
    assert text_image.required_inputs != image_only.required_inputs


# ---------------------------------------------------------------------------
# 3. audio-file-transcription: {audio} → text
# ---------------------------------------------------------------------------


def test_audio_file_transcription():
    cap = CapabilitySignature(
        operation="transcription",
        required_inputs=("audio",),
        output_representation="text",
    )
    model = _make_model_with_mechanisms(("encoder_decoder_generation", "transcriber"))
    _round_trip(cap)
    _round_trip(model)


# ---------------------------------------------------------------------------
# 4. audio-stream-transcription: {audio_stream} → text_delta + timestamps
# ---------------------------------------------------------------------------


def test_audio_stream_transcription():
    cap = CapabilitySignature(
        operation="transcription",
        required_inputs=("audio_stream",),
        output_representation="text_delta",
        streaming=True,
    )
    _make_model_with_mechanisms(("encoder_decoder_generation", "stream_transcriber"))
    _round_trip(cap)
    assert cap.streaming is True


# ---------------------------------------------------------------------------
# 5. text-to-embedding: {text} → embedding_vector
# ---------------------------------------------------------------------------


def test_text_to_embedding():
    cap = CapabilitySignature(
        operation="embedding",
        required_inputs=("text",),
        output_representation="embedding_vector",
    )
    model = _make_model_with_mechanisms(("single_pass_pooling", "pooler"))
    _round_trip(cap)
    _round_trip(model)


# ---------------------------------------------------------------------------
# 6. query-document-score: {text,text} → score
# ---------------------------------------------------------------------------


def test_query_document_score():
    cap = CapabilitySignature(
        operation="scoring",
        required_inputs=("text", "text"),
        output_representation="score",
    )
    _make_model_with_mechanisms(("single_pass_pooling", "scorer"))
    _round_trip(cap)
    assert cap.required_inputs == ("text", "text")


# ---------------------------------------------------------------------------
# 7. text-to-image: {text} → generated_image (generated-image fixture)
# ---------------------------------------------------------------------------


def test_text_to_image():
    """Resolves a real immutable pipeline into text-encoder, denoiser and
    decoder mechanisms without claiming a vLLM endpoint or executable
    media proof (exit gate item 16)."""
    cap = CapabilitySignature(
        operation="image_generation",
        required_inputs=("text",),
        output_representation="generated_image",
    )
    model = _make_model_with_mechanisms(
        ("media_encoder", "text_encoder"),
        ("latent_denoising", "denoiser"),
        ("vae_decode", "image_decoder"),
    )
    _round_trip(cap)
    _round_trip(model)
    assert len(model.components) == 3
    mechanism_types = {c.mechanism for c in model.components}
    assert mechanism_types == {"media_encoder", "latent_denoising", "vae_decode"}


# ---------------------------------------------------------------------------
# 8. audio-text-audio-compound: compound ASR + LLM + TTS
# ---------------------------------------------------------------------------


def test_audio_text_audio_compound():
    asr_cap = CapabilitySignature(
        operation="transcription",
        required_inputs=("audio",),
        output_representation="text",
    )
    llm_cap = CapabilitySignature(
        operation="text_generation",
        required_inputs=("text",),
        output_representation="generated_text",
    )
    tts_cap = CapabilitySignature(
        operation="speech_synthesis",
        required_inputs=("text",),
        output_representation="audio",
    )
    for cap in (asr_cap, llm_cap, tts_cap):
        _round_trip(cap)
    assert fingerprint_hex(asr_cap) != fingerprint_hex(llm_cap)
    assert fingerprint_hex(llm_cap) != fingerprint_hex(tts_cap)


# ---------------------------------------------------------------------------
# 9. checkpoint-not-exposed: model has capability but engine doesn't expose it
# ---------------------------------------------------------------------------


def test_checkpoint_not_exposed():
    """Model has a capability the engine cannot serve at the pinned image."""
    model_cap = CapabilitySignature(
        operation="embedding",
        required_inputs=("text",),
        output_representation="embedding_vector",
    )
    exposed_caps = [
        CapabilitySignature(
            operation="text_generation",
            required_inputs=("text",),
            output_representation="generated_text",
        ),
    ]
    model_cap_fp = fingerprint_hex(model_cap)
    exposed_fps = {fingerprint_hex(c) for c in exposed_caps}
    assert model_cap_fp not in exposed_fps


# ---------------------------------------------------------------------------
# 10. engine-resolved-unknown: engine discovers capability but calculator
#     has no mechanism branch for it
# ---------------------------------------------------------------------------


def test_engine_resolved_unknown():
    """Engine discovers a capability but the calculator has no branch."""
    from apron.domain.mechanisms import (
        CalculatorInput,
        TextWorkload,
        calculate,
        clear_calculator_registry,
    )
    from apron.domain.schemas.primitives import HardwareSpec

    clear_calculator_registry()

    novel_mechanism = ComponentMechanism(
        mechanism="novel_attention_variant",
        role="decoder",
    )
    inputs = CalculatorInput(
        mechanism=novel_mechanism,
        workload=TextWorkload(kind="text", input_length=512, output_length=128),
        artifact_metadata={},
        hardware=HardwareSpec(
            gpu_sku="RTX 4090",
            total_memory_bytes=25_769_803_776,
            compute_capability="8.9",
        ),
        execution_spec_data={},
    )
    result = calculate(inputs)
    assert result is None
