"""Apron follows vLLM releases (owner, 2026-09-27): one runner image per
version, and each plan runs on the newest pinned version that serves the
model's architecture.  The version is in the solution's identity through the
image digest."""

from __future__ import annotations

from apron.adapters.backends.vllm_quantization import engine_facts, engine_for
from apron.adapters.runner_image import (
    RUNNER_IMAGE_DIGEST,
    RUNNER_IMAGES,
    image_for_digest,
    newest_runner_image,
)


def test_the_newest_version_that_serves_the_architecture_is_chosen() -> None:
    both = ("v0.29.0", "v0.30.0")
    assert engine_for("Qwen3ForCausalLM", both) == "v0.30.0"  # both serve it: newest
    assert engine_for("Glm5NextForConditionalGeneration", both) == "v0.30.0"
    assert engine_for("Glm5NextForConditionalGeneration", ("v0.29.0",)) is None
    assert engine_for("NotAModelForCausalLM", both) is None


def test_each_version_has_its_own_facts() -> None:
    old, new = engine_facts("v0.29.0"), engine_facts("v0.30.0")
    assert (old.version, new.version) == ("v0.29.0", "v0.30.0")
    assert "deepseek_v41" in new.tokenizer_modes and "deepseek_v41" not in old.tokenizer_modes
    assert old.load_problems(["x.weight", "x.zz_invented"], {"quant_method": "fp8"}) != ()


def test_pinned_images_map_version_and_digest_both_ways() -> None:
    first = RUNNER_IMAGES["v0.29.0"]
    assert first.digest == RUNNER_IMAGE_DIGEST
    assert image_for_digest(first.digest).endswith("@" + first.digest)
    assert newest_runner_image().version in RUNNER_IMAGES
