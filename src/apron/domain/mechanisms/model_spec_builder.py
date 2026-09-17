"""Build a ModelSpec from config.json data and safetensors metadata.

Parses transformer config fields into ComponentMechanism instances and
constructs a ModelSpec with the component graph. This is the bridge between
raw HF Hub config.json and the domain's ModelSpec schema.
"""

from __future__ import annotations

from typing import Any

from apron.domain.mechanisms import ComponentMechanism
from apron.domain.schemas.models import ModelSpec

ARCHITECTURE_MECHANISMS: dict[str, str] = {
    "ForCausalLM": "autoregressive_decode",
    "LMHeadModel": "autoregressive_decode",
    "ForConditionalGeneration": "encoder_decoder_generation",
    "ForSequenceClassification": "single_pass_pooling",
    "ForTokenClassification": "single_pass_pooling",
    "ForImageClassification": "single_pass_pooling",
}


def _has_mla_fields(config: dict[str, Any]) -> bool:
    """Detect Multi-Latent Attention from config.json fields."""
    return config.get("kv_lora_rank") is not None and config.get("qk_rope_head_dim") is not None


def _infer_mechanism(architecture: str, config: dict[str, Any]) -> str:
    if _has_mla_fields(config):
        return "mla_decode"
    for suffix, mechanism in ARCHITECTURE_MECHANISMS.items():
        if architecture.endswith(suffix):
            return mechanism
    return architecture


def _infer_dtype_map(config: dict[str, Any]) -> dict[str, str]:
    torch_dtype = config.get("torch_dtype", "bfloat16")
    return {"decoder": torch_dtype}


def build_model_spec(
    config: dict[str, Any],
    *,
    repository: str | None = None,
    revision: str | None = None,
    license_id: str | None = None,
    total_weight_bytes: int | None = None,
    files: tuple[str, ...] = (),
) -> ModelSpec:
    """Construct a ModelSpec from config.json fields.

    The config dict must have at minimum: architectures, num_hidden_layers,
    num_attention_heads, hidden_size. Additional fields are used when present.
    """
    architectures = config.get("architectures", [])
    primary_arch = architectures[0] if architectures else "UnknownArchitecture"
    mechanism = _infer_mechanism(primary_arch, config)

    components: list[ComponentMechanism] = [
        ComponentMechanism(mechanism=mechanism, role="decoder"),
    ]

    edges: list[tuple[str, str]] = []

    return ModelSpec(
        repository=repository,
        immutable_revision=revision,
        license=license_id,
        files=files,
        component_bytes_dtype=_infer_dtype_map(config),
        remote_code_required=config.get("auto_map", {})
        .get("AutoModelForCausalLM", "")
        .startswith("--")
        if isinstance(config.get("auto_map"), dict)
        else False,
        components=tuple(components),
        edges=tuple(edges),
    )
