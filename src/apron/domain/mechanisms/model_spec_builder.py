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


UNKNOWN_MECHANISM = "unknown"


def _is_attention_free(config: dict[str, Any]) -> bool:
    """No attention heads declared: a state-space (Mamba) or other non-attention model.

    INV-32: such a model is not ``autoregressive_decode`` even though its
    architecture name ends in ``ForCausalLM``.  A Mamba-1 config is
    ``ssm_decode``; any other attention-free model is an explicit unknown.
    """
    return not config.get("num_attention_heads")


# Attention shapes the calculator does not count: a sparse-attention indexer
# (DeepSeek V3.2 / GLM-5 DSA: index_topk), compressed KV (DeepSeek V4:
# compress_ratios), and per-layer kinds other than full or sliding attention
# (linear attention, Mamba).  vLLM pages each with its own KV spec, so a plain
# estimate would be confidently wrong.  Models whose layout layered.py reads
# (``family``: DeepSeek V4 / V4.1, Qwen4Exp and GLM5Next among them, when their
# configs carry every field it reads) are ``layered_decode`` before this check;
# anything else with these fields stays unknown.
_UNMODELLED_ATTENTION_FIELDS = ("index_topk", "compress_ratios")
_MODELLED_LAYER_TYPES = frozenset({"full_attention", "sliding_attention"})


def unmodelled_attention(config: dict[str, Any]) -> str | None:
    """The config field that names an attention layout the calculator does
    not model, or None."""
    from apron.domain.mechanisms.layered import text_config

    for source in (config, text_config(config)):
        for key in _UNMODELLED_ATTENTION_FIELDS:
            if source.get(key):
                return key
        kinds = source.get("layer_types")
        if isinstance(kinds, list) and not set(kinds) <= _MODELLED_LAYER_TYPES:
            return "layer_types"
        for key in ("linear_num_value_heads", "hybrid_override_pattern", "layers_block_type"):
            if source.get(key):
                return key
    return None


def _has_mla_fields(config: dict[str, Any]) -> bool:
    """Detect Multi-Latent Attention from config.json fields."""
    return config.get("kv_lora_rank") is not None and config.get("qk_rope_head_dim") is not None


def _has_mamba1_state_fields(config: dict[str, Any]) -> bool:
    """A Mamba-1 state-space model: the fields vLLM sizes its state from
    (model_executor/models/mamba.py:234-245: intermediate_size, state_size,
    conv_kernel)."""
    return all(config.get(k) for k in ("intermediate_size", "state_size", "conv_kernel"))


def _infer_mechanism(architecture: str, config: dict[str, Any]) -> str:
    if _has_mla_fields(config):
        return "mla_decode"
    for suffix, mechanism in ARCHITECTURE_MECHANISMS.items():
        if architecture.endswith(suffix):
            if mechanism == "autoregressive_decode" and _is_attention_free(config):
                return "ssm_decode" if _has_mamba1_state_fields(config) else UNKNOWN_MECHANISM
            return mechanism
    return architecture


def _infer_dtype_map(config: dict[str, Any]) -> dict[str, str]:
    torch_dtype = config.get("torch_dtype") or config.get("dtype") or "bfloat16"
    return {"decoder": torch_dtype}


def build_model_spec(
    config: dict[str, Any],
    *,
    repository: str | None = None,
    revision: str | None = None,
    license_id: str | None = None,
    total_weight_bytes: int | None = None,
    files: tuple[str, ...] = (),
    unmodelled_architectures: frozenset[str] = frozenset(),
) -> ModelSpec:
    """Construct a ModelSpec from config.json fields.

    The config dict must have at minimum: architectures, num_hidden_layers,
    num_attention_heads, hidden_size. Additional fields are used when present.
    """
    architectures = config.get("architectures", [])
    primary_arch = architectures[0] if architectures else "UnknownArchitecture"
    # An architecture the engine runs with state the calculator does not model
    # (hybrid attention + Mamba/linear attention) is an explicit unknown, never
    # a plain-attention estimate: KV for every layer would be wrong.
    from apron.domain.mechanisms.layered import family

    if family(config) is not None:
        # Per-layer caches the calculator counts as vLLM pages them (layered.py).
        mechanism = "layered_decode"
    elif primary_arch in unmodelled_architectures or unmodelled_attention(config):
        mechanism = UNKNOWN_MECHANISM
    else:
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
