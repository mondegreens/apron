"""Which quantization formats vLLM v0.29.0 loads, and the GPU each needs.

``min_capability`` is what the engine checks before loading
(``config/vllm.py:791``: ``capability < quant_config.get_min_capability()``).
Every value below is the ``get_min_capability`` return of the class the
method name resolves to (``model_executor/layers/quantization/__init__.py:140-160``)
in the pinned source.  A method that is not listed is not loadable: v0.29.0
has no ``bitsandbytes`` entry, for example, although the latest vLLM
documentation lists it.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

# method -> (minimum capability, source of the value in vLLM v0.29.0)
QUANT_MIN_CAPABILITY: dict[str, tuple[int, str]] = {
    "fp8": (75, "model_executor/layers/quantization/fp8.py:144"),
    "fbgemm_fp8": (80, "model_executor/layers/quantization/fbgemm_fp8.py:67"),
    "fp_quant": (100, "model_executor/layers/quantization/fp_quant.py:64"),
    "awq": (75, "model_executor/layers/quantization/auto_awq.py:231"),
    "awq_marlin": (75, "model_executor/layers/quantization/auto_awq.py:231"),
    "auto_awq": (75, "model_executor/layers/quantization/auto_awq.py:231"),
    "gptq": (60, "model_executor/layers/quantization/auto_gptq.py:188"),
    "gptq_marlin": (60, "model_executor/layers/quantization/auto_gptq.py:188"),
    "auto_gptq": (60, "model_executor/layers/quantization/auto_gptq.py:188"),
    "modelopt": (80, "model_executor/layers/quantization/modelopt.py:407"),
    "mxfp4": (80, "model_executor/layers/quantization/mxfp4.py:62"),
}

# compressed-tensors checks the config (70, compressed_tensors.py:117), then
# the scheme each layer resolves to (compressed_tensors.py:930).
_CT = "model_executor/layers/quantization/compressed_tensors/schemes/"
COMPRESSED_TENSORS_SCHEMES: dict[str, tuple[int, str]] = {
    "wNa16": (75, _CT + "compressed_tensors_wNa16.py:100"),
    "w8a8_int8": (75, _CT + "compressed_tensors_w8a8_int8.py:37"),
    "w8a8_fp8": (89, _CT + "compressed_tensors_w8a8_fp8.py:85"),
}


def _compressed_tensors_scheme(quant: dict[str, Any]) -> str | None:
    """The scheme every config group resolves to, or None when mixed/unknown."""
    schemes: set[str | None] = set()
    for group in (quant.get("config_groups") or {}).values():
        weights = group.get("weights") or {}
        activations = group.get("input_activations")
        bits, kind = weights.get("num_bits"), weights.get("type")
        if activations is None and kind == "int" and bits in (4, 8):
            schemes.add("wNa16")
        elif activations and kind == "int" and bits == 8 and activations.get("type") == "int":
            schemes.add("w8a8_int8")
        elif activations and kind == "float" and bits == 8 and activations.get("type") == "float":
            schemes.add("w8a8_fp8")
        else:
            schemes.add(None)
    return schemes.pop() if len(schemes) == 1 else None


def min_capability(quantization_config: dict[str, Any] | None) -> int | None:
    """Capability (major*10+minor) v0.29.0 needs for this checkpoint's format.

    0 for an unquantized checkpoint; None when v0.29.0 cannot load the format
    (or the format is not recognised — never a guess).
    """
    if not quantization_config:
        return 0
    method = str(quantization_config.get("quant_method", ""))
    if method == "compressed-tensors":
        scheme = _compressed_tensors_scheme(quantization_config)
        return COMPRESSED_TENSORS_SCHEMES[scheme][0] if scheme else None
    entry = QUANT_MIN_CAPABILITY.get(method)
    return entry[0] if entry else None


# ---------------------------------------------------------------------------
# Will the checkpoint load?  (free: tensor names from the safetensors headers)
# ---------------------------------------------------------------------------
#
# vLLM loads a checkpoint tensor into the parameter of the same name and
# raises for a tensor it has no parameter for ("There is no module or
# parameter named ..."; the class 6 B200 boot, 2026-09-27).  Each quantized
# linear method registers its parameter names in ``create_weights``; plain
# layers hold ``weight`` and ``bias``.  A stored tensor whose last name part
# is none of these will not load.  Methods not listed here cannot be checked.

_Q = "model_executor/layers/quantization/"
REGISTERED_PARAMETERS: dict[str, tuple[frozenset[str], str]] = {
    "fp_quant": (
        frozenset(
            {
                "qweight",
                "scales",
                "weight_global_scale",
                "act_global_scale",
                "forward_hadamard_matrix",
            }
        ),
        _Q + "fp_quant.py:106 (FPQuantLinearMethod.create_weights)",
    ),
    "fp8": (
        frozenset({"weight", "weight_scale", "weight_scale_inv", "input_scale"}),
        _Q + "fp8.py:292 (Fp8LinearMethod.create_weights)",
    ),
    "fbgemm_fp8": (
        frozenset({"weight", "weight_scale"}),
        _Q + "fbgemm_fp8.py:99 (FBGEMMFp8LinearMethod.create_weights)",
    ),
    "gptq": (
        frozenset({"qweight", "qzeros", "scales", "g_idx"}),
        _Q + "auto_gptq.py:326 (AutoGPTQLinearMethod.create_weights)",
    ),
    "awq": (
        frozenset({"qweight", "qzeros", "scales"}),
        _Q + "auto_awq.py:414 (AutoAWQMarlinLinearMethod.create_weights)",
    ),
    "modelopt": (
        frozenset({"weight", "weight_scale", "weight_scale_2", "input_scale"}),
        _Q + "modelopt.py:455 (ModelOptFp8LinearMethod.create_weights)",
    ),
}
for _alias, _method in (
    ("gptq_marlin", "gptq"),
    ("auto_gptq", "gptq"),
    ("awq_marlin", "awq"),
    ("auto_awq", "awq"),
):
    REGISTERED_PARAMETERS[_alias] = REGISTERED_PARAMETERS[_method]

_CT_PARAMETERS: dict[str, tuple[frozenset[str], str]] = {
    "wNa16": (
        frozenset(
            {"weight_packed", "weight_scale", "weight_zero_point", "weight_shape", "weight_g_idx"}
        ),
        _CT + "compressed_tensors_wNa16.py:102 (CompressedTensorsWNA16.create_weights)",
    ),
    "w8a8_fp8": (
        frozenset({"weight", "weight_scale", "input_scale"}),
        _CT + "compressed_tensors_w8a8_fp8.py:87 (CompressedTensorsW8A8Fp8.create_weights)",
    ),
    "w8a8_int8": (
        frozenset({"weight", "weight_scale", "input_scale", "input_zero_point", "azp_adj"}),
        _CT + "compressed_tensors_w8a8_int8.py:39 (CompressedTensorsW8A8Int8.create_weights)",
    ),
}
_PLAIN = frozenset({"weight", "bias"})
# Rotary buffers some checkpoints store; model loaders skip them by name
# (e.g. models/deepseek_mtp.py:327: ``if "rotary_emb.inv_freq" in name``).
_SKIPPED = frozenset({"inv_freq", "cos_cached", "sin_cached"})


def load_problems(
    tensor_names: Iterable[str], quantization_config: dict[str, Any] | None
) -> tuple[str, ...] | None:
    """Why the pinned engine would refuse a quantized checkpoint's tensors;
    ``()`` when it loads them all; ``None`` when it cannot be told here (an
    unquantized checkpoint, or a method or scheme not listed)."""
    if not quantization_config:
        return None
    method = str(quantization_config.get("quant_method", ""))
    if method == "compressed-tensors":
        scheme = _compressed_tensors_scheme(quantization_config)
        if scheme is None:
            return None
        registered, source = _CT_PARAMETERS[scheme]
    elif method in REGISTERED_PARAMETERS:
        registered, source = REGISTERED_PARAMETERS[method]
    else:
        return None
    allowed = registered | _PLAIN | _SKIPPED
    unknown = Counter(name.rsplit(".", 1)[-1] for name in tensor_names)
    return tuple(
        f"{suffix} x{count}: no vLLM v0.29.0 parameter of that name ({source})"
        for suffix, count in sorted(unknown.items())
        if suffix not in allowed
    )
