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

from typing import Any

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
