"""Which quantization formats the pinned vLLM loads, and what each needs.

The facts come from ``vllm_facts.json``, generated from the pinned vLLM
source by ``scripts/generate_vllm_facts.py``: per method, the minimum compute
capability (``get_min_capability``, which ``config/vllm.py:791`` checks before
loading) and the parameter names its linear method registers.  A method not
in the file is not loadable (v0.29.0 has no ``bitsandbytes``, although the
latest vLLM documentation lists it).  A test fails when the file's vLLM
version differs from the runner image's pin, so a refusal is always "by this
version", never stale knowledge.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

# Generated from the pinned vLLM source by scripts/generate_vllm_facts.py; a
# test ties its version to the runner image's pin (docker/requirements.txt).
FACTS: dict[str, Any] = json.loads(Path(__file__).with_name("vllm_facts.json").read_text())
ENGINE_VERSION: str = FACTS["engine_version"]


def _method_table(key: str, source: str) -> dict[str, tuple[Any, str]]:
    table = {
        method: (entry[key], entry[source])
        for method, entry in FACTS["methods"].items()
        if key in entry
    }
    for alias, method in FACTS["aliases"].items():
        if method in table:
            table[alias] = table[method]
    return table


# method -> (minimum capability, source line in the pinned vLLM)
QUANT_MIN_CAPABILITY: dict[str, tuple[int, str]] = _method_table(
    "min_capability", "min_capability_source"
)
# compressed-tensors checks the config's minimum, then the scheme's.
COMPRESSED_TENSORS_SCHEMES: dict[str, tuple[int, str]] = {
    scheme: (entry["min_capability"], entry["min_capability_source"])
    for scheme, entry in FACTS["compressed_tensors"]["schemes"].items()
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

REGISTERED_PARAMETERS: dict[str, tuple[frozenset[str], str]] = {
    method: (frozenset(names), source)
    for method, (names, source) in _method_table("parameters", "parameters_source").items()
}
_CT_PARAMETERS: dict[str, tuple[frozenset[str], str]] = {
    scheme: (frozenset(entry["parameters"]), entry["parameters_source"])
    for scheme, entry in FACTS["compressed_tensors"]["schemes"].items()
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
        f"{suffix} x{count}: no vLLM {ENGINE_VERSION} parameter of that name ({source})"
        for suffix, count in sorted(unknown.items())
        if suffix not in allowed
    )


# ---------------------------------------------------------------------------
# Scheduler defaults `vllm serve` picks for a GPU (they size the profiling run)
# ---------------------------------------------------------------------------

_GIB = 1 << 30


def default_max_num_batched_tokens(total_memory_bytes: int, gpu_name: str) -> int:
    """``max_num_batched_tokens`` v0.29.0 uses for the OpenAI server on this GPU.

    ``engine/arg_utils.py:2698-2727`` (``get_batch_defaults``): 16384 at
    >= 160 GiB; 8192 at >= 70 GiB unless the name contains "a100"; else 2048.
    The startup profiling forward runs this many tokens.
    """
    if total_memory_bytes >= 160 * _GIB:
        return 16384
    if total_memory_bytes >= 70 * _GIB and "a100" not in gpu_name.lower():
        return 8192
    return 2048


def default_max_num_seqs(total_memory_bytes: int, gpu_name: str) -> int:
    """``max_num_seqs`` v0.29.0 uses for the OpenAI server on this GPU
    (engine/arg_utils.py:2698-2727): 1024 at >= 70 GiB unless "a100", else 256."""
    if total_memory_bytes >= 160 * _GIB:
        return 1024
    if total_memory_bytes >= 70 * _GIB and "a100" not in gpu_name.lower():
        return 1024
    return 256
