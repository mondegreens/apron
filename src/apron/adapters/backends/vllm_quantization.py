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

import gzip
import hashlib
import json
from collections import Counter
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

# Generated from the pinned vLLM source by scripts/generate_vllm_facts.py; a
# test ties its version to the runner image's pin (docker/requirements.txt).
FACTS: dict[str, Any] = json.loads(Path(__file__).with_name("vllm_facts.json").read_text())
ENGINE_VERSION: str = FACTS["engine_version"]
# Model classes the pinned vLLM marks IsHybrid (attention plus Mamba/linear
# attention state), generated from its source with the line of each class.
HYBRID_ARCHITECTURES: frozenset[str] = frozenset(FACTS["hybrid_architectures"])
# vLLM's own tokenizer/renderer modes named after a model type (DeepSeek V3.2
# and V4 ship no chat template; vLLM renders their chat only in that mode, and
# "auto" never picks it: tokenizers/registry.py:147-163).  Generic modes and
# the one "auto" already detects are left out.
MODEL_TOKENIZER_MODES: frozenset[str] = frozenset(FACTS["tokenizer_modes"]["modes"]) - {
    "auto",
    "hf",
    "slow",
    "mistral",
}


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


def _modelopt_algo(quant: dict[str, Any]) -> str:
    """``quant_algo`` as modelopt.py:240-257 reads it (either layout, upper-cased)."""
    nested = quant.get("quantization")
    if isinstance(nested, dict):
        return str(nested.get("quant_algo", "")).upper()
    return str(quant.get("quant_algo", "")).upper()


def _method(quant: dict[str, Any]) -> str:
    """The method name vLLM resolves the checkpoint to.  ModelOpt checkpoints
    all declare ``modelopt``; the ModelOpt configs' ``override_quantization_method``
    rename it by ``quant_algo`` (modelopt.py:1063-1070 FP4, 1730-1736 MXFP8,
    2224-2230 mixed precision); other algorithms stay with the FP8 config
    (modelopt.py:364).  A name the facts do not list stays unknown."""
    method = str(quant.get("quant_method", ""))
    if method != "modelopt":
        return method
    algo = _modelopt_algo(quant)
    if "NVFP4" in algo or "FP4" in algo:
        return "modelopt_fp4"
    if "MXFP8" in algo:
        return "modelopt_mxfp8"
    if algo == "MIXED_PRECISION":
        return "modelopt_mixed"
    return "modelopt"


def min_capability(quantization_config: dict[str, Any] | None) -> int | None:
    """Capability (major*10+minor) v0.29.0 needs for this checkpoint's format.

    0 for an unquantized checkpoint; None when v0.29.0 cannot load the format
    (or the format is not recognised — never a guess).
    """
    if not quantization_config:
        return 0
    method = _method(quantization_config)
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
# layers hold ``weight`` and ``bias``; a config that attaches a KV-cache method
# to attention layers adds its scales (``kv_cache``).  A stored tensor whose
# last name part is none of these will not load.  Methods not listed here
# cannot be checked.

REGISTERED_PARAMETERS: dict[str, tuple[frozenset[str], str]] = {
    method: (frozenset(names), source)
    for method, (names, source) in _method_table("parameters", "parameters_source").items()
}
_CT_PARAMETERS: dict[str, tuple[frozenset[str], str]] = {
    scheme: (frozenset(entry["parameters"]), entry["parameters_source"])
    for scheme, entry in FACTS["compressed_tensors"]["schemes"].items()
}
_KV_CACHE_SCALES = frozenset(FACTS["kv_cache"]["parameters"])
_KV_CACHE_METHODS = frozenset(
    method for method, entry in FACTS["methods"].items() if entry.get("kv_cache_scales")
) | frozenset(
    alias
    for alias, method in FACTS["aliases"].items()
    if FACTS["methods"].get(method, {}).get("kv_cache_scales")
)
_PLAIN = frozenset({"weight", "bias"})
# Rotary buffers some checkpoints store; model loaders skip them by name
# (e.g. models/deepseek_mtp.py:327: ``if "rotary_emb.inv_freq" in name``).
_SKIPPED = frozenset({"inv_freq", "cos_cached", "sin_cached"})


def _nvfp4_w4a4(quant: dict[str, Any]) -> bool:
    """NVFP4 with quantized activations; modelopt.py:1084-1098 routes a weight-only
    NVFP4 checkpoint (every group's ``input_activations`` null) to W4A16."""
    if _modelopt_algo(quant) != "NVFP4":
        return False
    groups = quant.get("config_groups")
    return not (
        isinstance(groups, dict)
        and groups
        and all(
            isinstance(g, dict) and g.get("input_activations") is None for g in groups.values()
        )
    )


@cache
def engine_names() -> frozenset[str]:
    """Every identifier-like name the pinned vLLM source mentions (generated)."""
    entry = FACTS["source_names"]
    blob = Path(__file__).with_name(entry["file"]).read_bytes()
    if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
        raise RuntimeError(f"{entry['file']} does not match vllm_facts.json — regenerate")
    return frozenset(gzip.decompress(blob).decode().split())


def load_problems(
    tensor_names: Iterable[str], quantization_config: dict[str, Any] | None
) -> tuple[str, ...] | None:
    """Why the pinned engine would refuse a quantized checkpoint's tensors;
    ``()`` when it loads them all; ``None`` when it cannot be told here (an
    unquantized checkpoint, or a method or scheme not listed)."""
    if not quantization_config:
        return None
    method = _method(quantization_config)
    if method == "compressed-tensors":
        scheme = _compressed_tensors_scheme(quantization_config)
        if scheme is None:
            return None
        registered, source = _CT_PARAMETERS[scheme]
        kv_cache = bool(FACTS["compressed_tensors"].get("kv_cache_scales"))
    elif method == "modelopt_fp4" and not _nvfp4_w4a4(quantization_config):
        return None  # W4A16_NVFP4 takes another linear method (modelopt.py:1084-1098)
    elif method == "modelopt" and _modelopt_algo(quantization_config) != "FP8":
        return None  # FP8 per-channel / block: other linear methods (modelopt.py:385-397)
    elif method in REGISTERED_PARAMETERS:
        registered, source = REGISTERED_PARAMETERS[method]
        kv_cache = method in _KV_CACHE_METHODS
    else:
        return None
    allowed = registered | _PLAIN | _SKIPPED | (_KV_CACHE_SCALES if kv_cache else frozenset())
    unknown = Counter(name.rsplit(".", 1)[-1] for name in tensor_names)
    names = engine_names()
    problems = []
    for suffix, count in sorted(unknown.items()):
        if suffix in allowed:
            continue
        if suffix in _KV_CACHE_SCALES:
            # A KV-cache scale with no KV-cache method to hold it.
            problems.append(
                f"{suffix} x{count}: no vLLM {ENGINE_VERSION} parameter of that name ({source})"
            )
        elif suffix not in names:
            # Not the quantization method's, and no model class of this vLLM
            # names it either (MoE router biases, sinks and the like are the
            # model's own parameters and are named).
            problems.append(
                f"{suffix} x{count}: named nowhere in vLLM {ENGINE_VERSION}'s source "
                f"and not registered by the method ({source})"
            )
    return tuple(problems)


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
