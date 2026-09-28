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

# Generated from each pinned vLLM source by scripts/generate_vllm_facts.py, one
# file per version (vllm_facts/<tag>.json); a test ties each to the runner
# image's pin for that version (docker/requirements*.txt).
FACTS_DIR = Path(__file__).with_name("vllm_facts")
# The version these module-level facts describe; per-plan engine selection
# reads other versions with ``load_facts``.
DEFAULT_ENGINE = "v0.29.0"
# vLLM's ``--gpu-memory-utilization`` when a plan sets none: CacheConfig's
# default, the same in every pinned version (config/cache.py:111 in v0.29.0,
# :103 in v0.30.0).  Fix-proof boots without it requested 0.92 of the GPU.
VLLM_DEFAULT_GPU_MEMORY_UTILIZATION = 0.92


def _version_key(tag: str) -> tuple[int, ...]:
    return tuple(int(x) for x in tag.lstrip("v").split("."))


def engine_versions() -> tuple[str, ...]:
    """Every vLLM version Apron has facts for, oldest first."""
    return tuple(sorted((p.stem for p in FACTS_DIR.glob("v*.json")), key=_version_key))


@cache
def load_facts(version: str) -> dict[str, Any]:
    return json.loads((FACTS_DIR / f"{version}.json").read_text())


# Checkpoint tensors every engine version loads without a method's help.
_PLAIN = frozenset({"weight", "bias"})
# Rotary buffers some checkpoints store; model loaders skip them by name
# (e.g. models/deepseek_mtp.py:327: ``if "rotary_emb.inv_freq" in name``).
_SKIPPED = frozenset({"inv_freq", "cos_cached", "sin_cached"})
# Generic tokenizer modes, and the one "auto" already detects (Mistral).
_GENERIC_TOKENIZER_MODES = frozenset({"auto", "hf", "slow", "mistral"})


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


class EngineFacts:
    """What one vLLM version knows, from its generated facts file.

    - quantization: each method's minimum compute capability (checked at
      ``config/vllm.py`` before loading) and the parameters its linear method
      registers;
    - the architectures its model registry serves, and which are hybrid
      (attention plus Mamba/linear-attention state);
    - its own tokenizer/renderer modes named after a model type (DeepSeek V3.2
      and V4 ship no chat template; vLLM renders their chat only in that mode,
      and "auto" never picks it: tokenizers/registry.py:147-163);
    - the reasoning and tool-call parsers it registers (none is on unless
      ``vllm serve`` names it: config/reasoning.py:22);
    - every identifier its source mentions (the load check's last resort).
    """

    def __init__(self, version: str) -> None:
        facts = load_facts(version)
        self.version: str = facts["engine_version"]
        self.facts = facts
        self.hybrid_architectures = frozenset(facts["hybrid_architectures"])
        self.tokenizer_modes = (
            frozenset(facts["tokenizer_modes"]["modes"]) - _GENERIC_TOKENIZER_MODES
        )
        self.architectures = frozenset(facts["architectures"]["names"])
        # Names `--reasoning-parser` / `--tool-call-parser` accept in this version
        # (reasoning/__init__.py, tool_parsers/__init__.py registries).
        self.reasoning_parsers = frozenset(facts["reasoning_parsers"]["names"])
        self.tool_parsers = frozenset(facts["tool_parsers"]["names"])
        # Architecture -> the reasoning parser vLLM's own recipes name for it.
        self.reasoning_parser_architectures: dict[str, str] = {
            arch: entry["parser"]
            for arch, entry in facts["reasoning_parser_architectures"]["architectures"].items()
        }
        # method ->(minimum capability, source line)
        self.quant_min_capability: dict[str, tuple[int, str]] = self._method_table(
            "min_capability", "min_capability_source"
        )
        # compressed-tensors checks the config's minimum, then the scheme's.
        self.compressed_tensors_schemes: dict[str, tuple[int, str]] = {
            scheme: (entry["min_capability"], entry["min_capability_source"])
            for scheme, entry in facts["compressed_tensors"]["schemes"].items()
        }
        self.registered_parameters: dict[str, tuple[frozenset[str], str]] = {
            method: (frozenset(names), source)
            for method, (names, source) in self._method_table(
                "parameters", "parameters_source"
            ).items()
        }
        self._ct_parameters: dict[str, tuple[frozenset[str], str]] = {
            scheme: (frozenset(entry["parameters"]), entry["parameters_source"])
            for scheme, entry in facts["compressed_tensors"]["schemes"].items()
        }
        # Scheduler defaults per GPU memory tier (get_batch_defaults), and where
        # the version needs one state block per decode sequence.
        self.batch_tiers: tuple[dict[str, Any], ...] = tuple(facts["batch_defaults"]["tiers"])
        self.state_block_check: str | None = facts["state_blocks"]["check"]
        self.state_block_profiling: tuple[str, ...] = tuple(
            facts["state_blocks"]["profiling_min_blocks"]
        )
        self.kv_cache_scales = frozenset(facts["kv_cache"]["parameters"])
        self._kv_cache_methods = frozenset(
            m for m, entry in facts["methods"].items() if entry.get("kv_cache_scales")
        ) | frozenset(
            alias
            for alias, m in facts["aliases"].items()
            if facts["methods"].get(m, {}).get("kv_cache_scales")
        )

    def _method_table(self, key: str, source: str) -> dict[str, tuple[Any, str]]:
        table = {
            method: (entry[key], entry[source])
            for method, entry in self.facts["methods"].items()
            if key in entry
        }
        for alias, method in self.facts["aliases"].items():
            if method in table:
                table[alias] = table[method]
        return table

    def names(self) -> frozenset[str]:
        return engine_names(self.version)

    def _batch_tier(self, total_memory_bytes: int, gpu_name: str) -> dict[str, Any]:
        """The first tier of ``get_batch_defaults`` this GPU falls in."""
        name = gpu_name.lower()
        for tier in self.batch_tiers:
            excluded = tier["excluded_name"]
            if total_memory_bytes >= tier["min_memory_gib"] * _GIB and not (
                excluded and excluded in name
            ):
                return tier
        raise ValueError(f"vLLM {self.version}: no batch-default tier for {gpu_name}")

    def default_max_num_batched_tokens(self, total_memory_bytes: int, gpu_name: str) -> int:
        """``max_num_batched_tokens`` this version's OpenAI server uses on the GPU;
        the startup profiling forward runs this many tokens."""
        return int(self._batch_tier(total_memory_bytes, gpu_name)["max_num_batched_tokens"])

    def default_max_num_seqs(self, total_memory_bytes: int, gpu_name: str) -> int:
        """``max_num_seqs`` this version's OpenAI server uses on the GPU."""
        return int(self._batch_tier(total_memory_bytes, gpu_name)["max_num_seqs"])

    def default_max_num_seqs_source(self, total_memory_bytes: int, gpu_name: str) -> str:
        return str(self._batch_tier(total_memory_bytes, gpu_name)["source"])

    def min_capability(self, quantization_config: dict[str, Any] | None) -> int | None:
        """Capability (major*10+minor) this version needs for the checkpoint's
        format: 0 unquantized; None when it cannot load the format or the format
        is not recognised (never a guess)."""
        if not quantization_config:
            return 0
        method = _method(quantization_config)
        if method == "compressed-tensors":
            scheme = _compressed_tensors_scheme(quantization_config)
            return self.compressed_tensors_schemes[scheme][0] if scheme else None
        entry = self.quant_min_capability.get(method)
        return entry[0] if entry else None

    def load_problems(
        self, tensor_names: Iterable[str], quantization_config: dict[str, Any] | None
    ) -> tuple[str, ...] | None:
        """Why this version would refuse a quantized checkpoint's tensors; ``()``
        when it loads them all; ``None`` when it cannot be told here."""
        if not quantization_config:
            return None
        method = _method(quantization_config)
        if method == "compressed-tensors":
            scheme = _compressed_tensors_scheme(quantization_config)
            if scheme is None:
                return None
            registered, source = self._ct_parameters[scheme]
            kv_cache = bool(self.facts["compressed_tensors"].get("kv_cache_scales"))
        elif method == "modelopt_fp4" and not _nvfp4_w4a4(quantization_config):
            return None  # W4A16_NVFP4 takes another linear method
        elif method == "modelopt" and _modelopt_algo(quantization_config) != "FP8":
            return None  # FP8 per-channel / block: other linear methods
        elif method in self.registered_parameters:
            registered, source = self.registered_parameters[method]
            kv_cache = method in self._kv_cache_methods
        else:
            return None
        allowed = (
            registered | _PLAIN | _SKIPPED | (self.kv_cache_scales if kv_cache else frozenset())
        )
        unknown = Counter(name.rsplit(".", 1)[-1] for name in tensor_names)
        names = self.names()
        problems = []
        for suffix, count in sorted(unknown.items()):
            if suffix in allowed:
                continue
            if suffix in self.kv_cache_scales:
                # A KV-cache scale with no KV-cache method to hold it.
                problems.append(
                    f"{suffix} x{count}: no vLLM {self.version} parameter of that name ({source})"
                )
            elif suffix not in names:
                # Not the quantization method's, and no model class of this vLLM
                # names it either (MoE router biases, sinks and the like are the
                # model's own parameters and are named).
                problems.append(
                    f"{suffix} x{count}: named nowhere in vLLM {self.version}'s source "
                    f"and not registered by the method ({source})"
                )
        return tuple(problems)


@cache
def engine_facts(version: str) -> EngineFacts:
    return EngineFacts(version)


@cache
def engine_names(version: str = DEFAULT_ENGINE) -> frozenset[str]:
    """Every identifier-like name that version's source mentions (generated)."""
    entry = load_facts(version)["source_names"]
    blob = (FACTS_DIR / entry["file"]).read_bytes()
    if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
        raise RuntimeError(f"{entry['file']} does not match {version}.json — regenerate")
    return frozenset(gzip.decompress(blob).decode().split())


def registered_architectures(version: str = DEFAULT_ENGINE) -> frozenset[str]:
    """Architectures that version's model registry serves."""
    return engine_facts(version).architectures


def engine_for(architecture: str, versions: Iterable[str]) -> str | None:
    """The newest of *versions* whose registry serves *architecture*; None if none."""
    for version in sorted(versions, key=_version_key, reverse=True):
        if architecture in registered_architectures(version):
            return version
    return None


# The default version's facts, for callers that do not choose a version yet
# (the first cohort's records were all made on it).
_DEFAULT = engine_facts(DEFAULT_ENGINE)
FACTS: dict[str, Any] = _DEFAULT.facts
ENGINE_VERSION: str = _DEFAULT.version
HYBRID_ARCHITECTURES: frozenset[str] = _DEFAULT.hybrid_architectures
MODEL_TOKENIZER_MODES: frozenset[str] = _DEFAULT.tokenizer_modes
QUANT_MIN_CAPABILITY: dict[str, tuple[int, str]] = _DEFAULT.quant_min_capability
COMPRESSED_TENSORS_SCHEMES: dict[str, tuple[int, str]] = _DEFAULT.compressed_tensors_schemes
REGISTERED_PARAMETERS: dict[str, tuple[frozenset[str], str]] = _DEFAULT.registered_parameters


def min_capability(
    quantization_config: dict[str, Any] | None, version: str = DEFAULT_ENGINE
) -> int | None:
    return engine_facts(version).min_capability(quantization_config)


def load_problems(
    tensor_names: Iterable[str],
    quantization_config: dict[str, Any] | None,
    version: str = DEFAULT_ENGINE,
) -> tuple[str, ...] | None:
    return engine_facts(version).load_problems(tensor_names, quantization_config)


# ---------------------------------------------------------------------------
# Scheduler defaults `vllm serve` picks for a GPU (they size the profiling run)
# ---------------------------------------------------------------------------

_GIB = 1 << 30


def default_max_num_batched_tokens(
    total_memory_bytes: int, gpu_name: str, version: str = DEFAULT_ENGINE
) -> int:
    """``max_num_batched_tokens`` *version*'s OpenAI server uses on this GPU
    (``get_batch_defaults``, v0.29.0 ``engine/arg_utils.py:2698-2727``: 16384 at
    >= 160 GiB; 8192 at >= 70 GiB unless the name contains "a100"; else 2048),
    read from that version's generated facts.  The startup profiling forward
    runs this many tokens.
    """
    return engine_facts(version).default_max_num_batched_tokens(total_memory_bytes, gpu_name)


def default_max_num_seqs(
    total_memory_bytes: int, gpu_name: str, version: str = DEFAULT_ENGINE
) -> int:
    """``max_num_seqs`` *version*'s OpenAI server uses on this GPU (v0.29.0
    ``engine/arg_utils.py:2698-2727``: 1024 at >= 70 GiB unless "a100", else
    256), read from that version's generated facts."""
    return engine_facts(version).default_max_num_seqs(total_memory_bytes, gpu_name)
