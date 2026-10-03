"""Generate the quantization facts the planner uses from a vLLM source tree.

For each quantization method the pinned engine registers, reads from the
source, never by hand:

- the minimum compute capability (the config class's ``get_min_capability``);
- the parameter names its linear method registers (``create_weights``);
- when its config attaches a KV-cache method to attention layers (a
  ``BaseKVCacheMethod`` subclass in its module), the scale names that method
  registers plus the checkpoint names vLLM maps onto them;

and, per version: the scheduler defaults `vllm serve` picks per GPU memory
tier (``get_batch_defaults``), and where the version needs one state (Mamba)
block per decode sequence (the check that refuses more, and the CUDA-graph
profiling cache of one block per sequence); and the names its
``--reasoning-parser`` and ``--tool-call-parser`` registries accept, and which
reasoning parser each architecture is served with (its test-registry example's
vllm-project recipe, ``reasoning_parser_architectures``);

and writes ``src/apron/adapters/backends/vllm_facts.json`` with the vLLM
version and commit they came from.  The index below only says where each
method lives; a moved or renamed class makes this script fail, never guess.
A test checks the file's version against the runner image's vLLM pin
(docker/requirements.txt) and, when the source is present, regenerates it.

    uv run python scripts/generate_vllm_facts.py /path/to/vllm   # checkout at the pin
"""

from __future__ import annotations

import ast
import gzip
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# One facts file per vLLM version: vllm_facts/<tag>.json (+ <tag>-names.txt.gz).
FACTS_DIR = Path(__file__).resolve().parents[1] / "src/apron/adapters/backends/vllm_facts"
# Every identifier-like name the engine's source mentions (identifiers, attributes,
# arguments, words in string constants).  A checkpoint tensor named nowhere in
# it cannot be loaded by any model class of this vLLM (the class 6 FPQuant
# tensor, backward_hadamard_matrix, is in no v0.29.0 file).
# Any identifier: parameter names are not all lower case (Mamba's A_log, dt_bias).
_SNAKE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_Q = "vllm/model_executor/layers/quantization/"
_CT = _Q + "compressed_tensors/schemes/"

# method -> (file, config class, linear method class).  Aliases share an entry.
METHODS: dict[str, tuple[str, str, str]] = {
    "fp_quant": (_Q + "fp_quant.py", "FPQuantConfig", "FPQuantLinearMethod"),
    "fp8": (_Q + "fp8.py", "Fp8Config", "Fp8LinearMethod"),
    "fbgemm_fp8": (_Q + "fbgemm_fp8.py", "FBGEMMFp8Config", "FBGEMMFp8LinearMethod"),
    "gptq": (_Q + "auto_gptq.py", "AutoGPTQConfig", "AutoGPTQLinearMethod"),
    "awq": (_Q + "auto_awq.py", "AutoAWQConfig", "AutoAWQMarlinLinearMethod"),
    "modelopt": (_Q + "modelopt.py", "ModelOptFp8Config", "ModelOptFp8LinearMethod"),
    # quant_method "modelopt" with an FP4 quant_algo: modelopt.py's
    # override_quantization_method renames it (the adapter mirrors that).
    "modelopt_fp4": (_Q + "modelopt.py", "ModelOptNvFp4Config", "ModelOptNvFp4LinearMethod"),
    "mxfp4": (_Q + "mxfp4.py", "Mxfp4Config", ""),
}
ALIASES = {
    "gptq_marlin": "gptq",
    "auto_gptq": "gptq",
    "awq_marlin": "awq",
    "auto_awq": "awq",
}
# compressed-tensors: the config checks 70, then each scheme its own minimum.
CT_CONFIG = (_Q + "compressed_tensors/compressed_tensors.py", "CompressedTensorsConfig")
# Model classes that declare ``IsHybrid`` (attention plus Mamba/linear
# attention state; model_executor/models/interfaces.py).  Apron's calculator
# has no memory model for them yet, so they must come out "unknown".
MODELS_DIR = "vllm/model_executor/models"
HYBRID_MARKER = "IsHybrid"

# The attention layer's KV-cache scales and the base config's name mapping.
KV_CACHE = (_Q + "kv_cache.py", "BaseKVCacheMethod")
CACHE_SCALE_MAPPER = (_Q + "base_config.py", "QuantizationConfig", "get_cache_scale_mapper")
CT_SCHEMES: dict[str, tuple[str, str]] = {
    "wNa16": (_CT + "compressed_tensors_wNa16.py", "CompressedTensorsWNA16"),
    "w8a8_int8": (_CT + "compressed_tensors_w8a8_int8.py", "CompressedTensorsW8A8Int8"),
    "w8a8_fp8": (_CT + "compressed_tensors_w8a8_fp8.py", "CompressedTensorsW8A8Fp8"),
}


def _class(tree: ast.Module, name: str, path: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise SystemExit(f"{path}: class {name} not found — update the index")


def _method(cls: ast.ClassDef, name: str, path: str) -> ast.FunctionDef:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise SystemExit(f"{path}: {cls.name}.{name} not found — update the index")


def _min_capability(fn: ast.FunctionDef, path: str) -> tuple[int, int]:
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, int)
        ):
            return node.value.value, node.lineno
    raise SystemExit(f"{path}:{fn.lineno}: get_min_capability is not a literal return")


def _registered(fn: ast.AST) -> list[str]:
    """Parameter names registered under *fn*: ``register_parameter("x", ...)``
    and, from v0.30.0's ModelOpt schemes, ``register_params(layer, "x", ...)``."""
    names = set()
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        index = {"register_parameter": 0, "register_params": 1}.get(node.func.attr)
        if index is None or len(node.args) <= index:
            continue
        arg = node.args[index]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            names.add(arg.value)
    return sorted(names)


def _linear_schemes(tree: ast.Module) -> list[ast.ClassDef]:
    """v0.30.0 serves every ModelOpt linear format with one generic linear
    method composed of per-key schemes (``QuantKeyScheme`` subclasses)."""
    bases = {"QuantKeyScheme"}
    found: list[ast.ClassDef] = []
    grew = True
    while grew:
        grew = False
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node not in found and _base_names(node) & bases:
                found.append(node)
                bases.add(node.name)
                grew = True
    return found


def _assigned_parameters(fn: ast.FunctionDef) -> list[str]:
    """``layer.<name> = KVCacheScaleParameter()`` targets."""
    names = set()
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "KVCacheScaleParameter"
        ):
            names.update(t.attr for t in node.targets if isinstance(t, ast.Attribute))
    return sorted(names)


def _mapped_suffixes(fn: ast.FunctionDef) -> list[str]:
    """Whole-suffix checkpoint names the mapper renames (``r"\\.kv_scale$"``)."""
    names = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
            if text.startswith("\\.") and text.endswith("$") and text[2:-1].isidentifier():
                names.add(text[2:-1])
    return sorted(names)


def _has_kv_cache_method(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ClassDef)
        and any(isinstance(b, ast.Name) and b.id == KV_CACHE[1] for b in node.bases)
        for node in tree.body
    )


def _base_names(cls: ast.ClassDef) -> set[str]:
    names = set()
    for base in cls.bases:
        if isinstance(base, ast.Name):
            names.add(base.id)
        elif isinstance(base, ast.Attribute):
            names.add(base.attr)
    return names


def _hybrid_classes(source: Path) -> dict[str, str]:
    """Class name -> ``file:line`` for every model class that is ``IsHybrid``,
    directly or through a base class in the models directory."""
    classes: dict[str, tuple[set[str], str]] = {}
    for path in sorted((source / MODELS_DIR).glob("*.py")):
        rel = f"{MODELS_DIR.removeprefix('vllm/')}/{path.name}"
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ClassDef):
                classes[node.name] = (_base_names(node), f"{rel}:{node.lineno}")
    hybrid = {n: where for n, (bases, where) in classes.items() if HYBRID_MARKER in bases}
    grew = True
    while grew:
        grew = False
        for name, (bases, where) in classes.items():
            if name not in hybrid and name != HYBRID_MARKER and bases & set(hybrid):
                hybrid[name] = where
                grew = True
    hybrid.pop(HYBRID_MARKER, None)
    if not hybrid:
        raise SystemExit(f"{MODELS_DIR}: no {HYBRID_MARKER} model classes found — update")
    return dict(sorted(hybrid.items()))


def source_names(source: Path) -> list[str]:
    names: set[str] = set()
    for path in sorted((source / "vllm").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.arg):
                names.add(node.arg)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                names.update(_WORD.findall(node.value))
    return sorted(n for n in names if _SNAKE.fullmatch(n))


def names_blob(names: list[str]) -> bytes:
    """Deterministic gzip (no timestamp) of one name per line."""
    return gzip.compress(("\n".join(names) + "\n").encode(), mtime=0)


def generate(source: Path) -> dict[str, Any]:
    trees: dict[str, ast.Module] = {}

    def tree(path: str) -> ast.Module:
        if path not in trees:
            trees[path] = ast.parse((source / path).read_text())
        return trees[path]

    kv_path, kv_cls = KV_CACHE
    kv_create = _method(_class(tree(kv_path), kv_cls, kv_path), "create_weights", kv_path)
    map_path, map_cls, map_fn = CACHE_SCALE_MAPPER
    mapper = _method(_class(tree(map_path), map_cls, map_path), map_fn, map_path)
    kv_cache = {
        "parameters": sorted({*_assigned_parameters(kv_create), *_mapped_suffixes(mapper)}),
        "source": f"{kv_path.removeprefix('vllm/')}:{kv_create.lineno} "
        f"({kv_cls}.create_weights), {map_path.removeprefix('vllm/')}:{mapper.lineno} "
        f"({map_cls}.{map_fn})",
    }
    if len(kv_cache["parameters"]) < 2:
        raise SystemExit(f"{kv_path}: KV-cache scale names not found — update the index")

    methods: dict[str, Any] = {}
    for method, (path, config_cls, linear_cls) in METHODS.items():
        cls = _class(tree(path), config_cls, path)
        minimum, line = _min_capability(_method(cls, "get_min_capability", path), path)
        entry: dict[str, Any] = {
            "min_capability": minimum,
            "min_capability_source": f"{path.removeprefix('vllm/')}:{line}",
        }
        linear = next(
            (n for n in tree(path).body if isinstance(n, ast.ClassDef) and n.name == linear_cls),
            None,
        )
        if linear_cls and linear is not None:
            create = _method(linear, "create_weights", path)
            entry["parameters"] = _registered(create)
            entry["parameters_source"] = (
                f"{path.removeprefix('vllm/')}:{create.lineno} ({linear_cls}.create_weights)"
            )
        elif linear_cls:
            schemes = _linear_schemes(tree(path))
            if not schemes:
                raise SystemExit(f"{path}: class {linear_cls} not found — update the index")
            names = sorted({n for cls in schemes for n in _registered(cls)})
            entry["parameters"] = names
            entry["parameters_source"] = (
                f"{path.removeprefix('vllm/')} ("
                + ", ".join(f"{c.name}:{c.lineno}" for c in schemes)
                + ")"
            )
        entry["kv_cache_scales"] = _has_kv_cache_method(tree(path))
        methods[method] = entry
    ct_path, ct_cls = CT_CONFIG
    ct_min, ct_line = _min_capability(
        _method(_class(tree(ct_path), ct_cls, ct_path), "get_min_capability", ct_path), ct_path
    )
    schemes = {}
    for scheme, (path, cls_name) in CT_SCHEMES.items():
        cls = _class(tree(path), cls_name, path)
        minimum, line = _min_capability(_method(cls, "get_min_capability", path), path)
        create = _method(cls, "create_weights", path)
        schemes[scheme] = {
            "min_capability": minimum,
            "min_capability_source": f"{path.removeprefix('vllm/')}:{line}",
            "parameters": _registered(create),
            "parameters_source": f"{path.removeprefix('vllm/')}:{create.lineno} "
            f"({cls_name}.create_weights)",
        }
    version = subprocess.run(
        ["git", "-C", str(source), "describe", "--tags"], capture_output=True, text=True
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    parsers = reasoning_parsers(source)
    tools = tool_parsers(source)
    return {
        "engine": "vllm",
        "engine_version": version,
        "source_commit": commit,
        "generated_by": "scripts/generate_vllm_facts.py",
        "methods": methods,
        "aliases": ALIASES,
        "kv_cache": kv_cache,
        "hybrid_architectures": _hybrid_classes(source),
        "source_names": _names_entry(version, source_names(source)),
        "architectures": registered_architectures(source),
        "tokenizer_modes": _tokenizer_modes(source),
        "reasoning_parsers": parsers,
        "reasoning_parser_architectures": reasoning_parser_architectures(source, parsers["names"]),
        "tool_parsers": tools,
        "tool_parser_architectures": parser_architectures(source, "tool", tools["names"]),
        "recipe_checkpoints": recipe_checkpoints(source, parsers["names"], tools["names"]),
        "batch_defaults": batch_defaults(source),
        "state_blocks": state_blocks(source),
        "compressed_tensors": {
            "kv_cache_scales": _has_kv_cache_method(tree(ct_path)),
            "config_min_capability": ct_min,
            "config_min_capability_source": f"{ct_path.removeprefix('vllm/')}:{ct_line}",
            "schemes": schemes,
        },
    }


BATCH_DEFAULTS = ("vllm/engine/arg_utils.py", "EngineArgs", "get_batch_defaults")
# `vllm serve` sizes the scheduler with the OpenAI server's entry.
SERVER_CONTEXT = "OPENAI_API_SERVER"


def _server_default(body: list[ast.stmt], name: str) -> int | None:
    """The OpenAI server's value in ``<name> = {UsageContext.X: n, ...}``."""
    for node in body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
            and isinstance(node.value, ast.Dict)
        ):
            for key, value in zip(node.value.keys, node.value.values, strict=True):
                if (
                    isinstance(key, ast.Attribute)
                    and key.attr == SERVER_CONTEXT
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, int)
                ):
                    return value.value
    return None


def _memory_tier(test: ast.expr) -> tuple[int, str | None] | None:
    """``device_memory >= N * GiB_bytes [and "x" not in device_name]`` ->
    (N, "x"); None for a test of anything else (the TPU / CPU branches)."""
    parts = (
        test.values if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And) else [test]
    )
    gib: int | None = None
    excluded: str | None = None
    for part in parts:
        if not (isinstance(part, ast.Compare) and len(part.ops) == 1):
            return None
        op, right = part.ops[0], part.comparators[0]
        if (
            isinstance(part.left, ast.Name)
            and part.left.id == "device_memory"
            and isinstance(op, ast.GtE)
            and isinstance(right, ast.BinOp)
            and isinstance(right.left, ast.Constant)
            and isinstance(right.left.value, int)
            and isinstance(right.right, ast.Name)
            and right.right.id == "GiB_bytes"
        ):
            gib = right.left.value
        elif (
            isinstance(part.left, ast.Constant)
            and isinstance(op, ast.NotIn)
            and isinstance(right, ast.Name)
            and right.id == "device_name"
        ):
            excluded = str(part.left.value)
        else:
            return None
    return None if gib is None else (gib, excluded)


def batch_defaults(source: Path) -> dict[str, Any]:
    """``max_num_batched_tokens`` and ``max_num_seqs`` `vllm serve` picks on a
    CUDA GPU: the device-memory tiers of ``get_batch_defaults``, first match
    wins (a tier may exclude GPUs whose lower-case name contains a string)."""
    path, cls_name, fn_name = BATCH_DEFAULTS
    fn = _method(_class(ast.parse((source / path).read_text()), cls_name, path), fn_name, path)
    chain = next(
        (n for n in fn.body if isinstance(n, ast.If) and _memory_tier(n.test) is not None),
        None,
    )
    tiers: list[dict[str, Any]] = []
    branch: ast.stmt | None = chain
    while branch is not None:
        if isinstance(branch, ast.If):
            tier = _memory_tier(branch.test)
            if tier is None:
                raise SystemExit(f"{path}:{branch.lineno}: unexpected batch-default test")
            body, line = branch.body, branch.lineno
            min_gib, excluded = tier
            nxt = branch.orelse
        else:
            raise SystemExit(f"{path}: unexpected batch-default branch")
        tokens = _server_default(body, "default_max_num_batched_tokens")
        seqs = _server_default(body, "default_max_num_seqs")
        if tokens is None or seqs is None:
            raise SystemExit(f"{path}:{line}: {SERVER_CONTEXT} defaults not found — update")
        tiers.append(
            {
                "min_memory_gib": min_gib,
                "excluded_name": excluded,
                "max_num_batched_tokens": tokens,
                "max_num_seqs": seqs,
                "source": f"{path.removeprefix('vllm/')}:{line}",
            }
        )
        if len(nxt) == 1 and isinstance(nxt[0], ast.If):
            branch = nxt[0]
            continue
        # The final else: every other CUDA GPU.
        tokens = _server_default(nxt, "default_max_num_batched_tokens")
        seqs = _server_default(nxt, "default_max_num_seqs")
        if tokens is None or seqs is None:
            raise SystemExit(f"{path}:{line}: fallback batch defaults not found — update")
        tiers.append(
            {
                "min_memory_gib": 0,
                "excluded_name": None,
                "max_num_batched_tokens": tokens,
                "max_num_seqs": seqs,
                "source": f"{path.removeprefix('vllm/')}:{nxt[0].lineno}",
            }
        )
        branch = None
    if not tiers:
        raise SystemExit(f"{path}: {cls_name}.{fn_name} memory tiers not found — update")
    return {
        "usage_context": SERVER_CONTEXT,
        "source": f"{path.removeprefix('vllm/')}:{fn.lineno} ({cls_name}.{fn_name})",
        "tiers": tiers,
    }


# The check that refuses more decode sequences than state (Mamba) blocks, by
# its message, and the CUDA-graph profiling allocation of one KV block per
# sequence (``min(max_num_reqs, max_cudagraph_capture_size)``).
STATE_BLOCK_CHECK = ("vllm/config/compilation.py", "exceeds available Mamba cache")
PROFILING_RUNNERS = ("vllm/v1/worker/gpu_model_runner.py", "vllm/v1/worker/gpu/cudagraph_utils.py")


def _raise_line(tree: ast.Module, text: str) -> int | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and any(
            isinstance(c, ast.Constant) and isinstance(c.value, str) and text in c.value
            for c in ast.walk(node)
        ):
            return node.lineno
    return None


def _profiling_min_blocks(tree: ast.Module) -> int | None:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "min"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Attribute)
            and node.args[0].attr == "max_num_reqs"
            and isinstance(node.args[1], ast.Attribute)
            and node.args[1].attr == "max_cudagraph_capture_size"
        ):
            return node.lineno
    return None


def state_blocks(source: Path) -> dict[str, Any]:
    """Where this version needs one state block per decode sequence: the
    check that refuses ``max_num_seqs`` above the Mamba cache's blocks (None
    when the version has none) and each runner's minimal profiling KV cache."""
    path, text = STATE_BLOCK_CHECK
    line = _raise_line(ast.parse((source / path).read_text()), text)
    profiling = []
    for runner in PROFILING_RUNNERS:
        if (source / runner).exists():
            at = _profiling_min_blocks(ast.parse((source / runner).read_text()))
            if at is not None:
                profiling.append(f"{runner.removeprefix('vllm/')}:{at}")
    return {
        "check": f"{path.removeprefix('vllm/')}:{line}" if line else None,
        "profiling_min_blocks": profiling,
    }


TOKENIZERS = ("vllm/tokenizers/registry.py", "_VLLM_TOKENIZERS")


def _tokenizer_modes(source: Path) -> dict[str, Any]:
    """vLLM's own tokenizer modes (a chat renderer for models whose repo has no
    chat template, e.g. DeepSeek V3.2 and V4): the keys of the registry dict."""
    path, name = TOKENIZERS
    for node in ast.parse((source / path).read_text()).body:
        if isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        elif isinstance(node, ast.Assign):
            target, value = node.targets[0], node.value
        else:
            continue
        if isinstance(target, ast.Name) and target.id == name and isinstance(value, ast.Dict):
            keys = sorted(str(k.value) for k in value.keys if isinstance(k, ast.Constant))
            return {"modes": keys, "source": f"{path.removeprefix('vllm/')}:{node.lineno}"}
    raise SystemExit(f"{path}: {name} not found — update the index")


# The names `--reasoning-parser` / `--tool-call-parser` accept: the keys of the
# lazy-registration dicts each package registers at import.
REASONING_PARSERS = ("vllm/reasoning/__init__.py", "_REASONING_PARSERS_TO_REGISTER")
TOOL_PARSERS = ("vllm/tool_parsers/__init__.py", "_TOOL_PARSERS_TO_REGISTER")


def _registry_names(source: Path, index: tuple[str, str]) -> dict[str, Any]:
    """The string keys of the module-level dict *index* names."""
    path, name = index
    for node in ast.parse((source / path).read_text()).body:
        if isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        elif isinstance(node, ast.Assign):
            target, value = node.targets[0], node.value
        else:
            continue
        if isinstance(target, ast.Name) and target.id == name and isinstance(value, ast.Dict):
            keys = sorted(
                str(k.value)
                for k in value.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            )
            if not keys:
                break
            return {"names": keys, "source": f"{path.removeprefix('vllm/')}:{node.lineno}"}
    raise SystemExit(f"{path}: {name} not found — update the index")


def reasoning_parsers(source: Path) -> dict[str, Any]:
    """Names ``--reasoning-parser`` accepts in this version."""
    return _registry_names(source, REASONING_PARSERS)


def tool_parsers(source: Path) -> dict[str, Any]:
    """Names ``--tool-call-parser`` accepts in this version."""
    return _registry_names(source, TOOL_PARSERS)


# Which reasoning parser vLLM serves an architecture with.  vLLM turns none on
# by itself (config/reasoning.py:22) and its source keys no parser on a model
# type (GLM-5's glm_moe_dsa is served with glm45, MiniMax-M3's minimax_m3_vl
# with minimax_m3).  The association is the project's own, in two places: the
# version's test registry names the example checkpoint of each architecture it
# serves (tests/models/registry.py), and the vllm-project/recipes serve command
# for that checkpoint names the parser (``features.reasoning.args``).  Joined
# on the checkpoint, the result is keyed on the architecture, never on a
# model id.  The recipes are read at a pinned commit, so the facts are
# reproducible; an architecture whose examples' recipes disagree names none.
EXAMPLE_REGISTRY = "tests/models/registry.py"
_EXAMPLE_TABLES = ("_TEXT_GENERATION_EXAMPLE_MODELS", "_MULTIMODAL_EXAMPLE_MODELS")
# vllm-project/recipes, checked out next to the vLLM source (.sources/recipes).
RECIPES_COMMIT = "f050a17eec51c7093727ddbc765c005647bc92f3"


def _example_checkpoints(source: Path) -> dict[str, set[str]]:
    """Architecture -> the checkpoints the version's test registry serves it with."""
    examples: dict[str, set[str]] = {}
    for node in ast.parse((source / EXAMPLE_REGISTRY).read_text()).body:
        if isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        elif isinstance(node, ast.Assign):
            target, value = node.targets[0], node.value
        else:
            continue
        if not (
            isinstance(target, ast.Name)
            and target.id in _EXAMPLE_TABLES
            and isinstance(value, ast.Dict)
        ):
            continue
        for key, info in zip(value.keys, value.values, strict=True):
            if not (isinstance(key, ast.Constant) and isinstance(info, ast.Call)):
                continue
            ids = examples.setdefault(str(key.value), set())
            extras = [kw.value for kw in info.keywords if kw.arg == "extras"]
            for arg in [*info.args[:2], *extras]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    ids.add(arg.value)
                elif isinstance(arg, ast.Dict):
                    ids.update(
                        str(v.value)
                        for v in arg.values
                        if isinstance(v, ast.Constant) and isinstance(v.value, str)
                    )
    if not examples:
        raise SystemExit(f"{EXAMPLE_REGISTRY}: no example tables found — update the index")
    return examples


def _version_tuple(text: str) -> tuple[int, ...] | None:
    match = re.fullmatch(r"v?(\d+(?:\.\d+)*)", text.strip())
    return tuple(int(p) for p in match.group(1).split(".")) if match else None


def _flag_value(args: list[str], flag: str) -> str | None:
    return args[args.index(flag) + 1] if flag in args[:-1] else None


def _recipe_parsers(recipes: Path) -> dict[str, dict[str, str | None]]:
    """Checkpoint -> its recipe's reasoning parser, tool-call parser and
    min_vllm_version, from the pinned recipes.

    The reasoning parser is in ``features.reasoning.args`` (or the base args
    of a recipe that always reasons); the tool-call parser in
    ``features.tool_calling.args``, which serve with
    ``--enable-auto-tool-choice`` (e.g. models/zai-org/GLM-5.1.yaml).
    """
    import io
    import tarfile

    import yaml

    archive = subprocess.run(
        ["git", "-C", str(recipes), "archive", RECIPES_COMMIT, "models"],
        capture_output=True,
        check=False,
    )
    if archive.returncode != 0:
        raise SystemExit(
            f"{recipes}: vllm-project/recipes commit {RECIPES_COMMIT} not found — fetch it"
        )
    parsers: dict[str, dict[str, str | None]] = {}
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        for member in sorted(tar.getmembers(), key=lambda m: m.name):
            if not (member.isfile() and member.name.endswith(".yaml")):
                continue
            handle = tar.extractfile(member)
            recipe = yaml.safe_load(handle.read()) if handle else None
            if not isinstance(recipe, dict):
                continue
            model = recipe.get("model") or {}
            features = recipe.get("features") or {}
            reasoning_args = [str(a) for a in (features.get("reasoning") or {}).get("args") or []]
            if not reasoning_args:  # a recipe that always reasons names it in its base args
                reasoning_args = [str(a) for a in model.get("base_args") or []]
            tool_args = [str(a) for a in (features.get("tool_calling") or {}).get("args") or []]
            reasoning = _flag_value(reasoning_args, "--reasoning-parser")
            tool = (
                _flag_value(tool_args, "--tool-call-parser")
                if "--enable-auto-tool-choice" in tool_args
                else None
            )
            if (reasoning or tool) and model.get("model_id"):
                minimum = model.get("min_vllm_version")
                parsers[str(model["model_id"])] = {
                    "reasoning": reasoning,
                    "tool": tool,
                    "min_vllm_version": str(minimum) if minimum else None,
                }
    if not any(entry["reasoning"] for entry in parsers.values()):
        raise SystemExit(f"{recipes}@{RECIPES_COMMIT}: no recipe names a reasoning parser")
    return parsers


def _recipes_checkout(source: Path) -> Path:
    recipes = source.parent / "recipes"
    if not (recipes / ".git").exists():
        raise SystemExit(f"{recipes}: vllm-project/recipes checkout not found next to vLLM")
    return recipes


def _source_version(source: Path) -> tuple[int, ...] | None:
    return _version_tuple(
        subprocess.run(
            ["git", "-C", str(source), "describe", "--tags"], capture_output=True, text=True
        ).stdout
    )


def recipe_parsers(source: Path, kind: str, registered: list[str]) -> dict[str, str]:
    """Checkpoint -> the *kind* (``reasoning`` / ``tool``) parser its recipe
    serves it with, for parsers this version registers and recipes it can run."""
    version = _source_version(source)
    known = set(registered)
    served: dict[str, str] = {}
    for checkpoint, entry in sorted(_recipe_parsers(_recipes_checkout(source)).items()):
        parser = entry[kind]
        needs = _version_tuple(entry["min_vllm_version"]) if entry["min_vllm_version"] else None
        if parser and parser in known and not (needs and version and needs > version):
            served[checkpoint] = parser
    return served


def parser_architectures(source: Path, kind: str, registered: list[str]) -> dict[str, Any]:
    """Architecture -> the *kind* parser its example checkpoint's recipe serves
    it with, for parsers this version registers and recipes it can run."""
    by_checkpoint = recipe_parsers(source, kind, registered)
    architectures: dict[str, Any] = {}
    for arch, checkpoints in sorted(_example_checkpoints(source).items()):
        served = {c: by_checkpoint[c] for c in sorted(checkpoints) if c in by_checkpoint}
        if len(set(served.values())) == 1:
            architectures[arch] = {
                "parser": next(iter(served.values())),
                "recipes": sorted(served),
            }
    field = "features.reasoning.args" if kind == "reasoning" else "features.tool_calling.args"
    return {
        "architectures": architectures,
        "source": f"{EXAMPLE_REGISTRY} ({', '.join(_EXAMPLE_TABLES)}) x "
        f"vllm-project/recipes@{RECIPES_COMMIT[:12]} models/*/*.yaml "
        f"({field})",
    }


def reasoning_parser_architectures(source: Path, registered: list[str]) -> dict[str, Any]:
    return parser_architectures(source, "reasoning", registered)


def recipe_checkpoints(
    source: Path, reasoning_registered: list[str], tool_registered: list[str]
) -> dict[str, Any]:
    """Checkpoint -> the parsers its own recipe serves it with.  For a
    checkpoint whose architecture has no example checkpoint in the test
    registry (Qwen4Exp's is ``""``, tests/models/registry.py:521, 1387), the
    recipe for that exact checkpoint is the project's only word on it."""
    reasoning = recipe_parsers(source, "reasoning", reasoning_registered)
    tool = recipe_parsers(source, "tool", tool_registered)
    return {
        "checkpoints": {
            c: {
                **({"reasoning_parser": reasoning[c]} if c in reasoning else {}),
                **({"tool_call_parser": tool[c]} if c in tool else {}),
            }
            for c in sorted(set(reasoning) | set(tool))
        },
        "source": f"vllm-project/recipes@{RECIPES_COMMIT[:12]} models/*/*.yaml "
        "(features.reasoning.args, features.tool_calling.args)",
    }


REGISTRY = "vllm/model_executor/models/registry.py"
# Registry tables of architectures this version can serve (not the lists of
# previously supported or out-of-tree ones).
_NOT_SERVED = {"_PREVIOUSLY_SUPPORTED_MODELS", "_OOT_SUPPORTED_MODELS"}


def registered_architectures(source: Path) -> dict[str, Any]:
    """Architecture names the version's model registry maps to a model class."""
    names: set[str] = set()
    for node in ast.parse((source / REGISTRY).read_text()).body:
        if isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        elif isinstance(node, ast.Assign):
            target, value = node.targets[0], node.value
        else:
            continue
        if (
            isinstance(target, ast.Name)
            and target.id.startswith("_")
            and target.id.endswith("_MODELS")
            and target.id not in _NOT_SERVED
            and isinstance(value, ast.Dict)
        ):
            names.update(str(k.value) for k in value.keys if isinstance(k, ast.Constant))
    if not names:
        raise SystemExit(f"{REGISTRY}: no architecture tables found — update")
    return {"names": sorted(names), "source": REGISTRY.removeprefix("vllm/")}


def names_file(version: str) -> Path:
    return FACTS_DIR / f"{version}-names.txt.gz"


def facts_file(version: str) -> Path:
    return FACTS_DIR / f"{version}.json"


def _names_entry(version: str, names: list[str]) -> dict[str, Any]:
    return {
        "file": names_file(version).name,
        "count": len(names),
        "sha256": hashlib.sha256(names_blob(names)).hexdigest(),
    }


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 1
    source = Path(sys.argv[1])
    facts = generate(source)
    version = facts["engine_version"]
    FACTS_DIR.mkdir(parents=True, exist_ok=True)
    names_file(version).write_bytes(names_blob(source_names(source)))
    facts_file(version).write_text(json.dumps(facts, indent=2, sort_keys=True) + "\n")
    print(f"{version} ({facts['source_commit'][:10]}) -> {facts_file(version)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
