"""Generate the quantization facts the planner uses from a vLLM source tree.

For each quantization method the pinned engine registers, reads from the
source, never by hand:

- the minimum compute capability (the config class's ``get_min_capability``);
- the parameter names its linear method registers (``create_weights``);
- when its config attaches a KV-cache method to attention layers (a
  ``BaseKVCacheMethod`` subclass in its module), the scale names that method
  registers plus the checkpoint names vLLM maps onto them;

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

OUT = Path(__file__).resolve().parents[1] / "src/apron/adapters/backends/vllm_facts.json"
# Every identifier-like name the engine's source mentions (identifiers, attributes,
# arguments, words in string constants).  A checkpoint tensor named nowhere in
# it cannot be loaded by any model class of this vLLM (the class 6 FPQuant
# tensor, backward_hadamard_matrix, is in no v0.29.0 file).
NAMES_OUT = OUT.with_name("vllm_source_names.txt.gz")
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


def _registered(fn: ast.FunctionDef) -> list[str]:
    names = set()
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "register_parameter"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            names.add(node.args[0].value)
    return sorted(names)


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
        if linear_cls:
            create = _method(_class(tree(path), linear_cls, path), "create_weights", path)
            entry["parameters"] = _registered(create)
            entry["parameters_source"] = (
                f"{path.removeprefix('vllm/')}:{create.lineno} ({linear_cls}.create_weights)"
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
    return {
        "engine": "vllm",
        "engine_version": version,
        "source_commit": commit,
        "generated_by": "scripts/generate_vllm_facts.py",
        "methods": methods,
        "aliases": ALIASES,
        "kv_cache": kv_cache,
        "hybrid_architectures": _hybrid_classes(source),
        "source_names": _names_entry(source_names(source)),
        "tokenizer_modes": _tokenizer_modes(source),
        "compressed_tensors": {
            "kv_cache_scales": _has_kv_cache_method(tree(ct_path)),
            "config_min_capability": ct_min,
            "config_min_capability_source": f"{ct_path.removeprefix('vllm/')}:{ct_line}",
            "schemes": schemes,
        },
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


def _names_entry(names: list[str]) -> dict[str, Any]:
    return {
        "file": NAMES_OUT.name,
        "count": len(names),
        "sha256": hashlib.sha256(names_blob(names)).hexdigest(),
    }


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 1
    source = Path(sys.argv[1])
    facts = generate(source)
    NAMES_OUT.write_bytes(names_blob(source_names(source)))
    OUT.write_text(json.dumps(facts, indent=2, sort_keys=True) + "\n")
    print(f"{facts['engine_version']} ({facts['source_commit'][:10]}) -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
