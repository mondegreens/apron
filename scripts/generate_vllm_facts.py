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
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

OUT = Path(__file__).resolve().parents[1] / "src/apron/adapters/backends/vllm_facts.json"
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
        "compressed_tensors": {
            "kv_cache_scales": _has_kv_cache_method(tree(ct_path)),
            "config_min_capability": ct_min,
            "config_min_capability_source": f"{ct_path.removeprefix('vllm/')}:{ct_line}",
            "schemes": schemes,
        },
    }


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 1
    facts = generate(Path(sys.argv[1]))
    OUT.write_text(json.dumps(facts, indent=2, sort_keys=True) + "\n")
    print(f"{facts['engine_version']} ({facts['source_commit'][:10]}) -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
