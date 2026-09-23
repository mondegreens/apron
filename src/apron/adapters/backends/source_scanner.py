"""Scan pinned engine source for correctable deployment errors.

Reads the engine source to find every raise site whose error message
contains a corrective suggestion. Outputs structured data per error
site: file, line, exception type, message, extraction fields, and
implied correction. Generates versioned rule files.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_SUGGESTION_PATTERNS = re.compile(
    r"try |use |set |increase |decrease |lower |enable |disable |"
    r"specify |pass |remove |instead|please |must be |should be |"
    r"--\w|VLLM_",
    re.IGNORECASE,
)

_CONFIG_MODULES = frozenset(
    {
        "config",
        "__init__",
        "__post_init__",
        "_validate",
        "_verify",
        "_check",
        "validate",
        "verify",
    }
)


def scan_source(source_dir: Path) -> list[dict[str, Any]]:
    """Scan a vLLM source tree for correctable error sites.

    Returns a list of dicts, one per correctable raise site:
    - file: relative path within source_dir
    - line: line number
    - exception_type: ValueError, RuntimeError, etc.
    - message: the error message template (f-string or literal)
    - format_vars: list of f-string variable names (extraction fields)
    - has_suggestion: True if message contains corrective text
    - module_context: enclosing function/class name
    - is_config_time: True if the raise is in a config/init/validate path
    """
    results: list[dict[str, Any]] = []

    for py_file in sorted(source_dir.rglob("*.py")):
        if "__pycache__" in str(py_file) or "/tests/" in str(py_file):
            continue

        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            continue

        rel_path = str(py_file.relative_to(source_dir))

        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue

            exc_info = _extract_raise_info(node, source)
            if exc_info is None:
                continue

            exc_type, message, format_vars, line_no = exc_info

            if not _SUGGESTION_PATTERNS.search(message):
                continue

            enclosing = _find_enclosing(tree, line_no)
            is_config = _is_config_time(rel_path, enclosing)

            results.append(
                {
                    "file": rel_path,
                    "line": line_no,
                    "exception_type": exc_type,
                    "message": message,
                    "format_vars": format_vars,
                    "has_suggestion": True,
                    "module_context": enclosing,
                    "is_config_time": is_config,
                }
            )

    return results


def _extract_raise_info(node: ast.Raise, source: str) -> tuple[str, str, list[str], int] | None:
    """Extract exception type, message text, format vars from a raise node."""
    exc = node.exc

    if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
        exc_type = exc.func.id
        if exc_type not in ("ValueError", "RuntimeError", "NotImplementedError"):
            return None
        if not exc.args:
            return None
        msg_node = exc.args[0]
    elif isinstance(exc, ast.Call) and isinstance(exc.func, ast.Attribute):
        exc_type = exc.func.attr
        if exc_type not in ("ValueError", "RuntimeError", "NotImplementedError"):
            return None
        if not exc.args:
            return None
        msg_node = exc.args[0]
    else:
        return None

    message, format_vars = _extract_message(msg_node, source)
    if not message:
        return None

    return exc_type, message, format_vars, node.lineno


def _extract_message(node: ast.expr, source: str) -> tuple[str, list[str]]:
    """Extract string content and format variables from an AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, []

    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        fmt_vars: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                var_name = _get_var_name(value.value)
                parts.append(f"{{{var_name}}}")
                if var_name:
                    fmt_vars.append(var_name)
        return "".join(parts), fmt_vars

    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Mod)
        and isinstance(node.left, ast.Constant)
        and isinstance(node.left.value, str)
    ):
        return node.left.value, []

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
        and isinstance(node.func.value, ast.Constant)
        and isinstance(node.func.value.value, str)
    ):
        return node.func.value.value, []

    lines = source.splitlines()
    start = getattr(node, "lineno", 0) - 1
    end = getattr(node, "end_lineno", start + 1)
    if 0 <= start < len(lines):
        chunk = " ".join(lines[start:end]).strip()
        return chunk[:500], []

    return "", []


def _get_var_name(node: ast.expr) -> str:
    """Get a readable name from a format variable AST node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        obj = _get_var_name(node.value)
        return f"{obj}.{node.attr}" if obj else node.attr
    if isinstance(node, ast.Subscript):
        obj = _get_var_name(node.value)
        return obj
    if isinstance(node, ast.Call):
        return _get_var_name(node.func)
    return ""


def _find_enclosing(tree: ast.Module, line_no: int) -> str:
    """Find the enclosing function or class name for a line number."""
    best = ""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", None) or node.lineno + 1000
            if hasattr(node, "lineno") and node.lineno <= line_no <= end:
                best = node.name
        elif isinstance(node, ast.ClassDef):
            end = getattr(node, "end_lineno", None) or node.lineno + 1000
            if hasattr(node, "lineno") and node.lineno <= line_no <= end and not best:
                best = node.name
    return best


_RUNTIME_DIRS = frozenset(
    {
        "benchmarks",
        "tool_parsers",
        "entrypoints/anthropic",
        "entrypoints/openai",
        "entrypoints/cli/benchmark",
    }
)


def _is_config_time(file_path: str, enclosing: str) -> bool:
    """Heuristic: is this raise likely at config/init time, not request time?"""
    for runtime_dir in _RUNTIME_DIRS:
        if runtime_dir in file_path:
            return False
    if "kernels/" in file_path or "kernel" in file_path.split("/")[-1]:
        return False
    if "/config/" in file_path:
        return True
    if "engine/arg_utils" in file_path:
        return True
    lower = enclosing.lower()
    for pattern in _CONFIG_MODULES:
        if pattern in lower:
            return True
    if any(kw in lower for kw in ("ensure", "setup", "load", "build", "create")):
        return True
    if "forward" in lower or "execute" in lower or "process_request" in lower:
        return False
    return False


def classify_error(entry: dict[str, Any]) -> str:
    """Classify a scanned error into a failure class name."""
    msg = entry["message"].lower()
    file_path = entry["file"].lower()

    if "out of memory" in msg or "outofmemoryerror" in msg:
        return "oom"
    if "kv cache" in msg and ("memory" in msg or "available" in msg):
        return "oom"
    if "max_model_len" in msg or "max_seq_len" in msg:
        return "max_model_len"
    if "divisible" in msg and ("tensor" in msg or "parallel" in msg or "head" in msg):
        return "tp_divisibility"
    if "dtype" in msg and "not supported" in msg:
        return "dtype_incompatible"
    if "float16" in msg and "not support" in msg:
        return "dtype_incompatible"
    if "compute capability" in msg or "minimum capability" in msg:
        return "quant_compute_capability"
    if "quantiz" in msg and ("not supported" in msg or "capability" in msg):
        return "quant_compute_capability"
    if "specul" in msg or "draft" in msg or "eagle" in msg or "mtp" in msg:
        return "speculative_config"
    if "lora" in msg.lower():
        return "lora_config"
    if "pipeline_parallel" in msg or "expert_parallel" in msg or "data_parallel" in msg:
        return "parallelism_config"
    if "world_size" in msg or "num_gpu" in msg or "nproc" in msg:
        return "parallelism_config"
    if "cuda_graph" in msg or "custom_op" in msg or "inductor" in msg:
        return "compilation_config"
    if "profil" in msg:
        return "profiler_config"
    if "multimodal" in msg or "vision" in msg or "mm_processor" in msg:
        return "multimodal_config"
    if "kv_transfer" in msg or "kv_connector" in msg:
        return "kv_transfer_config"
    if "platform" in msg and "not supported" in msg:
        return "platform_unsupported"
    if "device" in msg and ("not supported" in msg or "not available" in msg):
        return "platform_unsupported"
    if "scheduler" in msg or "batched_tokens" in msg:
        return "scheduler_config"
    if "tensor_parallel" in msg or "tp_size" in msg or "tp=" in msg:
        return "parallelism_config"
    if "not supported" in msg or "not compatible" in msg or "incompatible" in msg:
        return "config_incompatible"
    if "not enabled" in msg or "must be enabled" in msg:
        return "config_incompatible"
    if "backend" in msg and ("not" in msg or "unsupported" in msg or "invalid" in msg):
        return "compilation_config"
    if "image" in msg and ("size" in msg or "resolution" in msg or "format" in msg):
        return "multimodal_config"
    if "audio" in msg or "video" in msg:
        return "multimodal_config"
    if "distributed" in file_path:
        return "parallelism_config"

    if "config/" in file_path:
        return "config_incompatible"

    if "model_executor/" in file_path:
        return "model_runtime"

    return "other_correctable"


def derive_correction_spec(entry: dict[str, Any]) -> dict[str, Any]:
    """Derive a correction spec from a scanned error site."""
    msg = entry["message"].lower()

    if "try" in msg and ("lower" in msg or "decreas" in msg or "reduc" in msg):
        for var in entry["format_vars"]:
            return {"action": "scale_field", "field": var, "factor": 0.5}

    if "try" in msg and ("increas" in msg or "rais" in msg):
        for var in entry["format_vars"]:
            return {"action": "scale_field", "field": var, "factor": 2.0}

    if "set" in msg or "enable" in msg or "use" in msg:
        flag_match = re.search(r"--([\w-]+)", entry["message"])
        if flag_match:
            flag = flag_match.group(1).replace("-", "_")
            if "enable" in msg:
                return {"action": "set_field", "field": flag, "value": "true"}
            if "disable" in msg or "remove" in msg:
                return {"action": "remove_field", "field": flag}
            return {"action": "set_field", "field": flag, "value": "auto"}

    if "instead" in msg:
        flag_match = re.search(r"--([\w-]+)", entry["message"])
        if flag_match:
            return {"action": "remove_field", "field": flag_match.group(1).replace("-", "_")}
        return {"action": "use_alternative", "source": "error_suggestion"}

    flag_match = re.search(r"--([\w-]+)", entry["message"])
    if flag_match:
        flag = flag_match.group(1).replace("-", "_")
        if "not" in msg or "disable" in msg or "remove" in msg or "without" in msg:
            return {"action": "remove_field", "field": flag}
        return {"action": "set_field", "field": flag, "value": "auto"}

    env_match = re.search(r"(VLLM_\w+)", entry["message"])
    if env_match:
        return {"action": "set_env", "variable": env_match.group(1), "value": "1"}

    return {"action": "fallback", "description": "manual review needed"}


def scan_and_export(
    source_dir: Path,
    output_dir: Path,
    engine: str = "vllm",
    version: str = "v0.29.0",
) -> dict[str, Any]:
    """Scan source and export rule files grouped by failure class."""
    entries = scan_source(source_dir)
    config_entries = [e for e in entries if e["is_config_time"]]

    by_class: dict[str, list[dict[str, Any]]] = {}
    for entry in config_entries:
        cls = classify_error(entry)
        by_class.setdefault(cls, []).append(entry)

    v = version if version.startswith("v") else f"v{version}"
    parts = v.split(".")
    major_minor = ".".join(parts[:2]) if len(parts) > 2 else v
    rules_dir = output_dir / f"{engine}-{major_minor}"
    rules_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "total_raises": len(entries),
        "config_time": len(config_entries),
        "classes": {},
    }

    named_strategies = {
        "oom": "reduce_memory_pressure",
        "max_model_len": "clamp_max_model_len",
        "dtype_incompatible": "fallback_dtype",
        "tp_divisibility": "reduce_tensor_parallel",
        "quant_compute_capability": "remove_quantization",
        "lora_config": "fallback_engine_config",
    }

    for cls, class_entries in sorted(by_class.items()):
        extraction_fields: dict[str, str] = {}
        for entry in class_entries:
            for var in entry["format_vars"]:
                clean = var.split(".")[-1]
                if not clean or clean.startswith("_"):
                    continue
                if clean.isupper():
                    continue
                if clean in ("self", "type", "len", "str", "int", "cls"):
                    continue
                extraction_fields.setdefault(clean, "string")

        examples = [e["message"][:200] for e in class_entries[:5]]

        if cls in named_strategies:
            correction_field = {"correction_strategy": named_strategies[cls]}
        else:
            spec = {"action": "fallback", "description": "manual review needed"}
            for entry in class_entries:
                candidate = derive_correction_spec(entry)
                if candidate.get("action") != "fallback":
                    spec = candidate
                    break
            correction_field = {"correction_spec": spec}

        rule: dict[str, Any] = {
            "schema_version": 3,
            "engine": engine,
            "engine_version": version,
            "error_family": cls,
            **correction_field,
            "extraction_schema": {k: v for k, v in extraction_fields.items() if k},
            "source_sites": [{"file": e["file"], "line": e["line"]} for e in class_entries],
            "examples": examples,
            "status": "hypothesis",
            "count": len(class_entries),
        }

        rule_path = rules_dir / f"{cls}.json"
        rule_path.write_text(
            json.dumps(rule, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        summary["classes"][cls] = len(class_entries)

    return summary
