"""Extract version-pinned engine constraints from vLLM source files.

Parses pinned source files (not a running Docker image) to produce
a constraint dict used as ExecutionSpec input for the calculator.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


def extract_supported_architectures(supported_models_path: Path) -> list[str]:
    """Extract architecture class names from supported_models.md.

    Only matches backtick-quoted names inside markdown table rows
    (lines starting with ``|``), not prose/documentation examples.
    """
    text = supported_models_path.read_text()
    architectures: list[str] = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        for match in re.finditer(r"`(\w+)`", line):
            name = match.group(1)
            if name == "Architecture" or name.startswith("---"):
                continue
            first_char = name[0] if name else ""
            if first_char.isupper() and name not in architectures:
                architectures.append(name)
    return sorted(architectures)


def extract_task_registry(tasks_path: Path) -> list[str]:
    """Extract task names from tasks.py task-related Literal definitions.

    Scopes to GenerationTask, PoolingTask, FrontendTask, SupportedTask
    Literals only — excludes ScoreType and other non-task Literals.
    """
    text = tasks_path.read_text()
    tasks: list[str] = []

    task_literal_pattern = re.compile(
        r"(?:GenerationTask|PoolingTask|FrontendTask|SupportedTask)"
        r"\s*=\s*Literal\[([^\]]+)\]",
    )
    for match in task_literal_pattern.finditer(text):
        literal_content = match.group(1)
        for task_match in re.finditer(r'"([^"]+)"', literal_content):
            task = task_match.group(1)
            if task not in tasks:
                tasks.append(task)

    return sorted(tasks)


def extract_kv_cache_specs(kv_cache_path: Path) -> list[str]:
    """Extract KV cache spec class names from kv_cache_interface.py.

    Matches class definitions that contain 'Spec' anywhere in the name
    (including plurals like UniformTypeKVCacheSpecs).
    """
    text = kv_cache_path.read_text()
    specs: list[str] = []
    for match in re.finditer(
        r"^class (\w*(?:Spec|Specs)\w*)\b", text, re.MULTILINE
    ):
        name = match.group(1)
        if name not in specs:
            specs.append(name)
    return sorted(specs)


def extract_constraints(fixture_dir: Path) -> dict[str, Any]:
    """Extract all engine constraints from pinned vLLM source files.

    Returns a dict suitable as ExecutionSpec input:
    - supported_architectures: list of model architecture class names
    - task_registry: list of task names (generate, embed, classify, etc.)
    - kv_cache_specs: list of KV cache spec class names
    """
    constraints: dict[str, Any] = {}

    supported_models = fixture_dir / "supported_models.md"
    if supported_models.exists():
        constraints["supported_architectures"] = extract_supported_architectures(
            supported_models
        )

    tasks = fixture_dir / "tasks.py"
    if tasks.exists():
        constraints["task_registry"] = extract_task_registry(tasks)

    kv_cache = fixture_dir / "kv_cache_interface.py"
    if kv_cache.exists():
        constraints["kv_cache_specs"] = extract_kv_cache_specs(kv_cache)

    return constraints
