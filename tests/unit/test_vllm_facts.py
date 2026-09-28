"""The engine facts are each pinned engine's, never stale.

``vllm_facts/<tag>.json`` is generated from that vLLM version's source by
scripts/generate_vllm_facts.py, one file per version Apron runs.  Each must be
for the vLLM its runner image installs (docker/requirements.txt for the first
pinned version, docker/requirements-<tag>.txt for the others), and equal what
its source generates — otherwise a refusal ("will not load", "unknown
architecture") could be stale knowledge.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
from pathlib import Path

import pytest

from apron.adapters.backends.vllm_quantization import (
    DEFAULT_ENGINE,
    engine_versions,
    load_facts,
    registered_architectures,
)

REPO = Path(__file__).resolve().parents[2]


def _image_pin(version: str) -> str | None:
    """The vllm version the image for *version* installs."""
    own = REPO / "docker" / f"requirements-{version}.txt"
    path = own if own.exists() else REPO / "docker" / "requirements.txt"
    match = re.search(r"^vllm==([\w.]+)$", path.read_text(), re.MULTILINE)
    return f"v{match.group(1)}" if match else None


@pytest.mark.parametrize("version", engine_versions())
def test_each_facts_file_is_for_the_vllm_its_image_runs(version: str) -> None:
    assert load_facts(version)["engine_version"] == version
    assert _image_pin(version) == version, (
        f"the image for {version} pins vLLM {_image_pin(version)}: "
        "run scripts/generate_vllm_facts.py on a checkout at the pin"
    )


def test_the_default_engine_is_one_of_the_pinned_versions() -> None:
    assert DEFAULT_ENGINE in engine_versions()
    assert len(engine_versions()) >= 2  # v0.29.0 and v0.30.0 side by side


def test_a_newer_version_adds_the_architectures_it_was_taken_for() -> None:
    # GLM-5.3-Flash and DeepSeek-V4.1-Flash: #1 on OpenRouter (2026-09-27),
    # registered by v0.30.0, not by v0.29.0.
    for arch in ("Glm5NextForConditionalGeneration", "DeepseekV41ForCausalLM"):
        assert arch not in registered_architectures("v0.29.0")
        assert arch in registered_architectures("v0.30.0")
    assert registered_architectures("v0.29.0") <= registered_architectures("v0.30.0")


def _source_checkout(version: str) -> Path | None:
    package = os.environ.get("APRON_VLLM_SOURCE")  # .../vllm/vllm (the package dir)
    candidates = [Path(package).parent] if package else []
    for root in (REPO / ".sources", REPO.parents[2] / ".sources"):
        candidates += [root / f"vllm-{version}", root / "vllm"]
    for c in candidates:
        if not (c / "vllm").is_dir():
            continue
        tag = subprocess.run(
            ["git", "-C", str(c), "describe", "--tags"], capture_output=True, text=True
        ).stdout.strip()
        if tag == version:
            return c
    return None


@pytest.mark.parametrize("version", engine_versions())
def test_facts_equal_what_the_pinned_source_generates(version: str) -> None:
    source = _source_checkout(version)
    if source is None:
        pytest.skip(f"vLLM {version} source not available (.sources/vllm-{version})")
    spec = importlib.util.spec_from_file_location(
        "generate_vllm_facts", REPO / "scripts" / "generate_vllm_facts.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.generate(source) == load_facts(version), "run scripts/generate_vllm_facts.py"


@pytest.mark.parametrize("version", [v for v in engine_versions() if v != "v0.29.0"])
def test_pinned_requirements_are_what_the_procedure_derives(version: str) -> None:
    """docker/requirements-<tag>.txt is scripts/new_vllm_version.py's output
    for that tag's source (v0.29.0 predates the procedure; its file stays)."""
    source = _source_checkout(version)
    if source is None:
        pytest.skip(f"vLLM {version} source not available (.sources/vllm-{version})")
    spec = importlib.util.spec_from_file_location(
        "new_vllm_version", REPO / "scripts" / "new_vllm_version.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pinned = (REPO / "docker" / f"requirements-{version}.txt").read_text()
    assert module.pinned_requirements(version, source) == pinned
