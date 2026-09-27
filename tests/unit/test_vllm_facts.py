"""The quantization facts are the pinned engine's, never stale.

``vllm_facts.json`` is generated from the vLLM source by
scripts/generate_vllm_facts.py.  If the runner image's vLLM changes and the
facts are not regenerated, a refusal ("will not load") could be stale
knowledge — so the versions must match, and the file must equal what the
source generates.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
from pathlib import Path

import pytest

from apron.adapters.backends.vllm_quantization import ENGINE_VERSION, FACTS

REPO = Path(__file__).resolve().parents[2]


def _image_pin() -> str:
    text = (REPO / "docker" / "requirements.txt").read_text()
    match = re.search(r"^vllm==([\w.]+)$", text, re.MULTILINE)
    assert match, "docker/requirements.txt pins no vllm version"
    return f"v{match.group(1)}"


def test_facts_are_for_the_vllm_the_image_runs() -> None:
    assert _image_pin() == ENGINE_VERSION, (
        f"the image pins vLLM {_image_pin()} but the facts are for {ENGINE_VERSION}: "
        "run scripts/generate_vllm_facts.py on a checkout at the pin"
    )


def _source_checkout() -> Path | None:
    package = os.environ.get("APRON_VLLM_SOURCE")  # .../vllm/vllm (the package dir)
    candidates = [Path(package).parent] if package else []
    candidates += [REPO / ".sources" / "vllm", REPO.parents[2] / ".sources" / "vllm"]
    return next((c for c in candidates if (c / "vllm").is_dir()), None)


def test_facts_equal_what_the_pinned_source_generates() -> None:
    source = _source_checkout()
    if source is None:
        pytest.skip("pinned vLLM source not available (set APRON_VLLM_SOURCE)")
    version = subprocess.run(
        ["git", "-C", str(source), "describe", "--tags"], capture_output=True, text=True
    ).stdout.strip()
    assert version == ENGINE_VERSION, f"the source checkout is at {version}, not {ENGINE_VERSION}"
    spec = importlib.util.spec_from_file_location(
        "generate_vllm_facts", REPO / "scripts" / "generate_vllm_facts.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.generate(source) == FACTS, "run scripts/generate_vllm_facts.py"
