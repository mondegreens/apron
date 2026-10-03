"""Bring a new vLLM release into Apron: source, pinned requirements, facts.

One command per new tag (the manual form of phase-plan Phase 3 Tier 0):

1. shallow-clones the tag into ``.sources/vllm-<tag>`` (read locally from then
   on; nothing is read from the network at plan time);
2. writes ``docker/requirements-<tag>.txt`` from that tag's
   ``requirements/common.txt`` and ``cuda.txt``, the same layout as the first
   pinned file;
3. generates ``src/apron/adapters/backends/vllm_facts/<tag>.json`` (+ names).

Then build the image by pushing the tag ``runner-<tag>-rc1`` (the publish
workflow picks the pinned file) and pin its digest in ``runner_image.py``.

    uv run python scripts/new_vllm_version.py v0.30.0
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = "https://github.com/vllm-project/vllm.git"
# The runner image adds these to vLLM's own requirements.
APRON_ADDITIONS = ("hf-transfer>=0.1.8",)
# Prebuilt cubins exist only for x86_64 (the image's platform).
PLATFORM_MARKERS = {"flashinfer-cubin": '; platform_machine == "x86_64"'}


def sources_root() -> Path:
    for root in (REPO / ".sources", REPO.parents[2] / ".sources"):
        if root.is_dir():
            return root
    return REPO / ".sources"


def checkout(tag: str) -> Path:
    target = sources_root() / f"vllm-{tag}"
    if not target.exists():
        subprocess.run(
            ["git", "clone", "--quiet", "--depth", "1", "--branch", tag, UPSTREAM, str(target)],
            check=True,
        )
    return target


def _requirement_lines(path: Path) -> list[str]:
    lines = []
    for raw in path.read_text().splitlines():
        line = raw.split(" #", 1)[0].rstrip()
        if not line or line.lstrip().startswith("#") or line.startswith("-r "):
            continue
        name = re.split(r"[<>=!;\[ ]", line, maxsplit=1)[0]
        if name in PLATFORM_MARKERS and ";" not in line:
            line += PLATFORM_MARKERS[name]
        lines.append(line)
    return lines


def pinned_requirements(tag: str, source: Path) -> str:
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "--short=7", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    rel = source.relative_to(source.parents[1]) if len(source.parents) > 1 else source
    out = [
        f"# Apron runner requirements — pinned from vLLM {tag} tag",
        f"# Source: {rel} @ {tag} ({commit}) requirements/{{common,cuda}}.txt",
        "# When updating vLLM version: checkout the new tag, re-read requirements,",
        "# update every pin below. Do not hand-edit individual versions.",
        "",
        f"vllm=={tag.lstrip('v')}",
        "",
        f"# --- common.txt ({tag}) ---",
        *_requirement_lines(source / "requirements" / "common.txt"),
        "",
        f"# --- cuda.txt ({tag}) ---",
        *_requirement_lines(source / "requirements" / "cuda.txt"),
        "",
        "# --- apron runner additions ---",
        *APRON_ADDITIONS,
    ]
    return "\n".join(out) + "\n"


def main() -> int:
    if len(sys.argv) != 2 or not re.fullmatch(r"v\d+\.\d+\.\d+", sys.argv[1]):
        print(__doc__, file=sys.stderr)
        return 1
    tag = sys.argv[1]
    source = checkout(tag)
    (REPO / "docker" / f"requirements-{tag}.txt").write_text(pinned_requirements(tag, source))
    spec = importlib.util.spec_from_file_location(
        "generate_vllm_facts", REPO / "scripts" / "generate_vllm_facts.py"
    )
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    sys.argv = ["generate_vllm_facts.py", str(source)]
    generator.main()
    print(f"next: git tag runner-{tag}-rc1 && git push origin runner-{tag}-rc1; pin the digest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
