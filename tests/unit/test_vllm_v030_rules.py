"""The vLLM v0.30.0 diagnosis rules: every v0.29 rule ported, each a hypothesis.

A new engine version's rules start as hypotheses (phase plan V.4): the v0.29.0
proofs do not carry over. Scanner-generated families are exactly what the
source scanner writes for the pinned v0.30.0 source (with v0.29's hand-set
correction kept); hand-curated families re-cite every line in v0.30.0. When the
pinned source checkouts are present (.sources/ is not committed), every cited
line is read back and must hold the rule's message.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from apron.adapters.backends.rule_loader import load_rules
from apron.adapters.backends.source_scanner import (
    _extract_message,
    classify_error,
    scan_and_export,
)

REPO = Path(__file__).parents[2]
RULES = REPO / "rules"
V29 = {r["error_family"]: r for r in load_rules(RULES, "vllm", "v0.29.0")}
V30 = {r["error_family"]: r for r in load_rules(RULES, "vllm", "v0.30.0")}
RAW30 = {p.stem: json.loads(p.read_text()) for p in (RULES / "vllm-v0.30").glob("*.json")}
CURATED = sorted(f for f, r in V30.items() if r["curation"])
SCANNED = sorted(f for f, r in V30.items() if not r["curation"])


def _checkout(version: str) -> Path | None:
    """The pinned vLLM package dir for ``version`` (checked by its git tag)."""
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
            return c / "vllm"
    return None


def _need(version: str) -> Path:
    tree = _checkout(version)
    if tree is None:
        pytest.skip(f"vLLM {version} source not available (.sources/vllm-{version})")
    return tree


# ---------------------------------------------------------------------------
# The set loads and is a fresh hypothesis set
# ---------------------------------------------------------------------------


def test_every_v029_rule_has_a_v030_counterpart() -> None:
    assert set(V30) == set(V29)
    assert not (RULES / "vllm-v0.30" / "history").exists()


@pytest.mark.parametrize("family", sorted(V30))
def test_rule_is_a_v030_hypothesis(family: str) -> None:
    rule = V30[family]
    assert rule["engine"] == "vllm"
    assert rule["engine_version"] == "v0.30.0"
    assert rule["status"] == "hypothesis"
    assert rule["rule_version"] == 1
    assert rule["supersedes"] is None
    assert rule["promoting_verification_fingerprint"] is None
    # Observed on v0.30.0 so far: only the Mamba state-block check (Nemotron-3.5).
    assert rule["count"] == (1 if family == "mamba_cache_blocks" else 0)
    assert rule["source_sites"] and rule["examples"]


@pytest.mark.parametrize("family", sorted(V30))
def test_correction_is_kept_from_v029(family: str) -> None:
    for key in ("correction_strategy", "correction_spec"):
        assert V30[family][key] == V29[family][key]


@pytest.mark.parametrize("family", CURATED)
def test_curated_rule_keeps_its_typed_fields(family: str) -> None:
    """Only the cited lines move; field names, types and enums are v0.29's."""
    new, old = V30[family], V29[family]

    def shape(rule: dict[str, Any]) -> dict[str, Any]:
        return {n: (s["type"], s["enum"]) for n, s in rule["extraction_schema"].items()}

    assert shape(new) == shape(old)
    for key in ("float16_blocklist", "kernel_architectures", "exception_class", "replaces"):
        assert new[key] == old[key]
    assert "must not overwrite" in new["curation"]  # the scanner keeps curated files


@pytest.mark.parametrize("family", SCANNED)
def test_scanned_rule_is_in_the_scanner_file_layout(family: str) -> None:
    raw = RAW30[family]
    assert raw["schema_version"] == 4
    assert "curation" not in raw
    for spec in raw["extraction_schema"].values():
        assert spec == {"type": "string"}


# ---------------------------------------------------------------------------
# Against the pinned v0.30.0 source
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    tree = _need("v0.30.0")
    out = tmp_path_factory.mktemp("scan")
    scan_and_export(tree, out, version="v0.30.0")
    return {p.stem: json.loads(p.read_text()) for p in (out / "vllm-v0.30").glob("*.json")}


@pytest.mark.parametrize("family", SCANNED)
def test_scanned_rule_is_what_the_scanner_writes_for_v030(
    family: str, generated: dict[str, dict[str, Any]]
) -> None:
    ours, scan = RAW30[family], generated[family]
    assert ours["source_sites"] == scan["source_sites"]
    assert ours["examples"] == scan["examples"]
    assert ours["extraction_schema"] == {
        k: {"type": t} for k, t in scan["extraction_schema"].items()
    }


@cache
def _tree(version: str, file: str) -> tuple[list[str], ast.Module]:
    source = (_need(version) / file).read_text(encoding="utf-8")
    return source.splitlines(), ast.parse(source)


def _raise_message(version: str, file: str, line: int) -> list[str]:
    """The message template of the raise statement spanning ``line``, and the
    same template with each ``{name}`` that is a string local of the enclosing
    function (``estimated_msg = (...)``) spelled out."""
    lines, tree = _tree(version, file)
    source = "\n".join(lines)
    raises = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Raise)
        and isinstance(n.exc, ast.Call)
        and n.exc.args
        and n.lineno <= line <= (n.end_lineno or n.lineno)
    ]
    if not raises:
        return []
    node = raises[0]
    assert isinstance(node.exc, ast.Call)
    message, _ = _extract_message(node.exc.args[0], source)
    funcs = [
        f
        for f in ast.walk(tree)
        if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
        and f.lineno <= node.lineno <= (f.end_lineno or f.lineno)
    ]
    local: dict[str, str] = {}
    for a in ast.walk(funcs[-1]) if funcs else ():
        if (
            isinstance(a, ast.Assign)
            and a.lineno < node.lineno
            and len(a.targets) == 1
            and isinstance(a.targets[0], ast.Name)
            and isinstance(a.value, (ast.JoinedStr, ast.Constant))
        ):
            text, _ = _extract_message(a.value, source)
            if text:
                local[a.targets[0].id] = text
    spelled = re.sub(r"\{(\w+)\}", lambda m: local.get(m.group(1), m.group(0)), message)
    return [message, spelled]


# Cited lines that are not a raise: the allocation PyTorch's allocator fails in.
NOT_A_RAISE = {
    ("oom_weight_load", "model_executor/layers/linear.py"): "torch.empty(",
    ("oom_weight_load", "v1/worker/gpu/model_runner.py"): "model_loader.load_model(",
}
_PLACEHOLDER = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")


def _pattern(text: str) -> re.Pattern[str]:
    """``{...}`` placeholders and a ``...`` elision match any text."""
    parts = re.split(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\.\.\.)", text)
    return re.compile(
        "".join(".*?" if i % 2 else re.escape(p) for i, p in enumerate(parts)), re.DOTALL
    )


def _example_matches(example: str, template: str) -> bool:
    """A filled-in example fits the template, or an (elided, truncated)
    template-style example fits a prefix of it."""
    if template.startswith(example) or _pattern(template).fullmatch(example):
        return True
    return bool(_pattern(example).match(_PLACEHOLDER.sub("{}", template)))


def _site_messages(version: str, rule: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """(file, message templates) for every cited raise; non-raise citations are
    checked for their allocation fragment."""
    sites = []
    for site in rule["source_sites"]:
        lines, _ = _tree(version, site["file"])
        assert 1 <= site["line"] <= len(lines), f"{site} is past the end of the file"
        fragment = NOT_A_RAISE.get((rule["error_family"], site["file"]))
        if fragment is not None:
            assert fragment in lines[site["line"] - 1], site
            continue
        found = _raise_message(version, site["file"], site["line"])
        assert found, f"{rule['error_family']}: no raise at {site['file']}:{site['line']}"
        sites.append((site["file"], found))
    return sites


@pytest.mark.parametrize("family", sorted(V30))
def test_every_cited_line_raises_and_every_example_is_its_message(family: str) -> None:
    rule = V30[family]
    sites = _site_messages("v0.30.0", rule)
    messages = [m for _, found in sites for m in found]
    for example in rule["examples"]:
        if rule["exception_class"] == "torch.OutOfMemoryError":
            # printed by PyTorch's allocator, not by vLLM: the site is the allocation
            assert example.startswith("torch.OutOfMemoryError: CUDA out of memory. ")
            continue
        assert any(_example_matches(example, m) for m in messages), (
            f"{family}: no cited v0.30.0 line prints {example[:100]!r}"
        )
    # A cited raise with no example is still this family's message: a line that
    # drifted onto a neighbouring raise is caught here.
    for file, found in sites:
        if any(_example_matches(e, m) for e in rule["examples"] for m in found):
            continue
        got = classify_error({"message": found[0], "file": file})
        assert got == family, f"{family}: {file} raises {found[0][:100]!r} ({got})"


# A distinctive fragment of the message that prints each typed field.
FRAGMENTS = {
    ("oom_weight_load", "tried_to_allocate_bytes"): "torch.empty",
    ("oom_weight_load", "total_capacity_bytes"): "torch.empty",
    ("oom_weight_load", "free_bytes"): "torch.empty",
    ("oom_kv_cache", "estimated_max_model_len"): "estimated maximum model length is",
    ("oom_kv_cache", "max_num_seqs_attempted"): "when warming up sampler with",
    ("max_model_len", "derived_max"): "derived max_model_len",
    ("dtype_incompatible", "model_type"): "does not support float16",
    ("dtype_incompatible", "unsupported_dtype"): "does not support float16",
    ("tp_divisibility", "num_heads"): "Total number of attention heads",
    (
        "quant_compute_capability",
        "min_capability",
    ): "capability: {quant_config.get_min_capability()}",
    ("quant_compute_capability", "current_capability"): "Current capability",
    ("mamba_cache_blocks", "max_num_seqs"): "exceeds available Mamba cache",
    ("mamba_cache_blocks", "mamba_cache_blocks"): "kv_cache_config.num_blocks",
}


@pytest.mark.parametrize("family", CURATED)
def test_typed_field_citations_print_the_field(family: str) -> None:
    for name, spec in V30[family]["extraction_schema"].items():
        cite = re.match(r"^([\w/]+\.py):(\d+)", spec["source"] or "")
        assert cite, f"{family}.{name} cites no line"
        path, line = cite.groups()
        lines, _ = _tree("v0.30.0", path)
        assert FRAGMENTS[(family, name)] in lines[int(line) - 1], (
            f"{family}.{name} @ {path}:{line}"
        )
        for cited in re.findall(r"([\w/]+\.py):(\d+)", spec["source"])[1:]:
            lines, _ = _tree("v0.30.0", cited[0])
            assert int(cited[1]) <= len(lines)


def test_float16_blocklist_is_the_v030_table() -> None:
    rule = V30["dtype_incompatible"]
    cite = re.match(
        r"^([\w/]+\.py):(\d+)-(\d+) _FLOAT16_NOT_SUPPORTED_MODELS",
        rule["float16_blocklist_source"] or "",
    )
    assert cite, "float16_blocklist_source cites no line range"
    path, start, end = cite.groups()
    lines, _ = _tree("v0.30.0", path)
    block = lines[int(start) - 1 : int(end)]
    assert block[0].startswith("_FLOAT16_NOT_SUPPORTED_MODELS = {") and block[-1] == "}"
    assert re.findall(r'^\s+"(\w+)":', "\n".join(block), re.MULTILINE) == list(
        rule["float16_blocklist"]
    )


# ---------------------------------------------------------------------------
# Classification: each example lands in its family
# ---------------------------------------------------------------------------

# The scanner's taxonomy (classify_error) has no class for these hand-curated
# examples; the LLM classifier's class descriptions route them.  Pinned here so
# a change on either side is noticed.
SCANNER_ELSEWHERE = {
    ("mamba_cache_blocks", "max_num_seqs ("): "config_incompatible",
    ("oom_kv_cache", "CUDA out of memory occurred when warming up sampler"): "oom_weight_load",
}


@pytest.mark.parametrize("family", sorted(V30))
def test_each_example_classifies_into_its_family(family: str) -> None:
    rule = V30[family]
    files = [s["file"] for s in rule["source_sites"]]
    for example in rule["examples"]:
        got = {classify_error({"message": example, "file": f}) for f in files}
        expected = next(
            (
                c
                for (fam, pre), c in SCANNER_ELSEWHERE.items()
                if fam == family and example.startswith(pre)
            ),
            family,
        )
        assert expected in got, f"{family}: {example[:100]!r} classifies as {sorted(got)}"
