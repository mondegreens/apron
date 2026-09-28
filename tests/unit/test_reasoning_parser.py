"""A model that answers in a reasoning channel gets the engine's parser.

meta-models/Muse-Glimmer-30B on vLLM v0.30.0, H100, 2026-09-28 (solution
1220155171f73c...): every case answered right, every case scored wrong.  The
chat template ends the prompt at a bare ``<|start|>assistant``; the model
reasons in ``to=self<|message|> .. <|eom|>`` and answers in
``to=user<|message|> .. <|eot|>``.  vLLM turns no reasoning parser on by
itself (config/reasoning.py:22), so it returned the whole generation as
content (chat_completion/serving.py:982-985) with the special framing tokens
skipped (protocol.py:276), and the exact-match scorer compared
``" to=self...assistant to=user4"`` with ``"4"``.  The registered
``muse_glimmer`` parser keeps the framing (``adjust_request`` sets
``skip_special_tokens=False``) and splits the channels.

Measured offline with the checkpoint's tokenizer.json (a4e59da): the rendered
prompt below is 70 tokens and each reconstructed generation is exactly the
recorded output-token count (72, 78, 94) and decodes, special tokens skipped,
to the recorded content byte for byte.
"""

from __future__ import annotations

import ast
import json
import re
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import apron.domain.mechanisms.calculator  # noqa: F401 — registers calculators
from apron.adapters.backends.runpod import GPU_SPECS
from apron.adapters.backends.vllm_engine import VllmEngineAdapter
from apron.adapters.backends.vllm_quantization import engine_facts, engine_versions
from apron.adapters.evaluations.deterministic_scorer import DeterministicScorer, normalize
from apron.adapters.evidence.hf_hub import HFHubResolver
from apron.adapters.planning.calculator_source import CalculatorPlanningSource
from apron.application.orchestration.evidence import (
    chat_template_kwargs,
    reasoning_request_fields,
)
from apron.application.orchestration.plan_pipeline import (
    PlanPipelineResult,
    declares_reasoning_channel,
    reasoning_parser_for,
    run_plan_pipeline,
)
from apron.domain.schemas.primitives import HardwareSpec

REPO = Path(__file__).resolve().parents[2]
MUSE = REPO / "tests" / "fixtures" / "external-formats" / "muse-glimmer"
TEMPLATE = (MUSE / "chat_template.jinja").read_text()
TOKENIZER_CONFIG: dict[str, Any] = json.loads((MUSE / "tokenizer_config.json").read_text())
CONFIG: dict[str, Any] = json.loads((MUSE / "config.json").read_text())
# The plan's weight bytes (solutions.jsonl, solution 1220155171f73c...).
WEIGHT_BYTES = 59_553_253_376

H100_NAME = "NVIDIA H100 80GB HBM3"
H100 = HardwareSpec(
    gpu_sku=H100_NAME,
    total_memory_bytes=GPU_SPECS[H100_NAME]["total_memory_bytes"],
    compute_capability="9.0",
)

# The recorded content of each case (task-attempt records 12200349a8c7...,
# 122019a9e316..., 1220abba11b4...; 72, 78, 94 output tokens).
RECORDED = {
    "arith-2": (
        " to=selfWhat is 10 * 5? Answer with just the number.\n\nAnswer with just the "
        "number.\n\nWe should output 50\n\nProbably just 50. No extra text.\n\nFollow "
        "instruction: Answer with just the number.\n\nSo output 50.\n\nPotential nuance: "
        'just the number. So "50".assistant to=user50',
        "50",
    ),
    "arith-1": (
        " to=selfWhat is 2 + 2? Answer with just the number.\n\nAnswer with just the "
        'number.\n\nWe should output 4\n\nProbably just "4". No extra text.\n\nThe '
        "instruction says answer with just the number. So output 4.\n\nPotential nuance: "
        "maybe they want just number. So output 4.\n\nFollow policy.assistant to=user4",
        "4",
    ),
    "fact-1": (
        " to=selfWhat is the capital of France? Answer with just the city name.\n\nAnswer "
        'with just the city name.\n\nWe should output just city name. Probably "Paris". '
        "No extra punctuation? Just city name. Probably Paris.\n\nFollow instruction: "
        "Answer with just the city name.\n\nSo output: Paris\n\nNo extra text. Probably "
        "just Paris.\n\nEnsure no period? Might be okay. Safer to just Paris.\n\nProbably "
        "comply.\n\nassistant to=userParis",
        "Paris",
    ),
}
CHECKS = ("whitespace_normalized_exact_match", "strip_terminal_punctuation")


def _generation(recorded: str) -> str:
    """The generation with its special framing tokens back in place."""
    return (
        recorded.replace(" to=self", " to=self<|message|>", 1).replace(
            "assistant to=user", "<|eom|><|start|>assistant to=user<|message|>"
        )
        + "<|eot|>"
    )


# ---------------------------------------------------------------------------
# Engine facts: each version's parser registries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", engine_versions())
def test_each_version_lists_the_parsers_it_registers(version: str) -> None:
    facts = engine_facts(version)
    assert "muse_glimmer" in facts.reasoning_parsers
    assert "muse_glimmer" in facts.tool_parsers
    assert {"qwen3", "deepseek_r1", "openai_gptoss"} <= facts.reasoning_parsers
    assert {"hermes", "openai"} <= facts.tool_parsers
    # gpt-oss is served through harmony, not a parser named after its type.
    assert "gpt_oss" not in facts.reasoning_parsers


# ---------------------------------------------------------------------------
# The planner's rule
# ---------------------------------------------------------------------------

PARSERS = engine_facts("v0.30.0").reasoning_parsers


def test_muse_glimmer_declares_a_reasoning_channel_and_gets_its_parser() -> None:
    assert "enable_thinking" not in TEMPLATE
    assert (
        TOKENIZER_CONFIG["response_template"]["fields"]["reasoning_content"]["open_pattern"]
        == r"to=self<\|message\|>"
    )
    assert declares_reasoning_channel(TEMPLATE, TOKENIZER_CONFIG)
    assert declares_reasoning_channel(None, TOKENIZER_CONFIG)  # the schema alone
    assert declares_reasoning_channel(TEMPLATE, None)  # the template alone
    assert reasoning_parser_for(CONFIG["model_type"], PARSERS, TEMPLATE, TOKENIZER_CONFIG) == (
        "muse_glimmer"
    )


def test_a_type_match_alone_names_no_parser() -> None:
    # Mistral-7B-Instruct: "mistral" names a parser; its template has no channel.
    mistral = "{{ bos_token }}{% for m in messages %}[INST] {{ m['content'] }} [/INST]{% endfor %}"
    assert "mistral" in PARSERS
    assert reasoning_parser_for("mistral", PARSERS, mistral, {}) is None
    # An unregistered type, however it reasons.
    assert reasoning_parser_for("glm5_next", PARSERS, TEMPLATE, TOKENIZER_CONFIG) is None
    # enable_thinking: the request switches thinking off; the answer is the content.
    qwen3 = "{% if enable_thinking is false %}<think>\n\n</think>{% endif %}reasoning_content"
    assert reasoning_parser_for("qwen3", PARSERS, qwen3, None) is None
    assert reasoning_parser_for("muse_glimmer", frozenset(), TEMPLATE, TOKENIZER_CONFIG) is None


class _MuseResolver(HFHubResolver):
    """The pinned Muse-Glimmer files and the plan's weight bytes."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files

    def _get_model_info(self, model_id: str, revision: str | None) -> dict[str, Any]:
        return {"sha": "a4e59da52a7bc87ae7251dd5545c0dd437c44b68", "gated": False, "tags": []}

    def _download_file(self, model_id: str, filename: str, revision: str) -> bytes | None:
        return self.files.get(filename)

    def _tensor_meta(self, model_id: str, revision: str) -> dict[str, tuple[str, int]] | None:
        return {"weights": ("BF16", WEIGHT_BYTES)}


class _Clock:
    def now(self):  # type: ignore[no-untyped-def]
        from datetime import UTC, datetime

        return datetime(2026, 9, 28, tzinfo=UTC)


class _Ids:
    def generate(self) -> str:
        return "fixed"


def _muse_plan(parsers: frozenset[str], *, files: tuple[str, ...] = ()) -> PlanPipelineResult:
    names = files or ("config.json", "chat_template.jinja", "tokenizer_config.json")
    result = run_plan_pipeline(
        _MuseResolver({n: (MUSE / n).read_bytes() for n in names}),
        CalculatorPlanningSource(clock=_Clock()),
        "meta-models/Muse-Glimmer-30B",
        H100,
        clock=_Clock(),
        id_gen=_Ids(),
        reasoning_parsers=parsers,
    )
    assert result.ok, result.error
    assert result.plan is not None
    return result


def test_the_plan_names_the_parser_and_the_serve_command_carries_it() -> None:
    result = _muse_plan(PARSERS)
    assert result.plan is not None
    assert result.plan.engine_configuration["reasoning_parser"] == "muse_glimmer"
    assert result.chat_template == TEMPLATE
    command = VllmEngineAdapter()._build_serve_command(result.plan, None)
    assert "--reasoning-parser muse_glimmer" in command


def test_without_the_registry_the_plan_is_what_it_was() -> None:
    plain = _muse_plan(frozenset())
    named = _muse_plan(PARSERS)
    assert plain.plan is not None and named.plan is not None
    assert "reasoning_parser" not in plain.plan.engine_configuration
    others = {k: v for k, v in named.plan.engine_configuration.items() if k != "reasoning_parser"}
    assert others == plain.plan.engine_configuration


def test_the_tokenizer_config_schema_alone_is_enough() -> None:
    # A repo whose template never renders reasoning_content still declares it.
    result = _muse_plan(PARSERS, files=("config.json", "tokenizer_config.json"))
    assert result.plan is not None
    assert result.plan.engine_configuration["reasoning_parser"] == "muse_glimmer"
    # Neither the template nor the schema: no parser.
    bare = _muse_plan(PARSERS, files=("config.json",))
    assert bare.plan is not None
    assert "reasoning_parser" not in bare.plan.engine_configuration


# ---------------------------------------------------------------------------
# The request: minimal reasoning through the template's own variable
# ---------------------------------------------------------------------------


def test_the_request_asks_for_low_reasoning_strength() -> None:
    assert chat_template_kwargs(TEMPLATE) == {"reasoning_strength": "low"}
    # vLLM would drop reasoning_effort: the template has no such variable.
    assert reasoning_request_fields(TEMPLATE) is None
    assert chat_template_kwargs("{{ reasoning_strength }}{{ enable_thinking }}") == {
        "enable_thinking": False,
        "reasoning_strength": "low",
    }


def _render(**kwargs: Any) -> str:
    jinja2 = pytest.importorskip("jinja2")
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
    )

    def raise_exception(message: str) -> None:
        raise jinja2.TemplateError(message)

    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: date(2026, 9, 28).strftime(fmt)
    return env.from_string(TEMPLATE).render(
        messages=[{"role": "user", "content": "What is 10 * 5? Answer with just the number."}],
        add_generation_prompt=True,
        bos_token="<|begin_of_text|>",
        **kwargs,
    )


PROMPT_HIGH = (
    "<|begin_of_text|><|start|>system<|message|>You are a helpful AI assistant.\n"
    "Knowledge cutoff: 2026-01-04.\nCurrent date: 2026-09-28.\n\nReasoning strength: high."
    '\n\n# Valid recipients: "self", "user".<|eot|><|start|>user<|message|>'
    "What is 10 * 5? Answer with just the number.<|eot|><|start|>assistant"
)


def test_the_rendered_prompt_leaves_the_channel_to_the_model() -> None:
    # What the recorded run sent (70 tokens): no system message, so the
    # template's own, at its default strength; the prompt ends before a recipient.
    assert _render() == PROMPT_HIGH
    kwargs = chat_template_kwargs(TEMPLATE)
    assert kwargs is not None
    assert _render(**kwargs) == PROMPT_HIGH.replace(
        "Reasoning strength: high.", "Reasoning strength: low."
    )


# ---------------------------------------------------------------------------
# The split: content is the to=user body, and the scorer reads content only
# ---------------------------------------------------------------------------


def _declared_content(generation: str) -> str:
    """The body of the ``content`` field the model's response schema declares."""
    field = TOKENIZER_CONFIG["response_template"]["fields"]["content"]
    opened = re.search(field["open_pattern"], generation)
    assert opened is not None
    body = generation[opened.end() :]
    ends = [i for close in field["close"] if (i := body.find(close)) != -1]
    return body[: min(ends)] if ends else body


@pytest.mark.parametrize("case", sorted(RECORDED))
def test_the_recorded_content_fails_and_the_declared_answer_passes(case: str) -> None:
    recorded, expected = RECORDED[case]
    # No parser: the whole generation, framing stripped, is the content.
    assert re.sub(r"<\|[a-z_]+\|>", "", _generation(recorded)) == recorded
    assert normalize(recorded, CHECKS) != normalize(expected, CHECKS)
    # The model's declared answer channel holds the right answer.
    assert _declared_content(_generation(recorded)) == expected


def _parser_patterns(source: Path) -> dict[str, str]:
    """``_CONTENT_RE`` / ``_REASONING_RE`` from the pinned parser's source."""
    tree = ast.parse(source.read_text())
    patterns: dict[str, str] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ("_CONTENT_RE", "_REASONING_RE")
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.args[0], ast.Constant)
        ):
            patterns[node.targets[0].id] = str(node.value.args[0].value)
    return patterns


def test_the_pinned_parser_splits_the_recorded_generation() -> None:
    parser = None
    for root in (REPO / ".sources", REPO.parents[2] / ".sources"):
        candidate = (
            root / "vllm-v0.30.0" / "vllm" / "reasoning" / "muse_glimmer_reasoning_parser.py"
        )
        if candidate.exists():
            parser = candidate
            break
    if parser is None:
        pytest.skip("vLLM v0.30.0 source not available (.sources/vllm-v0.30.0)")
    patterns = _parser_patterns(parser)
    for recorded, expected in RECORDED.values():
        generation = _generation(recorded)
        content = re.search(patterns["_CONTENT_RE"], generation, re.DOTALL)
        assert content is not None and content.group(1) == expected
        reasoning = re.findall(patterns["_REASONING_RE"], generation, re.DOTALL)
        assert reasoning and reasoning[0].startswith("What is")


def test_the_scorer_reads_content_never_reasoning() -> None:
    class _Response:
        def raise_for_status(self) -> None: ...

        def json(self) -> dict[str, Any]:
            message = {"content": "4", "reasoning": "We should output 4"}
            return {"choices": [{"message": message, "finish_reason": "stop"}]}

    sent: list[dict[str, Any]] = []

    def post(url: str, json: dict[str, Any], timeout: float) -> _Response:
        sent.append(json)
        return _Response()

    scorer = DeterministicScorer()
    prepared = scorer.prepare(
        {
            "cases": [{"id": "arith-1", "prompt": "2 + 2?", "expected": "4", "max_tokens": "512"}],
            "model_id": "meta-models/Muse-Glimmer-30B",
            "endpoint": "http://x",
            "deterministic_checks": list(CHECKS),
            "chat_template_kwargs": chat_template_kwargs(TEMPLATE),
        }
    )
    with patch("apron.adapters.evaluations.deterministic_scorer.httpx.post", post):
        [result] = scorer.execute(prepared, "http://x")
    assert sent[0]["chat_template_kwargs"] == {"reasoning_strength": "low"}
    assert "reasoning_effort" not in sent[0]
    assert result["output"] == "4"
    assert result["accepted"] is True
