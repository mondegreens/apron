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
from apron.adapters.backends.vllm_quantization import engine_facts, engine_versions, load_facts
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
    served_parser,
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


# ===========================================================================
# The rule, general: the parser vLLM serves the architecture with
# ===========================================================================
#
# Group C/D on vLLM v0.30.0 would fail as Muse-Glimmer did: MiniMax-M3's type
# is minimax_m3_vl, GLM-5.3's glm_moe_dsa, GLM-5.3-Flash's glm5_next -- none
# names a parser -- and DeepSeek V4 / V4.1 ship no chat template (vLLM renders
# their chat and opens <think> by default).  The engine facts join the
# version's test registry (architecture -> example checkpoint) with the
# vllm-project recipe serving that checkpoint (-> --reasoning-parser); the
# plan names the parser for the architecture, never for a model id.

HEADERS = REPO / "tests" / "fixtures" / "cohort" / "headers"
FORMATS = REPO / "tests" / "fixtures" / "external-formats"
H200 = HardwareSpec(
    gpu_sku="NVIDIA H200", total_memory_bytes=150_754_820_096, compute_capability="9.0"
)
# model -> (template fixture or None when vLLM renders the chat, TP, parser)
GROUP_CD: dict[str, tuple[str | None, int, str]] = {
    "MiniMaxAI/MiniMax-M3": ("minimax-m3", 8, "minimax_m3"),
    "zai-org/GLM-5.3": ("glm-5.3", 8, "glm45"),
    "zai-org/GLM-5.3-Flash": ("glm-5.3-flash", 4, "glm45"),
    "deepseek-ai/DeepSeek-V4-Flash-0731": (None, 4, "deepseek_v4"),
    "deepseek-ai/DeepSeek-V4.1-Flash": (None, 4, "deepseek_v41"),
}


def _served_with(version: str) -> dict[str, str]:
    table = load_facts(version)["reasoning_parser_architectures"]["architectures"]
    return {arch: entry["parser"] for arch, entry in table.items()}


def _header(model_id: str) -> dict[str, Any]:
    record: dict[str, Any] = json.loads(
        (HEADERS / (model_id.replace("/", "__") + ".json")).read_text()
    )
    return record


def _template(model_id: str) -> str | None:
    fixture = GROUP_CD[model_id][0]
    return (FORMATS / fixture / "chat_template.jinja").read_text() if fixture else None


@pytest.mark.parametrize("version", engine_versions())
def test_each_version_serves_architectures_with_parsers_it_registers(version: str) -> None:
    facts = load_facts(version)
    table = facts["reasoning_parser_architectures"]
    assert "tests/models/registry.py" in table["source"]
    assert "vllm-project/recipes@" in table["source"]
    served = _served_with(version)
    assert set(served.values()) <= engine_facts(version).reasoning_parsers
    assert served["GlmMoeDsaForCausalLM"] == "glm45"
    assert served["MiniMaxM3SparseForConditionalGeneration"] == "minimax_m3"
    assert served["DeepseekV4ForCausalLM"] == "deepseek_v4"
    assert served["MuseGlimmerForConditionalGeneration"] == "muse_glimmer"
    # Each entry says which recipe it came from.
    assert table["architectures"]["GlmMoeDsaForCausalLM"]["recipes"] == ["zai-org/GLM-5"]
    # No recipe of an example names a parser: none (vLLM turns gpt-oss's on
    # itself, model_executor/models/config.py:406-407).
    assert "MistralForCausalLM" not in served
    assert "GptOssForCausalLM" not in served


def test_v0_30_0_adds_the_architectures_it_was_taken_for() -> None:
    served = _served_with("v0.30.0")
    assert served["Glm5NextForConditionalGeneration"] == "glm45"
    assert served["DeepseekV41ForCausalLM"] == "deepseek_v41"
    # v0.29.0 registers neither architecture, so serves neither.
    older = _served_with("v0.29.0")
    assert "Glm5NextForConditionalGeneration" not in older
    assert "DeepseekV41ForCausalLM" not in older


@pytest.mark.parametrize("model_id", sorted(GROUP_CD))
def test_group_c_d_models_get_the_parser_their_architecture_is_served_with(
    model_id: str,
) -> None:
    config = _header(model_id)["config"]
    template = _template(model_id)
    parser = GROUP_CD[model_id][2]
    facts = engine_facts("v0.30.0")
    # The old rule: the type names no parser (or the repo has no template).
    assert reasoning_parser_for(config["model_type"], PARSERS, template, None) is None
    kwargs: dict[str, Any] = {
        "architectures": config["architectures"],
        "served_with": _served_with("v0.30.0"),
        "engine_renders_chat": template is None and config["model_type"] in facts.tokenizer_modes,
    }
    assert reasoning_parser_for(config["model_type"], PARSERS, template, None, **kwargs) == parser


def test_the_architecture_comes_first_and_must_be_registered() -> None:
    served = {"MuseGlimmerForConditionalGeneration": "glm45"}
    assert served_parser(
        "muse_glimmer", ["MuseGlimmerForConditionalGeneration"], PARSERS, served
    ) == ("glm45")
    # A parser the version does not register is never named.
    assert served_parser("muse_glimmer", ["X"], PARSERS, {"X": "no_such_parser"}) is None
    assert served_parser("muse_glimmer", ["Unlisted"], PARSERS, served) == "muse_glimmer"
    # The gates still hold: Mistral-7B's architecture maps nowhere, its type has no channel.
    mistral = "{{ bos_token }}{% for m in messages %}[INST] {{ m['content'] }} [/INST]{% endfor %}"
    assert (
        reasoning_parser_for(
            "mistral",
            PARSERS,
            mistral,
            {},
            architectures=["MistralForCausalLM"],
            served_with=_served_with("v0.30.0"),
        )
        is None
    )
    # enable_thinking: Qwen3.x is served with qwen3, yet the request switches thinking off.
    qwen = "{% if enable_thinking is false %}<think>\n\n</think>{% endif %}reasoning_content"
    assert (
        reasoning_parser_for(
            "qwen3_5",
            PARSERS,
            qwen,
            None,
            architectures=["Qwen3_5ForConditionalGeneration"],
            served_with=_served_with("v0.30.0"),
        )
        is None
    )


class _RecordedResolver(HFHubResolver):
    """A recorded revision: config and tensor headers, plus the template fixture."""

    def __init__(self, record: dict[str, Any], template: str | None) -> None:
        self.record = record
        self.template = template

    def _get_model_info(self, model_id: str, revision: str | None) -> dict[str, Any]:
        return {"sha": self.record["revision"], "gated": False, "tags": [], "safetensors": {}}

    def _download_file(self, model_id: str, filename: str, revision: str) -> bytes | None:
        if filename == "config.json":
            return json.dumps(self.record["config"]).encode()
        if filename == "chat_template.jinja" and self.template is not None:
            return self.template.encode()
        return None

    def _tensor_meta(self, model_id: str, revision: str) -> dict[str, tuple[str, int]] | None:
        return {name: (dtype, size) for name, (dtype, size, _) in self.record["tensors"].items()}


def _group_plan(
    model_id: str, served_with: dict[str, str] | None, parsers: frozenset[str] = PARSERS
) -> PlanPipelineResult:
    facts = engine_facts("v0.30.0")
    result = run_plan_pipeline(
        _RecordedResolver(_header(model_id), _template(model_id)),
        CalculatorPlanningSource(clock=_Clock()),
        model_id,
        H200,
        clock=_Clock(),
        id_gen=_Ids(),
        tensor_parallel=GROUP_CD[model_id][1],
        max_num_batched_tokens=8192,
        max_num_seqs=256,
        tokenizer_modes=facts.tokenizer_modes,
        reasoning_parsers=parsers,
        reasoning_parser_architectures=served_with,
    )
    assert result.ok, result.error
    assert result.plan is not None
    return result


@pytest.mark.parametrize("model_id", sorted(GROUP_CD))
def test_the_group_c_d_plan_names_the_parser_and_nothing_else_changes(model_id: str) -> None:
    parser = GROUP_CD[model_id][2]
    named = _group_plan(model_id, _served_with("v0.30.0"))
    plain = _group_plan(model_id, None, frozenset())
    assert named.plan is not None and plain.plan is not None
    assert named.plan.engine_configuration["reasoning_parser"] == parser
    assert "reasoning_parser" not in plain.plan.engine_configuration
    # Without the architecture table only a type that is itself a parser gets
    # one: DeepSeek V4 / V4.1 (engine-rendered chat), not GLM-5.x or MiniMax-M3.
    by_type = _group_plan(model_id, None)
    assert by_type.plan is not None
    type_named = by_type.plan.engine_configuration.get("reasoning_parser")
    config = _header(model_id)["config"]
    assert type_named == (parser if config["model_type"] == parser else None)
    others = {k: v for k, v in named.plan.engine_configuration.items() if k != "reasoning_parser"}
    assert others == plain.plan.engine_configuration
    command = VllmEngineAdapter()._build_serve_command(named.plan, None)
    assert f"--reasoning-parser {parser}" in command


# ---------------------------------------------------------------------------
# The request: each template's own minimal-reasoning knob, rendered
# ---------------------------------------------------------------------------


def _render_template(template: str, **kwargs: Any) -> str:
    jinja2 = pytest.importorskip("jinja2")
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
    )

    def raise_exception(message: str) -> None:
        raise jinja2.TemplateError(message)

    env.globals["raise_exception"] = raise_exception
    return env.from_string(template).render(
        messages=[{"role": "user", "content": "What is 10 * 5? Answer with just the number."}],
        add_generation_prompt=True,
        **kwargs,
    )


def test_minimax_m3_gets_thinking_mode_disabled() -> None:
    template = _template("MiniMaxAI/MiniMax-M3")
    assert template is not None
    assert "enable_thinking" not in template and "reasoning_effort" not in template
    kwargs = chat_template_kwargs(template)
    assert kwargs == {"thinking_mode": "disabled"}
    assert kwargs is not None
    assert reasoning_request_fields(template) is None
    # Undefined renders adaptive: the model decides whether to think.
    assert _render_template(template).endswith("]~b]ai\n")
    assert "Current thinking mode: adaptive." in _render_template(template)
    # Disabled: the prompt closes the thinking block; the answer starts at once.
    sent = _render_template(template, **kwargs)
    assert sent.endswith("]~b]ai\n</mm:think>")
    assert "Current thinking mode: disabled. Do not output any thinking process." in sent
    # A template that only mentions a thinking_mode gets nothing it cannot honour.
    assert chat_template_kwargs("{{ thinking_mode }}") is None


@pytest.mark.parametrize("model_id", ["zai-org/GLM-5.3", "zai-org/GLM-5.3-Flash"])
def test_glm_5_gets_low_reasoning_effort(model_id: str) -> None:
    template = _template(model_id)
    assert template is not None
    assert chat_template_kwargs(template) is None
    fields = reasoning_request_fields(template)
    assert fields == {"reasoning_effort": "low"}
    assert fields is not None
    # The field reaches the template as its variable (protocol.py:585-589).
    assert "<|system|>Reasoning Effort: Max<|user|>" in _render_template(template)
    sent = _render_template(template, **fields)
    assert sent.startswith("[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>")
    # Thinking is always on: the generation starts inside <think>.
    assert sent.endswith("<|assistant|><think>")


# ---------------------------------------------------------------------------
# The split: the pinned parsers on a synthetic generation
# ---------------------------------------------------------------------------


def _vllm_source() -> Path:
    for root in (REPO / ".sources", REPO.parents[2] / ".sources"):
        candidate = root / "vllm-v0.30.0" / "vllm"
        if candidate.is_dir():
            return candidate
    pytest.skip("vLLM v0.30.0 source not available (.sources/vllm-v0.30.0)")


def _parser_engine(vllm: Path, parser_files: tuple[str, ...], config_name: str) -> Any:
    """The pinned parser-engine state machine with the parser's own config.

    The engine modules are plain Python (their ``regex`` use is ``re``'s:
    compile/escape); the parser files' module-level code (constants, config
    function) runs in one namespace, without their request-protocol imports
    (a later file may use an earlier one's names, as V4.1's does V4's).
    """
    import importlib.util
    import types

    modules: dict[str, Any] = {"regex": re}
    for pkg in ("vllm", "vllm.parser", "vllm.parser.engine"):
        package = types.ModuleType(pkg)
        package.__path__ = []
        modules[pkg] = package
    with patch.dict("sys.modules", modules):
        import sys

        for name in (
            "events",
            "parser_engine_config",
            "incremental_lexer",
            "token_id_scanner",
            "streaming_parser_engine",
        ):
            spec = importlib.util.spec_from_file_location(
                f"vllm.parser.engine.{name}", vllm / "parser" / "engine" / f"{name}.py"
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            modules[spec.name] = module
        engine = sys.modules["vllm.parser.engine.streaming_parser_engine"]
        events = sys.modules["vllm.parser.engine.events"]
        config = sys.modules["vllm.parser.engine.parser_engine_config"]
    namespace: dict[str, Any] = {
        "re": re,
        "EventType": events.EventType,
        "ParserState": config.ParserState,
        "Transition": config.Transition,
        "ParserEngineConfig": config.ParserEngineConfig,
        "functools": __import__("functools"),
        "json": json,
        "contextlib": __import__("contextlib"),
        "TYPE_CHECKING": False,
        "find_tool_properties": None,
        "replace": __import__("dataclasses").replace,
    }
    for parser_file in parser_files:
        tree = ast.parse((vllm / "parser" / parser_file).read_text())
        body = [
            node
            for node in tree.body
            if not isinstance(node, ast.Import | ast.ImportFrom | ast.ClassDef)
            or (isinstance(node, ast.ImportFrom) and node.module == "__future__")
        ]
        exec(compile(ast.Module(body=body, type_ignores=[]), parser_file, "exec"), namespace)

    def split(generation: str, *, thinking: bool) -> tuple[str | None, str | None]:
        # The engine imports its own modules lazily: keep them resolvable.
        with patch.dict("sys.modules", modules):
            machine = engine.StreamingParserEngine(namespace[config_name](thinking=thinking), None)
            out = machine.feed(generation, []) + machine.finish()
        kinds = events.EventType
        reasoning = "".join(e.value for e in out if e.type == kinds.REASONING_CHUNK)
        content = "".join(e.value for e in out if e.type == kinds.TEXT_CHUNK)
        return reasoning or None, content or None

    return split


def test_the_pinned_glm45_parser_splits_a_glm_5_generation() -> None:
    vllm = _vllm_source()
    registry = (vllm / "reasoning" / "__init__.py").read_text()
    # glm45 and glm47 are one parser in v0.30.0: the GLM-4.7 MoE engine parser.
    assert (
        registry.count('"glm47_moe_reasoning_parser",\n        "Glm47MoeParserReasoningAdapter"')
        == 2
    )
    split = _parser_engine(vllm, ("glm47_moe.py",), "glm47_moe_config")
    # The prompt ends inside <think>; the generation closes it, then answers.
    generation = "10 * 5 is 50. Answer with just the number.</think>50"
    assert normalize(generation, CHECKS) != normalize("50", CHECKS)  # no parser: scored wrong
    # Thinking is on unless the request says thinking/enable_thinking false
    # (parser/glm47_moe.py Glm47MoeParser.__init__); GLM-5's never does.
    assert split(generation, thinking=True) == ("10 * 5 is 50. Answer with just the number.", "50")


@pytest.mark.parametrize(
    ("encoding", "parser_files", "config_name"),
    [
        ("deepseek_v4_encoding.py", ("deepseek_v4.py",), "deepseek_v4_config"),
        # V4.1's config extends V4's (parser/deepseek_v41.py imports it).
        ("deepseek_v41_encoding.py", ("deepseek_v4.py", "deepseek_v41.py"), "deepseek_v41_config"),
    ],
)
def test_the_pinned_deepseek_v4_parsers_split_the_engine_rendered_chat(
    encoding: str, parser_files: tuple[str, ...], config_name: str
) -> None:
    import importlib.util

    vllm = _vllm_source()
    spec = importlib.util.spec_from_file_location(encoding[:-3], vllm / "tokenizers" / encoding)
    assert spec is not None and spec.loader is not None
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    # What the tokenizer mode renders with no thinking kwarg: thinking, effort high
    # (tokenizers/deepseek_v4.py:30-45, deepseek_v41.py:70-76).
    prompt = encoder.encode_messages(
        [{"role": "user", "content": "What is 10 * 5? Answer with just the number."}],
        thinking_mode="thinking",
        drop_thinking=True,
        reasoning_effort="high",
    )
    assert prompt.endswith("<\uff5cAssistant\uff5c><think>")  # fullwidth bars
    generation = "10 times 5 is 50.</think>50"
    assert normalize(generation, CHECKS) != normalize("50", CHECKS)  # no parser: scored wrong
    # The parser starts in reasoning on the renderer's default (parser/deepseek_v4.py:248-253).
    split = _parser_engine(vllm, parser_files, config_name)
    assert split(generation, thinking=True) == ("10 times 5 is 50.", "50")


def test_the_pinned_minimax_m3_parser_splits_with_thinking_disabled() -> None:
    vllm = _vllm_source()
    tree = ast.parse((vllm / "reasoning" / "minimax_m3_reasoning_parser.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    extract = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "extract_reasoning"
    )
    extract.returns = None
    for arg in extract.args.args:
        arg.annotation = None
    namespace: dict[str, Any] = {}
    exec(compile(ast.Module(body=[extract], type_ignores=[]), "minimax_m3", "exec"), namespace)

    class _Parser:
        start_token = "<mm:think>"
        end_token = "</mm:think>"

        def __init__(self, chat_template_kwargs: dict[str, Any]) -> None:
            # MiniMaxM3ReasoningParser.__init__ (minimax_m3_reasoning_parser.py:44-45)
            self._initial_in_reasoning = chat_template_kwargs.get("thinking_mode") == "enabled"

    template = _template("MiniMaxAI/MiniMax-M3")
    parser = _Parser(chat_template_kwargs(template) or {})
    for generation in ("50", "</mm:think>50", "<mm:think>10 * 5 = 50.</mm:think>50"):
        _, content = namespace["extract_reasoning"](parser, generation, None)
        assert content == "50"
    # Without the parser a stray closer stays in the content and scores wrong.
    assert normalize("</mm:think>50", CHECKS) != normalize("50", CHECKS)


def test_engine_rendered_deepseek_chats_ask_for_low_reasoning_effort() -> None:
    """DeepSeek V4/V4.1 ship no chat template; vLLM's own renderer reads
    ``reasoning_effort`` (v0.30.0 tokenizers/deepseek_v4.py:43-48) and thinks
    at "high" by default, which can run past a case's max_tokens."""
    from apron.application.orchestration.evidence import reasoning_request_fields

    assert reasoning_request_fields(None, "deepseek_v4") == {"reasoning_effort": "low"}
    assert reasoning_request_fields(None, "deepseek_v41") == {"reasoning_effort": "low"}
    assert reasoning_request_fields(None, "mistral") is None
    assert reasoning_request_fields(None) is None


@pytest.mark.parametrize(
    "model_id", ["deepseek-ai/DeepSeek-V4-Flash-0731", "deepseek-ai/DeepSeek-V4.1-Flash"]
)
def test_the_plans_own_renderer_asks_for_low_reasoning_effort(model_id: str) -> None:
    """The renderer as the plan names it reaches the request fields (a lookup
    by the bare mode missed ``vllm tokenizer mode deepseek_v4``: the DeepSeek
    requests went out at the default "high" effort, caught in a dry run
    before any GPU run, 2026-09-28)."""
    from apron.application.orchestration.evidence import reasoning_request_fields

    result = _group_plan(model_id, _served_with("v0.30.0"))
    assert result.chat_template is None and result.chat_renderer is not None
    assert reasoning_request_fields(result.chat_template, result.chat_renderer) == {
        "reasoning_effort": "low"
    }
