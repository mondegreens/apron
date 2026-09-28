"""What a plan tells the deployment checks (task suite v3), and the tool-call rule.

Suite v3 checks the serving configuration, not the model: whether the server
the plan starts serves a long context, tool calls, structured output, images
and a separate reasoning channel.  Which of those a plan can serve is a fact
of the plan (its engine flags) and of the checkpoint (its config and chat
template); ``deployment_facts`` reads them, GPU-free, into the string map the
scorer receives (``DeploymentCheckScorer``).  A check the plan cannot serve is
skipped with the reason these facts give, never sent and scored as wrong.

``tool_call_parser_for`` is the plan-side rule for tool calling, the analogue
of ``plan_pipeline.reasoning_parser_for``; ``with_tool_calling`` is the plan
flag it implies.  Both live here until the planner adopts them: a plan that
names a tool parser is a new solution identity, so the flag is added only to
plans that run suite v3 (v2's plans, and their records, stay as they are).
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from apron.domain.schemas.solutions import DeploymentPlan

#: The keys ``deployment_facts`` produces; the scorer refuses any other key.
FACT_KEYS = frozenset(
    {
        "max_model_len",
        "reasoning_parser",
        "thinking_switch",
        "tool_call_parser",
        "tool_reason",
        "vision",
        "vision_reason",
    }
)

#: vLLM's default per-prompt item limit for a modality the server was not
#: given a limit for (``MultiModalConfig.get_limit_per_prompt``: v0.29.0
#: config/multimodal.py:495-502, v0.30.0 :478-490).
VLLM_DEFAULT_MM_LIMIT = 999


def deployment_facts(
    plan: DeploymentPlan,
    model_config: Mapping[str, Any] | None,
    chat_template: str | None,
) -> dict[str, str]:
    """The facts suite v3's checks need from one plan, as strings.

    ``max_model_len`` sizes the long-context needle; ``reasoning_parser`` and
    ``thinking_switch`` say whether the model reasons (the reasoning split is
    checked only then); ``tool_call_parser`` is set only when the plan starts
    the server with ``--enable-auto-tool-choice --tool-call-parser``;
    ``vision`` says whether the server takes an image.
    """
    engine = dict(plan.engine_configuration)
    parser = engine.get("tool_call_parser", "")
    auto = engine.get("enable_auto_tool_choice", "").strip().lower() == "true"
    if parser and auto:
        tool_reason = f"the plan starts vLLM with --enable-auto-tool-choice --tool-call-parser {parser}"
    elif parser:
        tool_reason = (
            f"the plan names tool parser {parser} without --enable-auto-tool-choice: "
            "vLLM refuses tool_choice=auto"
        )
        parser = ""
    else:
        tool_reason = (
            "the plan starts vLLM without --enable-auto-tool-choice --tool-call-parser: "
            "vLLM refuses tool_choice=auto"
        )
    vision, vision_reason = vision_enabled(model_config or {}, engine)
    return {
        "max_model_len": engine.get("max_model_len", ""),
        "reasoning_parser": engine.get("reasoning_parser", ""),
        "thinking_switch": "true" if chat_template and "enable_thinking" in chat_template else "",
        "tool_call_parser": parser,
        "tool_reason": tool_reason,
        "vision": "true" if vision else "",
        "vision_reason": vision_reason,
    }


def vision_enabled(
    model_config: Mapping[str, Any], engine_configuration: Mapping[str, str]
) -> tuple[bool, str]:
    """Whether the planned server takes an image, and why.

    Both must hold: the checkpoint has a vision tower (a top-level
    ``vision_config``, as vLLM's multimodal wrappers do), and the plan leaves
    the image limit above zero.  vLLM sets every modality's limit to 0 under
    ``--language-model-only`` and otherwise defaults an unset modality to 999
    (see ``VLLM_DEFAULT_MM_LIMIT``); ``--limit-mm-per-prompt`` takes a JSON
    object, ``{"image": 0}`` or ``{"image": {"count": 0}}``.
    """
    if not model_config.get("vision_config"):
        return False, "the checkpoint has no vision_config: a text-only model"
    if engine_configuration.get("language_model_only", "").strip().lower() == "true":
        return False, "the plan starts vLLM with --language-model-only: image limit 0"
    raw = engine_configuration.get("limit_mm_per_prompt")
    limit = VLLM_DEFAULT_MM_LIMIT
    if raw:
        try:
            limits = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"limit_mm_per_prompt is not JSON: {raw!r}") from exc
        entry = limits.get("image", VLLM_DEFAULT_MM_LIMIT)
        limit = int(entry["count"] if isinstance(entry, dict) else entry)
    if limit <= 0:
        return False, f"the plan sets the image limit to {limit} (--limit-mm-per-prompt {raw})"
    source = "--limit-mm-per-prompt" if raw else "vLLM's default"
    return True, f"vision_config present; image limit {limit} ({source})"


# ---------------------------------------------------------------------------
# Tool calling: the plan-side rule
# ---------------------------------------------------------------------------

#: Parsers vLLM documents for a model type whose name is not a parser name.
#: Each entry is checked against the plan's engine version's registry before
#: use.  ``gpt_oss`` -> ``openai``: docs/features/tool_calling.md, "OpenAI OSS
#: Models (`openai`)", openai/gpt-oss-20b and -120b (v0.30.0, ced6857).
DOCUMENTED_TOOL_PARSERS: dict[str, str] = {"gpt_oss": "openai"}

#: A chat template that renders past tool calls as
#: ``<tool_call>\n<function=NAME>\n<parameter=...>``: the format whose model
#: cards name ``qwen3_coder`` (Qwen/Qwen3.6-35B-A3B-FP8 and
#: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16, both with exactly this
#: template block).  An inference from those two cards, applied only to a
#: template with the same block and no card of its own that names a parser.
_QWEN3_CODER_BLOCK = re.compile(r"<tool_call>(?:\\n|\n)<function=.*<parameter=", re.DOTALL)

_CARD_PARSER = re.compile(r"--tool-call-parser[ =]+([A-Za-z0-9_.-]+)")


def tool_call_parser_for(
    model_type: str,
    tool_parsers: frozenset[str],
    chat_template: str | None,
    model_card: str | None = None,
) -> tuple[str | None, str]:
    """The tool-call parser a plan names for this model, or None; and why.

    The template must render the request's ``tools`` (otherwise the model is
    never told about them).  Then, in order, the first that the plan's engine
    version registers (``EngineFacts.tool_parsers``):

    1. the parser the model card names (``--tool-call-parser X``), when it
       names exactly one;
    2. the model type itself (``gemma4``, ``muse_glimmer``, ``minimax_m3``);
    3. vLLM's documented parser for the type (``DOCUMENTED_TOOL_PARSERS``);
    4. ``qwen3_coder`` for a template with that format's block.

    Unlike the reasoning parser, a type match is not required: tool-parser
    names are formats (``hermes``, ``openai``, ``qwen3_coder``), not model
    types.
    """
    if not chat_template or "tools" not in chat_template:
        return None, "the chat template does not render tools"
    if model_card:
        named = sorted(set(_CARD_PARSER.findall(model_card)))
        if len(named) == 1 and named[0] in tool_parsers:
            return named[0], f"the model card names --tool-call-parser {named[0]}"
        if len(named) > 1:
            return None, f"the model card names several tool parsers: {named}"
    if model_type in tool_parsers:
        return model_type, f"the engine registers a tool parser under model type {model_type}"
    documented = DOCUMENTED_TOOL_PARSERS.get(model_type)
    if documented and documented in tool_parsers:
        return documented, f"vLLM documents --tool-call-parser {documented} for {model_type}"
    if "qwen3_coder" in tool_parsers and _QWEN3_CODER_BLOCK.search(chat_template):
        return "qwen3_coder", "the template renders the qwen3_coder tool-call block"
    return None, f"no tool parser is known for model type {model_type}"


def with_tool_calling(plan: DeploymentPlan, parser: str) -> DeploymentPlan:
    """*plan* started with ``--enable-auto-tool-choice --tool-call-parser parser``.

    ``engine_flag_args`` renders ``"true"`` as the bare flag.  The flags
    change no memory the calculator predicts; they do change the plan's
    identity, so only suite-v3 plans carry them.
    """
    return plan.model_copy(
        update={
            "engine_configuration": {
                **plan.engine_configuration,
                "enable_auto_tool_choice": "true",
                "tool_call_parser": parser,
            }
        }
    )
