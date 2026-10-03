"""What a plan tells the deployment checks (task suite v3), and the tool-call rule.

Suite v3 checks the serving configuration, not the model: whether the server
the plan starts serves a long context, tool calls, structured output, images
and a separate reasoning channel.  Which of those a plan can serve is a fact
of the plan (its engine flags) and of the checkpoint (its config and chat
template); ``deployment_facts`` reads them, GPU-free, into the string map the
scorer receives (``DeploymentCheckScorer``).  A check the plan cannot serve is
skipped with the reason these facts give, never sent and scored as wrong.

The plan-side rule for tool calling is the planner's
(``plan_pipeline.tool_call_parser_for``): a plan that names a tool parser is a
new solution identity, so only plans that run suite v3 carry the flag (v2's
plans, and their records, stay as they are).
"""

from __future__ import annotations

import json
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
        tool_reason = (
            f"the plan starts vLLM with --enable-auto-tool-choice --tool-call-parser {parser}"
        )
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
