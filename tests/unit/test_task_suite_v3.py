"""Task suite v3: deployment checks, each scored from a faked vLLM server.

Owner, PLAN §18.1 item 5: "suite v3 = deployment checks: long-context
needle, tool call, JSON, image input, reasoning split".  They test the serving
configuration a plan starts, not model quality; every check records pass,
fail or skipped with its reason.  The HTTP server here is ``httpx.MockTransport``:
the scorer's requests are real ``httpx`` requests, answered by a handler.
"""

from __future__ import annotations

import base64
import json
import struct
import zlib
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from apron.adapters.backends.vllm_quantization import engine_facts
from apron.adapters.evaluations.deployment_checks import (
    DeploymentCheckScorer,
    needle_answer_budget,
    needle_prompt_budget,
    schema_errors,
    solid_png,
)
from apron.adapters.evaluations.deployment_facts import deployment_facts, vision_enabled
from apron.application.orchestration.plan_pipeline import (
    tool_call_parser_for,
    with_tool_calling,
)
from apron.domain.schemas.solutions import DeploymentPlan
from apron.domain.schemas.tasks import TaskSuiteSpec
from apron.interfaces.cohort_root import TASK_SUITE_V2, TASK_SUITE_V3, load_inputs

if TYPE_CHECKING:
    from collections.abc import Callable

SUITE = TaskSuiteSpec.model_validate_json(TASK_SUITE_V3.read_text())
CASES = {c["id"]: dict(c) for c in SUITE.cases}
REASONING_PROMPT = CASES["reasoning-1"]["prompt"]
CHECKS = ["whitespace_normalized_exact_match", "strip_terminal_punctuation"]

ALL_ON = {
    "max_model_len": "640",
    "reasoning_parser": "openai_gptoss",
    "thinking_switch": "",
    "tool_call_parser": "openai",
    "tool_reason": "the plan starts vLLM with --enable-auto-tool-choice --tool-call-parser openai",
    "vision": "true",
    "vision_reason": "vision_config present; image limit 999 (vLLM's default)",
}


@dataclass
class Planned:
    """The SolutionPlan fields the scorer reads."""

    model_id: str
    plan: DeploymentPlan
    model_config: dict[str, Any] = field(default_factory=dict)
    chat_template: str | None = None
    chat_renderer: str | None = None


class FakeVllm:
    """/tokenize counts one token per 4 characters plus a 20-token template;
    /v1/chat/completions answers with the message the test sets."""

    def __init__(self, max_model_len: int = 640) -> None:
        self.max_model_len = max_model_len
        self.message: dict[str, Any] = {"role": "assistant", "content": ""}
        self.finish_reason = "stop"
        self.chat_status = 200
        self.chat_text = ""
        self.chats: list[dict[str, Any]] = []
        self.tokenized: list[int] = []

    def count(self, body: dict[str, Any]) -> int:
        text = "".join(m["content"] for m in body["messages"] if isinstance(m["content"], str))
        return 20 + len(text) // 4

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            n = self.count(body)
            self.tokenized.append(n)
            return httpx.Response(
                200, json={"count": n, "max_model_len": self.max_model_len, "tokens": []}
            )
        assert request.url.path == "/v1/chat/completions"
        self.chats.append(body)
        if self.chat_status != 200:
            return httpx.Response(self.chat_status, text=self.chat_text)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": self.message, "finish_reason": self.finish_reason}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 7},
            },
        )


def run(
    server: FakeVllm,
    case_ids: list[str],
    facts: dict[str, str] | None = ALL_ON,  # read only
    **protocol: Any,
) -> list[dict[str, Any]]:
    scorer = DeploymentCheckScorer(client=httpx.Client(transport=httpx.MockTransport(server)))
    data: dict[str, Any] = {
        "scorer_type": "deterministic_exact_match",
        "cases": [CASES[i] for i in case_ids],
        "model_id": "m",
        "deterministic_checks": CHECKS,
        **protocol,
    }
    if facts is not None:
        data["deployment"] = facts
    return scorer.execute(scorer.prepare(data), "http://pod")


def observed(result: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(result["output"])
    return out


# ---------------------------------------------------------------------------
# The suite file
# ---------------------------------------------------------------------------


def test_v3_is_five_deployment_checks_and_leaves_v2_alone() -> None:
    assert load_inputs(task_suite=TASK_SUITE_V3).task_suite == SUITE
    assert [c["check"] for c in SUITE.cases] == [
        "long_context_needle",
        "tool_call",
        "json_output",
        "image_input",
        "reasoning_split",
    ]
    v2 = TaskSuiteSpec.model_validate_json(TASK_SUITE_V2.read_text())
    assert v2.version == "2" and all("check" not in c for c in v2.cases)
    DeploymentCheckScorer().prepare(
        {"cases": list(SUITE.cases), "model_id": "m", "deterministic_checks": CHECKS}
    )


# ---------------------------------------------------------------------------
# 1. Long-context needle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_model_len", [640, 32768])
def test_the_needle_fills_the_plans_context_and_passes_on_the_key(max_model_len: int) -> None:
    server = FakeVllm(max_model_len)
    server.message = {"role": "assistant", "content": "71432."}
    facts = {**ALL_ON, "max_model_len": str(max_model_len)}
    [result] = run(server, ["needle-1"], facts)
    assert result["verdict"] == "pass" and result["accepted"] is True
    sent = server.chats[0]
    target = needle_prompt_budget(max_model_len, 512)
    prompt_tokens = observed(result)["prompt_tokens"]
    assert prompt_tokens <= target
    assert prompt_tokens >= target - 30  # filled, not a token count off by a unit
    assert server.count(sent) == prompt_tokens
    assert sent["max_tokens"] == needle_answer_budget(max_model_len, 512)
    assert sent["max_tokens"] + prompt_tokens <= max_model_len
    text = sent["messages"][0]["content"]
    middle = text.index("The pass key is 71432") / len(text)
    assert 0.4 < middle < 0.6  # depth 0.5
    assert text.endswith("What is the pass key? Answer with just the number.")


def test_needle_answer_budget_is_a_quarter_of_a_short_context() -> None:
    assert needle_answer_budget(640, 512) == 160
    assert needle_answer_budget(32768, 512) == 512
    assert needle_prompt_budget(32768, 512) == 32768 - 512 - 16


def test_the_needle_fails_when_the_server_does_not_serve_the_planned_context() -> None:
    server = FakeVllm(max_model_len=4096)
    [result] = run(server, ["needle-1"], {**ALL_ON, "max_model_len": "640"})
    assert result["verdict"] == "fail"
    assert "serves max_model_len 4096, the plan says 640" in result["reason"]
    assert server.chats == []


def test_a_needle_answered_in_the_reasoning_channel_fails_and_says_so() -> None:
    server = FakeVllm()
    server.message = {"role": "assistant", "content": None, "reasoning": "The key is 71432."}
    [result] = run(server, ["needle-1"])
    assert result["verdict"] == "fail"
    assert "the answer went to the reasoning channel" in result["reason"]


def test_the_needle_reads_the_context_from_the_server_without_plan_facts() -> None:
    server = FakeVllm(max_model_len=2048)
    server.message = {"role": "assistant", "content": "71432"}
    [result] = run(server, ["needle-1"], facts=None)
    assert result["verdict"] == "pass"
    assert observed(result)["max_model_len_source"] == "server"


# ---------------------------------------------------------------------------
# 2. Tool call
# ---------------------------------------------------------------------------

GOOD_CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
}


def test_a_well_formed_tool_call_passes_and_the_request_is_auto() -> None:
    server = FakeVllm()
    server.message = {"role": "assistant", "content": None, "tool_calls": [GOOD_CALL]}
    server.finish_reason = "tool_calls"
    [result] = run(server, ["tool-1"])
    assert result["verdict"] == "pass", result["reason"]
    sent = server.chats[0]
    assert sent["tool_choice"] == "auto"
    assert sent["tools"][0]["function"]["name"] == "get_weather"
    assert observed(result)["tool_calls"] == [GOOD_CALL]


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        (
            {"content": '<tool_call>{"name": "get_weather"}</tool_call>', "tool_calls": []},
            "no tool call",
        ),
        (
            {"tool_calls": [{**GOOD_CALL, "function": {"name": "search", "arguments": "{}"}}]},
            "none to 'get_weather'",
        ),
        (
            {
                "tool_calls": [
                    {**GOOD_CALL, "function": {"name": "get_weather", "arguments": "{city: Paris"}}
                ]
            },
            "arguments are not JSON",
        ),
        (
            {
                "tool_calls": [
                    {**GOOD_CALL, "function": {"name": "get_weather", "arguments": '"Paris"'}}
                ]
            },
            "not a JSON object",
        ),
        (
            {
                "tool_calls": [
                    {
                        **GOOD_CALL,
                        "function": {"name": "get_weather", "arguments": '{"town": "Paris"}'},
                    }
                ]
            },
            "missing 'city'",
        ),
    ],
)
def test_a_malformed_tool_call_fails_with_the_reason(message: dict[str, Any], reason: str) -> None:
    server = FakeVllm()
    server.message = {"role": "assistant", "content": None, **message}
    [result] = run(server, ["tool-1"])
    assert result["verdict"] == "fail" and result["accepted"] is False
    assert reason in result["reason"]


def test_the_tool_call_is_skipped_unsent_when_the_plan_has_no_tool_parser() -> None:
    server = FakeVllm()
    plan = DeploymentPlan(engine_configuration={"max_model_len": "640"})
    facts = deployment_facts(plan, {}, "{{ tools }}")
    [result] = run(server, ["tool-1"], facts)
    assert result["status"] == "skipped" and result["accepted"] is None and result["score"] is None
    assert "without --enable-auto-tool-choice --tool-call-parser" in result["reason"]
    assert observed(result)["verdict"] == "skipped"  # the reason reaches the record's output
    assert server.chats == []


def test_a_refused_tool_request_is_a_verdict_not_a_transport_failure() -> None:
    server = FakeVllm()
    server.chat_status = 400
    server.chat_text = '{"message": "\\"auto\\" tool choice requires --enable-auto-tool-choice"}'
    [result] = run(server, ["tool-1"])
    assert result["status"] == "completed" and result["verdict"] == "fail"
    assert "400" in result["reason"] and "enable-auto-tool-choice" in result["reason"]


def test_a_server_error_is_a_retryable_failure() -> None:
    server = FakeVllm()
    server.chat_status = 503
    [result] = run(server, ["tool-1"])
    assert result["status"] == "failed"  # _run_task_suite retries these once
    assert "503" in result["error"]


# ---------------------------------------------------------------------------
# 3. JSON output
# ---------------------------------------------------------------------------


def test_structured_output_passes_when_the_content_validates() -> None:
    server = FakeVllm()
    server.message = {"role": "assistant", "content": '{"city": "Paris", "country": "France"}'}
    [result] = run(server, ["json-1"], facts=None)  # needs nothing from the plan
    assert result["verdict"] == "pass"
    fmt = server.chats[0]["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"]["required"] == ["city", "country"]


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ("The capital is Paris.", "not JSON"),
        ('{"city": "Paris"}', "missing 'country'"),
        ('{"city": "Paris", "country": "France", "pop": 2}', "unexpected 'pop'"),
        ('{"city": 1, "country": "France"}', "expected string"),
    ],
)
def test_structured_output_fails_when_it_does_not(content: str, reason: str) -> None:
    server = FakeVllm()
    server.message = {"role": "assistant", "content": content}
    [result] = run(server, ["json-1"])
    assert result["verdict"] == "fail" and reason in result["reason"]


# ---------------------------------------------------------------------------
# 4. Image input
# ---------------------------------------------------------------------------


def _png_pixels(data: bytes) -> tuple[int, int, bytes]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    idat_len = struct.unpack(">I", data[33:37])[0]
    raw = zlib.decompress(data[41 : 41 + idat_len])
    return width, height, raw


def test_a_vision_plan_is_sent_a_generated_red_square_and_passes_on_red() -> None:
    server = FakeVllm()
    server.message = {"role": "assistant", "content": "Red."}
    [result] = run(server, ["image-1"])
    assert result["verdict"] == "pass", result["reason"]
    parts = server.chats[0]["messages"][0]["content"]
    url = parts[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    width, height, raw = _png_pixels(base64.b64decode(url.split(",", 1)[1]))
    assert (width, height) == (224, 224)
    assert raw[:4] == b"\x00\xff\x00\x00" and len(raw) == 224 * (1 + 224 * 3)
    assert parts[1] == {"type": "text", "text": CASES["image-1"]["prompt"]}


def test_the_png_is_byte_for_byte_reproducible() -> None:
    assert solid_png(2, 2, (1, 2, 3)) == solid_png(2, 2, (1, 2, 3))


def test_a_text_only_plan_skips_the_image_with_the_reason() -> None:
    server = FakeVllm()
    facts = deployment_facts(DeploymentPlan(), {"architectures": ["GptOssForCausalLM"]}, None)
    [result] = run(server, ["image-1"], facts)
    assert result["status"] == "skipped"
    assert result["reason"] == "the checkpoint has no vision_config: a text-only model"
    assert server.chats == []


def test_a_wrong_colour_fails() -> None:
    server = FakeVllm()
    server.message = {"role": "assistant", "content": "Blue"}
    [result] = run(server, ["image-1"])
    assert result["verdict"] == "fail" and "'Blue' is not 'red'" in result["reason"]


# ---------------------------------------------------------------------------
# 5. Reasoning split
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_reasoning_split_passes_when_both_channels_hold_their_part(field: str) -> None:
    server = FakeVllm()
    server.message = {"role": "assistant", "content": "42", field: "17 + 25 = 42."}
    [result] = run(server, ["reasoning-1"], request_fields={"reasoning_effort": "low"})
    assert result["verdict"] == "pass", result["reason"]
    # The suite keeps effort low; the case that asks for reasoning sends none
    # (the model's default), as it turns a thinking switch on: GLM-5.3-Flash
    # skipped thinking at low and at high and reasoned only at its default.
    assert "reasoning_effort" not in server.chats[0]


def test_reasoning_left_in_the_content_fails_and_names_the_missing_parser() -> None:
    server = FakeVllm()
    server.message = {"role": "assistant", "content": "<think>17 + 25 = 42</think>\n\n42"}
    facts = {**ALL_ON, "reasoning_parser": "", "thinking_switch": "true"}
    [result] = run(server, ["reasoning-1"], facts, chat_template_kwargs={"enable_thinking": False})
    assert result["verdict"] == "fail"
    assert result["reason"].startswith("message.reasoning is empty")
    assert "the plan names no reasoning parser" in result["reason"]
    # The other cases switch thinking off; this one turns it on.
    assert server.chats[0]["chat_template_kwargs"] == {"enable_thinking": True}


def test_a_right_answer_without_separated_reasoning_still_fails() -> None:
    server = FakeVllm()
    server.message = {"role": "assistant", "content": "42", "reasoning": None}
    [result] = run(server, ["reasoning-1"])
    assert result["verdict"] == "fail" and "message.reasoning is empty" in result["reason"]


def test_a_model_that_does_not_reason_skips_the_split() -> None:
    server = FakeVllm()
    [result] = run(server, ["reasoning-1"], {**ALL_ON, "reasoning_parser": ""})
    assert result["status"] == "skipped" and "not a reasoning model" in result["reason"]
    assert server.chats == []


# ---------------------------------------------------------------------------
# Plan facts reach the scorer
# ---------------------------------------------------------------------------


def test_without_plan_facts_the_plan_dependent_checks_skip_and_say_why() -> None:
    server = FakeVllm()
    results = run(server, ["tool-1", "image-1", "reasoning-1"], facts=None)
    assert {r["status"] for r in results} == {"skipped"}
    assert {r["reason"] for r in results} == {
        "the plan's deployment facts did not reach the scorer"
    }


def test_for_plans_gives_each_model_its_own_facts() -> None:
    def plan(model_id: str) -> DeploymentPlan:
        return DeploymentPlan(
            engine_configuration={"max_model_len": "640"},
            resource_allocation={"model_id": model_id},
        )

    tools, text = with_tool_calling(plan("a"), "openai"), plan("b")
    plans = [
        Planned("a", tools),
        Planned("b", text),
    ]
    scorer = DeploymentCheckScorer.for_plans(plans)
    a = scorer.prepare({"cases": [CASES["tool-1"]], "model_id": "a"})
    b = scorer.prepare({"cases": [CASES["tool-1"]], "model_id": "b"})
    assert a["deployment"]["tool_call_parser"] == "openai"
    assert b["deployment"]["tool_call_parser"] == ""
    clash = Planned("a", plan("a"))
    with pytest.raises(ValueError, match="two plans with different deployment facts"):
        DeploymentCheckScorer.for_plans([*plans, clash])


def test_the_tool_flag_renders_as_vllms_flags() -> None:
    from apron.adapters.renderers.engine_flags import engine_flag_args

    plan = with_tool_calling(DeploymentPlan(), "qwen3_coder")
    assert engine_flag_args(plan.engine_configuration) == [
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
    ]


# ---------------------------------------------------------------------------
# Facts: vision and tool-parser rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("engine", "on", "reason"),
    [
        ({}, True, "image limit 999 (vLLM's default)"),
        ({"limit_mm_per_prompt": '{"image": 2}'}, True, "image limit 2"),
        ({"limit_mm_per_prompt": '{"image": 0}'}, False, "image limit to 0"),
        ({"limit_mm_per_prompt": '{"image": {"count": 0}}'}, False, "image limit to 0"),
        ({"language_model_only": "true"}, False, "--language-model-only"),
    ],
)
def test_vision_needs_a_tower_and_a_nonzero_image_limit(
    engine: dict[str, str], on: bool, reason: str
) -> None:
    enabled, why = vision_enabled({"vision_config": {"hidden_size": 8}}, engine)
    assert enabled is on and reason in why


FACTS = engine_facts("v0.30.0")
QWEN_XML = "{% for tool in tools %}{% endfor %}'<tool_call>\\n<function=' ~ name ~ '<parameter='"


@pytest.mark.parametrize(
    ("model_type", "template", "card", "parser", "why"),
    [
        (
            "glm4_moe_lite",
            "{{ tools }}",
            "vllm serve --tool-call-parser glm47",
            "glm47",
            "model card",
        ),
        ("gemma4", "{{ tools }}", None, "gemma4", "under model type gemma4"),
        ("gpt_oss", "{{ tools }}", None, "openai", "vLLM documents"),
        ("qwen3_5", QWEN_XML, None, "qwen3_coder", "qwen3_coder tool-call block"),
        ("llama", "{{ tools }}", None, None, "no tool parser is known"),
        ("gemma4", "{{ messages }}", None, None, "does not render tools"),
        (
            "x",
            "{{ tools }}",
            "--tool-call-parser hermes / --tool-call-parser openai",
            None,
            "several",
        ),
    ],
)
def test_the_tool_parser_rule(
    model_type: str, template: str, card: str | None, parser: str | None, why: str
) -> None:
    got, reason = tool_call_parser_for(model_type, FACTS.tool_parsers, template, card)
    assert got == parser and why in reason


def test_deployment_facts_read_the_plan() -> None:
    plan = DeploymentPlan(
        engine_configuration={
            "max_model_len": "32768",
            "reasoning_parser": "openai_gptoss",
            "tool_call_parser": "openai",  # without --enable-auto-tool-choice: not served
        }
    )
    facts = deployment_facts(plan, {}, "{% if enable_thinking %}{% endif %}")
    assert facts["max_model_len"] == "32768"
    assert facts["reasoning_parser"] == "openai_gptoss"
    assert facts["thinking_switch"] == "true"
    assert facts["tool_call_parser"] == "" and "refuses tool_choice=auto" in facts["tool_reason"]


# ---------------------------------------------------------------------------
# Nothing a suite author wrote is ignored
# ---------------------------------------------------------------------------


def _prepare(case: dict[str, str], **extra: Any) -> Callable[[], Any]:
    return lambda: DeploymentCheckScorer().prepare({"cases": [case], "model_id": "m", **extra})


@pytest.mark.parametrize(
    ("case", "extra", "message"),
    [
        ({**CASES["json-1"], "temperature": "1"}, {}, "mean nothing to json_output"),
        ({**CASES["json-1"], "check": "vibes"}, {}, "unknown check"),
        ({**CASES["image-1"], "extra_checks": "stem"}, {}, "unknown extra_checks"),
        (
            {**CASES["json-1"], "schema": '{"type": "object", "minProperties": 1}'},
            {},
            "minProperties",
        ),
        ({**CASES["tool-1"], "expected_tool": "search"}, {}, "not in"),
        ({**CASES["image-1"], "image": "photo.jpg"}, {}, "solid:WxH:R,G,B"),
        (CASES["tool-1"], {"deployment": {"tool_parser": "openai"}}, "unknown deployment facts"),
    ],
)
def test_prepare_refuses_what_it_would_otherwise_drop(
    case: dict[str, str], extra: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _prepare(case, **extra)()


def test_skipped_checks_are_counted_apart_from_the_pass_rate() -> None:
    collected = DeploymentCheckScorer().collect(
        [
            {"case_id": "a", "score": 1, "status": "completed"},
            {"case_id": "b", "score": 0, "status": "completed"},
            {"case_id": "c", "score": None, "status": "skipped"},
        ]
    )
    assert collected["pass_rate"] == 0.5 and collected["skipped"] == 1
    assert len(collected["attempts"]) == 3


def test_a_recorded_check_is_not_rescored_as_text() -> None:
    with pytest.raises(ValueError, match="not rescored"):
        DeploymentCheckScorer().rescore(
            {"cases": [CASES["json-1"]]}, [{"case_id": "json-1", "output": "{}"}]
        )


def test_schema_errors_cover_arrays_and_enums() -> None:
    schema = {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}}
    assert schema_errors(["a", "b"], schema) == []
    assert schema_errors(["c"], schema) == ["$[0]: 'c' not in ['a', 'b']"]
    assert schema_errors(True, {"type": "integer"}) == ["$: expected integer, got bool"]


# ---------------------------------------------------------------------------
# Through the cohort's own task-suite step, unchanged
# ---------------------------------------------------------------------------


def test_the_cohort_step_runs_v3_with_a_for_plans_scorer_and_records_every_verdict() -> None:
    from pathlib import Path

    from apron.application.orchestration.cohort import _run_task_suite
    from apron.application.orchestration.evidence import EvidenceContext, build_task_attempt
    from apron.domain.protocols import EvaluationAdapter
    from apron.domain.schemas.authority import DecisionRequest
    from apron.domain.schemas.tasks import ApplicationSpec
    from apron.interfaces.cohort_root import PROTOCOL

    run_dir = Path(__file__).resolve().parents[1] / "fixtures" / "phase-1a-run"
    ctx = EvidenceContext.bind(
        request=DecisionRequest.model_validate_json(
            (run_dir / "decision-request.json").read_text()
        ),
        task_suite=SUITE,
        application=ApplicationSpec.model_validate_json(
            (run_dir / "application-spec.json").read_text()
        ),
        protocol_template=json.loads(PROTOCOL.read_text()),
        solution_fp="1220" + "ab" * 32,
    )
    gpt_oss_template = (
        "{{ tools }} {% set reasoning_effort = reasoning_effort | default('medium') %}"
    )
    plan = with_tool_calling(
        DeploymentPlan(
            engine_configuration={"max_model_len": "640", "reasoning_parser": "openai_gptoss"},
            resource_allocation={"model_id": "openai/gpt-oss-20b"},
        ),
        "openai",
    )
    sp = Planned("openai/gpt-oss-20b", plan, {"model_type": "gpt_oss"}, gpt_oss_template)
    server = FakeVllm()
    server.message = {"role": "assistant", "content": "42", "reasoning": "17 + 25 = 42."}
    scorer = DeploymentCheckScorer.for_plans(
        [sp], client=httpx.Client(transport=httpx.MockTransport(server))
    )
    assert isinstance(scorer, EvaluationAdapter)
    ports = SimpleNamespace(evaluator=scorer)
    scored = _run_task_suite(ctx, sp, ports, "openai/gpt-oss-20b", "http://pod")  # type: ignore[arg-type]

    verdicts = {a["case_id"]: a["verdict"] for a, _ in scored}
    assert verdicts == {
        "needle-1": "fail",  # the fake answers "42", not the key
        "tool-1": "fail",  # ... and makes no tool call
        "json-1": "fail",  # ... and "42" is not the object
        "image-1": "skipped",  # gpt-oss has no vision_config
        "reasoning-1": "pass",
    }
    # Every case keeps effort low but the one that asks for reasoning.
    efforts = [
        (body["messages"][-1]["content"] == REASONING_PROMPT, body.get("reasoning_effort"))
        for body in server.chats
    ]
    assert all(effort == (None if asks else "low") for asks, effort in efforts)
    assert sum(asks for asks, _ in efforts) == 1
    for attempt, retry in scored:
        record = build_task_attempt(
            ctx, attempt, attempt_id=f"a-{attempt['case_id']}", retry=retry
        )
        output = json.loads(record.output or "")
        assert output["verdict"] == attempt["verdict"] and output["reason"] == attempt["reason"]
        # A skipped check is neither accepted nor rejected.
        skipped = attempt["verdict"] == "skipped"
        assert record.accepted is (None if skipped else attempt["verdict"] == "pass")
