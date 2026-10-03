"""Deployment checks (task suite v3): does the planned server serve what clients send?

Five checks an inference engineer runs after a deployment, each a few
requests, each scored without an LLM judge:

``long_context_needle``
    A pass key inside filler sized from the server's context length
    (``max_model_len``): the answer must be the key.  The filler is the
    passkey-retrieval text (Mohtashami & Jaggi 2023, "Landmark Attention");
    its size comes from vLLM's ``/tokenize`` with the same chat messages, so
    the prompt fills the context whatever the tokenizer.
``tool_call``
    An OpenAI ``tools`` request with ``tool_choice="auto"``: the response must
    hold a tool call with the expected name whose arguments are a JSON object
    valid against that tool's parameters.  vLLM serves ``auto`` only when
    started with ``--enable-auto-tool-choice --tool-call-parser``.
``json_output``
    ``response_format`` ``json_schema`` (vLLM structured output): the content
    must parse and validate against the schema.
``image_input``
    A generated solid-colour PNG and an exact-answer question; only for a
    plan whose server takes images.
``reasoning_split``
    Reasoning left on: ``message.reasoning`` (``reasoning_content`` before
    v0.30) must be non-empty *and* the content must be the exact answer.  It
    checks the reasoning parser, not the answer alone.

Each case records its own verdict: ``pass``, ``fail`` (the server answered,
or refused with a 4xx, and the check did not hold) or ``skipped`` (the plan
cannot serve it; the reason says why), and the observation behind it, as the
attempt's ``output`` (a JSON object).  A request that failed in transport
(connection, timeout, 5xx) is ``status: failed`` and is retried by the cohort
like any other case.

What each check needs from the plan arrives as the ``deployment`` facts
(``deployment_facts.deployment_facts``).  Without them the checks that depend
on the plan (tool call, image, reasoning) are skipped with that reason; the
needle reads the context length from the server and the JSON check needs
nothing.
"""

from __future__ import annotations

import base64
import json
import logging
import struct
import time
import zlib
from typing import TYPE_CHECKING, Any, Protocol

import httpx

from apron.adapters.evaluations.deployment_facts import FACT_KEYS, deployment_facts
from apron.adapters.evaluations.deterministic_scorer import (
    DEFAULT_CHECKS,
    NORMALIZATIONS,
    DeterministicScorer,
    normalize,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from apron.domain.schemas.solutions import DeploymentPlan

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The suite's case vocabulary
# ---------------------------------------------------------------------------

_COMMON = frozenset({"id", "check", "max_tokens", "extra_checks"})

#: Required fields per check; a case with a field outside required | optional
#: is refused (nothing a suite author wrote is silently ignored).
CHECK_FIELDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "long_context_needle": (
        frozenset({"needle", "question", "expected", "filler", "depth"}),
        frozenset(),
    ),
    "tool_call": (frozenset({"prompt", "tools", "expected_tool"}), frozenset()),
    "json_output": (frozenset({"prompt", "schema", "schema_name"}), frozenset()),
    "image_input": (frozenset({"prompt", "image", "expected"}), frozenset()),
    "reasoning_split": (frozenset({"prompt", "expected"}), frozenset()),
}

#: Tokens kept free between the needle prompt and the answer budget: the
#: /tokenize count is taken without the request's sampling fields, and
#: gpt-oss's template renders ``reasoning_effort`` into the system turn.
NEEDLE_MARGIN_TOKENS = 16

_UNIT_PROBE = 64  # filler units in the probe that measures tokens per unit


def needle_answer_budget(max_model_len: int, answer_tokens: int) -> int:
    """Tokens reserved for the needle's answer: the case's budget, at most a
    quarter of the context, so a short context still holds a real prompt."""
    return max(1, min(answer_tokens, max_model_len // 4))


def needle_prompt_budget(max_model_len: int, answer_tokens: int) -> int:
    """Prompt tokens the needle may fill: the context less the answer and a margin."""
    return (
        max_model_len - needle_answer_budget(max_model_len, answer_tokens) - NEEDLE_MARGIN_TOKENS
    )


# ---------------------------------------------------------------------------
# A generated image (no file, no randomness)
# ---------------------------------------------------------------------------


def solid_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """A width x height PNG of one colour, byte-for-byte reproducible."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    row = b"\x00" + bytes(rgb) * width  # filter type 0 per scanline
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(row * height, 9))
        + chunk(b"IEND", b"")
    )


def parse_image_spec(spec: str) -> tuple[int, int, tuple[int, int, int]]:
    """``solid:WxH:R,G,B`` -> (W, H, (R, G, B)); anything else is refused."""
    kind, _, rest = spec.partition(":")
    size, _, colour = rest.partition(":")
    if kind != "solid" or "x" not in size or colour.count(",") != 2:
        raise ValueError(f"image spec {spec!r}: expected 'solid:WxH:R,G,B'")
    width, height = (int(v) for v in size.split("x"))
    red, green, blue = (int(v) for v in colour.split(","))
    if not (0 < width <= 4096 and 0 < height <= 4096):
        raise ValueError(f"image spec {spec!r}: size out of range")
    if any(not 0 <= v <= 255 for v in (red, green, blue)):
        raise ValueError(f"image spec {spec!r}: colour out of range")
    return width, height, (red, green, blue)


def image_data_url(spec: str) -> str:
    width, height, rgb = parse_image_spec(spec)
    return "data:image/png;base64," + base64.b64encode(solid_png(width, height, rgb)).decode()


# ---------------------------------------------------------------------------
# JSON Schema: the subset the suite uses, validated without a dependency
# ---------------------------------------------------------------------------

SCHEMA_KEYWORDS = frozenset(
    {"type", "properties", "required", "additionalProperties", "enum", "items", "description"}
)
_TYPES: dict[str, Callable[[Any], bool]] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def check_schema_supported(schema: Any, path: str = "$") -> None:
    """Refuse a schema keyword the validator does not implement (never skip one)."""
    if not isinstance(schema, dict):
        raise ValueError(f"{path}: a schema must be an object")
    unknown = sorted(set(schema) - SCHEMA_KEYWORDS)
    if unknown:
        raise ValueError(f"{path}: unsupported JSON Schema keywords {unknown}")
    if "type" in schema and schema["type"] not in _TYPES:
        raise ValueError(f"{path}: unsupported type {schema['type']!r}")
    if not isinstance(schema.get("additionalProperties", False), bool):
        raise ValueError(f"{path}: additionalProperties must be true or false")
    for name, sub in (schema.get("properties") or {}).items():
        check_schema_supported(sub, f"{path}.{name}")
    if "items" in schema:
        check_schema_supported(schema["items"], f"{path}[]")


def schema_errors(value: Any, schema: Mapping[str, Any], path: str = "$") -> list[str]:
    """Why *value* does not satisfy *schema* (empty when it does)."""
    kind = schema.get("type")
    if kind is not None and not _TYPES[kind](value):
        return [f"{path}: expected {kind}, got {type(value).__name__}"]
    errors: list[str] = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in {schema['enum']}")
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        errors += [f"{path}: missing {k!r}" for k in schema.get("required", ()) if k not in value]
        if schema.get("additionalProperties") is False:
            errors += [f"{path}: unexpected {k!r}" for k in value if k not in properties]
        for key, sub in properties.items():
            if key in value:
                errors += schema_errors(value[key], sub, f"{path}.{key}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors += schema_errors(item, schema["items"], f"{path}[{i}]")
    return errors


def _parse_tools(raw: str) -> list[dict[str, Any]]:
    tools = json.loads(raw)
    if not isinstance(tools, list) or not tools:
        raise ValueError("tools: a non-empty JSON array of OpenAI tool definitions")
    for tool in tools:
        if tool.get("type") != "function" or "name" not in tool.get("function", {}):
            raise ValueError(f"tools: not a function tool: {tool}")
        check_schema_supported(tool["function"].get("parameters", {}), tool["function"]["name"])
    return tools


# ---------------------------------------------------------------------------
# The scorer
# ---------------------------------------------------------------------------


class _Refused(Exception):
    """The server answered 4xx: a verdict about the deployment, not transport."""

    def __init__(self, status: int, text: str) -> None:
        super().__init__(f"{status}: {text[:500]}")


class _PlanLike(Protocol):
    """The ``SolutionPlan`` fields the facts come from (read only)."""

    @property
    def model_id(self) -> str: ...
    @property
    def plan(self) -> DeploymentPlan: ...
    @property
    def model_config(self) -> Mapping[str, Any]: ...
    @property
    def chat_template(self) -> str | None: ...


class DeploymentCheckScorer(DeterministicScorer):
    """Scores suite-v3 deployment checks; a case without ``check`` is scored
    exactly as ``DeterministicScorer`` scores it.

    ``facts_for(model_id)`` supplies the plan's facts when the protocol input
    carries none (the cohort's evaluator port is shared by every plan of a run).
    """

    _harness_name = "deployment_checks"
    _harness_version = "0.1"

    def __init__(
        self,
        timeout: int = 300,
        *,
        facts_for: Callable[[str], Mapping[str, str] | None] | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(timeout=timeout)
        self._facts_for = facts_for
        self._client = client

    @classmethod
    def for_plans(cls, plans: Iterable[_PlanLike], **kwargs: Any) -> DeploymentCheckScorer:
        """A scorer that knows each plan's facts by its served model id.

        Two plans of one model with different facts cannot share a run's
        scorer: that is refused rather than scored under the wrong facts.
        """
        facts: dict[str, dict[str, str]] = {}
        for sp in plans:
            model_id = sp.plan.resource_allocation.get("model_id", sp.model_id)
            these = deployment_facts(sp.plan, sp.model_config, sp.chat_template)
            if facts.setdefault(model_id, these) != these:
                raise ValueError(f"{model_id}: two plans with different deployment facts")
        return cls(facts_for=facts.get, **kwargs)

    # -- EvaluationAdapter ------------------------------------------------

    def prepare(self, protocol: dict[str, Any]) -> dict[str, Any]:
        cases = list(protocol.get("cases", []))
        checks = tuple(protocol.get("deterministic_checks") or DEFAULT_CHECKS)
        for case in cases:
            if "check" in case:
                _validate_case(case)
                extra = _extra_checks(case)
                unknown = [c for c in extra if c not in NORMALIZATIONS]
                if unknown:
                    raise ValueError(f"{case['id']}: unknown extra_checks {unknown}")
        plain = [c for c in cases if "check" not in c]
        prepared = super().prepare({**protocol, "cases": plain})
        prepared["cases"] = cases
        prepared["deterministic_checks"] = list(checks)
        facts = protocol.get("deployment")
        if facts is None and self._facts_for is not None:
            facts = self._facts_for(prepared["model_id"])
        if facts is not None:
            unknown_keys = sorted(set(facts) - FACT_KEYS)
            if unknown_keys:
                raise ValueError(f"unknown deployment facts {unknown_keys}")
            prepared["deployment"] = {k: str(v) for k, v in facts.items()}
        return prepared

    def collect(self, attempts: list[dict[str, Any]]) -> dict[str, Any]:
        """Pass rate over the checks that ran; skipped checks are counted apart."""
        ran = [a for a in attempts if a.get("status") != "skipped"]
        collected = super().collect(ran)
        collected["skipped"] = len(attempts) - len(ran)
        collected["attempts"] = attempts
        return collected

    def rescore(
        self, protocol: dict[str, Any], recorded: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        checked = {c.get("id") for c in protocol.get("cases", []) if "check" in c}
        if any(item.get("case_id") in checked for item in recorded):
            raise ValueError(
                "deployment checks are scored from the live response; a recorded "
                "observation is not rescored"
            )
        return super().rescore(protocol, recorded)

    # -- dispatch ---------------------------------------------------------

    def _execute_case(
        self,
        case: dict[str, Any],
        model_id: str,
        endpoint: str,
        protocol: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        protocol = protocol or {}
        check = case.get("check")
        if check is None:
            return super()._execute_case(case, model_id, endpoint, protocol)
        run = {
            "long_context_needle": self._needle,
            "tool_call": self._tool_call,
            "json_output": self._json_output,
            "image_input": self._image_input,
            "reasoning_split": self._reasoning_split,
        }[check]
        start = time.monotonic()
        try:
            result = run(case, model_id, endpoint, protocol)
        except _Refused as exc:
            result = _verdict(case, "fail", f"the server refused the request: {exc}", {})
        except (httpx.HTTPError, KeyError, ValueError, TypeError, IndexError) as exc:
            # Transport, a 5xx, or a response that is not the OpenAI shape:
            # nothing was learned about the check.  Retried by the cohort.
            error = f"{type(exc).__name__}: {exc}"
            result = {
                "case_id": case["id"],
                "check": check,
                "score": 0,
                "accepted": False,
                "status": "failed",
                "error": error,
                "output": json.dumps({"check": check, "verdict": "error", "reason": error}),
            }
        result.setdefault("time_seconds", round(time.monotonic() - start, 3))
        return result

    # -- the checks ---------------------------------------------------------

    def _needle(
        self, case: dict[str, Any], model_id: str, endpoint: str, protocol: dict[str, Any]
    ) -> dict[str, Any]:
        facts = protocol.get("deployment") or {}
        kwargs = protocol.get("chat_template_kwargs")

        def messages(units: int) -> list[dict[str, Any]]:
            before = round(units * float(case["depth"]))
            text = (
                case["filler"] * before
                + case["needle"]
                + " "
                + case["filler"] * (units - before)
                + "\n\n"
                + case["question"]
            )
            return [{"role": "user", "content": text}]

        def count(units: int) -> tuple[int, int]:
            body: dict[str, Any] = {"model": model_id, "messages": messages(units)}
            if kwargs:
                body["chat_template_kwargs"] = kwargs
            data = self._post(endpoint, "/tokenize", body)
            return int(data["count"]), int(data["max_model_len"])

        base, served_len = count(0)
        planned = facts.get("max_model_len")
        if planned and int(planned) != served_len:
            return _verdict(
                case,
                "fail",
                f"the server serves max_model_len {served_len}, the plan says {planned}",
                {"max_model_len": served_len, "planned_max_model_len": int(planned)},
            )
        target = needle_prompt_budget(served_len, int(case["max_tokens"]))
        if base > target:
            return _verdict(
                case,
                "fail",
                f"max_model_len {served_len} leaves {target} prompt tokens; the needle "
                f"and question alone take {base}",
                {"max_model_len": served_len, "needle_tokens": base},
            )
        probe, _ = count(_UNIT_PROBE)
        per_unit = max(1e-9, (probe - base) / _UNIT_PROBE)
        units = int((target - base) / per_unit)
        tokens, _ = count(units)
        for _ in range(8):  # merges at unit boundaries: shrink until it fits
            if tokens <= target or units == 0:
                break
            units = max(0, units - int((tokens - target) / per_unit) - 1)
            tokens, _ = count(units)
        answer = needle_answer_budget(served_len, int(case["max_tokens"]))
        body = self._body(protocol, model_id, messages(units), answer)
        data, elapsed = self._chat(endpoint, body)
        message, content = _message(data)
        observed = {
            "max_model_len": served_len,
            "max_model_len_source": "plan and server" if planned else "server",
            "prompt_tokens": tokens,
            "answer_budget": answer,
            "filler_units": units,
            "depth": float(case["depth"]),
            "content": content,
            "reasoning": _reasoning(message),
        }
        ok, why = _exact(content, case["expected"], protocol, case)
        if not ok and not content and _reasoning(message):
            why += "; the answer went to the reasoning channel"
        return _verdict(case, "pass" if ok else "fail", why, observed, data, elapsed)

    def _tool_call(
        self, case: dict[str, Any], model_id: str, endpoint: str, protocol: dict[str, Any]
    ) -> dict[str, Any]:
        facts = protocol.get("deployment")
        if facts is None:
            return _no_facts(case)
        if not facts.get("tool_call_parser"):
            return _verdict(case, "skipped", facts.get("tool_reason", "no tool parser"), {})
        tools = _parse_tools(case["tools"])
        body = self._body(
            protocol,
            model_id,
            [{"role": "user", "content": case["prompt"]}],
            int(case["max_tokens"]),
        )
        body["tools"] = tools
        body["tool_choice"] = "auto"
        data, elapsed = self._chat(endpoint, body)
        message, content = _message(data)
        calls = message.get("tool_calls") or []
        observed = {
            "tool_call_parser": facts["tool_call_parser"],
            "tool_calls": calls,
            "content": content,
            "reasoning": _reasoning(message),
        }
        verdict, why = _judge_tool_calls(calls, tools, case["expected_tool"])
        return _verdict(case, verdict, why, observed, data, elapsed)

    def _json_output(
        self, case: dict[str, Any], model_id: str, endpoint: str, protocol: dict[str, Any]
    ) -> dict[str, Any]:
        schema = json.loads(case["schema"])
        check_schema_supported(schema)
        body = self._body(
            protocol,
            model_id,
            [{"role": "user", "content": case["prompt"]}],
            int(case["max_tokens"]),
        )
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": case["schema_name"], "schema": schema, "strict": True},
        }
        data, elapsed = self._chat(endpoint, body)
        message, content = _message(data)
        observed = {"content": content, "reasoning": _reasoning(message)}
        try:
            value = json.loads(content or "")
        except json.JSONDecodeError as exc:
            return _verdict(case, "fail", f"content is not JSON: {exc}", observed, data, elapsed)
        errors = schema_errors(value, schema)
        if errors:
            reason = "content does not match the schema: " + "; ".join(errors)
            return _verdict(case, "fail", reason, observed, data, elapsed)
        return _verdict(
            case, "pass", "content parses and matches the schema", observed, data, elapsed
        )

    def _image_input(
        self, case: dict[str, Any], model_id: str, endpoint: str, protocol: dict[str, Any]
    ) -> dict[str, Any]:
        facts = protocol.get("deployment")
        if facts is None:
            return _no_facts(case)
        if not facts.get("vision"):
            return _verdict(case, "skipped", facts.get("vision_reason", "not a vision plan"), {})
        content_parts = [
            {"type": "image_url", "image_url": {"url": image_data_url(case["image"])}},
            {"type": "text", "text": case["prompt"]},
        ]
        body = self._body(
            protocol,
            model_id,
            [{"role": "user", "content": content_parts}],
            int(case["max_tokens"]),
        )
        data, elapsed = self._chat(endpoint, body)
        message, content = _message(data)
        observed = {"image": case["image"], "content": content, "reasoning": _reasoning(message)}
        ok, why = _exact(content, case["expected"], protocol, case)
        return _verdict(case, "pass" if ok else "fail", why, observed, data, elapsed)

    def _reasoning_split(
        self, case: dict[str, Any], model_id: str, endpoint: str, protocol: dict[str, Any]
    ) -> dict[str, Any]:
        facts = protocol.get("deployment")
        if facts is None:
            return _no_facts(case)
        parser = facts.get("reasoning_parser", "")
        switch = bool(facts.get("thinking_switch"))
        if not parser and not switch:
            return _verdict(
                case,
                "skipped",
                "not a reasoning model: the plan names no reasoning parser and the "
                "chat template has no thinking switch",
                {},
            )
        body = self._body(
            protocol,
            model_id,
            [{"role": "user", "content": case["prompt"]}],
            int(case["max_tokens"]),
        )
        if switch:  # the suite's other cases switch thinking off; this one needs it on
            body["chat_template_kwargs"] = {
                **body.get("chat_template_kwargs", {}),
                "enable_thinking": True,
            }
        # Where effort, not a switch, governs thinking (GLM-5.x, gpt-oss,
        # DeepSeek V4), the suite's "low" lets the model skip it.  This case
        # asks for reasoning, so it sends no effort: the model's own default.
        # GLM-5.3-Flash on 4xH200 (2026-09-29) answered "17 + 25" with an empty
        # <think></think> at low and at high, and reasoned (113 characters)
        # only at its template default, max; the glm45 parser split all three.
        body.pop("reasoning_effort", None)
        data, elapsed = self._chat(endpoint, body)
        message, content = _message(data)
        reasoning = _reasoning(message)
        observed = {"reasoning_parser": parser, "content": content, "reasoning": reasoning}
        ok, why = _exact(content, case["expected"], protocol, case)
        problems = [] if ok else [why]
        if not (reasoning or "").strip():
            problems.insert(0, "message.reasoning is empty")
            if not parser:
                problems.append("the plan names no reasoning parser")
        verdict = "fail" if problems else "pass"
        reason = "; ".join(problems) or "reasoning separated and the content is the answer"
        return _verdict(case, verdict, reason, observed, data, elapsed)

    # -- HTTP -------------------------------------------------------------

    def _body(
        self,
        protocol: dict[str, Any],
        model_id: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": protocol.get("temperature", 0),
            "seed": protocol.get("seed", 42),
        }
        if protocol.get("chat_template_kwargs"):
            body["chat_template_kwargs"] = dict(protocol["chat_template_kwargs"])
        for field, value in (protocol.get("request_fields") or {}).items():
            body.setdefault(field, value)
        if protocol.get("top_p") is not None:
            body["top_p"] = protocol["top_p"]
        if protocol.get("stop"):
            body["stop"] = list(protocol["stop"])
        return body

    def _chat(self, endpoint: str, body: dict[str, Any]) -> tuple[dict[str, Any], float]:
        start = time.monotonic()
        data = self._post(endpoint, "/v1/chat/completions", body)
        return data, round(time.monotonic() - start, 3)

    def _post(self, endpoint: str, path: str, body: dict[str, Any]) -> dict[str, Any]:
        client = self._client or httpx
        response = client.post(f"{endpoint}{path}", json=body, timeout=self._timeout)
        if 400 <= response.status_code < 500:
            raise _Refused(response.status_code, response.text)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_case(case: Mapping[str, str]) -> None:
    case_id = case.get("id", "<unknown>")
    check = case["check"]
    if check not in CHECK_FIELDS:
        raise ValueError(f"{case_id}: unknown check {check!r}; known: {sorted(CHECK_FIELDS)}")
    required, optional = CHECK_FIELDS[check]
    missing = sorted((required | {"id", "max_tokens"}) - set(case))
    if missing:
        raise ValueError(f"{case_id}: {check} needs {missing}")
    unknown = sorted(set(case) - required - optional - _COMMON)
    if unknown:
        raise ValueError(f"{case_id}: fields {unknown} mean nothing to {check}")
    if check == "tool_call":
        tools = _parse_tools(case["tools"])
        names = {t["function"]["name"] for t in tools}
        if case["expected_tool"] not in names:
            raise ValueError(f"{case_id}: expected_tool {case['expected_tool']!r} not in {names}")
    if check == "json_output":
        check_schema_supported(json.loads(case["schema"]))
    if check == "image_input":
        parse_image_spec(case["image"])
    if check == "long_context_needle" and not 0.0 <= float(case["depth"]) <= 1.0:
        raise ValueError(f"{case_id}: depth must be within [0, 1]")


def _extra_checks(case: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(c.strip() for c in case.get("extra_checks", "").split(",") if c.strip())


def _message(data: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
    message = dict(data["choices"][0]["message"])
    return message, message.get("content")


def _reasoning(message: Mapping[str, Any]) -> str | None:
    """vLLM's reasoning field: ``reasoning`` (v0.30), ``reasoning_content`` before."""
    return message.get("reasoning") or message.get("reasoning_content")


def _exact(
    content: str | None, expected: str, protocol: Mapping[str, Any], case: Mapping[str, str]
) -> tuple[bool, str]:
    checks = tuple(protocol.get("deterministic_checks") or DEFAULT_CHECKS) + _extra_checks(case)
    if normalize(content or "", checks) == normalize(expected, checks):
        return True, "the content is the expected answer"
    return False, f"content {content!r} is not {expected!r}"


def _judge_tool_calls(
    calls: list[dict[str, Any]], tools: list[dict[str, Any]], expected: str
) -> tuple[str, str]:
    if not calls:
        return "fail", "no tool call in the response (the parser left it in the content?)"
    parameters = {t["function"]["name"]: t["function"].get("parameters", {}) for t in tools}
    matching = [c for c in calls if (c.get("function") or {}).get("name") == expected]
    if not matching:
        names = [(c.get("function") or {}).get("name") for c in calls]
        return "fail", f"tool calls {names}, none to {expected!r}"
    call = matching[0]
    if call.get("type", "function") != "function":
        return "fail", f"tool call type {call.get('type')!r}, not 'function'"
    raw = call["function"].get("arguments")
    try:
        arguments = json.loads(raw) if isinstance(raw, str) else None
    except json.JSONDecodeError as exc:
        return "fail", f"arguments are not JSON: {exc}"
    if not isinstance(arguments, dict):
        return "fail", f"arguments are not a JSON object: {raw!r}"
    errors = schema_errors(arguments, parameters[expected])
    if errors:
        return "fail", "arguments do not match the tool's parameters: " + "; ".join(errors)
    return "pass", f"a well-formed call to {expected} with valid JSON arguments"


def _no_facts(case: Mapping[str, str]) -> dict[str, Any]:
    return _verdict(case, "skipped", "the plan's deployment facts did not reach the scorer", {})


def _verdict(
    case: Mapping[str, str],
    verdict: str,
    reason: str,
    observed: Mapping[str, Any],
    data: Mapping[str, Any] | None = None,
    elapsed: float | None = None,
) -> dict[str, Any]:
    """One case's result: the verdict, its reason and the observation, as the
    attempt's ``output``; token counts and finish reason from the response."""
    choice = (data or {}).get("choices", [{}])[0] if data else {}
    usage = (data or {}).get("usage") or {}
    output = {"check": case["check"], "verdict": verdict, "reason": reason, **observed}
    result: dict[str, Any] = {
        "case_id": case["id"],
        "check": case["check"],
        "verdict": verdict,
        "reason": reason,
        "output": json.dumps(output, sort_keys=True, ensure_ascii=False),
        "expected": case.get("expected") or case.get("expected_tool"),
        "status": "skipped" if verdict == "skipped" else "completed",
        "score": None if verdict == "skipped" else int(verdict == "pass"),
        "accepted": None if verdict == "skipped" else verdict == "pass",
    }
    if data is not None:
        result.update(
            {
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "finish_reason": choice.get("finish_reason"),
            }
        )
    if elapsed is not None:
        result["time_seconds"] = elapsed
    return result
