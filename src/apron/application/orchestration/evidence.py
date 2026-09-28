"""Evidence identity — fingerprints built from validated domain objects (§8).

Every fingerprint an evidence record carries is ``fingerprint_hex`` of a
validated domain object, never a placeholder.  The solution fingerprint is
the digest of a model spec, a deployment plan and a *requested* execution,
all known before provisioning (F4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from apron.domain.canonical import canonicalize, digest_hex
from apron.domain.fingerprints import fingerprint_hex
from apron.domain.schemas.migrations import load_record
from apron.domain.schemas.records import TaskAttemptRecord
from apron.domain.schemas.solutions import (
    EvaluationProtocol,
    RequestedExecutionSpec,
    SolutionIdentity,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from apron.domain.protocols import RecordStore
    from apron.domain.schemas.authority import DecisionRequest
    from apron.domain.schemas.models import ModelSpec
    from apron.domain.schemas.solutions import DeploymentPlan
    from apron.domain.schemas.tasks import ApplicationSpec, TaskSuiteSpec


def solution_identity(
    model_spec: ModelSpec,
    plan: DeploymentPlan,
    requested: RequestedExecutionSpec,
) -> SolutionIdentity:
    return SolutionIdentity(
        model_spec_fingerprint=fingerprint_hex(model_spec),
        deployment_plan_fingerprint=fingerprint_hex(plan),
        requested_execution=requested,
    )


def solution_fingerprint(
    model_spec: ModelSpec,
    plan: DeploymentPlan,
    requested: RequestedExecutionSpec,
) -> str:
    """Digest of model spec + deployment plan + requested execution spec."""
    return fingerprint_hex(solution_identity(model_spec, plan, requested))


def decision_request_digest(request: DecisionRequest) -> str:
    """Full-record digest of the accepted request (same as ``build_report``)."""
    return digest_hex(canonicalize(request.model_dump(mode="json")))


BOUND_PROTOCOL_FIELDS = frozenset(
    {
        "decision_request_digest",
        "task_suite_fingerprint",
        "application_fingerprint",
        "solution_fingerprint",
    }
)


def protocol_template(protocol: EvaluationProtocol) -> dict[str, Any]:
    """The solution-independent part of an accepted protocol (no digests)."""
    dumped = protocol.model_dump(mode="json")
    return {k: v for k, v in dumped.items() if k not in BOUND_PROTOCOL_FIELDS}


def build_evaluation_protocol(
    template: Mapping[str, Any],
    *,
    request: DecisionRequest,
    task_suite: TaskSuiteSpec,
    application: ApplicationSpec,
    solution_fp: str,
) -> EvaluationProtocol:
    """Bind an accepted protocol template to one solution.

    The template supplies harness, scorer, checks and sampling and carries no
    digests; the digests come from the validated request, suite, application
    and solution.  A template that tries to set a digest is rejected.
    """
    preset = sorted(BOUND_PROTOCOL_FIELDS & set(template))
    if preset:
        raise ValueError(f"protocol template must not set bound fields: {preset}")
    return EvaluationProtocol.model_validate(
        {
            **template,
            "decision_request_digest": decision_request_digest(request),
            "task_suite_fingerprint": fingerprint_hex(task_suite),
            "application_fingerprint": fingerprint_hex(application),
            "solution_fingerprint": solution_fp,
        }
    )


def load_typed[M: BaseModel](
    store: RecordStore, digest: str, cls: type[M]
) -> tuple[M | None, str | None]:
    """Load a stored record by digest through the migration path.

    Returns ``(record, None)`` or ``(None, reason)``: a missing or unloadable
    digest is a named reason, never a silent pass.
    """
    try:
        raw = store.retrieve(digest)
    except (ValueError, OSError) as exc:
        return None, f"record {digest[:16]}… unreadable: {exc}"
    if raw is None:
        return None, f"record {digest[:16]}… not found"
    try:
        return load_record(cls, raw), None
    except (ValidationError, KeyError, ValueError) as exc:
        return None, f"record {digest[:16]}… is not a valid {cls.__name__}: {exc}"


@dataclass(frozen=True)
class EvidenceContext:
    """The five INV-25 fingerprints a task attempt carries, from validated objects."""

    request: DecisionRequest
    task_suite: TaskSuiteSpec
    application: ApplicationSpec
    protocol: EvaluationProtocol
    solution_fingerprint: str

    @classmethod
    def bind(
        cls,
        *,
        request: DecisionRequest,
        task_suite: TaskSuiteSpec,
        application: ApplicationSpec,
        protocol_template: Mapping[str, Any],
        solution_fp: str,
    ) -> EvidenceContext:
        protocol = build_evaluation_protocol(
            protocol_template,
            request=request,
            task_suite=task_suite,
            application=application,
            solution_fp=solution_fp,
        )
        return cls(request, task_suite, application, protocol, solution_fp)

    @property
    def decision_fingerprint(self) -> str:
        return fingerprint_hex(self.request)

    @property
    def task_suite_fingerprint(self) -> str:
        return fingerprint_hex(self.task_suite)

    @property
    def application_fingerprint(self) -> str:
        return fingerprint_hex(self.application)

    @property
    def evaluation_protocol_fingerprint(self) -> str:
        return fingerprint_hex(self.protocol)


def build_task_attempt(
    ctx: EvidenceContext,
    scored: dict[str, Any],
    *,
    attempt_id: str,
    retry: int = 0,
    infrastructure_cost: float | None = None,
    trace_references: tuple[str, ...] = (),
    failures: tuple[str, ...] = (),
    reason: str = "task_evaluation",
    provenance: str | None = None,
) -> TaskAttemptRecord:
    """One validated TaskAttemptRecord for one scored case — failures included.

    ``scored`` is an evaluation adapter's per-case result.  A case that errored
    (``status == "failed"``) is still a first-class attempt with
    ``accepted=False`` and the error in ``failures``.  Cost fields follow §7:
    ``infrastructure_cost`` is this case's share of task-evaluation seconds x
    rate; the market-equivalent and out-of-pocket prices equal it here
    because RunPod Secure is paid at list price with no subsidy.
    """
    errors = list(failures)
    if scored.get("status") == "failed":
        errors.append(f"evaluation:{scored.get('error', 'unknown error')}")
    if scored.get("finish_reason") == "length":
        # Cut at the case's max_tokens: a mismatch says nothing about the answer.
        errors.append("evaluation:truncated at max_tokens")
    score = scored.get("score")
    return TaskAttemptRecord(
        decision_fingerprint=ctx.decision_fingerprint,
        task_suite_fingerprint=ctx.task_suite_fingerprint,
        application_fingerprint=ctx.application_fingerprint,
        evaluation_protocol_fingerprint=ctx.evaluation_protocol_fingerprint,
        solution_fingerprint=ctx.solution_fingerprint,
        case_id=str(scored.get("case_id", "")),
        attempt_id=attempt_id,
        output=scored.get("output"),
        criterion_scores={} if score is None else {ctx.protocol.scorer: float(score)},
        accepted=bool(scored.get("accepted", False)),
        trace_references=trace_references,
        retries=retry,
        input_tokens=scored.get("input_tokens"),
        output_tokens=scored.get("output_tokens"),
        time_seconds=scored.get("time_seconds"),
        infrastructure_cost=infrastructure_cost,
        market_equivalent_price=infrastructure_cost,
        project_out_of_pocket_cost=infrastructure_cost,
        failures=tuple(errors),
        claim_scope="task_outcome",
        production_mode=False,
        reason=reason,
        raw_observation_provenance=provenance,
        lifecycle="observed",
    )


# A template that switches thinking off by ``thinking_mode`` (MiniMax-M3).
_THINKING_MODE_DISABLED = re.compile(r"thinking_mode\s*==\s*['\"]disabled['\"]")


def chat_template_kwargs(chat_template: str | None) -> dict[str, Any] | None:
    """Template kwargs for a model whose chat template accepts them (§9.1 step 4).

    Qwen3-style templates branch on ``enable_thinking``; for them thinking is
    switched off so the deterministic scorer sees the answer, not the
    reasoning.  Muse-Glimmer's template always reasons and reads
    ``reasoning_strength`` (default ``high``; the model card lists low / medium
    / high / xhigh); it gets ``low``, the suite's minimal reasoning.  vLLM
    drops a ``reasoning_effort`` request field for it: the field reaches the
    template only as a variable of that name (chat_completion/protocol.py:
    585-598, renderers/hf.py:726-737, v0.30.0).  MiniMax-M3's template reads
    ``thinking_mode`` (enabled / disabled / adaptive; undefined renders
    adaptive) and, disabled, ends the prompt with ``</mm:think>``: the answer
    starts at once, and the ``minimax_m3`` parser reads the same kwarg
    (reasoning/minimax_m3_reasoning_parser.py:44-45).  Only a template that
    knows the ``disabled`` value gets it.  Templates that read none of these
    variables get no kwargs, so the request body is exactly what an ordinary
    client sends.
    """
    kwargs: dict[str, Any] = {}
    if chat_template and "enable_thinking" in chat_template:
        kwargs["enable_thinking"] = False
    if chat_template and "reasoning_strength" in chat_template:
        kwargs["reasoning_strength"] = "low"
    if chat_template and _THINKING_MODE_DISABLED.search(chat_template):
        kwargs["thinking_mode"] = "disabled"
    return kwargs or None


# Tokenizer modes whose chat renderer reads ``reasoning_effort`` from the
# request (vLLM v0.30.0 tokenizers/deepseek_v4.py:43-48 and deepseek_v41.py):
# DeepSeek V4/V4.1 repos ship no chat template, vLLM renders the chat, and
# thinking defaults to "high" unless the request says otherwise.
REASONING_EFFORT_RENDERERS = frozenset({"deepseek_v4", "deepseek_v41"})


def reasoning_request_fields(
    chat_template: str | None, renderer: str | None = None
) -> dict[str, Any] | None:
    """Request fields that keep reasoning short where it cannot be switched off.

    gpt-oss always reasons in its analysis channel; its template reads
    ``reasoning_effort`` and has no ``enable_thinking``.  vLLM takes the
    effort as the OpenAI request field (not a template kwarg) for it.  GLM-5.x
    templates always open ``<think>`` and read the same variable (``low`` or
    ``high``, else ``max``); the field reaches them as it (chat_completion/
    protocol.py:585-589, v0.30.0).  A template with ``enable_thinking`` gets
    that switch instead.
    """
    if (
        chat_template
        and "enable_thinking" not in chat_template
        and ("reasoning_effort" in chat_template)
    ):
        return {"reasoning_effort": "low"}
    if chat_template is None and renderer in REASONING_EFFORT_RENDERERS:
        return {"reasoning_effort": "low"}
    return None


def scorer_input(
    ctx: EvidenceContext,
    *,
    model_id: str,
    chat_template_kwargs: Mapping[str, Any] | None = None,
    request_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The evaluation adapter's input for this context's suite and protocol.

    ``chat_template_kwargs`` is included only when the caller established that
    the model's chat template accepts it (§9.1 step 4).
    """
    protocol = ctx.protocol
    # The deterministic harness sends each case once, one request at a time.
    # A protocol asking for more is refused, never run as if it had not asked.
    if protocol.repetitions != 1:
        raise ValueError(f"repetitions={protocol.repetitions}: the harness runs each case once")
    if protocol.concurrency not in (None, 1):
        raise ValueError(
            f"concurrency={protocol.concurrency}: the harness sends one request at a time"
        )
    data: dict[str, Any] = {
        "scorer_type": protocol.scorer,
        "cases": [dict(case) for case in ctx.task_suite.cases],
        "model_id": model_id,
        "deterministic_checks": list(protocol.deterministic_checks),
        "sampling_temperature": protocol.sampling_temperature,
        "seeds": list(protocol.seeds),
    }
    if protocol.sampling_top_p is not None:
        data["sampling_top_p"] = protocol.sampling_top_p
    if protocol.stopping_rules:
        data["stopping_rules"] = list(protocol.stopping_rules)
    if chat_template_kwargs:
        data["chat_template_kwargs"] = dict(chat_template_kwargs)
    if request_fields:
        data["request_fields"] = dict(request_fields)
    return data


class InvalidRecordError(ValueError):
    """A record failed validation at the storage boundary; nothing was stored."""


def store_validated(store: RecordStore, record: BaseModel) -> str:
    """Validate *record* through its own model and store its JSON form (F10).

    Re-validation catches objects built with ``model_copy(update=...)``,
    which skips validation.  An invalid record raises and is never stored;
    the cohort run stops on it.
    """
    data = record.model_dump(mode="json")
    try:
        type(record).model_validate(data)
    except ValidationError as exc:
        raise InvalidRecordError(f"{type(record).__name__} failed validation: {exc}") from exc
    return store.store(data)
