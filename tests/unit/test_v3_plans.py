"""Suite-v3 plans: what the deployment checks need, and nothing for suite v2.

A plan that runs task suite v3 serves a long context, tool calls and a
reasoning split (``ServingFeatures``): ``max_model_len`` 32768 (predicted at
that length), ``--enable-auto-tool-choice --tool-call-parser`` from the
vllm-project recipes, and the reasoning parser also for a template with a
thinking switch.  A plan without features is built exactly as before, so
v2's plans and their records keep their identities.  The groups C/D models are
planned from their recorded revisions (config, tensor headers, templates).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import apron.domain.mechanisms.calculator  # noqa: F401 — registers calculators
from apron.adapters.backends.vllm_quantization import engine_facts, load_facts
from apron.adapters.evaluations.deployment_checks import DeploymentCheckScorer
from apron.adapters.evaluations.deployment_facts import deployment_facts
from apron.adapters.evidence.hf_hub import HFHubResolver
from apron.adapters.planning.calculator_source import CalculatorPlanningSource
from apron.application.orchestration.evidence import build_task_attempt
from apron.application.orchestration.plan_pipeline import (
    PlanPipelineResult,
    ServingFeatures,
    reasoning_parser_for,
    run_plan_pipeline,
    served_context,
    served_parser,
    tool_call_parser_for,
)
from apron.domain.schemas.primitives import HardwareSpec
from apron.domain.verdicts import task_verdict

REPO = Path(__file__).resolve().parents[2]
HEADERS = REPO / "tests" / "fixtures" / "cohort" / "headers"
FORMATS = REPO / "tests" / "fixtures" / "external-formats"
H200 = HardwareSpec(
    gpu_sku="NVIDIA H200", total_memory_bytes=150_754_820_096, compute_capability="9.0"
)
FACTS = engine_facts("v0.30.0")
V3 = ServingFeatures(
    max_model_len=32768,
    tool_parsers=FACTS.tool_parsers,
    tool_parser_architectures=FACTS.tool_parser_architectures,
    recipe_checkpoints=FACTS.recipe_checkpoints,
)
# model -> (template fixture or None when vLLM renders the chat, TP,
#           reasoning parser, tool-call parser) -- what each recipe serves.
GROUP_CD: dict[str, tuple[str | None, int, str, str]] = {
    "MiniMaxAI/MiniMax-M3": ("minimax-m3", 8, "minimax_m3", "minimax_m3"),
    "zai-org/GLM-5.3": ("glm-5.3", 8, "glm45", "glm47"),
    "zai-org/GLM-5.3-Flash": ("glm-5.3-flash", 4, "glm45", "glm47"),
    "deepseek-ai/DeepSeek-V4-Flash-0731": (None, 4, "deepseek_v4", "deepseek_v4"),
    "deepseek-ai/DeepSeek-V4.1-Flash": (None, 4, "deepseek_v41", "deepseek_v41"),
    "Qwen/Qwen3.8-Flash-Next": ("qwen3.8-flash-next", 4, "qwen3", "qwen3_xml"),
}


def _header(model_id: str) -> dict[str, Any]:
    record: dict[str, Any] = json.loads(
        (HEADERS / (model_id.replace("/", "__") + ".json")).read_text()
    )
    return record


def _template(model_id: str) -> str | None:
    fixture = GROUP_CD[model_id][0]
    return (FORMATS / fixture / "chat_template.jinja").read_text() if fixture else None


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


class _Clock:
    def now(self) -> Any:
        from datetime import UTC, datetime

        return datetime(2026, 9, 28, tzinfo=UTC)


class _Ids:
    def generate(self) -> str:
        return "fixed"


class _Capturing(CalculatorPlanningSource):
    """The calculator, recording the workload shape each prediction gets."""

    def __init__(self) -> None:
        super().__init__(clock=_Clock())
        self.shapes: list[dict[str, Any]] = []

    def predict(self, model_spec: Any, hardware_spec: Any, workload_shape: Any) -> Any:
        self.shapes.append(dict(workload_shape))
        return super().predict(model_spec, hardware_spec, workload_shape)


def _plan(
    model_id: str, features: ServingFeatures | None
) -> tuple[PlanPipelineResult, _Capturing]:
    source = _Capturing()
    result = run_plan_pipeline(
        _RecordedResolver(_header(model_id), _template(model_id)),
        source,
        model_id,
        H200,
        clock=_Clock(),
        id_gen=_Ids(),
        tensor_parallel=GROUP_CD[model_id][1],
        max_num_batched_tokens=8192,
        max_num_seqs=256,
        tokenizer_modes=FACTS.tokenizer_modes,
        reasoning_parsers=FACTS.reasoning_parsers,
        reasoning_parser_architectures=FACTS.reasoning_parser_architectures,
        features=features,
    )
    assert result.ok, result.error
    assert result.plan is not None
    return result, source


# ---------------------------------------------------------------------------
# Facts: the recipes' tool parsers, per architecture and per checkpoint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", ["v0.29.0", "v0.30.0"])
def test_each_version_lists_recipe_tool_parsers_it_registers(version: str) -> None:
    facts = engine_facts(version)
    raw = load_facts(version)
    assert "features.tool_calling.args" in raw["tool_parser_architectures"]["source"]
    assert set(facts.tool_parser_architectures.values()) <= facts.tool_parsers
    # GLM-5's recipe: --tool-call-parser glm47 --enable-auto-tool-choice.
    assert facts.tool_parser_architectures["GlmMoeDsaForCausalLM"] == "glm47"
    assert facts.tool_parser_architectures["DeepseekV4ForCausalLM"] == "deepseek_v4"
    for entry in facts.recipe_checkpoints.values():
        assert set(entry) <= {"reasoning_parser", "tool_call_parser"}
        assert entry.get("reasoning_parser", "qwen3") in facts.reasoning_parsers | {"qwen3"}
        assert entry.get("tool_call_parser", "hermes") in facts.tool_parsers | {"hermes"}


def test_a_checkpoint_the_registry_has_no_example_of_has_its_own_recipe() -> None:
    # Qwen4Exp's example checkpoint is "" (tests/models/registry.py:521, 1387).
    assert "Qwen4ExpForConditionalGeneration" not in FACTS.reasoning_parser_architectures
    assert FACTS.recipe_checkpoints["Qwen/Qwen3.8-Flash-Next"] == {
        "reasoning_parser": "qwen3",
        "tool_call_parser": "qwen3_xml",
    }


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------

TOOLS_TEMPLATE = "{% if tools %}{{ tools }}{% endif %}"


def test_the_recipe_comes_before_the_template_inference() -> None:
    qwen = _template("Qwen/Qwen3.8-Flash-Next")
    # The template's own block alone infers qwen3_coder ...
    assert tool_call_parser_for("qwen4_exp", FACTS.tool_parsers, qwen)[0] == "qwen3_coder"
    # ... its recipe serves it with qwen3_xml, and wins.
    parser, reason = tool_call_parser_for(
        "qwen4_exp", FACTS.tool_parsers, qwen, checkpoint_parser="qwen3_xml"
    )
    assert parser == "qwen3_xml"
    assert "recipe" in reason


def test_the_architecture_recipe_names_a_parser_the_type_does_not() -> None:
    assert tool_call_parser_for("glm_moe_dsa", FACTS.tool_parsers, TOOLS_TEMPLATE)[0] is None
    parser, _ = tool_call_parser_for(
        "glm_moe_dsa",
        FACTS.tool_parsers,
        TOOLS_TEMPLATE,
        architectures=["GlmMoeDsaForCausalLM"],
        served_with=FACTS.tool_parser_architectures,
    )
    assert parser == "glm47"


def test_an_engine_rendered_chat_takes_tools_only_on_a_recipes_word() -> None:
    # DeepSeek V4 ships no template; vLLM renders its chat.
    none, reason = tool_call_parser_for(
        "deepseek_v4", FACTS.tool_parsers, None, engine_renders_chat=True
    )
    assert none is None and "does not render tools" in reason
    parser, _ = tool_call_parser_for(
        "deepseek_v4",
        FACTS.tool_parsers,
        None,
        architectures=["DeepseekV4ForCausalLM"],
        served_with=FACTS.tool_parser_architectures,
        engine_renders_chat=True,
    )
    assert parser == "deepseek_v4"
    # A template that does not render tools is refused even with a recipe.
    refused, reason = tool_call_parser_for(
        "glm_moe_dsa", FACTS.tool_parsers, "{{ messages }}", checkpoint_parser="glm47"
    )
    assert refused is None and "does not render tools" in reason


def test_a_recipe_parser_the_version_does_not_register_is_refused_with_why() -> None:
    parser, reason = tool_call_parser_for(
        "x", frozenset({"hermes"}), TOOLS_TEMPLATE, checkpoint_parser="qwen3_xml"
    )
    assert parser is None
    assert "does not register" in reason


def test_the_checkpoint_recipe_is_the_last_reasoning_source() -> None:
    parsers = FACTS.reasoning_parsers
    assert served_parser("qwen4_exp", ["Qwen4ExpForConditionalGeneration"], parsers, {}) is None
    assert (
        served_parser(
            "qwen4_exp",
            ["Qwen4ExpForConditionalGeneration"],
            parsers,
            {},
            checkpoint_parser="qwen3",
        )
        == "qwen3"
    )
    # The architecture's own still comes first.
    assert (
        served_parser("x", ["A"], parsers, {"A": "glm45"}, checkpoint_parser="qwen3") == "glm45"
    )
    assert served_parser("x", [], parsers, {}, checkpoint_parser="no_such") is None


def test_a_thinking_switch_gets_its_parser_only_when_thinking_is_per_request() -> None:
    qwen = _template("Qwen/Qwen3.8-Flash-Next")
    assert qwen is not None and "enable_thinking" in qwen
    kwargs: dict[str, Any] = {"checkpoint_parser": "qwen3"}
    assert reasoning_parser_for("qwen4_exp", FACTS.reasoning_parsers, qwen, None, **kwargs) is None
    assert (
        reasoning_parser_for(
            "qwen4_exp", FACTS.reasoning_parsers, qwen, None, thinking_per_request=True, **kwargs
        )
        == "qwen3"
    )


def test_the_context_stops_at_the_checkpoints_positions() -> None:
    assert served_context(32768, {"max_position_embeddings": 262144}) == (32768, None)
    assert served_context(32768, {}) == (32768, None)
    context, note = served_context(32768, {"text_config": {"max_position_embeddings": 8192}})
    assert context == 8192
    assert note is not None and "max_position_embeddings" in note


# ---------------------------------------------------------------------------
# The plans: groups C/D from their recorded revisions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_id", sorted(GROUP_CD))
def test_a_v3_plan_serves_the_checks(model_id: str) -> None:
    _, _, reasoning, tool = GROUP_CD[model_id]
    result, source = _plan(model_id, V3)
    assert result.plan is not None
    engine = result.plan.engine_configuration
    assert engine["max_model_len"] == "32768"
    assert engine["enable_auto_tool_choice"] == "true"
    assert engine["tool_call_parser"] == tool
    assert engine["reasoning_parser"] == reasoning
    # Predicted at the length it serves, never at 640.
    assert source.shapes
    assert all(s["max_model_len"] == 32768 for s in source.shapes)
    assert all(s["isl"] + s["osl"] == 32768 for s in source.shapes)
    # ... and must hold one request of it, as vLLM checks at startup.
    assert all(s["max_batch_size"] == 1 for s in source.shapes)
    facts = deployment_facts(result.plan, result.model_config, result.chat_template)
    assert facts["max_model_len"] == "32768"
    assert facts["tool_call_parser"] == tool


@pytest.mark.parametrize("model_id", sorted(GROUP_CD))
def test_without_features_the_plan_is_what_it_was(model_id: str) -> None:
    result, source = _plan(model_id, None)
    assert result.plan is not None
    engine = result.plan.engine_configuration
    assert engine["max_model_len"] == "640"
    assert "tool_call_parser" not in engine
    assert "enable_auto_tool_choice" not in engine
    # The workload shape the claim is fingerprinted on has no new key.
    assert all(
        "max_model_len" not in s and s["isl"] == 512 and s["max_batch_size"] == 4
        for s in source.shapes
    )


def test_the_qwen_thinking_switch_has_no_parser_in_v2_and_one_in_v3() -> None:
    v2, _ = _plan("Qwen/Qwen3.8-Flash-Next", None)
    v3, _ = _plan("Qwen/Qwen3.8-Flash-Next", V3)
    assert v2.plan is not None and v3.plan is not None
    assert "reasoning_parser" not in v2.plan.engine_configuration
    assert v3.plan.engine_configuration["reasoning_parser"] == "qwen3"


def test_the_v3_scorer_knows_each_plans_facts() -> None:
    result, _ = _plan("zai-org/GLM-5.3-Flash", V3)

    class _Sp:
        model_id = "zai-org/GLM-5.3-Flash"
        plan = result.plan
        model_config = result.model_config
        chat_template = result.chat_template

    scorer = DeploymentCheckScorer.for_plans([_Sp()])  # type: ignore[arg-type]
    prepared = scorer.prepare({"model_id": "zai-org/GLM-5.3-Flash", "cases": []})
    assert prepared["deployment"]["tool_call_parser"] == "glm47"
    assert prepared["deployment"]["max_model_len"] == "32768"


# ---------------------------------------------------------------------------
# Skipped checks: recorded with no acceptance, left out of the pass rate
# ---------------------------------------------------------------------------

FP = "1220" + "ab" * 32


class _Ctx:
    decision_fingerprint = FP
    task_suite_fingerprint = FP
    application_fingerprint = FP
    evaluation_protocol_fingerprint = FP
    solution_fingerprint = FP

    class protocol:  # noqa: N801 — the attribute the builder reads
        scorer = "deployment_checks"


def _attempt(case_id: str, status: str, accepted: bool | None) -> Any:
    scored = {
        "case_id": case_id,
        "status": status,
        "accepted": accepted,
        "score": None if accepted is None else int(accepted),
        "output": json.dumps({"verdict": status}),
    }
    return build_task_attempt(_Ctx(), scored, attempt_id=f"a-{case_id}")  # type: ignore[arg-type]


def test_a_skipped_check_has_no_acceptance_and_no_score() -> None:
    record = _attempt("image-1", "skipped", None)
    assert record.accepted is None
    assert record.criterion_scores == {}
    assert record.failures == ()


def test_the_pass_rate_leaves_skipped_checks_out_and_says_so() -> None:
    attempts = [
        _attempt("needle-1", "completed", True),
        _attempt("tool-1", "completed", True),
        _attempt("image-1", "skipped", None),
    ]
    cases = ["needle-1", "tool-1", "image-1"]
    verdict = task_verdict(attempts, solution_fingerprint=FP, case_ids=cases, quality_floor=None)
    assert verdict.passed
    assert "accepted 2/2" in verdict.detail
    assert "skipped image-1" in verdict.detail
    failing = [*attempts[:1], _attempt("tool-1", "completed", False), attempts[2]]
    verdict = task_verdict(failing, solution_fingerprint=FP, case_ids=cases, quality_floor=None)
    assert not verdict.passed
    everything_skipped = [_attempt(c, "skipped", None) for c in cases]
    verdict = task_verdict(
        everything_skipped, solution_fingerprint=FP, case_ids=cases, quality_floor=None
    )
    assert not verdict.passed and verdict.reason == "every case was skipped"
