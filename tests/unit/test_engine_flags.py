"""Engine flags render the way vLLM parses them (BooleanOptionalAction for bools)."""

from __future__ import annotations

from apron.adapters.backends.vllm_engine import VllmEngineAdapter
from apron.adapters.renderers.engine_flags import engine_flag_args
from apron.application.orchestration.remediation import SIX_CLASSES


def test_booleans_are_bare_or_negated_flags() -> None:
    assert engine_flag_args(
        {"allow_deprecated_quantization": "true", "enforce_eager": "False", "max_model_len": "640"}
    ) == [
        "--allow-deprecated-quantization",
        "--no-enforce-eager",
        "--max-model-len",
        "640",
    ]


def test_class6_broken_plan_reaches_the_capability_check() -> None:
    """The bypass must be on the command line, or vLLM stops at the deprecation check."""
    case = SIX_CLASSES[5]
    command = VllmEngineAdapter()._build_serve_command(case.broken_plan, None)
    assert "--allow-deprecated-quantization" in command.split()
    assert "--allow-deprecated-quantization true" not in command
