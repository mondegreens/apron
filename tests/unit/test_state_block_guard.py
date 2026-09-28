"""A state (Mamba) cache serves one decode sequence per KV block: the plan
lowers vLLM's default ``max_num_seqs`` to what the predicted cache holds.

Two hybrids failed on one H100 with vLLM's defaults (max_num_seqs 1024):
nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 on v0.30.0 ("max_num_seqs
(1024) exceeds available Mamba cache blocks (733)", config/compilation.py:1535;
report 1220be1b1f6c...) and Qwen/Qwen3.8-27B on v0.29.0 (CUDA-graph profiling
allocated min(1024, 512) blocks x 51,380,224 bytes = 24.5 GiB and ran out of
memory, v1/worker/gpu_model_runner.py:6598-6602; report 122001a0c3f9...).

Configs: the fields the formulas read, from the Hub (2026-09-28).  Weights: what
the planner resolves from each checkpoint's safetensors headers.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import apron.domain.mechanisms.calculator  # noqa: F401 — registers calculators
from apron.adapters.backends.rule_loader import load_rules
from apron.adapters.backends.runpod import GPU_SPECS
from apron.adapters.backends.vllm_quantization import (
    default_max_num_batched_tokens,
    default_max_num_seqs,
    engine_facts,
    engine_versions,
)
from apron.adapters.evidence.hf_hub import HFHubResolver
from apron.adapters.planning.calculator_source import CalculatorPlanningSource
from apron.application.orchestration.correction import apply_correction_spec
from apron.application.orchestration.diagnosis_pipeline import run_diagnosis_pipeline
from apron.application.orchestration.plan_pipeline import (
    PlanPipelineResult,
    StateBlockFacts,
    run_plan_pipeline,
    state_block_limit,
)
from apron.domain.mechanisms.layered import (
    KV_BUDGET_SAFETY_BUFFER_BYTES,
    block_accounting,
    kv_layout,
    layer_kinds,
    state_block_capacity,
)
from apron.domain.schemas.primitives import HardwareSpec
from apron.domain.schemas.solutions import DeploymentPlan

REPO = Path(__file__).resolve().parents[2]
H100_NAME = "NVIDIA H100 80GB HBM3"
H100 = HardwareSpec(
    gpu_sku=H100_NAME,
    total_memory_bytes=GPU_SPECS[H100_NAME]["total_memory_bytes"],
    compute_capability="9.0",
)

_BLOCKS = {"M": "mamba", "E": "moe", "*": "attention"}
NEMOTRON_35 = {
    "architectures": ["NemotronHForCausalLM"],
    "model_type": "nemotron_h",
    "dtype": "bfloat16",
    "vocab_size": 131072,
    "hidden_size": 2688,
    "num_hidden_layers": 52,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "mamba_num_heads": 64,
    "mamba_head_dim": 64,
    "ssm_state_size": 128,
    "n_groups": 8,
    "conv_kernel": 4,
    "mamba_ssm_cache_dtype": "float32",
    "layers_block_type": [
        _BLOCKS[c] for c in "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
    ],
}
QWEN38 = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "torch_dtype": "bfloat16",
    # A vision tower: the embedding is not in the compiled graph (no compile segment).
    "vision_config": {"depth": 27, "hidden_size": 1152, "out_hidden_size": 5120},
    "text_config": {
        "model_type": "qwen3_5_text",
        "vocab_size": 248320,
        "num_hidden_layers": 64,
        "full_attention_interval": 4,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "hidden_size": 5120,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "mamba_ssm_dtype": "float32",
    },
}
QWEN36 = {
    **QWEN38,
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe",
    "text_config": {
        **QWEN38["text_config"],
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "linear_num_value_heads": 32,
    },
}
# Sliding + full attention, no state (meta-models/Muse-Glimmer-30B's shape).
MUSE = {
    "architectures": ["MuseGlimmerForCausalLM"],
    "model_type": "muse_glimmer",
    "torch_dtype": "bfloat16",
    "vocab_size": 202048,
    "hidden_size": 6656,
    "num_hidden_layers": 4,
    "num_attention_heads": 52,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "sliding_window": 4096,
    "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
}
WEIGHTS = {
    # 61.31 GiB stored, less the 2.49 GiB of mtp.layers.* vLLM's NemotronH
    # mapper drops (models/nemotron_h.py:728)
    "nemotron": 63_155_880_576,  # 58.82 GiB
    "qwen38": 54_713_457_120,  # 50.96 GiB
    "qwen36": 36_601_120_992,  # 34.09 GiB
    "muse": 59_553_253_376,
}


class _Clock:
    def now(self) -> datetime:
        return datetime(2026, 9, 28, tzinfo=UTC)


class _Ids:
    def generate(self) -> str:
        return "fixed"


class _Resolver(HFHubResolver):
    """One recorded revision: the config and one tensor of the loaded bytes."""

    def __init__(self, config: dict[str, Any], weight_bytes: int) -> None:
        self.config = config
        self.weight_bytes = weight_bytes

    def _get_model_info(self, model_id: str, revision: str | None) -> dict[str, Any]:
        return {"sha": "0" * 40, "gated": False, "tags": [], "safetensors": {}}

    def _download_file(self, model_id: str, filename: str, revision: str) -> bytes | None:
        return json.dumps(self.config).encode() if filename == "config.json" else None

    def _tensor_meta(self, model_id: str, revision: str) -> dict[str, tuple[str, int]] | None:
        return {"weights": ("BF16", self.weight_bytes)}


def _state_facts(version: str) -> StateBlockFacts:
    facts = engine_facts(version)
    return StateBlockFacts(
        engine_version=facts.version,
        check=facts.state_block_check,
        profiling=facts.state_block_profiling,
        default_source=facts.default_max_num_seqs_source(H100.total_memory_bytes, H100_NAME),
    )


def _plan(config: dict[str, Any], weights: int, version: str) -> PlanPipelineResult:
    facts = engine_facts(version)
    result = run_plan_pipeline(
        _Resolver(config, weights),
        CalculatorPlanningSource(clock=_Clock()),
        "recorded/model",
        H100,
        clock=_Clock(),
        id_gen=_Ids(),
        max_num_batched_tokens=facts.default_max_num_batched_tokens(
            H100.total_memory_bytes, H100_NAME
        ),
        max_num_seqs=facts.default_max_num_seqs(H100.total_memory_bytes, H100_NAME),
        state_blocks=_state_facts(version),
    )
    assert result.ok, result.error
    return result


# ---------------------------------------------------------------------------
# Engine facts: defaults and the state-block sites, per version
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", engine_versions())
def test_batch_defaults_come_from_each_versions_facts(version: str) -> None:
    facts = engine_facts(version)
    gib = 1 << 30
    # get_batch_defaults' tiers, OpenAI server: >= 160 GiB; >= 70 GiB but not
    # an A100; everything else.
    for memory, name, tokens, seqs in (
        (180 * gib, "NVIDIA B200", 16384, 1024),
        (80 * gib, H100_NAME, 8192, 1024),
        (80 * gib, "NVIDIA A100-SXM4-80GB", 2048, 256),
        (24 * gib, "NVIDIA GeForce RTX 4090", 2048, 256),
    ):
        assert facts.default_max_num_batched_tokens(memory, name) == tokens
        assert facts.default_max_num_seqs(memory, name) == seqs
    # The module functions read the default version's facts: the values the
    # hand-written table held for every GPU Apron rents.
    for name, spec in GPU_SPECS.items():
        memory = spec["total_memory_bytes"]
        big = memory >= 70 * gib and "a100" not in name.lower()
        assert default_max_num_seqs(memory, name) == (1024 if big else 256)
        expected = 16384 if memory >= 160 * gib else (8192 if big else 2048)
        assert default_max_num_batched_tokens(memory, name) == expected


def test_each_version_names_its_state_block_check() -> None:
    v29, v30 = engine_facts("v0.29.0"), engine_facts("v0.30.0")
    assert v29.state_block_check == "config/compilation.py:1513"
    assert v30.state_block_check == "config/compilation.py:1535"
    assert v29.state_block_profiling == (
        "v1/worker/gpu_model_runner.py:6600",
        "v1/worker/gpu/cudagraph_utils.py:845",
    )
    assert v30.state_block_profiling == (
        "v1/worker/gpu_model_runner.py:6550",
        "v1/worker/gpu/cudagraph_utils.py:974",
    )


# ---------------------------------------------------------------------------
# The block vLLM sizes the pool with
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "per_block", "per_sequence"),
    [
        # 6 groups x 2096-token pages of 1,024 B per token (hybrid-memory-trace.md)
        (NEMOTRON_35, 12_877_824, 1_725_628_416),
        (QWEN38, 51_380_224, 17_520_656_384),
        (QWEN36, 21_626_880, 5_514_854_400),
    ],
)
def test_grouped_layout_block_is_group_size_pages(
    config: dict[str, Any], per_block: int, per_sequence: int
) -> None:
    kinds = layer_kinds(config, tp=1, kv_dtype_bytes=2, model_dtype_bytes=2)
    assert kinds is not None
    blocks = block_accounting(
        kinds, 262_144, in_flight_tokens=16_384, layout=kv_layout(config)
    )
    assert blocks is not None
    assert blocks.bytes_per_block == per_block
    assert blocks.bytes_per_sequence == per_sequence


def test_state_block_capacity_is_the_floor_less_the_margin() -> None:
    assert state_block_capacity(9_390_155_392, 12_877_824, margin_bytes=0) == 729
    # The default margin is the safety buffer: (9,390,155,392 - 2,351,494,595) // 12,877,824.
    assert state_block_capacity(9_390_155_392, 12_877_824) == 546
    assert state_block_capacity(1 << 30, 12_877_824) == 0
    assert state_block_capacity(1 << 40, 0) == 0


# ---------------------------------------------------------------------------
# The planner, against the two failed boots
# ---------------------------------------------------------------------------


def _unguarded(config: dict[str, Any], weights: int, version: str) -> dict[str, Any]:
    """The prediction at vLLM's default max_num_seqs, before the guard."""
    facts = engine_facts(version)
    result = run_plan_pipeline(
        _Resolver(config, weights),
        CalculatorPlanningSource(clock=_Clock()),
        "recorded/model",
        H100,
        clock=_Clock(),
        id_gen=_Ids(),
        max_num_batched_tokens=facts.default_max_num_batched_tokens(
            H100.total_memory_bytes, H100_NAME
        ),
        max_num_seqs=facts.default_max_num_seqs(H100.total_memory_bytes, H100_NAME),
    )
    assert result.ok, result.error
    return dict(result.claim.proposed_configuration)


def test_nemotron_35_guard_stays_under_the_blocks_vllm_measured() -> None:
    """v0.30.0 on an H100 with 1024 sequences: 733 blocks measured (report
    1220be1b1f6c...).  The prediction at those settings is 793 blocks, 60
    (0.72 GiB) over -- the 0.66 GiB compile segment this text-only TP 1 plan
    may hold is most of it -- and inside the safety buffer plus that segment;
    the plan's 556 sequences stay under 733."""
    claim = _unguarded(NEMOTRON_35, WEIGHTS["nemotron"], "v0.30.0")
    assert claim["available_kv_cache_bytes"] == 10_216_498_970
    assert claim["kv_bytes_per_block"] == 12_877_824
    assert claim["compile_segment_bytes"] == 131_072 * 2688 * 2
    assert claim["available_kv_cache_bytes"] // claim["kv_bytes_per_block"] == 793
    over = claim["available_kv_cache_bytes"] - 733 * 12_877_824
    assert 0 < over < KV_BUDGET_SAFETY_BUFFER_BYTES + claim["compile_segment_bytes"]
    result = _plan(NEMOTRON_35, WEIGHTS["nemotron"], "v0.30.0")
    assert result.plan is not None
    assert result.plan.engine_configuration["max_num_seqs"] == "556"
    assert 556 <= 733
    (note,) = result.notes
    assert note.startswith("max_num_seqs 556: vLLM v0.30.0 defaults to 1024 on this GPU")
    assert "engine/arg_utils.py:2740" in note
    assert "holds 793 blocks of 12,877,824 bytes, 556 after a 2.85 GiB safety buffer" in note
    assert "2.19 GiB, plus a 0.66 GiB compile segment vLLM may hold" in note
    assert "config/compilation.py:1535" in note


def test_qwen38_minimal_profiling_cache_does_not_fit_the_predicted_memory() -> None:
    """v0.29.0: the profiling cache of min(1024, 512) blocks is 24.5 GiB, more
    than the 16.70 GiB the prediction leaves for the KV cache."""
    claim = _unguarded(QWEN38, WEIGHTS["qwen38"], "v0.29.0")
    available, per_block = claim["available_kv_cache_bytes"], claim["kv_bytes_per_block"]
    assert (available, per_block) == (17_928_519_610, 51_380_224)
    assert claim["compile_segment_bytes"] == 0  # a vision wrapper: no compiled embedding
    assert min(1024, 512) * per_block == 26_306_674_688 > available
    assert available // per_block == 348
    result = _plan(QWEN38, WEIGHTS["qwen38"], "v0.29.0")
    assert result.plan is not None
    assert result.plan.engine_configuration["max_num_seqs"] == "303"
    assert 303 * per_block <= available  # the profiling cache now fits
    encoder, note = result.notes
    assert note.startswith("max_num_seqs 303: vLLM v0.29.0 defaults to 1024")
    # No processor files recorded here: the plan says the encoder peak is left out.
    assert encoder == _no_processor_note("qwen3_5")
    assert "config/compilation.py:1513" in note
    assert "v1/worker/gpu_model_runner.py:6600" in note
    # v0.30.0 has the same check and the same defaults: the same plan.
    v30 = _plan(QWEN38, WEIGHTS["qwen38"], "v0.30.0")
    assert v30.plan is not None
    assert v30.plan.engine_configuration["max_num_seqs"] == "303"


def test_qwen36_keeps_the_default_it_booted_with() -> None:
    """v0.30.0 on an H100 booted healthy with 1024 sequences (report
    12208654b66b...): 34,821,447,352 bytes of KV memory, 1610 blocks."""
    result = _plan(QWEN36, WEIGHTS["qwen36"], "v0.30.0")
    claim = result.claim.proposed_configuration
    assert claim["available_kv_cache_bytes"] == 36_814_581_946
    assert state_block_capacity(claim["available_kv_cache_bytes"], 21_626_880) == 1593
    assert 1593 <= 34_821_447_352 // 21_626_880 == 1610
    assert result.plan is not None
    assert "max_num_seqs" not in result.plan.engine_configuration
    assert result.notes == (_no_processor_note("qwen3_5_moe"),)


def _no_processor_note(model_type: str) -> str:
    """This file's configs carry no processor files (test_encoder_startup_plan
    plans Qwen3.6 with the recorded ones)."""
    return (
        f"encoder startup peak not modelled for the {model_type} vision tower (no "
        "processor_config.json, preprocessor_config.json or video_preprocessor_config.json "
        "at the revision); activation may be under-predicted"
    )


def test_the_buffer_covers_qwen36s_measured_over_prediction() -> None:
    """Qwen3.6's available KV memory, predicted from this file's config (no
    processor fields, so without the encoder moment) minus measured, is
    under the safety buffer measured over every healthy record."""
    assert 0 < 36_814_581_946 - 34_821_447_352 < KV_BUDGET_SAFETY_BUFFER_BYTES


def test_models_without_a_state_cache_are_untouched() -> None:
    result = _plan(MUSE, WEIGHTS["muse"], "v0.30.0")
    assert result.model_spec is not None
    assert result.model_spec.components[0].mechanism == "layered_decode"
    assert "kv_bytes_per_block" not in result.claim.proposed_configuration
    assert result.plan is not None
    assert "max_num_seqs" not in result.plan.engine_configuration
    assert result.notes == ()


def test_no_guard_without_facts_or_default() -> None:
    predicted = {"kv_bytes_per_block": 12_877_824, "available_kv_cache_bytes": 9_390_155_392}
    facts = _state_facts("v0.30.0")
    assert state_block_limit(predicted, 1024, None) is None
    assert state_block_limit(predicted, None, facts) is None
    assert state_block_limit(predicted, 256, facts) is None  # fits: the default stays
    no_sites = StateBlockFacts(engine_version="vX", check=None, profiling=())
    assert state_block_limit(predicted, 1024, no_sites) is None
    limit = state_block_limit(predicted, 1024, facts)
    assert limit is not None and limit[0] == 546  # (9,390,155,392 - buffer) // 12,877,824


# ---------------------------------------------------------------------------
# The diagnosis rule for the runtime error (hypothesis)
# ---------------------------------------------------------------------------

OBSERVED = (
    "ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (733). Each "
    "decode sequence requires one Mamba cache block, so CUDA graph capture cannot "
    "proceed. Please lower max_num_seqs to at most 733 or increase gpu_memory_utilization."
)


def _rule(version: str) -> dict[str, Any]:
    (rule,) = [
        r
        for r in load_rules(REPO / "rules", "vllm", version)
        if r["error_family"] == "mamba_cache_blocks"
    ]
    return rule


@pytest.mark.parametrize(("version", "line"), [("v0.29.0", 1513), ("v0.30.0", 1535)])
def test_rule_is_a_typed_hypothesis_citing_the_check(version: str, line: int) -> None:
    rule = _rule(version)
    assert rule["engine_version"] == version
    assert rule["status"] == "hypothesis"
    assert rule["source_sites"] == [{"file": "config/compilation.py", "line": line}]
    assert engine_facts(version).state_block_check == f"config/compilation.py:{line}"
    schema = rule["extraction_schema"]
    assert {k: v["type"] for k, v in schema.items()} == {
        "max_num_seqs": "int",
        "mamba_cache_blocks": "int",
    }
    assert schema["mamba_cache_blocks"]["source"].startswith(
        f"config/compilation.py:{line + 2} "
    )
    # The pinned checkout, when present (.sources/ is not committed).
    for root in (REPO / ".sources", REPO.parents[2] / ".sources"):
        tree = root / ("vllm" if version == "v0.29.0" else f"vllm-{version}") / "vllm"
        if (tree / "config" / "compilation.py").exists():
            lines = (tree / "config" / "compilation.py").read_text().splitlines()
            assert lines[line - 1].strip() == "raise ValueError("
            assert "exceeds available Mamba cache" in lines[line]
            assert "kv_cache_config.num_blocks" in lines[line + 1]
            break


def test_v30_rule_names_the_verification_report_it_came_from() -> None:
    rule = _rule("v0.30.0")
    report = "1220be1b1f6ceab3adcfac8e4e200c754c4c2d3f22e5ef2e70ba7fc05e864522d7f6"
    assert report in rule["curation"]
    assert OBSERVED.removeprefix("ValueError: ") in rule["examples"]
    path = REPO / "_dev_notes" / "cohort-run" / "records" / "verification-reports"
    if (path / f"{report}.json").exists():
        log = json.loads((path / f"{report}.json").read_text())["log_tail"]
        assert OBSERVED in log


def test_correction_sets_max_num_seqs_to_the_blocks_the_message_names() -> None:
    plan = DeploymentPlan(engine_configuration={"max_model_len": "640"})
    corrected = apply_correction_spec(
        _rule("v0.30.0")["correction_spec"],
        {"max_num_seqs": 1024, "mamba_cache_blocks": 733},
        plan,
    )
    assert corrected is not None
    assert corrected.engine_configuration == {"max_model_len": "640", "max_num_seqs": "733"}


class _Engine:
    """Classification as the classifier would return it for the observed error."""

    engine_version = "v0.30.0"

    def classify(self, error: str) -> dict[str, Any]:
        return {"failure_class": "mamba_cache_blocks", "confidence": 0.95}

    def extract(self, error: str, failure_class: str) -> dict[str, int]:
        return {"max_num_seqs": 1024, "mamba_cache_blocks": 733}


def test_diagnosis_pipeline_corrects_the_observed_failure() -> None:
    rules = load_rules(REPO / "rules", "vllm", "v0.30.0")
    plan = DeploymentPlan(engine_configuration={"max_model_len": "640"})
    result = run_diagnosis_pipeline(OBSERVED, _Engine(), plan, {}, H100, rules)
    assert result.rule_matched
    assert result.correction_strategy == "set_field"
    assert result.corrected_plan is not None
    assert result.corrected_plan.engine_configuration["max_num_seqs"] == "733"
    assert result.extraction_confidence == 1.0
