"""The GPU-free load check (adapters/backends/vllm_quantization.load_problems).

Validated on real checkpoints (safetensors headers, 2026-09-27): the class 6
FPQuant checkpoint stores backward_hadamard_matrix x196, which vLLM v0.29.0
registers nowhere — the B200 boot failed exactly there; the FP8, GPTQ, AWQ
and compressed-tensors checkpoints that loaded in the cohort store nothing
unregistered.
"""

from __future__ import annotations

from apron.adapters.backends.vllm_quantization import load_problems, min_capability

_FPQ = {"quant_method": "fp_quant", "forward_dtype": "mxfp4"}
_LAYER = "model.layers.0.mlp.down_proj"


def test_the_class6_checkpoint_is_refused_before_any_boot() -> None:
    names = [
        f"{_LAYER}.{s}"
        for s in ("qweight", "scales", "forward_hadamard_matrix", "backward_hadamard_matrix")
    ]
    problems = load_problems([*names, "model.norm.weight"], _FPQ)
    assert problems is not None and len(problems) == 1
    assert problems[0].startswith("backward_hadamard_matrix x1")
    assert "fp_quant.py:106" in problems[0]


def test_formats_that_loaded_in_the_cohort_pass() -> None:
    fp8 = [f"{_LAYER}.weight", f"{_LAYER}.weight_scale_inv", "lm_head.weight"]
    assert load_problems(fp8, {"quant_method": "fp8"}) == ()
    gptq = [f"{_LAYER}.{s}" for s in ("qweight", "qzeros", "scales", "g_idx")]
    assert load_problems(gptq, {"quant_method": "gptq", "bits": 4}) == ()
    groups = {"g": {"weights": {"num_bits": 4, "type": "int"}, "input_activations": None}}
    wna16 = [f"{_LAYER}.{s}" for s in ("weight_packed", "weight_scale", "weight_shape")]
    ct = {"quant_method": "compressed-tensors", "config_groups": groups}
    assert load_problems(wna16, ct) == ()


def test_rotary_buffers_the_loaders_skip_are_not_problems() -> None:
    names = [f"{_LAYER}.qweight", "model.layers.0.self_attn.rotary_emb.inv_freq"]
    assert load_problems(names, {"quant_method": "gptq", "bits": 4}) == ()


def test_what_cannot_be_checked_says_so() -> None:
    assert load_problems(["x.weight"], None) is None  # unquantized: the main load path
    assert load_problems(["x.weight"], {"quant_method": "hqq"}) is None
    mixed = {"g": {"weights": {"num_bits": 4, "type": "float"}}}
    assert (
        load_problems(["x.weight"], {"quant_method": "compressed-tensors", "config_groups": mixed})
        is None
    )


# nvidia/Qwen3-8B-NVFP4 (config.json and index read 2026-09-27): quant_method
# "modelopt" with quant_algo "NVFP4", and FP8 KV-cache scales stored as
# self_attn.k_proj.k_scale / v_proj.v_scale.  The first class 6 lineage search
# on Qwen3-8B dropped it as "would not load"; vLLM loads it.
_NVFP4 = {
    "quant_method": "modelopt",
    "quant_algo": "NVFP4",
    "config_groups": {
        "group_0": {
            "weights": {"num_bits": 4, "type": "float"},
            "input_activations": {"num_bits": 4, "type": "float"},
        }
    },
    "kv_cache_scheme": {"num_bits": 8, "type": "float"},
}
_NVFP4_NAMES = [
    f"model.layers.0.self_attn.k_proj.{s}"
    for s in ("weight", "input_scale", "weight_scale", "weight_scale_2", "k_scale")
] + ["model.layers.0.self_attn.v_proj.v_scale"]


def test_modelopt_nvfp4_resolves_to_its_own_method_and_loads() -> None:
    assert load_problems(_NVFP4_NAMES, _NVFP4) == ()
    assert min_capability(_NVFP4) == 75  # ModelOptNvFp4Config, not the FP8 config's 80
    assert min_capability({"quant_method": "modelopt", "quant_algo": "FP8"}) == 80


def test_kv_cache_scales_count_only_for_configs_with_a_kv_cache_method() -> None:
    fp8 = [f"{_LAYER}.weight", f"{_LAYER}.weight_scale_inv", "model.layers.0.self_attn.k_scale"]
    assert load_problems(fp8, {"quant_method": "fp8"}) == ()
    awq = [f"{_LAYER}.{s}" for s in ("qweight", "qzeros", "scales")]
    problems = load_problems([*awq, "model.layers.0.self_attn.k_scale"], {"quant_method": "awq"})
    assert problems is not None and problems[0].startswith("k_scale x1")


def test_modelopt_algorithms_without_listed_facts_stay_unknown() -> None:
    weight_only = {**_NVFP4, "config_groups": {"g": {"weights": {}, "input_activations": None}}}
    assert load_problems(_NVFP4_NAMES, weight_only) is None  # W4A16 linear method
    assert load_problems(_NVFP4_NAMES, {**_NVFP4, "quant_algo": "FP8_PB_WO"}) is None
    assert min_capability({"quant_method": "modelopt", "quant_algo": "MXFP8"}) is None


def test_model_level_tensors_the_engine_names_are_not_problems() -> None:
    # MiniMax-M2.7 / GLM-5.3 / DeepSeek-V3.2 / Kimi K2 (indexes read 2026-09-27):
    # the MoE router's e_score_correction_bias is the model's own parameter, not
    # the fp8 method's.  The first GPU-free check called all four unloadable.
    names = [
        f"{_LAYER}.weight",
        f"{_LAYER}.weight_scale_inv",
        "model.layers.3.mlp.gate.e_score_correction_bias",
    ]
    assert load_problems(names, {"quant_method": "fp8"}) == ()


def test_a_tensor_no_vllm_file_names_is_still_refused() -> None:
    problems = load_problems(
        [f"{_LAYER}.weight", f"{_LAYER}.zz_invented_tensor"], {"quant_method": "fp8"}
    )
    assert problems is not None and problems[0].startswith("zz_invented_tensor x1")
    assert "named nowhere" in problems[0]


def test_state_parameters_with_capitals_are_named_by_the_engine() -> None:
    # Qwen3.6-35B-A3B-FP8 stores the gated delta net's A_log (and dt_bias):
    # a lower-case-only dictionary missed it and called the model unloadable.
    names = [f"{_LAYER}.weight", f"{_LAYER}.weight_scale_inv", "model.layers.0.linear_attn.A_log"]
    assert load_problems(names, {"quant_method": "fp8"}) == ()
