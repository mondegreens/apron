"""The GPU-free load check (adapters/backends/vllm_quantization.load_problems).

Validated on real checkpoints (safetensors headers, 2026-09-27): the class 6
FPQuant checkpoint stores backward_hadamard_matrix x196, which vLLM v0.29.0
registers nowhere — the B200 boot failed exactly there; the FP8, GPTQ, AWQ
and compressed-tensors checkpoints that loaded in the cohort store nothing
unregistered.
"""

from __future__ import annotations

from apron.adapters.backends.vllm_quantization import load_problems

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
