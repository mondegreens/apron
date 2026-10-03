"""Weight bytes the engine loads (plan_pipeline._resolve_weight_bytes).

L5 (2026-09-27) found two errors against the measured records: a tied
``lm_head`` counted twice (Qwen3-1.7B +17%) and, without an index, only the
first dtype of the Hub's parameter counts (Qwen3-0.6B-FP8 -22%).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from apron.application.orchestration.plan_pipeline import _resolve_weight_bytes


@dataclass
class _Observation:
    resolved_revision: str = "r"
    publisher_metadata: dict[str, str] | None = None


class _Resolver:
    def __init__(self, tensors: dict[str, int] | None = None, index: bytes | None = None) -> None:
        self.tensors, self.index = tensors, index

    def _tensor_bytes(self, model_id: str, revision: str) -> dict[str, int] | None:
        return self.tensors

    def _download_file(self, model_id: str, filename: str, revision: str) -> bytes | None:
        return self.index


_TENSORS = {"model.embed_tokens.weight": 300, "lm_head.weight": 300, "model.layers.0": 1000}


def test_headers_count_every_stored_tensor() -> None:
    assert _resolve_weight_bytes(_Resolver(_TENSORS), "m", _Observation(), {}) == 1600


def test_tied_lm_head_is_loaded_once() -> None:
    config: dict[str, Any] = {"tie_word_embeddings": True}
    assert _resolve_weight_bytes(_Resolver(_TENSORS), "m", _Observation(), config) == 1300


def test_tied_flag_nested_under_text_config() -> None:
    config: dict[str, Any] = {"text_config": {"tie_word_embeddings": True}}
    assert _resolve_weight_bytes(_Resolver(_TENSORS), "m", _Observation(), config) == 1300


def test_index_total_when_no_headers() -> None:
    resolver = _Resolver(None, b'{"metadata": {"total_size": 4242}}')
    assert _resolve_weight_bytes(resolver, "m", _Observation(), {}) == 4242


def test_parameter_counts_sum_every_dtype_at_its_width() -> None:
    observation = _Observation(
        publisher_metadata={"parameters_BF16": "100", "parameters_F8_E4M3": "400"}
    )
    assert _resolve_weight_bytes(_Resolver(), "m", observation, {}) == 100 * 2 + 400


def test_an_unknown_dtype_makes_the_total_unknown() -> None:
    observation = _Observation(publisher_metadata={"parameters_BF16": "100", "parameters_Q3": "9"})
    assert _resolve_weight_bytes(_Resolver(), "m", observation, {}) == 0


def test_an_unquantized_float32_checkpoint_loads_at_16_bits() -> None:
    """vLLM's dtype=auto serves a float32 checkpoint in bfloat16/float16
    (config/model.py:2285-2287): Mamba-2.8B stores 11.07 GB and loads ~5.5."""
    from apron.application.orchestration.plan_pipeline import loaded_tensor_bytes, runtime_dtype

    meta = {"a.weight": ("F32", 400), "b.weight": ("BF16", 200)}
    assert loaded_tensor_bytes(meta, {"torch_dtype": "float32"}) == {
        "a.weight": 200,
        "b.weight": 200,
    }
    quantized = {"torch_dtype": "bfloat16", "quantization_config": {"quant_method": "fp8"}}
    scale = {"a.weight_scale": ("F32", 4)}
    assert loaded_tensor_bytes(scale, quantized) == {"a.weight_scale": 4}
    assert runtime_dtype({"torch_dtype": "float32"}) == "bfloat16"
    assert runtime_dtype({"torch_dtype": "float16"}) == "float16"
