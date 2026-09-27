"""HubLineage.search with the Hub replaced by in-memory repos (no network)."""

from __future__ import annotations

import struct
from typing import Any

from apron.adapters.evidence.hf_lineage import HubLineage, decode_floats, weight_bits

_SHAPE = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 1024}
_NORM = (1.0, 0.5, 0.25)
_FP8 = {"quant_method": "fp8"}


class _Hub(HubLineage):
    def __init__(self, repos: dict[str, dict[str, Any]], quantized: list[str]) -> None:
        super().__init__(api=object())
        self.repos, self.quantized = repos, quantized

    def config(self, model_id: str) -> dict[str, Any]:
        return dict(self.repos[model_id]["config"])

    def card(self, model_id: str) -> tuple[dict[str, Any], int]:
        return dict(self.repos[model_id].get("card", {})), 10

    def identity_tensor(self, model_id: str) -> tuple[float, ...] | None:
        return self.repos[model_id].get("norm")

    def tensor_names(self, model_id: str) -> list[str] | None:
        return self.repos[model_id].get(
            "tensors", ["m.layers.0.weight", "m.layers.0.weight_scale"]
        )

    def quantized_from(self, base_model_id: str) -> list[tuple[str, int, dict[str, Any]]]:
        return [(m, 5, self.repos[m]["config"]) for m in self.quantized]


def _repos(**overrides: dict[str, Any]) -> dict[str, dict[str, Any]]:
    repos = {
        "lab/Q-FPQuant": {
            "config": {
                **_SHAPE,
                "quantization_config": {"quant_method": "fp_quant", "forward_dtype": "mxfp4"},
            },
            "norm": _NORM,
        },
        "org/Q": {"config": dict(_SHAPE), "norm": _NORM},
        "org/Q-FP8": {"config": {**_SHAPE, "quantization_config": _FP8}, "norm": _NORM},
        "x/Q-tuned-FP8": {"config": {**_SHAPE, "quantization_config": _FP8}, "norm": (9.0,)},
    }
    repos.update(overrides)
    return repos


def _propose(model_id: str, config: dict[str, Any]) -> dict[str, Any]:
    return {"base_model_id": "org/Q"}


def test_a_listed_quantized_model_with_other_weights_is_dropped() -> None:
    hub = _Hub(_repos(), ["org/Q-FP8", "x/Q-tuned-FP8"])
    search, evidence = hub.search("lab/Q-FPQuant", _propose)
    assert search is not None
    assert [c.model_id for c in search.candidates] == ["org/Q", "org/Q-FP8"]
    dropped = {c["model_id"]: c.get("dropped") for c in evidence["considered"]}
    assert "another model" in (dropped["x/Q-tuned-FP8"] or "")


def test_a_proposed_base_with_other_weights_is_rejected() -> None:
    hub = _Hub(_repos(**{"org/Q": {"config": dict(_SHAPE), "norm": (7.0,)}}), [])
    search, evidence = hub.search("lab/Q-FPQuant", _propose)
    assert search is None
    assert evidence["base_model"]["confirmed"] is False


def test_a_declared_base_needs_no_proposal() -> None:
    repos = _repos()
    repos["lab/Q-FPQuant"]["card"] = {"base_model": "org/Q"}

    def refuse(model_id: str, config: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("the card declares the base; no classifier call")

    search, evidence = _Hub(repos, []).search("lab/Q-FPQuant", refuse)
    assert search is not None and evidence["base_model"]["source"] == "card"


def test_decode_floats_reads_bf16_f16_f32() -> None:
    bf16 = struct.pack("<2H", 0x3F80, 0x3F00)  # 1.0, 0.5
    assert decode_floats(bf16, "BF16") == (1.0, 0.5)
    assert decode_floats(struct.pack("<2e", 1.0, 0.25), "F16") == (1.0, 0.25)
    assert decode_floats(struct.pack("<f", 2.0), "F32") == (2.0,)
    assert decode_floats(b"\x00", "I8") is None


def test_weight_bits_by_format() -> None:
    assert weight_bits({"torch_dtype": "bfloat16"}) == 16
    assert weight_bits({"quantization_config": _FP8}) == 8
    assert weight_bits({"quantization_config": {"quant_method": "gptq", "bits": 4}}) == 4
    groups = {"g": {"weights": {"num_bits": 4}}}
    assert (
        weight_bits(
            {
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "config_groups": groups,
                }
            }
        )
        == 4
    )
    assert weight_bits({"quantization_config": {"quant_method": "hqq"}}) is None


def test_a_candidate_the_engine_would_not_load_is_dropped() -> None:
    repos = _repos()
    repos["org/Q-FP8"]["tensors"] = ["m.layers.0.weight", "m.layers.0.backward_hadamard_matrix"]
    search, evidence = _Hub(repos, ["org/Q-FP8"]).search("lab/Q-FPQuant", _propose)
    assert search is not None
    assert [c.model_id for c in search.candidates] == ["org/Q"]
    dropped = {c["model_id"]: c.get("dropped") for c in evidence["considered"]}
    assert "would not load" in (dropped["org/Q-FP8"] or "")


def test_a_listed_model_with_another_architecture_is_dropped() -> None:
    repos = _repos()
    repos["x/Q-draft-FP8"] = {
        "config": {"architectures": ["DFlashDraftModel"], "quantization_config": _FP8},
        "norm": None,
    }
    search, evidence = _Hub(repos, ["org/Q-FP8", "x/Q-draft-FP8"]).search(
        "lab/Q-FPQuant", _propose
    )
    assert search is not None
    assert "x/Q-draft-FP8" not in [c.model_id for c in search.candidates]
    dropped = {c["model_id"]: c.get("dropped") for c in evidence["considered"]}
    assert "another architecture" in (dropped["x/Q-draft-FP8"] or "")
