"""Same-lineage checkpoints on the Hugging Face Hub (PLAN §10.1 class 6 fallback).

Finds the model a checkpoint was derived from and the published checkpoints
the Hub records as *quantized* from that model (the model tree's
``base_model:quantized:<id>`` relation; fine-tunes are a different relation
and never listed).  Every candidate's config.json is read so the strategy can
compare architectures; the pinned engine's format table says what each needs.

The base model comes from the checkpoint's card when it declares one.  When
it does not, a proposer (the classifier model) names one, and the proposal is
used only if the Hub confirms it: it exists, is unquantized, has the same
architecture as the checkpoint, and stores the same final-norm weights.

The Hub's "quantized" relation is declared by each publisher and is sometimes
wrong (``buttercoconut/Qwen3-ko-alpaca-0.6B-Q4`` is a fine-tune).  Quantizers
leave norm layers unquantized, so a true quantized copy stores the base
model's final norm bit for bit; a candidate whose final norm differs is a
different model and is dropped (2 KB range read per repo).  Limit: a LoRA
fine-tune merged into the linear layers leaves the norm untouched and passes.
``evidence`` records every step, including each dropped candidate and why.
"""

from __future__ import annotations

import json
import struct
from collections import Counter
from typing import TYPE_CHECKING, Any

from apron.adapters.backends.vllm_quantization import (
    QUANT_MIN_CAPABILITY,
    load_problems,
    min_capability,
)
from apron.application.orchestration.correction import (
    SHAPE_FIELDS,
    ArtifactCandidate,
    ArtifactSearch,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# Formats the pinned engine can load: only these are worth reading in full.
_LOADABLE = frozenset(QUANT_MIN_CAPABILITY) | {"compressed-tensors"}
# Unquantized in every quantized format; identical across copies of one model.
_IDENTITY_TENSORS = ("model.norm.weight", "model.language_model.norm.weight")


def decode_floats(raw: bytes, dtype: str) -> tuple[float, ...] | None:
    """A float tensor's values; None for a dtype that is not a plain float."""
    if dtype == "BF16":
        halves = struct.unpack(f"<{len(raw) // 2}H", raw)
        return tuple(struct.unpack("<f", struct.pack("<I", h << 16))[0] for h in halves)
    if dtype == "F16":
        return struct.unpack(f"<{len(raw) // 2}e", raw)
    if dtype == "F32":
        return struct.unpack(f"<{len(raw) // 4}f", raw)
    return None


def weight_bits(config: dict[str, Any]) -> int | None:
    """Stored weight precision from config.json; None when not recognised."""
    quant = config.get("quantization_config")
    if not quant:
        dtype = str(config.get("torch_dtype") or config.get("dtype") or "bfloat16")
        return 32 if dtype == "float32" else 16
    method = str(quant.get("quant_method", ""))
    if method in ("fp8", "fbgemm_fp8"):
        return 8
    if method in ("gptq", "gptq_marlin", "auto_gptq", "awq", "awq_marlin", "auto_awq"):
        bits = quant.get("bits") or quant.get("w_bit")
        return int(bits) if bits else None
    if method == "fp_quant":
        return 4 if str(quant.get("forward_dtype", "")).lower() in ("mxfp4", "nvfp4") else None
    if method == "mxfp4":
        return 4
    if method == "modelopt":
        algo = str(quant.get("quant_algo", "")).upper()
        return 8 if "FP8" in algo else 4 if "FP4" in algo else None
    if method == "compressed-tensors":
        groups = (quant.get("config_groups") or {}).values()
        bits = [(g.get("weights") or {}).get("num_bits") for g in groups]
        ints = [b for b in bits if isinstance(b, int)]
        return min(ints) if ints and len(ints) == len(bits) else None
    if method == "bitsandbytes":
        return 4 if quant.get("load_in_4bit") else 8 if quant.get("load_in_8bit") else None
    return None


def shape_of(config: dict[str, Any]) -> dict[str, Any]:
    return {k: config.get(k) for k in SHAPE_FIELDS}


def _declared_base(card: dict[str, Any]) -> str | None:
    base = card.get("base_model")
    if isinstance(base, list):
        return str(base[0]) if len(base) == 1 else None
    return str(base) if base else None


class HubLineage:
    """Reads the Hub (the token, if any, comes from the environment)."""

    def __init__(self, api: Any = None, *, limit: int = 500) -> None:
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi()
        self._api = api
        self._limit = limit

    def config(self, model_id: str) -> dict[str, Any]:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model_id, "config.json")
        with open(path, encoding="utf-8") as fh:
            return dict(json.load(fh))

    def card(self, model_id: str) -> tuple[dict[str, Any], int]:
        info = self._api.model_info(model_id, expand=["cardData", "downloads"])
        card = info.card_data.to_dict() if info.card_data else {}
        return card, int(info.downloads or 0)

    def identity_tensor(self, model_id: str) -> tuple[float, ...] | None:
        """The final-norm weights, read by byte range (no weights downloaded)."""
        from huggingface_hub import get_safetensors_metadata, hf_hub_url
        from huggingface_hub.utils import build_hf_headers, get_session

        try:
            meta = get_safetensors_metadata(model_id)
        except Exception:
            return None
        name = next((n for n in _IDENTITY_TENSORS if n in meta.weight_map), None)
        if name is None:
            return None
        filename = meta.weight_map[name]
        info = meta.files_metadata[filename].tensors[name]
        url = hf_hub_url(model_id, filename)
        session, headers = get_session(), build_hf_headers()

        def byte_range(start: int, end: int) -> bytes:
            response = session.get(url, headers={**headers, "Range": f"bytes={start}-{end}"})
            response.raise_for_status()
            return response.content

        (header_len,) = struct.unpack("<Q", byte_range(0, 7))
        begin, end = info.data_offsets
        return decode_floats(
            byte_range(8 + header_len + begin, 8 + header_len + end - 1), info.dtype
        )

    def tensor_names(self, model_id: str) -> list[str] | None:
        """Every stored tensor's name, from the safetensors headers."""
        from huggingface_hub import get_safetensors_metadata

        try:
            return list(get_safetensors_metadata(model_id).weight_map)
        except Exception:
            return None

    def quantized_from(self, base_model_id: str) -> list[tuple[str, int, dict[str, Any]]]:
        """(id, downloads, listed config) of models the Hub marks as quantized from the base."""
        models = self._api.list_models(
            filter=f"base_model:quantized:{base_model_id}",
            sort="downloads",
            limit=self._limit,
            expand=["downloads", "config"],
        )
        return [(m.id, int(m.downloads or 0), dict(m.config or {})) for m in models]

    def search(
        self,
        model_id: str,
        propose_base: Callable[[str, dict[str, Any]], dict[str, Any]],
    ) -> tuple[ArtifactSearch | None, dict[str, Any]]:
        """The checkpoint's lineage and its loadable same-architecture siblings."""
        config = self.config(model_id)
        card, _ = self.card(model_id)
        evidence: dict[str, Any] = {"requested_model_id": model_id}
        requested_bits = weight_bits(config)
        evidence["requested_weight_bits"] = requested_bits

        base = _declared_base(card)
        if base:
            base_evidence: dict[str, Any] = {"id": base, "source": "card"}
        else:
            proposal = propose_base(model_id, config)
            base = proposal.get("base_model_id") or None
            base_evidence = {"id": base, "source": "proposed", "proposal": proposal}
        evidence["base_model"] = base_evidence
        if not base or base == model_id or requested_bits is None:
            evidence["result"] = "no base model identified"
            return None, evidence

        try:
            base_config = self.config(base)
            _, base_downloads = self.card(base)
        except Exception as exc:  # the Hub has no such model: the proposal is rejected
            evidence["result"] = f"base model not on the Hub: {type(exc).__name__}"
            return None, evidence
        base_norm = self.identity_tensor(base)
        same_weights = base_norm is not None and self.identity_tensor(model_id) == base_norm
        confirmed = (
            shape_of(base_config) == shape_of(config)
            and not base_config.get("quantization_config")
            and same_weights
        )
        base_evidence["confirmed"] = confirmed
        base_evidence["same_final_norm"] = same_weights
        if not confirmed:
            evidence["result"] = (
                "base model has another architecture, is itself quantized, or stores other weights"
            )
            return None, evidence

        candidates = [
            ArtifactCandidate(
                model_id=base,
                shape=shape_of(base_config),
                weight_bits=weight_bits(base_config) or 16,
                min_capability=0,
                quant_method=None,
                downloads=base_downloads,
            )
        ]
        dropped: Counter[str] = Counter()
        considered: list[dict[str, Any]] = []
        for repo, downloads, listed in self.quantized_from(base):
            method = str((listed.get("quantization_config") or {}).get("quant_method", ""))
            if method not in _LOADABLE:
                dropped[f"format not loadable by vLLM v0.29.0: {method or 'none'}"] += 1
                continue
            try:
                full = self.config(repo)
            except Exception as exc:
                dropped[f"config.json unreadable: {type(exc).__name__}"] += 1
                continue
            bits = weight_bits(full)
            needs = min_capability(full.get("quantization_config"))
            entry = {"model_id": repo, "method": method, "bits": bits, "min_capability": needs}
            considered.append(entry)
            if bits is None or needs is None:
                entry["dropped"] = "scheme not recognised"
                continue
            if shape_of(full) == shape_of(config) and self.identity_tensor(repo) != base_norm:
                entry["dropped"] = "final-norm weights differ from the base: another model"
                continue
            names = self.tensor_names(repo)
            refused = load_problems(names, full.get("quantization_config")) if names else None
            if refused:
                entry["dropped"] = "the engine would not load it: " + "; ".join(refused)
                continue
            candidates.append(
                ArtifactCandidate(
                    model_id=repo,
                    shape=shape_of(full),
                    weight_bits=bits,
                    min_capability=needs,
                    quant_method=method,
                    downloads=downloads,
                )
            )
        evidence["dropped_by_format"] = dict(dropped)
        evidence["considered"] = considered
        evidence["result"] = f"{len(candidates)} candidates"
        return (
            ArtifactSearch(
                requested_model_id=model_id,
                base_model_id=base,
                requested_weight_bits=requested_bits,
                requested_shape=shape_of(config),
                candidates=tuple(candidates),
            ),
            evidence,
        )
