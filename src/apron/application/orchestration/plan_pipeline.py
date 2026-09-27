"""Plan pipeline — orchestrate resolution → calculator → plan → render.

Extracted from CLI to satisfy INV-11: no recommendation, calculation,
ranking or promotion logic in interfaces.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from apron.application.orchestration.plan_builder import build_plan, served_dtype
from apron.application.orchestration.resolution import ResolutionChain
from apron.domain.mechanisms.model_spec_builder import build_model_spec
from apron.domain.schemas.solutions import DeploymentPlan, RenderContext

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from apron.domain.artifacts import ArtifactSourceObservation
    from apron.domain.ports import Clock, IdGenerator
    from apron.domain.protocols import ArtifactSourceResolver, PlanningSource
    from apron.domain.schemas.models import ModelSpec
    from apron.domain.schemas.primitives import HardwareSpec


class PlanPipelineResult:
    """Outcome of the plan pipeline.

    ``observation`` is the resolved artifact observation (F12), ``model_spec``
    the spec built from ``config.json``, and ``model_config`` the resolved
    ``config.json`` itself, which diagnosis needs (F5).  A prediction-error
    candidate (unknown mechanism) keeps ``claim``, ``model_spec`` and
    ``observation`` with ``error`` set, so the claim can be stored.
    """

    __slots__ = (
        "chat_renderer",
        "chat_template",
        "claim",
        "context",
        "error",
        "load_problems",
        "model_config",
        "model_spec",
        "observation",
        "plan",
    )

    def __init__(
        self,
        *,
        plan: DeploymentPlan | None = None,
        context: RenderContext | None = None,
        claim: Any = None,
        error: str | None = None,
        observation: ArtifactSourceObservation | None = None,
        model_spec: ModelSpec | None = None,
        model_config: dict[str, Any] | None = None,
        chat_template: str | None = None,
        chat_renderer: str | None = None,
        load_problems: tuple[str, ...] | None = None,
    ) -> None:
        self.plan = plan
        # Why the engine would refuse the checkpoint's tensors; () when it
        # would load them; None when no check applies (GPU-free, pre-boot).
        self.load_problems = load_problems
        self.context = context
        self.claim = claim
        self.error = error
        self.observation = observation
        self.model_spec = model_spec
        self.model_config = model_config
        self.chat_template = chat_template
        # Who turns chat messages into the prompt when the repo has no
        # template: the engine's own mode (None: the template does).
        self.chat_renderer = chat_renderer

    @property
    def ok(self) -> bool:
        return self.plan is not None and self.error is None


def run_plan_pipeline(
    resolver: ArtifactSourceResolver,
    planning_source: PlanningSource,
    model_id: str,
    hardware: HardwareSpec,
    *,
    clock: Clock,
    id_gen: IdGenerator,
    tensor_parallel: int = 1,
    max_num_batched_tokens: int | None = None,
    max_num_seqs: int | None = None,
    load_check: Callable[[Iterable[str], dict[str, Any] | None], tuple[str, ...] | None]
    | None = None,
    unmodelled_architectures: frozenset[str] = frozenset(),
    tokenizer_modes: frozenset[str] = frozenset(),
) -> PlanPipelineResult:
    """Run the full plan pipeline: resolve → calculate → build plan.

    ``tensor_parallel`` > 1 predicts per-GPU memory for a plan that already
    fixes TP (a fix proof); 1 predicts the whole model on one GPU.
    ``load_check`` (the engine adapter's) reads the checkpoint's tensor names
    and says whether the engine would load them, before any GPU is paid for.
    """
    chain = ResolutionChain(resolver)
    result = chain.resolve(model_id)

    if not result.ok:
        return PlanPipelineResult(error=str(result.error))

    assert result.observation is not None

    config_content = _download_config(resolver, model_id, result.observation.resolved_revision)
    if config_content is None:
        return PlanPipelineResult(error="config.json not found")

    config = json.loads(config_content)

    read_meta = getattr(resolver, "_tensor_meta", None)
    meta = read_meta(model_id, result.observation.resolved_revision) if read_meta else None
    if meta:
        tensors: dict[str, int] | None = loaded_tensor_bytes(meta, config)
    else:
        read_tensors = getattr(resolver, "_tensor_bytes", None)
        tensors = (
            read_tensors(model_id, result.observation.resolved_revision) if read_tensors else None
        )
    total_weight_bytes = _resolve_weight_bytes(
        resolver, model_id, result.observation, config, tensors=tensors
    )
    load_problems = (
        load_check(tensors, config.get("quantization_config"))
        if load_check is not None and tensors
        else None
    )

    model_spec = build_model_spec(
        config,
        repository=model_id,
        revision=result.observation.resolved_revision,
        license_id=result.observation.license_observed,
        unmodelled_architectures=unmodelled_architectures,
    )

    # A multimodal checkpoint keeps its language model in ``text_config``;
    # vLLM sizes caches from it (``hf_text_config``).
    text = config.get("text_config")
    calc_metadata = {**config, **text} if isinstance(text, dict) else dict(config)
    calc_metadata["torch_dtype"] = runtime_dtype(config)
    calc_metadata["total_weight_bytes"] = total_weight_bytes
    calc_metadata["components"] = [c.model_dump(mode="json") for c in model_spec.components]

    claim = planning_source.predict(
        calc_metadata,
        hardware,
        {
            "isl": 512,
            "osl": 128,
            "max_batch_size": 4,
            "tensor_parallel": tensor_parallel,
            # The engine's profiling run size on this GPU (sizes the activation peak).
            **(
                {"max_num_batched_tokens": max_num_batched_tokens}
                if max_num_batched_tokens
                else {}
            ),
            **({"max_num_seqs": max_num_seqs} if max_num_seqs else {}),
        },
    )

    if claim.proposed_configuration.get("status") == "unknown":
        return PlanPipelineResult(
            claim=claim,
            error="unknown model mechanism — calculator cannot predict memory",
            observation=result.observation,
            model_spec=model_spec,
            model_config=config,
            load_problems=load_problems,
        )

    deployment_plan = build_plan(
        claim,
        model_spec,
        hardware,
        result.execution_spec,
        None,
        clock=clock,
        id_gen=id_gen,
    )

    # A model type the engine has its own tokenizer mode for (no chat template
    # in the repo): the plan names that mode, and the engine renders the chat.
    chat_renderer = None
    model_type = str(config.get("model_type") or "")
    if model_type in tokenizer_modes:
        chat_renderer = f"vllm tokenizer mode {model_type}"
        deployment_plan = deployment_plan.model_copy(
            update={
                "engine_configuration": {
                    **deployment_plan.engine_configuration,
                    "tokenizer_mode": model_type,
                }
            }
        )

    assert result.locator is not None
    ctx = RenderContext(plan=deployment_plan, locator=result.locator, hardware=hardware)

    return PlanPipelineResult(
        plan=deployment_plan,
        context=ctx,
        claim=claim,
        observation=result.observation,
        model_spec=model_spec,
        model_config=config,
        load_problems=load_problems,
        chat_template=_download_chat_template(
            resolver, model_id, result.observation.resolved_revision
        ),
        chat_renderer=chat_renderer,
    )


def _download_config(resolver: Any, model_id: str, revision: str) -> bytes | None:
    if hasattr(resolver, "_download_file"):
        return resolver._download_file(model_id, "config.json", revision)
    return None


def _download_chat_template(resolver: Any, model_id: str, revision: str) -> str | None:
    """The model's chat template: ``chat_template.jinja``, else the
    ``chat_template`` field of ``tokenizer_config.json`` (a string or a list
    of named templates, the ``default`` one first).  ``None`` if absent."""
    if not hasattr(resolver, "_download_file"):
        return None
    jinja = resolver._download_file(model_id, "chat_template.jinja", revision)
    if jinja is not None:
        return jinja.decode("utf-8")
    raw = resolver._download_file(model_id, "tokenizer_config.json", revision)
    if raw is None:
        return None
    template = json.loads(raw).get("chat_template")
    if isinstance(template, list):
        named = {t.get("name"): t.get("template") for t in template if isinstance(t, dict)}
        template = named.get("default") or next(iter(named.values()), None)
    return template if isinstance(template, str) else None


# Stored bytes per element, by safetensors dtype name.
SAFETENSORS_DTYPE_BYTES: dict[str, int] = {
    "F64": 8,
    "I64": 8,
    "U64": 8,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "BF16": 2,
    "F16": 2,
    "I16": 2,
    "U16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


def loaded_tensor_bytes(
    meta: dict[str, tuple[str, int]], config: dict[str, Any]
) -> dict[str, int]:
    """Bytes each tensor occupies once the engine has loaded it.

    With the default ``dtype=auto`` vLLM serves a float32 checkpoint in a
    16-bit dtype (config/model.py:2285-2287: "Downcast for float32 models";
    bfloat16 from compute capability 8.0, else float16 — 2 bytes either way,
    platforms/cuda.py:237-243), so an unquantized checkpoint's F32 tensors
    load at half their stored size.  Quantized checkpoints keep their stored
    sizes (their scales are created in float32 by the method).
    """
    if config.get("quantization_config"):
        return {name: size for name, (_, size) in meta.items()}
    return {name: size // 2 if dtype == "F32" else size for name, (dtype, size) in meta.items()}


def runtime_dtype(config: dict[str, Any]) -> str:
    """The dtype the engine serves in with dtype=auto (float32 downcast to 16 bits)."""
    # Newer configs name it "dtype" (transformers 5); vLLM reads either.
    return served_dtype(config.get("torch_dtype") or config.get("dtype"))


def _tied_embeddings(config: dict[str, Any]) -> bool:
    text = config.get("text_config")
    nested = text if isinstance(text, dict) else {}
    return bool(config.get("tie_word_embeddings", nested.get("tie_word_embeddings", False)))


def _resolve_weight_bytes(
    resolver: Any,
    model_id: str,
    observation: Any,
    config: dict[str, Any] | None = None,
    *,
    tensors: dict[str, int] | None = None,
) -> int:
    """Bytes the engine loads: every stored tensor at its stored dtype.

    1. The safetensors headers (exact per-tensor bytes).  With tied word
       embeddings the engine shares one matrix, so a stored ``lm_head.weight``
       is not loaded twice (Qwen3-1.7B: predicted 3.78 GiB, measured 3.22 —
       the 0.58 GiB lm_head; L5, 2026-09-27).
    2. The index's ``total_size``.
    3. The Hub's per-dtype parameter counts, every dtype summed at its width
       (the old path counted only the first dtype: Qwen3-0.6B-FP8 lost its
       FP8 half).  An unknown dtype makes the total unknown (0), never a guess.
    """
    rev = observation.resolved_revision
    if tensors is None and hasattr(resolver, "_tensor_bytes"):
        tensors = resolver._tensor_bytes(model_id, rev)
    if tensors:
        total = sum(tensors.values())
        if _tied_embeddings(config or {}) and "lm_head.weight" in tensors:
            total -= tensors["lm_head.weight"]
        return total

    if hasattr(resolver, "_download_file"):
        index_content = resolver._download_file(model_id, "model.safetensors.index.json", rev)
        if index_content is not None:
            total_size = json.loads(index_content).get("metadata", {}).get("total_size", 0)
            if total_size:
                return int(total_size)

    total_weight_bytes = 0
    for key, val in (observation.publisher_metadata or {}).items():
        if key.startswith("parameters_"):
            width = SAFETENSORS_DTYPE_BYTES.get(key.split("_", 1)[1])
            if width is None:
                return 0
            total_weight_bytes += int(val) * width
    return total_weight_bytes


def recorded_loaded_bytes(entry: dict[str, Any], plan_dtype: str | None) -> int:
    """Weights a recorded checkpoint loads at under a plan's dtype, through the
    planner's own resolution (tests/fixtures/cohort/weight-bytes.json rows)."""
    total, f32, lm_head = (int(entry[k]) for k in ("total_bytes", "f32_bytes", "lm_head_bytes"))
    meta = {"lm_head.weight": ("BF16", lm_head)} if lm_head else {}
    meta |= {"f32": ("F32", f32), "rest": ("BF16", total - lm_head - f32)}
    config: dict[str, Any] = {"tie_word_embeddings": entry["tie_word_embeddings"]}
    if entry["quantized"]:
        config["quantization_config"] = {"quant_method": "recorded"}
    tensors = (
        {name: size for name, (_, size) in meta.items()}
        if plan_dtype == "float32"
        else loaded_tensor_bytes(meta, config)
    )
    return _resolve_weight_bytes(None, "recorded", _RecordedObservation(), config, tensors=tensors)


class _RecordedObservation:
    resolved_revision = "recorded"
    publisher_metadata = None
