"""Plan pipeline — orchestrate resolution → calculator → plan → render.

Extracted from CLI to satisfy INV-11: no recommendation, calculation,
ranking or promotion logic in interfaces.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from apron.application.orchestration.plan_builder import (
    PLAN_GPU_MEMORY_UTILIZATION,
    build_plan,
    served_dtype,
)
from apron.application.orchestration.resolution import ResolutionChain
from apron.domain.mechanisms.calculator import (
    ENCODER_FIELDS,
    activation_config,
    declares_towers,
    encoder_peak_note,
)
from apron.domain.mechanisms.layered import (
    KV_BUDGET_SAFETY_BUFFER_BYTES,
    required_block_size,
    state_block_capacity,
)
from apron.domain.mechanisms.model_spec_builder import build_model_spec
from apron.domain.schemas.solutions import DeploymentPlan, RenderContext

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from apron.domain.artifacts import ArtifactSourceObservation
    from apron.domain.ports import Clock, IdGenerator
    from apron.domain.protocols import ArtifactSourceResolver, PlanningSource
    from apron.domain.schemas.models import ModelSpec
    from apron.domain.schemas.primitives import HardwareSpec


#: How a plan names a chat the engine renders in its own tokenizer mode
#: (``PlanPipelineResult.chat_renderer``): this prefix, then the mode.
ENGINE_RENDERER_PREFIX = "vllm tokenizer mode "


class PlanPipelineResult:
    """Outcome of the plan pipeline.

    ``observation`` is the resolved artifact observation (F12), ``model_spec``
    the spec built from ``config.json``, and ``model_config`` the resolved
    ``config.json`` itself, which diagnosis needs (F5).  A prediction-error
    candidate (unknown mechanism) keeps ``claim``, ``model_spec`` and
    ``observation`` with ``error`` set, so the claim can be stored.
    ``host_memory_bytes`` are the checkpoint's tables the engine keeps in
    pinned host memory (all ranks together), and ``notes`` say what the
    prediction needs besides the GPUs.
    """

    __slots__ = (
        "chat_renderer",
        "chat_template",
        "claim",
        "context",
        "error",
        "host_memory_bytes",
        "load_problems",
        "model_config",
        "model_spec",
        "notes",
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
        host_memory_bytes: int = 0,
        notes: tuple[str, ...] = (),
    ) -> None:
        self.plan = plan
        self.host_memory_bytes = host_memory_bytes
        self.notes = notes
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


@dataclass(frozen=True)
class ServingFeatures:
    """What a plan serves besides chat at the smallest context (task suite v3).

    Suite v3's deployment checks send a long-context needle, tool calls and a
    reasoning split, so a plan that runs them starts the server for them:
    ``max_model_len`` tokens of context, ``--enable-auto-tool-choice
    --tool-call-parser`` where a parser is known (``tool_call_parser_for``),
    and the reasoning parser also for a chat template with a thinking switch
    (the checks switch thinking per request).  The rest are the plan's engine
    version's facts: the tool parsers it registers, the one its recipes name
    per architecture, and each recipe checkpoint's own parsers.  A plan
    without features (suite v2's) is built exactly as before.
    """

    max_model_len: int
    tool_parsers: frozenset[str] = frozenset()
    tool_parser_architectures: Mapping[str, str] = field(default_factory=dict)
    recipe_checkpoints: Mapping[str, Mapping[str, str]] = field(default_factory=dict)


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
    reasoning_parsers: frozenset[str] = frozenset(),
    reasoning_parser_architectures: Mapping[str, str] | None = None,
    state_blocks: StateBlockFacts | None = None,
    gpu_memory_utilization: float = PLAN_GPU_MEMORY_UTILIZATION,
    dtype: str | None = None,
    features: ServingFeatures | None = None,
) -> PlanPipelineResult:
    """Run the full plan pipeline: resolve → calculate → build plan.

    ``tensor_parallel`` > 1 predicts per-GPU memory for a plan that already
    fixes TP (a fix proof); 1 predicts the whole model on one GPU.
    ``load_check`` (the engine adapter's) reads the checkpoint's tensor names
    and says whether the engine would load them, before any GPU is paid for.
    ``max_num_seqs`` is the engine's default on this GPU; ``state_blocks``
    (the engine version's facts) says where that version needs one state
    block per decode sequence, and the plan lowers ``max_num_seqs`` to what
    the predicted cache holds (``state_block_limit``).  ``tokenizer_modes``
    and ``reasoning_parsers`` are the engine version's registries, and
    ``reasoning_parser_architectures`` the parser that version serves each
    architecture with (``reasoning_parser_for`` says when a plan names one).
    ``gpu_memory_utilization`` and ``dtype`` are what the plan boots with: the
    planner's own plans set ``PLAN_GPU_MEMORY_UTILIZATION`` and serve the
    checkpoint's runtime dtype (``dtype=None``); a given plan (a fix proof) is
    predicted with its own values, or vLLM's defaults where it sets none.
    A multimodal wrapper (a ``vision_config`` / ``audio_config``) also has its
    processor files read at the resolved revision (``download_processor_config``):
    they size the vision tower's startup peak; where that peak is not modelled
    the notes say so (``encoder_peak_note``).  ``features`` (task suite v3)
    predicts and plans at its ``max_model_len`` and adds the tool-call and
    reasoning parsers the deployment checks need (``ServingFeatures``).
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
    if meta and dtype == "float32":
        # Served in float32: every tensor loads at its stored size (the
        # 16-bit downcast of ``loaded_tensor_bytes`` is dtype=auto's).
        tensors: dict[str, int] | None = {name: size for name, (_, size) in meta.items()}
    elif meta:
        tensors = loaded_tensor_bytes(meta, config)
    else:
        read_tensors = getattr(resolver, "_tensor_bytes", None)
        tensors = (
            read_tensors(model_id, result.observation.resolved_revision) if read_tensors else None
        )
    total_weight_bytes = _resolve_weight_bytes(
        resolver, model_id, result.observation, config, tensors=tensors
    )
    host_bytes = host_tensor_bytes(tensors, config) if tensors else 0
    notes = _host_memory_notes(host_bytes, config, tensor_parallel)
    # Only a multimodal wrapper's processor files size anything: a text-only
    # checkpoint costs no extra requests.
    processor = (
        download_processor_config(resolver, model_id, result.observation.resolved_revision)
        if declares_towers(config)
        else None
    )
    encoder_note = encoder_peak_note(config, processor)
    if encoder_note is not None:
        notes = (*notes, encoder_note)
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

    calc_metadata = calculator_metadata(
        config,
        model_spec,
        total_weight_bytes=total_weight_bytes,
        replicated_weight_bytes=replicated_tensor_bytes(tensors, config) if tensors else 0,
        kv_head_weight_bytes=kv_head_tensor_bytes(tensors, config) if tensors else 0,
        dtype=dtype,
        processor=processor,
    )

    # The context the plan serves: 640 tokens (512 in, 128 out) unless the
    # plan serves features, at most what the checkpoint's positions reach.
    context = None
    if features is not None:
        context, context_note = served_context(features.max_model_len, config)
        if context_note is not None:
            notes = (*notes, context_note)

    def predict(seqs: int | None) -> Any:
        return planning_source.predict(
            calc_metadata,
            hardware,
            {
                "isl": 512 if context is None else context - 128,
                "osl": 128,
                # The calculator reads max_model_len where vLLM sizes by it
                # (MLA prefill workspace, encoder budget); unset, isl + osl.
                # At a served context the plan must hold what vLLM checks at
                # startup: one request of max_model_len
                # (v1/core/kv_cache_utils.py:975-1010, "To serve at least one
                # request"); four such requests is a bar vLLM never sets.
                **({"max_model_len": context} if context is not None else {}),
                "max_batch_size": 4 if context is None else 1,
                "tensor_parallel": tensor_parallel,
                "gpu_memory_utilization": gpu_memory_utilization,
                # The engine's profiling run size on this GPU (sizes the activation peak).
                **(
                    {"max_num_batched_tokens": max_num_batched_tokens}
                    if max_num_batched_tokens
                    else {}
                ),
                **({"max_num_seqs": seqs} if seqs else {}),
            },
        )

    claim = predict(max_num_seqs)
    limit = state_block_limit(claim.proposed_configuration, max_num_seqs, state_blocks)
    if limit is not None:
        # The prediction the plan is booted with: fewer sequences never raise
        # the activation, so the cache still holds them.
        claim = predict(limit[0])
        notes = (*notes, limit[1])

    if claim.proposed_configuration.get("status") == "unknown":
        return PlanPipelineResult(
            claim=claim,
            error="unknown model mechanism — calculator cannot predict memory",
            observation=result.observation,
            model_spec=model_spec,
            model_config=config,
            load_problems=load_problems,
            host_memory_bytes=host_bytes,
            notes=notes,
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

    if context is not None:
        deployment_plan = deployment_plan.model_copy(
            update={
                "engine_configuration": {
                    **deployment_plan.engine_configuration,
                    "max_model_len": str(context),
                }
            }
        )

    if limit is not None:
        deployment_plan = deployment_plan.model_copy(
            update={
                "engine_configuration": {
                    **deployment_plan.engine_configuration,
                    "max_num_seqs": str(limit[0]),
                }
            }
        )

    # A model type the engine has its own tokenizer mode for (no chat template
    # in the repo): the plan names that mode, and the engine renders the chat.
    chat_renderer = None
    model_type = str(config.get("model_type") or "")
    if model_type in tokenizer_modes:
        chat_renderer = f"{ENGINE_RENDERER_PREFIX}{model_type}"
        deployment_plan = deployment_plan.model_copy(
            update={
                "engine_configuration": {
                    **deployment_plan.engine_configuration,
                    "tokenizer_mode": model_type,
                }
            }
        )

    # A model vLLM cannot boot at its own block size: the plan names the one
    # that works (MiniMax-M3's 128-token sparse pages, ``required_block_size``).
    block_size = required_block_size(config)
    if block_size is not None:
        deployment_plan = deployment_plan.model_copy(
            update={
                "engine_configuration": {
                    **deployment_plan.engine_configuration,
                    "block_size": str(block_size),
                }
            }
        )

    chat_template = _download_chat_template(
        resolver, model_id, result.observation.resolved_revision
    )
    # A model that answers in a reasoning channel the engine has a parser for:
    # without the parser vLLM returns the whole generation as content, framing
    # stripped (Muse-Glimmer-30B, 2026-09-28: every answer right, every case
    # scored wrong).  The plan names the parser; the engine splits the channels.
    architectures = [str(a) for a in config.get("architectures") or []]
    served_with = reasoning_parser_architectures or {}
    recipe = dict((features.recipe_checkpoints.get(model_id) or {}) if features else {})
    engine_renders_chat = chat_template is None and chat_renderer is not None
    if (
        served_parser(
            model_type,
            architectures,
            reasoning_parsers,
            served_with,
            checkpoint_parser=recipe.get("reasoning_parser"),
        )
        is not None
    ):
        parser = reasoning_parser_for(
            model_type,
            reasoning_parsers,
            chat_template,
            _download_json(
                resolver, model_id, "tokenizer_config.json", result.observation.resolved_revision
            ),
            architectures=architectures,
            served_with=served_with,
            engine_renders_chat=engine_renders_chat,
            checkpoint_parser=recipe.get("reasoning_parser"),
            thinking_per_request=features is not None,
        )
        if parser is not None:
            deployment_plan = deployment_plan.model_copy(
                update={
                    "engine_configuration": {
                        **deployment_plan.engine_configuration,
                        "reasoning_parser": parser,
                    }
                }
            )

    if features is not None:
        card = _download_text(
            resolver, model_id, "README.md", result.observation.resolved_revision
        )
        tool_parser, tool_reason = tool_call_parser_for(
            model_type,
            features.tool_parsers,
            chat_template,
            card,
            architectures=architectures,
            served_with=features.tool_parser_architectures,
            checkpoint_parser=recipe.get("tool_call_parser"),
            engine_renders_chat=engine_renders_chat,
        )
        if tool_parser is not None:
            deployment_plan = with_tool_calling(deployment_plan, tool_parser)
        else:
            notes = (*notes, f"no tool calling: {tool_reason}")

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
        chat_template=chat_template,
        chat_renderer=chat_renderer,
        host_memory_bytes=host_bytes,
        notes=notes,
    )


def calculator_metadata(
    config: dict[str, Any],
    model_spec: ModelSpec,
    *,
    total_weight_bytes: int,
    replicated_weight_bytes: int = 0,
    kv_head_weight_bytes: int = 0,
    dtype: str | None = None,
    processor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """What the calculator reads for one checkpoint: ``config.json`` with the
    language model's ``text_config`` merged over it (vLLM sizes caches from
    ``hf_text_config``), the dtype it is served in, the loaded weight bytes
    and the mechanisms.  ``dtype`` None: the runtime dtype of dtype=auto.
    *processor* (``download_processor_config``) adds the fields that size a
    traced vision tower's dummy encoder batch (``ENCODER_FIELDS``)."""
    text = config.get("text_config")
    metadata = {**config, **text} if isinstance(text, dict) else dict(config)
    if processor is not None:
        encoder = activation_config(config, processor)
        metadata.update({k: encoder[k] for k in ENCODER_FIELDS if k in encoder})
    metadata["torch_dtype"] = dtype or runtime_dtype(config)
    metadata["total_weight_bytes"] = total_weight_bytes
    if replicated_weight_bytes:
        metadata["replicated_weight_bytes"] = replicated_weight_bytes
    if kv_head_weight_bytes:
        metadata["kv_head_weight_bytes"] = kv_head_weight_bytes
    metadata["components"] = [c.model_dump(mode="json") for c in model_spec.components]
    return metadata


@dataclass(frozen=True)
class StateBlockFacts:
    """Where one engine version needs a state (Mamba) block per decode sequence
    (the version's generated facts; ``None`` / empty where it has none)."""

    engine_version: str
    check: str | None  # the check that refuses more sequences than blocks
    profiling: tuple[str, ...] = ()  # CUDA-graph profiling: one block per sequence
    default_source: str | None = None  # where the default max_num_seqs is set


_GIB = 1 << 30


def state_block_limit(
    predicted: dict[str, Any], default_seqs: int | None, facts: StateBlockFacts | None
) -> tuple[int, str] | None:
    """``max_num_seqs`` the plan must set, and why; None to keep the default.

    A state cache serves one decode sequence per KV block.  With full CUDA
    graphs vLLM refuses ``max_num_seqs`` above the pool's blocks
    (``facts.check``), and its CUDA-graph profiling allocates
    ``min(max_num_seqs, max_cudagraph_capture_size)`` blocks before the real
    cache (``facts.profiling``): Nemotron-3.5 on an H100 failed the first
    (1024 > 733 blocks, vLLM v0.30.0), Qwen3.8-27B the second (512 blocks x
    51,380,224 bytes, vLLM v0.29.0).  The pool is ``available //
    bytes_per_block``; the predicted available memory less the safety buffer
    (``KV_BUDGET_SAFETY_BUFFER_BYTES``, the largest error measured over every
    healthy record) and less the compile segment vLLM may hold for this plan
    (``compile_segment_bytes``) bounds it from below.  A default at or under
    that bound satisfies both, since the profiling blocks never exceed
    ``max_num_seqs``.
    """
    per_block = int(predicted.get("kv_bytes_per_block") or 0)
    available = predicted.get("available_kv_cache_bytes")
    if (
        facts is None
        or not (facts.check or facts.profiling)
        or not default_seqs
        or per_block <= 0
        or not isinstance(available, int | float)
    ):
        return None
    segment = int(predicted.get("compile_segment_bytes") or 0)
    margin = KV_BUDGET_SAFETY_BUFFER_BYTES + segment
    capacity = state_block_capacity(int(available), per_block, margin_bytes=margin)
    if default_seqs <= capacity:
        return None
    seqs = max(1, capacity)
    blocks = max(0, int(available)) // per_block
    sites = ", ".join(s for s in (facts.check, *facts.profiling) if s)
    default_at = f", {facts.default_source}" if facts.default_source else ""
    held = f", plus a {segment / _GIB:.2f} GiB compile segment vLLM may hold" if segment else ""
    note = (
        f"max_num_seqs {seqs}: vLLM {facts.engine_version} defaults to {default_seqs} on this "
        f"GPU{default_at}, but a state (Mamba) cache serves one decode sequence per KV block "
        f"and the predicted {int(available) / _GIB:.2f} GiB of KV memory holds {blocks} blocks "
        f"of {per_block:,} bytes, {capacity} after a {margin / _GIB:.2f} GiB safety buffer "
        f"(the largest KV-budget error measured over the cohort's healthy boots, "
        f"{KV_BUDGET_SAFETY_BUFFER_BYTES / _GIB:.2f} GiB{held}); vLLM refuses more sequences "
        f"than blocks with full CUDA graphs and allocates one block per sequence to profile "
        f"them ({sites})"
    )
    return seqs, note


def _host_memory_notes(
    host_bytes: int, config: dict[str, Any], tensor_parallel: int
) -> tuple[str, ...]:
    """What the pod needs in host RAM for tables the engine keeps off the GPU."""
    if not host_bytes:
        return ()
    ranks = max(1, tensor_parallel)
    return (
        f"host RAM: {host_bytes / 2**30:.2f} GiB of n-gram embedding tables stay in pinned "
        f"host memory, {host_bytes / ranks / 2**30:.2f} GiB per tensor-parallel rank x {ranks} "
        "(vLLM v0.30.0 VLLM_PLE_CPU_OFFLOAD=1 default, config/engram.py:40-47); "
        "not counted as GPU weights; the pod needs this RAM besides the GPUs",
    )


#: The processor files vLLM's multimodal processors read, and where
#: ``activation_config`` expects each: processor_config.json holds both
#: processors; the older layout has one file each.
PROCESSOR_FILES: tuple[tuple[str, str | None], ...] = (
    ("processor_config.json", None),
    ("preprocessor_config.json", "image_processor"),
    ("video_preprocessor_config.json", "video_processor"),
)


def download_processor_config(
    resolver: Any, model_id: str, revision: str
) -> dict[str, Any] | None:
    """The checkpoint's processor configuration in the shape
    ``activation_config`` takes: processor_config.json's ``image_processor`` /
    ``video_processor``, else preprocessor_config.json and
    video_preprocessor_config.json under those keys.  None when there is none."""
    found: dict[str, Any] = {}
    for filename, key in PROCESSOR_FILES:
        data = _download_json(resolver, model_id, filename, revision)
        if data is None:
            continue
        if key is None:
            found.update({k: data[k] for k in ("image_processor", "video_processor") if k in data})
        elif key not in found:
            found[key] = data
    return found or None


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


def _download_json(
    resolver: Any, model_id: str, filename: str, revision: str
) -> dict[str, Any] | None:
    if not hasattr(resolver, "_download_file"):
        return None
    raw = resolver._download_file(model_id, filename, revision)
    if raw is None:
        return None
    data = json.loads(raw)
    return data if isinstance(data, dict) else None


def _download_text(resolver: Any, model_id: str, filename: str, revision: str) -> str | None:
    if not hasattr(resolver, "_download_file"):
        return None
    raw = resolver._download_file(model_id, filename, revision)
    if raw is None:
        return None
    return raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)


def served_context(requested: int, config: Mapping[str, Any]) -> tuple[int, str | None]:
    """The ``max_model_len`` a plan serves for *requested*, and a note when it
    is less: vLLM refuses a length past the checkpoint's position limit
    (``max_position_embeddings``, top level or in ``text_config``) unless
    told to override it, so the plan serves that limit instead."""
    text = config.get("text_config")
    nested = text if isinstance(text, dict) else {}
    limit = config.get("max_position_embeddings") or nested.get("max_position_embeddings")
    if limit and int(limit) < requested:
        return int(limit), (
            f"max_model_len {int(limit)}: the checkpoint's max_position_embeddings "
            f"(requested {requested})"
        )
    return requested, None


def declares_reasoning_channel(
    chat_template: str | None, tokenizer_config: dict[str, Any] | None
) -> bool:
    """Whether the model says it answers with a separate reasoning channel.

    Either its tokenizer config's response schema has a ``reasoning_content``
    field (Muse-Glimmer: ``to=self<|message|>`` .. ``<|eom|>``), or its chat
    template renders an assistant turn's ``reasoning_content``.
    """
    template = (tokenizer_config or {}).get("response_template")
    fields = template.get("fields") if isinstance(template, dict) else None
    if isinstance(fields, dict) and "reasoning_content" in fields:
        return True
    return bool(chat_template and "reasoning_content" in chat_template)


def served_parser(
    model_type: str,
    architectures: Sequence[str],
    reasoning_parsers: frozenset[str],
    served_with: Mapping[str, str],
    *,
    checkpoint_parser: str | None = None,
) -> str | None:
    """The reasoning parser the engine version serves this model with, or None.

    The architecture's own (``served_with``: the parser the version's example
    checkpoint for it is served with, from the engine facts) comes first:
    GLM-5.x is ``glm_moe_dsa`` / ``glm5_next`` and served with ``glm45``,
    MiniMax-M3 is ``minimax_m3_vl`` and served with ``minimax_m3``.  Otherwise
    a parser registered under the model's type (Muse-Glimmer's
    ``muse_glimmer``).  Then the checkpoint's own recipe
    (``checkpoint_parser``), for an architecture the version's test registry
    has no example checkpoint of (Qwen4Exp: Qwen3.8-Flash-Next).  Each must be
    a parser the version registers.
    """
    for arch in architectures:
        parser = served_with.get(arch)
        if parser is not None:
            return parser if parser in reasoning_parsers else None
    if model_type in reasoning_parsers:
        return model_type
    return checkpoint_parser if checkpoint_parser in reasoning_parsers else None


def reasoning_parser_for(
    model_type: str,
    reasoning_parsers: frozenset[str],
    chat_template: str | None,
    tokenizer_config: dict[str, Any] | None,
    *,
    architectures: Sequence[str] = (),
    served_with: Mapping[str, str] | None = None,
    engine_renders_chat: bool = False,
    checkpoint_parser: str | None = None,
    thinking_per_request: bool = False,
) -> str | None:
    """The engine reasoning parser a plan names for this model, or None.

    All three must hold: the plan's engine version serves the model with a
    parser it registers (``served_parser``; vLLM turns none on by itself,
    config/reasoning.py:22); the template has no ``enable_thinking`` (the
    request switches thinking off there, and the answer is already the
    content); and the model declares a reasoning channel
    (``declares_reasoning_channel``).  A parser match alone is not enough:
    ``mistral`` names a parser and Mistral-7B-Instruct has no reasoning
    channel.  Where the repo has no template and the engine renders the chat
    in its own tokenizer mode, that renderer is the template and the parser
    comes with it: DeepSeek V4 / V4.1's opens ``<think>`` unless the request
    switches thinking off (v0.30.0 tokenizers/deepseek_v4.py:30-35,
    deepseek_v41.py:70-72), and their parsers start in reasoning on the same
    default (parser/deepseek_v4.py:248-253).  A renderer that does not think
    by default is served with a parser that passes the content through
    (DeepSeek V3.2's ``deepseek_v3``: reasoning/deepseek_v3_reasoning_parser.py:30-38).
    ``thinking_per_request`` (suite v3, whose reasoning split switches
    thinking on while its other checks switch it off) names the parser for a
    template with a thinking switch too: the parser reads the request's
    ``enable_thinking`` and passes the content through when it is off
    (v0.30.0 parser/qwen3.py:231, 255-256); a parser that did not would show
    as the other checks' answers in the reasoning field, never silently.
    """
    parser = served_parser(
        model_type,
        architectures,
        reasoning_parsers,
        served_with or {},
        checkpoint_parser=checkpoint_parser,
    )
    if parser is None:
        return None
    if engine_renders_chat:
        return parser
    if chat_template and "enable_thinking" in chat_template and not thinking_per_request:
        return None
    if not declares_reasoning_channel(chat_template, tokenizer_config):
        return None
    return parser


# ---------------------------------------------------------------------------
# Tool calling: the plan-side rule (task suite v3)
# ---------------------------------------------------------------------------

#: Parsers vLLM documents for a model type whose name is not a parser name.
#: Each entry is checked against the plan's engine version's registry before
#: use.  ``gpt_oss`` -> ``openai``: docs/features/tool_calling.md, "OpenAI OSS
#: Models (`openai`)", openai/gpt-oss-20b and -120b (v0.30.0, ced6857).
DOCUMENTED_TOOL_PARSERS: dict[str, str] = {"gpt_oss": "openai"}

#: A chat template that renders past tool calls as
#: ``<tool_call>\n<function=NAME>\n<parameter=...>``: the format whose model
#: cards name ``qwen3_coder`` (Qwen/Qwen3.6-35B-A3B-FP8 and
#: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16, both with exactly this
#: template block).  An inference from those two cards, applied only to a
#: template with the same block and no recipe or card that names a parser.
_QWEN3_CODER_BLOCK = re.compile(r"<tool_call>(?:\\n|\n)<function=.*<parameter=", re.DOTALL)

_CARD_PARSER = re.compile(r"--tool-call-parser[ =]+([A-Za-z0-9_.-]+)")


def tool_call_parser_for(
    model_type: str,
    tool_parsers: frozenset[str],
    chat_template: str | None,
    model_card: str | None = None,
    *,
    architectures: Sequence[str] = (),
    served_with: Mapping[str, str] | None = None,
    checkpoint_parser: str | None = None,
    engine_renders_chat: bool = False,
) -> tuple[str | None, str]:
    """The tool-call parser a plan names for this model, or None; and why.

    The first that the plan's engine version registers
    (``EngineFacts.tool_parsers``):

    1. the parser the checkpoint's own vllm-project recipe serves it with
       (``checkpoint_parser``; ``features.tool_calling.args``);
    2. the parser the recipes name for its architecture (``served_with``,
       joined on the version's example checkpoints like the reasoning parser);
    3. the parser the model card names (``--tool-call-parser X``), when it
       names exactly one;
    4. the model type itself (``gemma4``, ``muse_glimmer``, ``minimax_m3``);
    5. vLLM's documented parser for the type (``DOCUMENTED_TOOL_PARSERS``);
    6. ``qwen3_coder`` for a template with that format's block.

    The chat template must render the request's ``tools`` (otherwise the
    model is never told about them).  Where the repo has no template and the
    engine renders the chat in its own tokenizer mode, only a recipe says the
    renderer takes tools (DeepSeek V4's recipe serves ``--tokenizer-mode
    deepseek_v4 --tool-call-parser deepseek_v4``), so rules 3-6 do not apply.
    Unlike the reasoning parser, a type match is not required: tool-parser
    names are formats (``hermes``, ``openai``, ``qwen3_coder``), not types.
    """
    recipe_rules: list[tuple[str | None, str]] = [
        (checkpoint_parser, "the checkpoint's vllm-project recipe serves it with"),
        *(
            (
                (served_with or {}).get(arch),
                f"the vllm-project recipes serve architecture {arch} with",
            )
            for arch in architectures
        ),
    ]
    for parser, source in recipe_rules:
        if parser is None:
            continue
        if parser not in tool_parsers:
            return None, f"{source} {parser}, which this engine version does not register"
        if not engine_renders_chat and not (chat_template and "tools" in chat_template):
            return None, "the chat template does not render tools"
        return parser, f"{source} --enable-auto-tool-choice --tool-call-parser {parser}"
    if not chat_template or "tools" not in chat_template:
        return None, "the chat template does not render tools"
    if model_card:
        named = sorted(set(_CARD_PARSER.findall(model_card)))
        if len(named) == 1 and named[0] in tool_parsers:
            return named[0], f"the model card names --tool-call-parser {named[0]}"
        if len(named) > 1:
            return None, f"the model card names several tool parsers: {named}"
    if model_type in tool_parsers:
        return model_type, f"the engine registers a tool parser under model type {model_type}"
    documented = DOCUMENTED_TOOL_PARSERS.get(model_type)
    if documented and documented in tool_parsers:
        return documented, f"vLLM documents --tool-call-parser {documented} for {model_type}"
    if "qwen3_coder" in tool_parsers and _QWEN3_CODER_BLOCK.search(chat_template):
        return "qwen3_coder", "the template renders the qwen3_coder tool-call block"
    return None, f"no tool parser is known for model type {model_type}"


def with_tool_calling(plan: DeploymentPlan, parser: str) -> DeploymentPlan:
    """*plan* started with ``--enable-auto-tool-choice --tool-call-parser parser``.

    ``engine_flag_args`` renders ``"true"`` as the bare flag.  The flags
    change no memory the calculator predicts; they do change the plan's
    identity, so only suite-v3 plans carry them.
    """
    return plan.model_copy(
        update={
            "engine_configuration": {
                **plan.engine_configuration,
                "enable_auto_tool_choice": "true",
                "tool_call_parser": parser,
            }
        }
    )


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
    sizes (their scales are created in float32 by the method).  A few tensors
    load at another width than either rule gives (``_loaded_size``).
    """
    quantized = bool(config.get("quantization_config"))
    model_type = str(config.get("model_type") or "")
    return {
        name: _loaded_size(name, dtype, size, model_type, quantized)
        for name, (dtype, size) in meta.items()
    }


# Tensors vLLM v0.30.0 loads at another width than the checkpoint stores.
# GLM-5.x DSA: the indexer's fp8 ``wk`` is dequantized into the unquantized
# bf16 ``wk_weights_proj`` and its scale is not kept
# (models/deepseek_v32/attention.py:79-86; model_executor/models/
# deepseek_v2.py:850-907).  MiniMax-M3: the router gate and its routing bias
# are float32 parameters, not downcast with dtype=auto
# (models/minimax_m3/nvidia/model.py:214-235).
_DSA_INDEXER_WK = re.compile(r"\.indexer\.wk\.weight$")
_DSA_INDEXER_WK_SCALE = re.compile(r"\.indexer\.wk\.weight_scale_inv$")
_MINIMAX_M3_FP32 = re.compile(r"\.block_sparse_moe\.(?:gate\.weight|e_score_correction_bias)$")


def _loaded_size(name: str, dtype: str, size: int, model_type: str, quantized: bool) -> int:
    if model_type == "glm_moe_dsa":
        if dtype == "F8_E4M3" and _DSA_INDEXER_WK.search(name):
            return 2 * size
        if _DSA_INDEXER_WK_SCALE.search(name):
            return 0
    if model_type == "minimax_m3_vl" and dtype == "F32" and _MINIMAX_M3_FP32.search(name):
        return size
    if quantized or dtype != "F32":
        return size
    return size // 2


def runtime_dtype(config: dict[str, Any]) -> str:
    """The dtype the engine serves in with dtype=auto (float32 downcast to 16 bits)."""
    # Newer configs name it "dtype" (transformers 5); vLLM reads either.
    return served_dtype(config.get("torch_dtype") or config.get("dtype"))


def _tied_embeddings(config: dict[str, Any]) -> bool:
    text = config.get("text_config")
    nested = text if isinstance(text, dict) else {}
    return bool(config.get("tie_word_embeddings", nested.get("tie_word_embeddings", False)))


_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")

# Model types whose vLLM v0.30.0 loader drops every checkpoint tensor whose name
# contains "mtp." (a WeightsMapper substring mapped to None): DeepSeek V4
# (models/deepseek_v4/nvidia/model.py:1793,1803), V4.1
# (models/deepseek_v41/nvidia/model.py:1006,1016; nvidia/vl_model.py:94,108),
# Qwen4Exp, whose MTP layers ``mtp_num_hidden_layers`` counts
# (models/qwen4_exp/nvidia/model.py:849-852, 1044-1047), and the Qwen3.5
# family, whose mapper sends the ``mtp.`` prefix to None in v0.29.0 and v0.30.0
# alike (model_executor/models/qwen3_5.py:321-323, 476-479; the MoE classes
# inherit it).  Qwen3.6-35B-A3B-FP8 measured 34.23 GiB against 34.88 predicted
# with its 0.80 GiB of mtp.* counted (group B, 2026-09-28).  NemotronH maps the
# ``mtp`` prefix to None (models/nemotron_h.py:711 in v0.29.0, :728 in v0.30.0):
# Nemotron-3.5-Lightning-30B-A3B stores 2.49 GiB of ``mtp.layers.*`` the engine
# never loads.  Qwen3-Next drops ``mtp.`` the same way (models/qwen3_next.py:808
# in v0.29.0, :810 in v0.30.0), and MiniMax-M3's loader skips any name with
# ``mtp.`` (models/minimax_m3/nvidia/model.py:942-944; the MiniMax-M3
# checkpoint stores none).
_MTP_NAMES_DROPPED = frozenset(
    {
        "deepseek_v4",
        "deepseek_v41",
        "qwen4_exp",
        "qwen3_5",
        "qwen3_5_moe",
        "nemotron_h",
        "qwen3_next",
        "minimax_m3_vl",
    }
)


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) else config


def mtp_tensor_bytes(tensors: dict[str, int], config: dict[str, Any]) -> int:
    """Bytes of the multi-token-prediction layers the main model skips.

    GLM-4.x MoE and DeepSeek V3 checkpoints store ``num_nextn_predict_layers``
    extra layers after the last decoder layer; vLLM loads them only for
    speculative decoding and skips them otherwise
    (``model_executor/models/utils.py:542``, ``glm4_moe_lite.py:361-363``).
    GLM-4.7-Flash measured 55.87 GiB against 58.15 predicted with them
    counted (2026-09-28).  DeepSeek V4 / V4.1 and Qwen4Exp name them ``mtp.*``
    instead, which their loaders drop (``_MTP_NAMES_DROPPED``).
    """
    is_mtp = _mtp_names(config)
    return sum(size for name, size in tensors.items() if is_mtp(name))


def _mtp_names(config: dict[str, Any]) -> Callable[[str], bool]:
    """Whether a tensor name belongs to the MTP layers the main model skips."""
    text = _text_config(config)
    extra = int(text.get("num_nextn_predict_layers") or 0)
    layers = int(text.get("num_hidden_layers") or 0)
    by_name = str(config.get("model_type") or "") in _MTP_NAMES_DROPPED

    def is_mtp(name: str) -> bool:
        if by_name and "mtp." in name:
            return True
        match = _LAYER.search(name)
        return bool(extra and layers and match and layers <= int(match.group(1)) < layers + extra)

    return is_mtp


# N-gram embedding tables vLLM v0.30.0 keeps in pinned host memory, split
# across the tensor-parallel ranks (EngramConfig.cpu_offload defaults to
# VLLM_PLE_CPU_OFFLOAD=1: config/engram.py:18-47, envs.py:2047).  Keyed by
# architecture and the config field naming the n-gram layers, as vLLM keys it
# (config/engram.py:18-23): DeepSeek V4.1's Engram tables
# (models/deepseek_v41/nvidia/engram.py:217-296, the ``engram.embed`` weight and
# scale) and Qwen4Exp's PLE n-gram table
# (models/qwen4_exp/nvidia/ngram_embedding.py:430-443, 690-708).
_HOST_TABLES: dict[str, tuple[str, re.Pattern[str]]] = {
    "DeepseekV41ForCausalLM": ("engram_layer_ids", re.compile(r"\.engram\.embed\.")),
    "Qwen4ExpForCausalLM": ("ple_layer_ids", re.compile(r"\.ple_embedding\.ngram_embedding\.")),
    "Qwen4ExpForConditionalGeneration": (
        "ple_layer_ids",
        re.compile(r"\.ple_embedding\.ngram_embedding\."),
    ),
}


def _host_table(config: dict[str, Any]) -> re.Pattern[str] | None:
    """The tensor names the engine keeps in host memory for this model, or None."""
    architecture = (config.get("architectures") or [""])[0]
    table = _HOST_TABLES.get(architecture)
    if table is None or not _text_config(config).get(table[0]):
        return None
    return table[1]


def host_tensor_bytes(tensors: dict[str, int], config: dict[str, Any]) -> int:
    """Bytes of the checkpoint vLLM keeps in pinned host memory, not on a GPU."""
    names = _host_table(config)
    if names is None:
        return 0
    return sum(size for name, size in tensors.items() if names.search(name))


# DeepSeek V4 / V4.1 weights every tensor-parallel rank holds whole: the fused
# wq_a + wkv projection (disable_tp: models/deepseek_v4/attention.py:243-250,
# deepseek_v41/attention.py:316-323), the compressors (disable_tp:
# deepseek_v4/compressor.py:270-278, deepseek_v41/compressor.py:219-226), the
# indexer (ReplicatedLinear: deepseek_v4/attention.py:958-970,
# deepseek_v41/attention.py:1089-1127), the router gate (GateLinear is a
# ReplicatedLinear: layers/fused_moe/router/gate_linear.py:14;
# deepseek_v4/nvidia/model.py:830-878), the hyper-connection parameters
# (deepseek_v4/nvidia/model.py:1169-1210, 1420-1438), V4.1's Engram projection
# (ReplicatedLinear: deepseek_v41/common/engram.py:926-941) and the RMSNorms.
_DEEPSEEK_REPLICATED = re.compile(
    r"^(?:layers\.\d+\.(?:"
    r"attn\.(?:wq_a|wkv|compressor|indexer)\."
    r"|ffn\.gate\."
    r"|hc_"
    r"|engram\.(?:wkv|q_weight|k_weight)"
    r"|(?:attn|ffn)_norm\."
    r"|attn\.(?:q|kv)_norm\."
    r")|hc_head_|norm\.)"
)


# GLM-5.x DSA (models/deepseek_v32): the fused q_a + kv_a projection
# (DeepSeekV2FusedQkvAProjLinear, disable_tp: model_executor/models/
# deepseek_v2.py:948-963), the whole indexer (wq_b ReplicatedLinear,
# wk_weights_proj disable_tp, k_norm: models/deepseek_v32/attention.py:71-87),
# the router gate and its bias (GateLinear: deepseek_v2.py:320-332) and the
# RMSNorms.  MTP layers are left out (``mtp_tensor_bytes``).
_GLM_DSA_REPLICATED = re.compile(
    r"^model\.(?:layers\.\d+\.(?:"
    r"self_attn\.(?:q_a_proj|kv_a_proj_with_mqa|indexer|q_a_layernorm|kv_a_layernorm)\."
    r"|mlp\.gate\."
    r"|(?:input|post_attention)_layernorm\."
    r")|norm\.)"
)

# MiniMax-M3: the index key's single head (model_executor/layers/linear.py:
# 1366-1367, 1407-1418), the float32 router gate and routing bias (GateLinear
# is a ReplicatedLinear: layers/fused_moe/router/gate_linear.py:14;
# models/minimax_m3/nvidia/model.py:214-235), the norms; in the vision tower
# the patch embedding (nn.Conv3d: common/vision_tower.py:57-63), the
# LayerNorms and the biases of row-parallel layers, which every rank adds whole.
_MINIMAX_M3_REPLICATED = re.compile(
    r"^(?:language_model\.model\.(?:layers\.\d+\.(?:"
    r"self_attn\.(?:index_k_proj|q_norm|k_norm|index_q_norm|index_k_norm)\."
    r"|block_sparse_moe\.(?:gate\.|e_score_correction_bias)"
    r"|(?:input|post_attention)_layernorm\."
    r")|norm\.)"
    r"|vision_tower\.vision_model\.(?:embeddings\.patch_embedding\.|pre_layrnorm\."
    r"|encoder\.layers\.\d+\.(?:layer_norm[12]\.|self_attn\.out_proj\.bias|mlp\.fc2\.bias))"
    r"|(?:multi_modal_projector|patch_merge_mlp)\.linear_2\.bias)"
)

_REPLICATED: dict[str, re.Pattern[str]] = {
    "deepseek_v4": _DEEPSEEK_REPLICATED,
    "deepseek_v41": _DEEPSEEK_REPLICATED,
    "glm_moe_dsa": _GLM_DSA_REPLICATED,
    "minimax_m3_vl": _MINIMAX_M3_REPLICATED,
}


def replicated_tensor_bytes(tensors: dict[str, int], config: dict[str, Any]) -> int:
    """Bytes of loaded weights vLLM keeps whole on every tensor-parallel rank.

    Only DeepSeek V4 / V4.1, GLM-5.x DSA and MiniMax-M3 are traced; every
    other model's weights are divided by TP (the calculator's ``_per_gpu``).
    """
    names = _REPLICATED.get(str(config.get("model_type") or ""))
    if names is None:
        return 0
    is_mtp = _mtp_names(config)
    return sum(size for name, size in tensors.items() if names.match(name) and not is_mtp(name))


# MiniMax-M3's projections split by KV head: k and v, and the index query,
# which shards like them (models/minimax_m3/nvidia/model.py:441-452;
# model_executor/layers/linear.py:1396-1405).  At TP 8 its 4 KV heads leave
# each rank one head, the same one on two ranks.
_MINIMAX_M3_KV_HEAD = re.compile(
    r"^language_model\.model\.layers\.\d+\.self_attn\.(?:k_proj|v_proj|index_q_proj)\.weight$"
)


def kv_head_tensor_bytes(tensors: dict[str, int], config: dict[str, Any]) -> int:
    """Bytes of loaded weights vLLM splits by KV head (the calculator shares
    them across ``min(TP, KV heads)`` ranks).  Only MiniMax-M3 is traced;
    every other model's are divided by TP."""
    if str(config.get("model_type") or "") != "minimax_m3_vl":
        return 0
    return sum(size for name, size in tensors.items() if _MINIMAX_M3_KV_HEAD.match(name))


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

    Tables the engine keeps in host memory (``host_tensor_bytes``) are not
    GPU weights; without headers they cannot be told apart, so a model that
    has them is unknown (0) rather than counted whole.
    """
    rev = observation.resolved_revision
    if tensors is None and hasattr(resolver, "_tensor_bytes"):
        tensors = resolver._tensor_bytes(model_id, rev)
    if tensors:
        total = (
            sum(tensors.values())
            - mtp_tensor_bytes(tensors, config or {})
            - host_tensor_bytes(tensors, config or {})
        )
        if _tied_embeddings(config or {}) and "lm_head.weight" in tensors:
            total -= tensors["lm_head.weight"]
        return total
    if _host_table(config or {}) is not None:
        return 0

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
    total -= int(entry.get("mtp_bytes") or 0)  # skipped by the main model (mtp_tensor_bytes)
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
