"""Layer 2 — ComponentMechanism, WorkloadShape, and calculator dispatch.

The calculator is a plain function (Callable), not a Protocol object
(engineering-standards.md §1).  Dispatch is by ComponentMechanism tag
lookup in a strategy-pattern registry (INV-32).  Unimplemented mechanisms
return ``None`` — no fallback formula.
"""

from collections.abc import Callable
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from apron.domain.fingerprints import DISPLAY, IDENTITY
from apron.domain.schemas.primitives import HardwareSpec

# ---------------------------------------------------------------------------
# ComponentMechanism (ADR-007 §4)
# ---------------------------------------------------------------------------


class ComponentMechanism(BaseModel):
    """A typed execution mechanism with a role in the component graph.

    ``mechanism`` is an open string with documented conventions:
    autoregressive_decode, discrete_diffusion_decode, single_pass_pooling,
    encoder_decoder_generation, media_encoder, projector, latent_denoising,
    vae_decode, vocoder, or a namespaced extension.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Annotated[int, DISPLAY] = 1
    mechanism: Annotated[str, IDENTITY]
    role: Annotated[str, IDENTITY]


# ---------------------------------------------------------------------------
# WorkloadShape (discriminated union, ADR-011 §2)
# ---------------------------------------------------------------------------


class TextWorkload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: Annotated[Literal["text"], IDENTITY]
    input_length: Annotated[int, IDENTITY]
    output_length: Annotated[int, IDENTITY]


class AudioWorkload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: Annotated[Literal["audio"], IDENTITY]
    duration_seconds: Annotated[float, IDENTITY]
    chunk_size_ms: Annotated[int | None, IDENTITY] = None


class ImageWorkload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: Annotated[Literal["image"], IDENTITY]
    width: Annotated[int, IDENTITY]
    height: Annotated[int, IDENTITY]
    steps: Annotated[int | None, IDENTITY] = None


WorkloadShape = Annotated[
    TextWorkload | AudioWorkload | ImageWorkload,
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Calculator registry and dispatch
# ---------------------------------------------------------------------------


class CalculatorInput(BaseModel):
    """Consumed inputs for a calculator function."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mechanism: ComponentMechanism
    workload: TextWorkload | AudioWorkload | ImageWorkload
    artifact_metadata: dict[str, Any]
    hardware: HardwareSpec
    execution_spec_data: dict[str, Any]
    calibration_records: tuple[dict[str, Any], ...] = ()


CalculatorFn = Callable[[CalculatorInput], dict[str, Any] | None]

_CALCULATOR_REGISTRY: dict[str, CalculatorFn] = {}


def register_calculator(
    mechanism: str,
) -> Callable[[CalculatorFn], CalculatorFn]:
    def decorator(fn: CalculatorFn) -> CalculatorFn:
        if mechanism in _CALCULATOR_REGISTRY:
            msg = f"calculator already registered: {mechanism}"
            raise ValueError(msg)
        _CALCULATOR_REGISTRY[mechanism] = fn
        return fn

    return decorator


def calculate(inputs: CalculatorInput) -> dict[str, Any] | None:
    """Dispatch to the registered calculator for the mechanism.

    Returns ``None`` for unimplemented mechanisms — no fallback formula.
    """
    fn = _CALCULATOR_REGISTRY.get(inputs.mechanism.mechanism)
    if fn is None:
        return None
    return fn(inputs)


def clear_calculator_registry() -> None:
    _CALCULATOR_REGISTRY.clear()
