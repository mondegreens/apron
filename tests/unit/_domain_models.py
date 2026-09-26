"""Enumerate every pydantic model defined under ``apron.domain``.

Shared by the strict-schema property test (F2) and the fingerprint vector
tests (F1), so neither relies on a hand-picked list.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import types
import typing
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel

import apron.domain

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo


def all_domain_models() -> list[type[BaseModel]]:
    """Every ``BaseModel`` subclass whose defining module is under ``apron.domain``."""
    modules = [apron.domain]
    for info in pkgutil.walk_packages(apron.domain.__path__, "apron.domain."):
        modules.append(importlib.import_module(info.name))
    found: dict[str, type[BaseModel]] = {}
    for module in modules:
        for obj in vars(module).values():
            if (
                inspect.isclass(obj)
                and issubclass(obj, BaseModel)
                and obj.__module__.startswith("apron.domain")
            ):
                found[f"{obj.__module__}.{obj.__qualname__}"] = obj
    return [found[name] for name in sorted(found)]


# ---------------------------------------------------------------------------
# Minimal valid input synthesis from field annotations
# ---------------------------------------------------------------------------

FP_SAMPLE = "1220" + "00" * 31 + "01"


def _is_fingerprint(info: FieldInfo | None) -> bool:
    return info is not None and any(
        "1220" in (getattr(m, "pattern", None) or "") for m in info.metadata
    )


def _value_for(annotation: Any, info: FieldInfo | None = None) -> Any:
    if _is_fingerprint(info):
        return FP_SAMPLE
    origin = get_origin(annotation)
    if origin is Annotated:
        base, *meta = get_args(annotation)
        for m in meta:
            pattern = getattr(m, "pattern", None)
            if pattern and "1220" in pattern:
                return FP_SAMPLE
        return _value_for(base)
    if origin is Literal:
        return get_args(annotation)[0]
    if origin in (Union, types.UnionType):
        options = [a for a in get_args(annotation) if a is not type(None)]
        return _value_for(options[0]) if options else None
    if origin in (tuple, list):
        return []
    if origin is dict:
        return {}
    if annotation is type(None):
        return None
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return minimal_input(annotation)
    if annotation is str:
        return "x"
    if annotation is bool:
        return False
    if annotation is int:
        return 1
    if annotation is float:
        return 1.0
    if annotation is typing.Any:
        return "x"
    raise TypeError(f"no synthesizer for {annotation!r}")


def minimal_input(model: type[BaseModel]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for name, info in model.model_fields.items():
        if info.is_required():
            data[name] = _value_for(info.annotation, info)
    return data
