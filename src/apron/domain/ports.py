"""Determinism ports for wall-clock time, identifier generation and randomness.

INV-42: these reach the deterministic core only through injection.
"""

from __future__ import annotations

import random as _random_mod
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable
from uuid import uuid4


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime: ...


@runtime_checkable
class IdGenerator(Protocol):
    def generate(self) -> str: ...


@runtime_checkable
class Rng(Protocol):
    def random(self) -> float: ...


class WallClock:
    """Default Clock that reads the system clock."""

    def now(self) -> datetime:
        return datetime.now(UTC)


class UuidIdGenerator:
    """Default IdGenerator using uuid4."""

    def generate(self) -> str:
        return uuid4().hex


class SystemRng:
    """Default Rng using the standard library."""

    def random(self) -> float:
        return _random_mod.random()
