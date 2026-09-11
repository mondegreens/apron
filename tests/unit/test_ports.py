from datetime import datetime, timezone

from apron.domain.ports import (
    Clock,
    IdGenerator,
    Rng,
    SystemRng,
    UuidIdGenerator,
    WallClock,
)


def test_wall_clock_returns_utc_datetime():
    clock = WallClock()
    now = clock.now()
    assert isinstance(now, datetime)
    assert now.tzinfo is not None


def test_uuid_id_generator_returns_hex_string():
    gen = UuidIdGenerator()
    id1 = gen.generate()
    id2 = gen.generate()
    assert isinstance(id1, str)
    assert len(id1) == 32
    assert id1 != id2


def test_system_rng_returns_float_in_range():
    rng = SystemRng()
    val = rng.random()
    assert isinstance(val, float)
    assert 0.0 <= val < 1.0


def test_defaults_satisfy_protocols():
    assert isinstance(WallClock(), Clock)
    assert isinstance(UuidIdGenerator(), IdGenerator)
    assert isinstance(SystemRng(), Rng)


class FixedClock:
    def __init__(self, dt: datetime):
        self._dt = dt

    def now(self) -> datetime:
        return self._dt


class FixedIdGenerator:
    def __init__(self, value: str):
        self._value = value

    def generate(self) -> str:
        return self._value


class FixedRng:
    def __init__(self, value: float):
        self._value = value

    def random(self) -> float:
        return self._value


def test_fixed_implementations_satisfy_protocols():
    dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert isinstance(FixedClock(dt), Clock)
    assert isinstance(FixedIdGenerator("abc"), IdGenerator)
    assert isinstance(FixedRng(0.5), Rng)


def test_determinism_with_fixed_ports():
    dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
    clock = FixedClock(dt)
    gen = FixedIdGenerator("deadbeef")
    rng = FixedRng(0.42)

    assert clock.now() == clock.now()
    assert gen.generate() == gen.generate()
    assert rng.random() == rng.random()
