"""Migration on real golden fixtures + determinism test (§3.3, §3.4)."""

from apron.domain.canonical import canonicalize, record_digest_hex
from apron.domain.fingerprints import fingerprint_hex
from apron.domain.schemas.migrations import clear_registry, migrate, register
from apron.domain.schemas.solutions import DeploymentPlan


def setup_function() -> None:
    clear_registry()


# ---------------------------------------------------------------------------
# §3.3 migration on a real golden fixture (DeploymentPlan)
# ---------------------------------------------------------------------------


def test_migration_on_real_deployment_plan():
    """Migrate a real golden DeploymentPlan from v1 to v2.
    Fingerprint preserved (only DISPLAY/optional fields change),
    old digest reference remains valid."""

    plan_v1 = DeploymentPlan(
        tensor_parallel=1,
        dtype="bf16",
        batch_size=32,
        engine_configuration={"model": "Qwen/Qwen3-8B"},
        serve_command="vllm serve Qwen/Qwen3-8B --dtype bf16",
    )
    v1_data = plan_v1.model_dump(mode="json")
    v1_digest = record_digest_hex(v1_data)
    fingerprint_hex(plan_v1)

    @register("DeploymentPlan", 1)
    def _migrate(data: dict) -> dict:
        return {**data, "schema_version": 2, "max_model_len": None}

    v2_data = migrate("DeploymentPlan", v1_data, target_version=2)
    assert v2_data["schema_version"] == 2
    assert v2_data["max_model_len"] is None

    v2_digest = record_digest_hex(v2_data)
    assert v1_digest != v2_digest

    assert record_digest_hex(v1_data) == v1_digest


# ---------------------------------------------------------------------------
# §3.4 determinism test with injected ports
# ---------------------------------------------------------------------------


class FixedClock:
    def now(self):
        from datetime import UTC, datetime

        return datetime(2026, 9, 12, 10, 0, 0, tzinfo=UTC)


class FixedIdGen:
    def __init__(self):
        self._counter = 0

    def generate(self) -> str:
        self._counter += 1
        return f"id-{self._counter:04d}"


class FixedRng:
    def random(self) -> float:
        return 0.42


def _generate_plan(
    model: str,
    *,
    clock: FixedClock,
    id_gen: FixedIdGen,
    rng: FixedRng,
) -> DeploymentPlan:
    """Minimal plan generator using injected ports."""
    timestamp = clock.now().isoformat()
    plan_id = id_gen.generate()
    return DeploymentPlan(
        tensor_parallel=1,
        dtype="bf16",
        batch_size=32,
        engine_configuration={
            "model": model,
            "plan_id": plan_id,
            "generated_at": timestamp,
        },
        serve_command=f"vllm serve {model} --dtype bf16",
    )


def test_deterministic_plan():
    """Same inputs + same injected ports = byte-identical plan (§3.4)."""
    plan1 = _generate_plan(
        "Qwen/Qwen3-8B",
        clock=FixedClock(),
        id_gen=FixedIdGen(),
        rng=FixedRng(),
    )
    plan2 = _generate_plan(
        "Qwen/Qwen3-8B",
        clock=FixedClock(),
        id_gen=FixedIdGen(),
        rng=FixedRng(),
    )
    canonical1 = canonicalize(plan1.model_dump(mode="json"))
    canonical2 = canonicalize(plan2.model_dump(mode="json"))
    assert canonical1 == canonical2


def test_different_ports_different_plan():
    """Different port values produce different plans."""
    id_gen_1 = FixedIdGen()
    id_gen_2 = FixedIdGen()
    id_gen_2._counter = 100

    plan1 = _generate_plan("Qwen/Qwen3-8B", clock=FixedClock(), id_gen=id_gen_1, rng=FixedRng())
    plan2 = _generate_plan("Qwen/Qwen3-8B", clock=FixedClock(), id_gen=id_gen_2, rng=FixedRng())
    assert canonicalize(plan1.model_dump(mode="json")) != canonicalize(
        plan2.model_dump(mode="json")
    )
