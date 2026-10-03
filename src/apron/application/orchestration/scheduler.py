"""Evidence-gap scheduler (§9.2): rank candidates by what evidence they add.

No provider, model or GPU is privileged or forbidden by name (exit gate 7).
A candidate is ranked by the coverage obligations it would satisfy that the
existing records do not (gate items 1-4), then by mechanism diversity, then
by estimated cost.  Candidates already measured are skipped; candidates the
remaining budget cannot afford are skipped with a reason.

The seed list is data (``CandidateSeed`` values), not code paths.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

SizeClass = Literal["small", "mid", "large"]
HardwareClass = Literal["consumer", "professional", "datacenter"]

# The estimate is the slowest measured case, never a central guess: a run
# that costs more than 1.5 x its estimate stops the cohort after it
# (budget.OVERRUN_FACTOR; cohort._run_plans).  A flat 25 minutes stopped a
# 4-GPU run at $13.62 against $7.65 (2026-09-29).  The measurements behind
# each rate: tests/unit/test_run_cost_vs_cohort.py.
#
# Boot time scales with size (H1).  H1 said 15 / 25 / 35 min; a cold first
# boot also JIT-compiles FlashInfer kernels after graph capture: L0-A3's
# small model was still compiling 16 min after launch (notebook, 2026-09-26).
_BOOT_MINUTES: dict[str, float] = {"small": 25.0, "mid": 35.0, "large": 45.0}
# Hub downloads measured (events.jsonl "staged", records' phase_seconds):
# 0.21-1.06 GB/s; the slowest complete one is used.  H1's 6 GB/min predates
# them.
DOWNLOAD_GB_PER_MINUTE = 12.5
# Engine start (load, profile, capture) per GB of checkpoint, the slowest
# large boot (a 328.4 GB checkpoint in 28.4 min from its pod's own
# network-backed volume).  The time follows the checkpoint, not the per-GPU
# share: the same checkpoint on 2 and 4 GPUs started within 4 min.
LOAD_GB_PER_MINUTE = 11.5
# Task suite and serving measurement: 4.2 min for suite v3, the slowest.
EVALUATION_MINUTES = 5.0
# Every pod pulls the 9.1 GiB runner image first: 9 min (L0-A) and 13 min
# (L0-A3) from pod creation to SSH, 2026-09-26 (the run notebook);
# 3.8-6.5 min to the first container line since (events.jsonl "provisioned").
# The slowest is used.
IMAGE_PULL_MINUTES = 13.0


@dataclass(frozen=True)
class CostModel:
    """The rates a cost estimate is made with.  A ranking records the model it
    used (``ranking_record``), so ``rerank`` re-derives it with the same one."""

    image_pull_minutes: float
    download_gb_per_minute: float
    boot_minutes_small: float
    boot_minutes_mid: float
    boot_minutes_large: float
    load_gb_per_minute: float | None  # None: no load term
    evaluation_minutes: float

    def boot_minutes(self, size_class: SizeClass) -> float:
        return {
            "small": self.boot_minutes_small,
            "mid": self.boot_minutes_mid,
            "large": self.boot_minutes_large,
        }[size_class]


COST_MODEL = CostModel(
    image_pull_minutes=IMAGE_PULL_MINUTES,
    download_gb_per_minute=DOWNLOAD_GB_PER_MINUTE,
    boot_minutes_small=_BOOT_MINUTES["small"],
    boot_minutes_mid=_BOOT_MINUTES["mid"],
    boot_minutes_large=_BOOT_MINUTES["large"],
    load_gb_per_minute=LOAD_GB_PER_MINUTE,
    evaluation_minutes=EVALUATION_MINUTES,
)
# H1's rates (download 6 GB/min, no load or evaluation term).  Every ranking
# recorded before 2026-09-29 was made with them and names no cost model.
H1_COST_MODEL = CostModel(
    image_pull_minutes=13.0,
    download_gb_per_minute=6.0,
    boot_minutes_small=25.0,
    boot_minutes_mid=35.0,
    boot_minutes_large=45.0,
    load_gb_per_minute=None,
    evaluation_minutes=0.0,
)


@dataclass(frozen=True)
class CandidateSeed:
    """One (model, GPU) point the cohort may measure."""

    model_id: str
    gpu_sku: str
    size_class: SizeClass
    hardware_class: HardwareClass
    mechanism: str
    weight_gb: float
    gpu_count: int = 1
    quantized: bool = False
    prediction_error: bool = False  # GPU-free: unknown mechanism or predicted infeasible
    gated: bool = False
    features: tuple[str, ...] = ()  # e.g. "gqa", "sliding_window", "mla", "moe"

    @property
    def key(self) -> str:
        return f"{self.model_id}@{self.gpu_sku}x{self.gpu_count}"


@dataclass(frozen=True)
class Coverage:
    """What the existing records already cover."""

    sizes: frozenset[str] = frozenset()
    hardware: frozenset[str] = frozenset()
    mechanisms: frozenset[str] = frozenset()
    features: frozenset[str] = frozenset()
    quantized: bool = False
    prediction_error: bool = False
    models: frozenset[str] = frozenset()
    gpus: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RankedCandidate:
    seed: CandidateSeed
    estimated_cost: float
    obligations: tuple[str, ...]
    score: tuple[int, int, float]


@dataclass
class Ranking:
    ranked: list[RankedCandidate] = field(default_factory=list)
    skipped: list[tuple[CandidateSeed, str]] = field(default_factory=list)


def run_minutes(
    weight_gb: float, size_class: SizeClass, *, download: bool, model: CostModel = COST_MODEL
) -> float:
    """Pod minutes for one model: image pull + download (when the pod fetches
    the weights itself; none from a staged volume) + engine start (the size
    class's boot time, or the checkpoint at the measured load rate when that
    is longer) + evaluation."""
    fetch = weight_gb / model.download_gb_per_minute if download else 0.0
    boot = model.boot_minutes(size_class)
    if model.load_gb_per_minute:
        boot = max(boot, weight_gb / model.load_gb_per_minute)
    return model.image_pull_minutes + fetch + boot + model.evaluation_minutes


def run_cost(
    weight_gb: float,
    size_class: SizeClass,
    gpu_count: int,
    hourly_rate: float,
    *,
    download: bool,
    model: CostModel = COST_MODEL,
) -> float:
    """``run_minutes`` x the per-GPU rate x the GPU count."""
    minutes = run_minutes(weight_gb, size_class, download=download, model=model)
    return round(minutes / 60 * hourly_rate * gpu_count, 4)


def estimate_cost(
    seed: CandidateSeed,
    hourly_rate: float,
    *,
    download: bool = True,
    model: CostModel = COST_MODEL,
) -> float:
    """``run_cost`` for a seed (H1).  *download*: the pod downloads the
    weights (the default since the staged volume was dropped, 2026-09-29).

    Prediction-error candidates spend no GPU and cost nothing.
    """
    if seed.prediction_error:
        return 0.0
    return run_cost(
        seed.weight_gb,
        seed.size_class,
        seed.gpu_count,
        hourly_rate,
        download=download,
        model=model,
    )


def obligations_met(seed: CandidateSeed, coverage: Coverage) -> tuple[str, ...]:
    """Coverage obligations (gate items 1-4) this candidate would newly satisfy."""
    met: list[str] = []
    if seed.size_class not in coverage.sizes:
        met.append(f"size:{seed.size_class}")
    if not seed.prediction_error and seed.hardware_class not in coverage.hardware:
        met.append(f"hardware:{seed.hardware_class}")
    if seed.mechanism not in coverage.mechanisms:
        met.append(f"mechanism:{seed.mechanism}")
    if seed.quantized and not coverage.quantized:
        met.append("quantization")
    if seed.prediction_error and not coverage.prediction_error:
        met.append("prediction_error")
    if seed.model_id not in coverage.models:
        met.append("new_model")
    if not seed.prediction_error and seed.gpu_sku not in coverage.gpus:
        met.append("new_gpu")
    return tuple(met)


def rank_candidates(
    candidates: Iterable[CandidateSeed],
    existing: Coverage,
    *,
    measured: Iterable[str],
    remaining_budget: float,
    rates: Mapping[str, float],
    download: bool = True,
    cost_model: CostModel = COST_MODEL,
) -> Ranking:
    """Order candidates by evidence value inside the budget.

    Score (higher first): number of coverage obligations met, then number of
    new architecture features, then lower cost.  Greedy: each pick updates the
    coverage the next candidates are scored against, so the list reads as a
    plan, not a static sort.  *download*, *cost_model*: as in ``estimate_cost``.
    """
    done = set(measured)
    pool: list[CandidateSeed] = []
    ranking = Ranking()
    for seed in candidates:
        if seed.key in done:
            ranking.skipped.append((seed, "already measured"))
        elif not seed.prediction_error and seed.gpu_sku not in rates:
            ranking.skipped.append((seed, f"no rate for {seed.gpu_sku}"))
        else:
            pool.append(seed)

    coverage = existing
    budget = remaining_budget
    while pool:
        scored = []
        for seed in pool:
            cost = estimate_cost(
                seed, rates.get(seed.gpu_sku, 0.0), download=download, model=cost_model
            )
            met = obligations_met(seed, coverage)
            new_features = len(set(seed.features) - coverage.features)
            scored.append((seed, cost, met, (len(met), new_features, -cost)))
        scored.sort(key=lambda s: (s[3], s[0].key), reverse=True)
        seed, cost, met, score = scored[0]
        pool.remove(seed)
        if cost > budget:
            ranking.skipped.append((seed, f"estimate ${cost:.2f} exceeds remaining ${budget:.2f}"))
            continue
        budget -= cost
        ranking.ranked.append(RankedCandidate(seed, cost, met, score))
        coverage = _cover(coverage, seed)
    return ranking


def _cover(coverage: Coverage, seed: CandidateSeed) -> Coverage:
    return Coverage(
        sizes=coverage.sizes | {seed.size_class},
        hardware=coverage.hardware
        if seed.prediction_error
        else coverage.hardware | {seed.hardware_class},
        mechanisms=coverage.mechanisms | {seed.mechanism},
        features=coverage.features | set(seed.features),
        quantized=coverage.quantized or seed.quantized,
        prediction_error=coverage.prediction_error or seed.prediction_error,
        models=coverage.models | {seed.model_id},
        gpus=coverage.gpus if seed.prediction_error else coverage.gpus | {seed.gpu_sku},
    )


def coverage_of(seeds: Sequence[CandidateSeed]) -> Coverage:
    coverage = Coverage()
    for seed in seeds:
        coverage = _cover(coverage, seed)
    return coverage


# ---------------------------------------------------------------------------
# The ranking as a run artifact (exit gate §1 item 7)
# ---------------------------------------------------------------------------


def _coverage_dict(coverage: Coverage) -> dict[str, Any]:
    return {k: sorted(v) if isinstance(v, frozenset) else v for k, v in asdict(coverage).items()}


def ranking_record(
    ranking: Ranking,
    *,
    candidates: Sequence[CandidateSeed],
    existing: Coverage,
    measured: Sequence[str],
    remaining_budget: float,
    rates: Mapping[str, float],
    download: bool = True,
    cost_model: CostModel = COST_MODEL,
) -> dict[str, Any]:
    """Inputs and output of one ranking, so the choice can be re-derived later:
    the cost model and the download flag the estimates were made with too."""
    return {
        "inputs": {
            "candidates": [{**asdict(s), "features": list(s.features)} for s in candidates],
            "existing": _coverage_dict(existing),
            "measured": list(measured),
            "remaining_budget": remaining_budget,
            "rates": dict(rates),
            "download": download,
            "cost_model": asdict(cost_model),
        },
        "ranked": [
            {"key": r.seed.key, "cost": r.estimated_cost, "obligations": list(r.obligations)}
            for r in ranking.ranked
        ],
        "skipped": [{"key": s.key, "reason": reason} for s, reason in ranking.skipped],
    }


def rerank(record: Mapping[str, Any]) -> Ranking:
    """Run the scheduler again on a recorded ranking's inputs, with the cost
    model it names (``H1_COST_MODEL`` for the rankings recorded before one was
    named: they were made with it, downloading on the pod)."""
    inputs = record["inputs"]
    named = inputs.get("cost_model")
    cost_model = CostModel(**named) if named else H1_COST_MODEL
    seeds = [
        CandidateSeed(**{**c, "features": tuple(c.get("features", ()))})
        for c in inputs["candidates"]
    ]
    recorded: dict[str, Any] = inputs["existing"]
    existing = Coverage(
        sizes=frozenset(recorded["sizes"]),
        hardware=frozenset(recorded["hardware"]),
        mechanisms=frozenset(recorded["mechanisms"]),
        features=frozenset(recorded["features"]),
        quantized=bool(recorded["quantized"]),
        prediction_error=bool(recorded["prediction_error"]),
        models=frozenset(recorded["models"]),
        gpus=frozenset(recorded["gpus"]),
    )
    return rank_candidates(
        seeds,
        existing,
        measured=inputs["measured"],
        remaining_budget=float(inputs["remaining_budget"]),
        rates=inputs["rates"],
        download=bool(inputs.get("download", True)),
        cost_model=cost_model,
    )
