"""Evidence-gap scheduler (§9.2): rank candidates by what evidence they add.

No provider, model or GPU is privileged or forbidden by name (exit gate 7).
A candidate is ranked by the coverage obligations it would satisfy that the
existing records do not (gate items 1-4), then by mechanism diversity, then
by estimated cost.  Candidates already measured are skipped; candidates the
remaining budget cannot afford are skipped with a reason.

The seed list is data (``CandidateSeed`` values), not code paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

SizeClass = Literal["small", "mid", "large"]
HardwareClass = Literal["consumer", "professional", "datacenter"]

# Boot time scales with size (H1): 15 min for small models, up to 35 for large.
_BOOT_MINUTES: dict[str, float] = {"small": 15.0, "mid": 25.0, "large": 35.0}
DOWNLOAD_GB_PER_MINUTE = 6.0


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


def estimate_cost(seed: CandidateSeed, hourly_rate: float) -> float:
    """(download minutes + boot minutes, scaled by size) x pod rate (H1).

    Prediction-error candidates spend no GPU and cost nothing.
    """
    if seed.prediction_error:
        return 0.0
    minutes = seed.weight_gb / DOWNLOAD_GB_PER_MINUTE + _BOOT_MINUTES[seed.size_class]
    return round(minutes / 60 * hourly_rate * seed.gpu_count, 4)


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
) -> Ranking:
    """Order candidates by evidence value inside the budget.

    Score (higher first): number of coverage obligations met, then number of
    new architecture features, then lower cost.  Greedy: each pick updates the
    coverage the next candidates are scored against, so the list reads as a
    plan, not a static sort.
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
            cost = estimate_cost(seed, rates.get(seed.gpu_sku, 0.0))
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
