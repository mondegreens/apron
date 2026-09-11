# Pipeline cost model — verified pricing, September 2026

**Date:** 2026-09-07 · **Source:** RunPod pricing page (runpod.io/pricing), Modal pricing (spheron.network comparison), GitHub Actions docs
**Decision status:** Provider unit-price inputs remain useful. ADR-011 makes these infrastructure costs inputs to accepted-outcome economics, not the product's final cost metric. The fixed Phase 1b and Phase 3 matrices below are superseded by evidence-gap scheduling and retained only as historical sensitivity scenarios; they are not execution commitments.
**Rule:** every number below has a source. No estimates without labels.

## Outcome-economics contract (ADR-011)

Provider price, GPU-hour, per-token, per-image, per-audio/video-second, per-request and cost per attempted task answer different questions. The canonical records retain the provider's raw billed quantity, unit, tier and price snapshot and report them separately:

- `attempt_cost`: endpoint/API usage, judge usage, tool/provider charges and attributable infrastructure for one attempt;
- `observed_cost_per_accepted_outcome`: total attributable cost of all included attempts, including failures and retries, divided by accepted outcomes;
- `observed_time_per_accepted_outcome`: the protocol's declared elapsed-time boundary divided by accepted outcomes, with parallel execution disclosed;
- `expected_cost_or_time_to_acceptance`: a prediction with its probability/retry model, calibration scope and uncertainty, never relabeled as an observation;
- `steady_state_solution_cost`: startup/transfer, replicas, idle capacity, utilization, interruption, storage and availability costs over the accepted serving horizon.
- `market_equivalent_cost`: reproducible public/list or contracted price for the same execution boundary before project-only credits;
- `gross_attributable_execution_cost`: the full cost assigned to the attempt before subsidy;
- `subsidy_or_credit_applied`: provider/model-lab contribution applied to this attempt, with contributor and terms relevant to interpretation;
- `project_out_of_pocket_cost`: what the Apron project actually paid after that contribution.

Managed and self-hosted candidates are comparison-eligible only when task/application acceptance, serving horizon, availability target, included cost components and currency/time snapshot match. Provider-opaque charges or infrastructure fields remain unknown rather than being normalized away. Artificial Analysis Optima's cost/time per task is useful evaluation evidence, but Apron additionally retains rejected outcomes and reports accepted-outcome economics when the protocol defines acceptance. A project-only grant never enters solution ranking as a user discount unless the accepted user has the same durable entitlement. Sources reviewed: <https://artificialanalysis.ai/optima> and <https://artificialanalysis.ai/methodology/coding-agents-benchmarking/>.

`MaintainerBaselineAllocation` and `ContributedResourcePool` are distinct. Before Phase 1a execution, the accepted run plan is priced against current provider quotes and must fit the owner-controlled baseline with the contributed pool set to zero. Contributed credits, scoped credentials, quota or capacity can reduce project spend or expand evidence breadth; they cannot reduce evidence obligations, change candidate ranking or become assumed future funding (ADR-010, INV-29).

## GPU rental pricing (per-second billing, both providers)

### RunPod — current as of September 2026 [W: runpod.io/pricing]

| GPU | Community Cloud | Secure Cloud | 15-min cost (Secure) |
|---|---|---|---|
| RTX 3090 | $0.22/hr | $0.50/hr | $0.13 |
| RTX 4090 | $0.34/hr | $0.74/hr | $0.19 |
| L4 | $0.44/hr | $0.49/hr | $0.12 |
| RTX A6000 | $0.33/hr | $0.53/hr | $0.13 |
| L40 | $0.69/hr | $0.82/hr | $0.21 |
| A100 PCIe | $1.19/hr | $1.59/hr | $0.40 |
| A100 SXM | $1.39/hr | $1.59/hr | $0.40 |
| H100 PCIe | $1.99/hr | $2.89/hr | $0.72 |
| H100 SXM | $2.69/hr | $3.49/hr | $0.87 |
| H200 SXM | $3.59/hr | $4.59/hr | $1.15 |

Billing: per-second. A 16-minute job on a $1.99/hr GPU costs $0.53, not $1.99.

Community Cloud is cheaper but unvetted datacenters. Secure Cloud is vetted. The concept uses Secure Cloud pricing for cost estimates (conservative).

### Modal — current as of 2026 [W: spheron.network/blog/modal-gpu-pricing-2026]

| GPU | per-second | effective $/hr | 15-min cost |
|---|---|---|---|
| A100 40GB | $0.000583/sec | $2.10/hr | $0.52 |
| A100 80GB | $0.000694/sec | $2.50/hr | $0.63 |
| H100 | $0.001097/sec | $3.95/hr | $0.99 |
| H200 | $0.001261/sec | $4.54/hr | $1.14 |

Modal: per-millisecond billing, no idle charges. More expensive than RunPod for sustained work. Better for short bursts (<5 min). No RTX consumer GPUs available.

### Which provider for what

| use case | provider | why |
|---|---|---|
| Consumer GPU boots (4090, 3090, A6000) | RunPod | Only provider with consumer GPUs |
| Quick infeasibility checks (<1 min) | Modal | Per-millisecond, no minimum |
| Sustained benchmarks (15+ min) | RunPod Secure | Cheaper per-hour for sustained |
| Phase 1b bulk boots | RunPod Community | $0.34/hr for 4090 halves the GPU cost |

## Storage pricing [W: runpod.io/pricing]

| item | cost |
|---|---|
| RunPod network volume (standard), under 1 TB | $0.07/GB/month |
| RunPod network volume (standard), over 1 TB | $0.05/GB/month |
| RunPod network volume (high-performance) | $0.14/GB/month |
| Modal persistent volume | $0.09/GiB/month, 1 TiB free |

Weight pre-staging: upload via RunPod S3 API (no pod needed, $0 compute). Models are downloaded once to the volume and reused across boots.

Estimated storage for 20 models: ~500 GB = $35/month on RunPod standard.

## Free infrastructure [W: verified sources]

| component | cost | source |
|---|---|---|
| GitHub Actions (standard runners, public repo) | $0 — unlimited, unmetered | [W: github.blog/changelog/2025-12-16, cicdcalculator.com] |
| HuggingFace dataset hosting | $0 | HF free tier |
| GitHub Pages (docs + catalogue) | $0 | GitHub free tier for public repos |
| PostHog telemetry | $0 | Free tier: 1M events/month |
| Scarf install tracking | $0 | Free |
| Embedded SurrealDB diagnostic graph | no separate database service | Runs in-process with persistent SurrealKV; local storage/compute and any future hosted operation are accounted separately |

## Superseded Phase 1b fixed-matrix sensitivity scenario (20 models × 3 GPUs)

Assumptions:
- 60 model×GPU combinations. Planning assumption: ~40% may be rejected as provably infeasible by the free static prediction, leaving 36 GPU candidates. A passing prediction is not vLLM validation.
- Weights pre-staged to RunPod network volume via S3 API before GPU instances start.
- Boot + benchmark: ~15 min per feasible combination (includes vLLM startup, profile_run, short benchmark).
- Failure injection: 6 classes × 4 attempts average = 24 additional boots.
- Diagnosis convergence: ~10 extra boots for agent-proposed corrections on hard failure classes.
- Using RunPod Secure Cloud pricing (conservative).

| item | count | unit cost | total |
|---|---|---|---|
| Weight pre-staging storage (500 GB) | 1 month | $35/month | $35 |
| RTX 4090 boots (Secure) | 12 feasible + 8 failure/diag | $0.19/boot | $3.80 |
| A100 SXM boots (Secure) | 12 feasible + 8 failure/diag | $0.40/boot | $8.00 |
| H100 SXM boots (Secure) | 12 feasible + 8 failure/diag | $0.87/boot | $17.40 |
| LLM reasoning (agent API calls) | ~200K tokens | ~$0.50 total | $0.50 |
| **Total GPU + storage** | | | **$64.70** |

With Community Cloud pricing (4090 at $0.34/hr instead of $0.74/hr): total drops to ~$50.

**Phase 1b fits within $100 comfortably.** The $150-200 revised budget has room for additional GPU types (L4, L40, A6000, 3090) and more benchmark presets.

## Superseded Phase 3 fixed-matrix monthly sensitivity scenario

Assumptions:
- 2 vLLM releases per month (biweekly cadence).
- Top 20 models × 5 GPU types per release = 100 boots per release. ~50% infeasible = 50 feasible boots per release.
- 100 feasible boots/month total.
- Storage stable at ~500 GB (models accumulate slowly, old ones pruned).

| item | monthly cost |
|---|---|
| Tier 0 (GitHub Actions) | $0 |
| Tier 1 boots: 50 on consumer GPUs (avg $0.19/boot) | $9.50 |
| Tier 1 boots: 30 on A100 (avg $0.40/boot) | $12.00 |
| Tier 1 boots: 20 on H100 (avg $0.87/boot) | $17.40 |
| Weight storage (500 GB) | $35.00 |
| LLM reasoning (agent) | $4.00 |
| Issue scanning + response drafting (agent) | $2.00 |
| **Total monthly** | **$79.90** |

At $200/month budget: covers the full matrix PLUS additional GPUs (L4, L40, 3090, A6000) and extended benchmark presets.

At $100/month budget: covers top 15 models × 3 GPUs per release. Storage + reasoning = $41 fixed, leaving $59 for ~80 boots.

At $50/month budget: covers top 5 models × 2 GPUs per release. Storage + reasoning = $41 fixed, leaving $9 for ~12 boots. Tight but functional.

At $0/month budget: Tier 0 still runs ($0). Knowledge base still serves ($0). Calculated verdicts still work ($0). Issue scanning still runs ($2-4 LLM cost — the only nonzero item). No new measured records until budget resumes.

## Historical fixed-matrix budget sensitivity

This table describes the superseded matrix assumptions above. Under D10, a budget constrains the evidence-gap scheduler; it does not promise a fixed number of models, boots, or GPU types. Operative coverage is reported by claim scope and exact execution fingerprint.

| monthly budget | what it buys | coverage |
|---|---|---|
| $0 | Tier 0 diffs + calculated verdicts + issue scanning | All models calculated, no new measurements |
| $50 | + 12 measured boots per month | Top 5 models, 2 GPU types |
| $100 | + 80 measured boots per month | Top 15 models, 3 GPU types |
| $200 | + 160 measured boots per month | Full matrix: 20 models, 5+ GPU types, multiple presets |

## Sources

- RunPod pricing: [runpod.io/pricing](https://www.runpod.io/pricing) — fetched 2026-09-07
- RunPod storage and S3 API: [docs.runpod.io/storage/network-volumes](https://docs.runpod.io/storage/network-volumes)
- Modal pricing: [spheron.network/blog/modal-gpu-pricing-2026](https://www.spheron.network/blog/modal-gpu-pricing-2026-per-second-billing/) — fetched 2026-09-07
- GitHub Actions free tier: [cicdcalculator.com/github-actions-free-tier](https://cicdcalculator.com/github-actions-free-tier) — public repos unlimited
- RunPod per-second billing: [hivenet.com/post/runpod-pricing](https://www.hivenet.com/post/runpod-pricing-complete-guide-to-gpu-cloud-costs)
- Provider-resource mechanisms, used only to establish that contributed capacity exists and never assumed in the baseline: [Baseten startup program](https://www.baseten.co/startup-program/), [Modal startup program](https://modal.com/startups), [Together AI startup accelerator](https://www.together.ai/startup-accelerator), [RunPod startup program](https://www.runpod.io/startup-program)
