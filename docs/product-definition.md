# Product definition v1 — what "usable" means and why it is achievable

**Date:** 2026-09-06 · **Reframed:** 2026-09-08 by ADR-011 · **Amended:** 2026-09-09 by ADR-002 registry/planner/host independence and ADR-012 repository-native distribution · **Source:** ADR-011 primary-source review, local [AIConfigurator](https://github.com/ai-dynamo/aiconfigurator) and Hugging Face Hub documentation review · **Authority:** ADR-002, ADR-004, ADR-005, ADR-011, ADR-012

## 0a. Why this product has to be shipped

### The problem

Teams first have to decide which model or model system can perform their real work, at what quality, time and cost. They then have to turn that choice into an endpoint that fits, boots, meets the serving SLO and preserves the measured task outcome. Public leaderboards, custom evaluations, managed model APIs and self-hosted deployment tools each answer part of this decision, but the evidence is rarely joined across the exact task, application scaffold, model/API, artifact, runtime and hardware that produced it.

When someone tries to self-host a model with vLLM, they may additionally download large weight artifacts, wait for the engine to boot, and only then discover whether the selected artifact, hardware and configuration work together. Volatile package-download counts are not used as a proxy for the number of deployment users.

vLLM already catches many config errors before boot: its pinned source contains checks for TP divisibility, incompatible dtypes and other known misconfigurations. It also auto-derives max-model-len when passed `-1`, solving one common failure class. These exist, they work, and the product acknowledges them; a syntactic count of `raise` statements is not called a count of semantic validators.

What vLLM does NOT catch: whether the model fits in VRAM at all before loading, every scheme/backend/kernel compatibility failure, and what to do when a boot fails. Issue #54318 is the concrete counterexample: a Qwen3.8 FP8 artifact passed generic configuration construction on A100/SM80 and failed later in a Triton fused-MoE kernel. It does not establish a universal FP8-versus-Ampere rule or a verified correction. GitHub issue snapshots contain repeated startup-failure reports, but no defensible query establishes a total number of deployment failures or the subset Apron can correct. The addressable gap between static prediction and GPU execution is real and deliberately left unquantified.

Each failure costs 30 minutes to hours — downloading weights, booting, crashing, diagnosing, retrying. An IBM engineer described it in issue #29325: *"30 minutes to several hours, only to realize the setup cannot run."*

### The 2am moment

It's 2am. You rented a 4090 on RunPod at $0.69/hour. You downloaded 65 GB of Qwen3-32B — that took 45 minutes. You ran `vllm serve` with flags from a Reddit thread. Crash: "Engine core initialization failed." You asked Claude. "Try `--enforce-eager`." Different error. "Try lowering `max-model-len`." Still crashes. Two hours gone. $1.38 burned on a GPU that served nothing. 65 GB downloaded for nothing. You still don't know if this model fits on this card.

Before you rent the GPU, before you download anything, you type one command. Apron calculates a prediction from immutable model metadata, version-pinned engine constraints, and the declared hardware target, then returns a candidate plan with its uncertainty and unknowns exposed. It does not call this a vLLM test. When vLLM executes on the GPU, the product records the actual result as a measurement; if boot fails, it matches the failure fingerprint to a deterministic correction, proves the mechanism by corrected boot, and separately checks that the accepted request still passes.

### What already works (and what doesn't)

The ecosystem is working on this problem. Acknowledging what exists:

| solution | what it does well | the gap |
|---|---|---|
| Artificial Analysis / Optima | Public model intelligence plus custom benchmarks over user tasks, agents, quality, cost per task and time per task | Does not close provider-neutral exact artifact/runtime/hardware deployment, boot diagnosis and requalification |
| Baseten Model APIs and dedicated inference | Pre-optimized managed endpoints, autoscaling, observability and production operations | One provider's execution plane, not an independent comparison and evidence layer across managed and self-hosted solutions |
| Inspect AI, Harbor, Phoenix | Reusable task/scorer/agent, sandbox/trial, trace/dataset and experiment machinery | Evaluation engines and observability systems, not the task-to-qualified-deployment decision and remediation loop |
| vLLM itself | pinned config-time checks and `--max-model-len -1` auto-fit | No pre-load weight estimate, no end-to-end preflight, no diagnosis loop |
| NVIDIA aiconfigurator | Mature analytical memory/SLO planning and artifact rendering for supported datacenter and professional accelerators | Its documentation still requires real benchmark validation; no log-to-proven-correction loop or Apron's RunPod history |
| vllm-project/recipes | Official-org recipes, including explicitly `verified` consumer configurations such as 2×RTX 4090 | Curated coverage; no general prediction-beside-measurement or diagnosis loop |
| club-3090 | validated Compose files with measured throughput for a curated consumer-GPU catalogue | Curated coverage; hardcoded minimum-VRAM metadata rather than general computed feasibility; manual maintenance burden |
| LLMs (Claude, GPT) | Fast, free advice | ~50% accuracy on exact flags; "countless trial and error" (Reddit users' own words); can't verify or measure |

The product fills the joint gap none of them covers alone: accepted task outcome → solution comparison → computed feasibility before download → exact deployment → task and serving reproduction → diagnosis after failure → version-pinned requalification. Existing evaluation and serving systems are adapters, evidence sources and execution targets; they make Apron stronger rather than redundant.

Inference providers, GPU platforms and model labs may contribute scoped credits, credentials, quota or dedicated capacity because reproducible records, integrations and qualified workloads can benefit their ecosystems. They never purchase placement or conclusions. The first conformance loop and minimum evidence operation have an owner-controlled budget path with no contributed resources; grants only expand coverage or reduce project out-of-pocket cost. Every affected record discloses resource provenance and separates market-equivalent/user-relevant cost from subsidy and project spend. Relationship terms such as `partner` or `sponsor` require an actual agreement; ordinary adapter support does not imply endorsement.

### Why no LLM can give this answer

This project's own LLM-generated competitive analysis got 5 of 8 vLLM version facts wrong and concluded "DeepSeek-V3 FP8 needs 8×H100 minimum" for weights that don't fit in 640 GB. vLLM releases every two weeks; a flag that exists in v0.28 may not exist in v0.29. Reddit users describe using Claude and Codex for vLLM config as "trial and error." The product makes the answer auditable: Claude calls the MCP tool, receives a version-pinned prediction, and receives a verified answer only when matching GPU execution evidence exists. Claude is the consumer, not the authority. Sources: vLLM source at the pinned commit; reddit.com/r/BlackwellPerformance/1snk6h8; reddit.com/r/LocalLLM/1vtsid0; internal research findings.

### Why nobody else has built the full loop

In issue #29325, at least five distinct commenters explicitly offered to collaborate, contribute or implement. One proposed `vllm serve --dry-run`—an upstream preflight adjacent to this product's prediction-and-measurement loop, not proof that vLLM can be validated without a GPU. A project member responded, but no project owner committed to implementation before the stale bot closed the issue. The issue itself, rather than a mutable volunteer total, is the demand evidence.

This does not mean the ecosystem is failing — it means the engine team builds the engine, the hardware vendor builds for datacenter customers, and the evidence-plus-diagnosis layer is nobody's priority. If any of them build part of it, the product calls their code and gets better (ADR-003, ADR-008). Sources: github.com/vllm-project/vllm/issues/29325; github.com/vllm-project/vllm/issues/39749; github.com/ai-dynamo/aiconfigurator; internal research findings.

### Why "it boots" and "it scored well" are both insufficient

The difference between a basic setup and the right config for a workload (inference engineering knowledge base):

- Decode is memory-bandwidth-bound: the same model on the same GPU gives 5 tok/s or 40 depending on `max_num_seqs`, `max_num_batched_tokens`, and quantization choice.
- A coding agent needs low total response time; a chat app needs low TTFT; a RAG pipeline needs high concurrent throughput. Different workloads need different configs.
- A public benchmark or task score belongs to the exact task suite, prompt/agent/tool scaffold, judge and model/API configuration that produced it. It does not prove that a different quantization, endpoint or application preserves the result.
- Cost per token or per attempted task is not cost per accepted result. Failures, retries, tool calls, judges, idle capacity and availability requirements can reverse the ranking.
- FP8 and FP4 compatibility depends on artifact, scheme, backend, kernel, hardware, model, and engine version. Issue #54318 is retained as a scoped external GPU failure observation with no verified correction; issue #55008's successful Marlin remediation belongs to a different incident and is not silently transferred.
- KV cache takes 80%+ of post-weights VRAM. A "2× params" estimate is wrong by up to 16× for GQA/MQA/MLA models.

The product keeps task evidence, serving evidence, version-pinned prediction and GPU-executed engine evidence separate, then joins them only through exact fingerprints. Every execution record carries predicted beside measured, and every qualified solution shows which accepted outcome and serving constraints it actually reproduced.

### How trust is earned

The 2am person doesn't need to trust the maintainer. They need the answer to be right once.

1. **Try** — cost is 2 minutes (one command or one Compose file). Cost of NOT trying is 30-60 minutes of trial-and-error.
2. **Verify** — `nvidia-smi` shows the VRAM. The record said the predicted value. It matches. They checked a number, not trusted a claim.
3. **Come back** — vLLM releases every two weeks. The product tracks per-tag diffs and tells them what changed.
4. **Find it without searching** — the MCP tool the agent already calls, the catalogue the search finds, the vLLM issue the record is attached to (posted by the maintainer to existing open issues).

Precedent: the Open LLM Leaderboard demonstrates trust in auto-generated benchmark results through reproducible methodology rather than human attestation per entry. club-3090 demonstrates adoption of per-model Compose files with measured numbers. These are precedents, not guarantees or universal comparisons between every planner and catalogue. The supported product inference is narrower: people can directly reuse a working configuration.

Expected 90-day outcome: 0–100 stars. The base rate in this niche for a solo tool is near zero. The concept says this upfront and has a decision rule for when the targets fail.

### The selling points

1. **Each failure is expensive** — 30 min to hours per attempt; 16–65 GB downloaded before you learn the config is wrong
2. **The gap is specific** — model evaluation, managed serving and engine benchmarking exist; the missing product joins task outcome to an exact provider-neutral solution, deployment, diagnosis and requalification
3. **No LLM can do it** — ground truth changes biweekly; Claude is the consumer via MCP, not the competitor
4. **People copy configs, not install tools** — the catalogue form factor outperforms every tool in this niche (one precedent: club-3090)
5. **Trust comes from reproducibility** — predicted beside measured, verifiable with `nvidia-smi`

## 0. What the product is

The product is a task-to-qualified-inference-solution decision and evidence system. Given an accepted task outcome, quality floor, serving SLO, policy and authorization envelope, it discovers managed, self-hosted or compound candidates; evaluates their task behavior; calculates and deploys feasible endpoints; reproduces the outcome on the exact retained solution; measures total outcome economics; and returns a qualified solution, bounded comparison, or proven fix with prediction beside measurement.

| face | who | what they get |
|---|---|---|
| the tool | a person or a coding agent | an accepted task in; a scoped solution comparison and qualified endpoint/system, a corrected plan with its proving record, or an honest unknown out |
| the decision record | the application owner | exact candidates, exclusions, task and serving evidence, outcome economics, uncertainty and trade-offs behind the selected solution |
| the oracle | upstream projects and evaluation/deployment systems | records in reusable formats that their planners, evaluators and CI can consume, plus the contributions that fall out of them (ADR-008) |
| the lab | the maintainer or authorized user | the harness that regenerates deployment, evaluation and compatibility evidence on permitted resources at bounded cost |

It is an evidence, qualification, planning, verification and placement-optimization layer across evaluation systems, managed APIs and inference engines. Apron owns the canonical decision and evidence graph, deterministic selection rules and a GPU-free, mechanism-aware predictor built from immutable artifact metadata, explicit architecture dispatch, versioned engine constraints and calibrated observations. It integrates established evaluation engines rather than treating generated benchmarks or LLM judges as unexamined truth. vLLM's physical cache specification comes from instantiated model layers, while its memory profile and actual compatibility paths require GPU execution; those are the conformance and measurement oracle, not a universal CPU calculator.

It is not merely a calculator, platform, website, or competitor to an engine. A resource calculator is nevertheless a required product component: it creates useful pre-download candidates where no matching record exists. The evidence and replay loop is what makes those calculations auditable and improves them over time.

It is a product rather than a leaderboard or dataset because the loop closes: an accepted outcome produces a replayable task protocol; candidates are evaluated; a solution and its endpoint plans are derived; deployment either works or becomes diagnosis evidence; task and serving outcomes are reproduced on the retained solution; and every material task, application, model, runtime or provider change invalidates or requalifies the scoped result.

## 1. The experience

A person or coding agent may begin with the task and let Apron compare authorized solutions. The following uses `qualify` as illustrative syntax; the CLI verb is not frozen by ADR-011 and requires its own naming review:

```
apron qualify --tasks ./evals/support-agent --application ./agent.yaml --quality success_rate:0.95 --slo p95_e2e:8s --optimize cost_per_accepted_task
```

The resulting `DecisionReport` exposes the compared managed, self-hosted and compound candidates, exclusions, scores, serving evidence, outcome economics and uncertainty. Once accepted, `apron deploy DECISION_ID` retains the selected solution and reproduces the accepted task outcome against it.

An explicitly selected model remains a valid constrained request rather than a separate product:

```
apron deploy Qwen/Qwen3.8-27B --workload coding-agents --concurrency 6 --context 32768 --optimize balanced
```

The same operation is available over local MCP:

```
deploy_model(model="Qwen/Qwen3.8-27B", workload="coding-agents", concurrency=6, context=32768, optimize="balanced")
```

The MCP server is the same binary in server mode:

```
apron mcp
```

Claude Code configuration:

```json
{"command": "apron", "args": ["mcp"]}
```

After that, when a user asks Claude "deploy Qwen3.8-27B on my 4090," Claude calls the MCP tool silently. The user never types `apron` — the agent does. One package, two doors into the same engine.

The public service receives only what the user chooses to share. Task data, prompts, traces, outputs, model files, credentials and filesystem contents remain local by default. Sending task material to an external candidate, judge or evaluation service requires an accepted data policy and matching authorization; evaluation never implies telemetry or publication.

## 1a. Telemetry

The 90-day checkpoint requires instrumented, privacy-respecting event counting with published counting rules. No custom scaffolding — two existing tools:

**Install tracking:** Scarf gateway wrapping the PyPI package. Zero code. Counts installs and shows which organizations are using it.

**Usage telemetry:** PostHog open-source analytics. Python SDK, opt-in, free tier (1M events/month). Events carry only bounded categories the product already defines — capability-signature class, mechanism class, hardware class, outcome and engine tag — never media contents, exact high-cardinality private signatures, model files, credentials, prompts, or filesystem contents. The counting rules are published in the docs.

Events that map to the 90-day checkpoint:

| event | checkpoint target |
|---|---|
| `plan_completed` | ≥10 plans for unseeded model IDs |
| `diagnosis_completed` | ≥3 diagnoses leading to modified plans |
| `artifact_used` | ≥5 artifacts used through a measurable action |
| `verification_completed` | ≥1 signed external verification report |
| `session_return` | ≥1 returning user or agent |
| `run_wrapped` | ≥1 user or agent invokes `apron run` or the MCP tool more than once for different models or engine versions (distinguishes workflow adoption from one-time use) |
| `community_record` | ≥1 locally verified record explicitly published via `apron submit` from hardware outside the maintainer-rented lab |
| `repository_check_completed` | a pinned Apron Action completes a canonical plan/invalidation check; workflow references and badge views alone do not count |

PostHog can be self-hosted if needed. The dashboard is built-in — no custom analytics backend, no custom event pipeline.

**Demand signals for the agentic pipeline.** Every CLI, MCP or repository-Action invocation may capture the same categorical deployment signals locally: capability-signature class, mechanism class, workload type, hardware class, concurrency, applicable token/media shape bands and outcome. Accepted requests and local failures are the direct demand inputs to Tier 1 evidence-gap ranking (`phase-plan.md` §demand-driven), alongside claim scope, missing/stale/contested evidence, prediction uncertainty/error, unresolved failures, freshness, adequate external coverage and cost. With opt-in telemetry, only the bounded categories above are transmitted. Without opt-in, they feed only the invoking local or repository workflow and are not sent to Mondegreens. No prompts, media, filesystem contents, exact private signatures or credentials are captured. Dated fallback signals may come from multiple model registries, provider catalogues, engine issues/releases, recipes, external planners and evidence systems. Every source's snapshot, normalization, deduplication and contribution are recorded; no single popularity counter, including Hub downloads, defines eligibility or silently wins a tie.

## 1b. Tech stack

No custom frameworks. Proven libraries, well-maintained, well-documented.

| layer | library | role |
|---|---|---|
| CLI framework | **Typer** | subcommands (`deploy`, `diagnose`, `mcp`), auto-completion, `--help` generation, type-safe arguments |
| Terminal output | **Rich** | verdict tables, colored ✓/✗, VRAM usage bars, progress indicators, Compose file preview, startup banner |
| MCP server | **mcp** (Anthropic SDK) | `apron mcp` mode — same planning engine, accessed by coding agents |
| Schemas and validation | **Pydantic** | `DecisionRequest`, task/application/serving specs, `ArtifactLocator`/source observation/content identity, `InferenceSolution`, `PlanningClaim`, evaluation/task-attempt records, endpoint `DeploymentPlan`, `VerificationReport`, `EvidenceReleaseManifest`, `DecisionReport` — typed, validated, JSON-serializable |
| HTTP client | **httpx** | artifact-source and provider APIs, including the initial Hugging Face Hub, RunPod and Modal adapters |
| Configuration | **tomli/tomllib** | user config, provider credentials, workload presets |
| Telemetry | **PostHog** | opt-in usage events (see §1a) |
| Install tracking | **Scarf** | PyPI gateway, zero code |
| Canonical serialization and digest | **RFC 8785 (JCS)** + `hashlib` SHA-256, multihash-prefixed | record, manifest and evidence-release identity (INV-43, ADR-006) |
| Canonical record store | content-addressed files under `records/`, with a local index for queries | the append-only source of truth; an evidence release is reconstructable from it without any host (INV-35) |
| Diagnostic graph | **embedded SurrealDB** (persistent SurrealKV) | the §3 projection of `failure-knowledge-base.md`; behind an internal repository interface, rebuildable from records, and replaceable without changing public schemas, record identity or evidence semantics |
| Durable orchestration state | local durable job store behind the `orchestration/` contracts | stable ids, deduplication keys, state transitions and replay safety (ADR-010, INV-21) |
| Property and stateful testing | **Hypothesis** | round-trip, calculator laws and the qualification-graph state machine; imported only inside test packages (`engineering-standards.md` §11) |
| External-contract recording | HTTP record/replay in tests | artifact-source and provider drift detection; Tier 0 covers the engine only |

Typer + Rich is the standard Python CLI combo in 2026. Typer handles the interface, Rich handles the output. Pydantic handles the data contracts. httpx handles the network. Nothing custom, nothing exotic.

## 1c. Repository setup (2026-09-07; stack decisions corrected 2026-09-11)

Notes on §1b: Scarf's PyPI gateway records IP-derived location and company on every download before any opt-in; either disclose that in §1a or drop Scarf for PyPI's public statistics. Credentials never live in the TOML config (INV-13): environment or OS keyring only; TOML holds presets and preferences.

The GitHub account boundary is fixed by ADR-012:

| repository | visibility | authority and contents |
|---|---|---|
| `mondegreens/.github` | public | organization profile and shared community policy; no product logic |
| `mondegreens/apron` | public | canonical core, CLI, MCP, schemas, adapters, public records and documentation |
| `mondegreens/apron-research` | private | investigations, rejected hypotheses, unpublished evidence and vendor working material; separate history, never flipped public as the product bootstrap |
| `mondegreens/apron-action` | public | independently released GitHub/Marketplace adapter pinned to the canonical Apron core; no recommendation logic |

| tool | role |
|---|---|
| uv | environment, lockfile, build backend, publish; a PEP 751 `pylock.toml` export is committed beside `uv.lock` as the tool-neutral auditable record and the input to dependency scanning. The two can drift, because Dependabot updates `uv.lock` natively through its `uv` ecosystem support but does not consume `pylock.toml` (dependabot-core#12094, open since 2025-04-18), so the `supply-chain` job regenerates the export with `uv export --format pylock.toml` and fails on a diff (`engineering-standards.md` §10a, §11) |
| Python ≥ 3.12 | The floor is stated on its own terms and set by lifecycle, not by another project's floor: 3.11 is security-only with end of life in October 2027, 3.12 reaches end of life in October 2028, and 3.15.0 is scheduled for 2026-10-01 (https://devguide.python.org/versions/, fetched 2026-09-11). Nothing is written yet, so a 3.11 floor would guarantee a raise inside the first year. `tomllib` is in the standard library from 3.11 and is therefore no longer the reason. vLLM's own floor at the pinned commit is 3.10; the two are not equal and are not claimed to be |
| ruff | lint and format |
| pyright (strict on `schemas/`) | the Pydantic models are the public contract, so the type gate blocks and must be stable. pyright 1.1.414 (2026-09-09) is that gate. Astral's `ty` is faster and shipped first-class Pydantic support in 0.0.57–0.0.61 (July 2026), but 0.0.80 (2026-09-09) is still beta with no stable release since the beta opened on 2025-12-16 (https://astral.sh/blog/ty, https://github.com/astral-sh/ty/releases), and pydantic-ai's own adoption tracker (pydantic-ai#3970) remains open with dozens of unresolved errors. `ty` therefore runs as a non-blocking CI lane that measures the distance to a switch, rather than as a substitution |
| pytest | conformance suites as test packages (INV-3, INV-5, INV-8), published as the separate `apron-conformance` distribution (ADR-012) |
| import-linter | layer contracts enforcing the INV-11 import boundaries (`engineering-standards.md` §3) |
| prek | the hook runner for ruff, pyright, secret scan and the schema-version bump check (INV-13, INV-16). It reads the same `.pre-commit-config.yaml` as `pre-commit` and is documented as fully compatible with its hooks (prek 0.5.2, 2026-09-02, https://prek.j178.dev); CPython's lint workflow and ruff's own CI run it. The configuration is the portable asset, so reverting to `pre-commit` 4.6.2 costs one line |
| git-cliff | changelog from conventional commits, generated into the release body. Generating needs only `contents: read`; the write scope belongs to the commit-the-changelog-back pattern, which this project does not use (git-cliff 2.14.1 and git-cliff-action v4.9.0, both 2026-09-01) |

Version and dependency policy (`engineering-standards.md` §11): the matrix tests the declared floor and the newest stable Python release rather than the versions between them; a new release is added within a stated window and a release is dropped at upstream end of life. Direct dependencies update as grouped minor/patch proposals with security updates separated and a cooldown before an update becomes eligible for automatic merge. ADR-009 requires full-commit-SHA action pins, which are kept current by automated update proposals — a pin policy without an update mechanism produces stale dependencies rather than reviewed ones. The number of simultaneously supported engine minor versions is stated explicitly: the engine releases every two weeks with backward compatibility guaranteed for a limited number of minors, so an unstated window is either an unaffordable revalidation obligation or an unsupported currency claim.

| workflow | trigger | content |
|---|---|---|
| `ci.yml` | every pull request | the named job graph of `engineering-standards.md` §10a — `format`, `types`, `boundaries`, `schemas`, `migrations`, `identity`, `determinism`, `conformance`, `interop`, `policy`, `isolation`, `rules`, `secrets`, `supply-chain`, `docs`, `spec-drift`, `commit-grammar` — over a matrix of the declared Python floor and the newest stable release. Each job guards shape, contract or claim and names the invariants it enforces. GPU-free; it never presents its result as engine validation (INV-37). |
| `tier0.yml` | schedule | poll vLLM tags; extract versioned schemas and constraints; recompute Apron predictions on a CPU runner; open a pull request with the diff; produce no vLLM validation evidence |
| `tier1.yml` | manual dispatch, protected Environment with required reviewer | provider tokens as environment secrets; spend cap and teardown deadline as required inputs (INV-14) |
| `publish-dataset.yml` | merge to main | build a schema-versioned, content-addressed CDLA evidence release and publish identical records through authorized mirrors; Hugging Face Hub is the initial dataset mirror |
| `release.yml` | tag | one tag releases both distributions at the same version. Jobs in order: build `apron` and `apron-conformance`; attest build provenance and the applicable SBOM with an explicit `actions/attest-build-provenance` step (GitHub artifact attestations are not enabled by default); create the GitHub Release with the assets and the `git-cliff` changelog as its body, the one job holding `contents: write`; publish to PyPI by trusted publishing in environment `pypi` with no stored token, which produces PEP 740 attestations by default — these are Sigstore bundles, and PyPI accepts no separate signature upload, so no cosign step exists. Attestation precedes release creation because immutable releases lock assets at publication. A PyPI release rejects new files added more than 14 days after it was created (policy effective 2026-07-22), so a release is published complete. Corrected 2026-09-11: the separate `attest-release.yml` previously listed here ran after the release existed, which immutable releases make too late; it is folded into this workflow |

Repository settings: a repository ruleset on the default branch (pull requests only, CI required, linear history), not classic branch protection; secret scanning with push protection, which is the only protection a new public repository receives automatically; CodeQL default setup, Dependabot alerts and security updates, and private vulnerability reporting, each enabled by hand; Dependabot for dependencies and Actions with a cooldown; the per-user open-pull-request cap for users without write access; fork pull-request workflow approval set to all external contributors; immutable releases enabled before the first tag; `SECURITY.md`; `CODEOWNERS` over the claim-bearing paths; issue template "numbers from your rig" in the recipes-compatible shape; pull-request template asking for the record id behind any rule change (INV-2) and carrying a one-line AI-disclosure field (`public-development-strategy.md` §4c); `CITATION.cff`; repository topics, social preview and a compact evidence-bearing badge set. Licenses in commit one: Apache-2.0 code, CDLA-Permissive-2.0 records, CC-BY-4.0 docs. `public-development-strategy.md` §5 carries the source for each setting.

Documentation: a Zensical site on GitHub Pages (methodology, invariants, metrics, ADRs) with mkdocstrings for the API reference and the mike fork for versioning; catalogue pages generated into the same site from records. The generator changed from mkdocs-material on 2026-09-11 and the reasoning is in `public-development-strategy.md` §8; the choice is not an architecture decision and does not change what the site publishes.

Community routing: GitHub Discussions receives Q&A and exploratory integration conversations; Issues receive actionable bugs/features and structured failure/evidence intake. A public organization Project may expose evidence gaps and contribution-ready work without importing private research planning.

`mondegreens/apron-action` is released when the canonical structured input/output schema for its first repository check is stable and an external fixture repository reproduces the expected Check Summary. The Action's hosted-runner path is GPU-free and must distinguish calculation, transferred evidence, proven constraints and measurement. GPU work is reachable only through the ordinary accepted-protocol, execution-target and authorization contracts; fork pull requests receive no secrets or side-effect authority. The Action is eligible for GitHub Marketplace because repository-native installation is the distribution mechanism, not because Marketplace presence is evidence of use.

**Corrections applied 2026-09-11.** Five statements in this section were changed, each because a primary source checked that day contradicted them, and each is recorded here because `framework-spec.md` §4 forbids a silent correction. The Python floor moves from 3.11 to 3.12: `tomllib` was the stated reason and it is in the standard library from 3.11, so the reason no longer distinguishes the two, and 3.11 is security-only with end of life in October 2027. The documentation generator moves from mkdocs-material to Zensical, because mkdocs-material's security fixes end on 2026-11-05 and MkDocs 2.0 removes the plugin system. The hook runner moves from `pre-commit` to `prek`, which reads the same configuration. The Sigstore cosign step is removed from `release.yml`, because PyPI accepts no separate signature upload and trusted publishing already produces PEP 740 attestations. "Branch protection on main" becomes a repository ruleset, and the claim that a new public repository arrives with Dependabot and code scanning is corrected: only secret scanning and push protection do. The `ty` rationale is rewritten, because the sentence that its stable path "still lists first-class Pydantic support as outstanding" was overtaken when that support shipped in July 2026; the decision is unchanged and the reason for it is now beta status rather than a missing feature. `public-development-strategy.md` §8 and its Corrections section carry the sources.

Not needed: self-hosted runners (Tier 1 calls the provider API from a hosted runner), coverage or code-quality SaaS, a repository per adapter, separate docs/MCP/evidence repositories, or a republished engine-image registry. Publish an Apron OCI image to GHCR only when a supported hermetic Apron runner or remote MCP deployment requires one. Introduce a GitHub App only when a required installation-scoped asynchronous or cross-repository workflow cannot be expressed safely as the Action. GitHub Discussions remains the public community surface unless support and moderation demand justifies another channel. `FUNDING.yml` and GitHub Sponsors remain disabled while the project accepts contributed compute access but not money.

## 2. The ten-step product loop

| step | what "usable" requires | implementation | status |
|---|---|---|---|
| 1 Accept outcome | immutable `DecisionRequest` with task success, quality, serving, policy, authorization and objective | human or authorized caller accepts an AI-assisted or explicit request | contract defined by ADR-011 |
| 2 Define evidence | versioned task suite, application/agent/tool graph, serving workload and evaluation protocol | canonical specs; adapters for Inspect, Harbor, trace experiments, HTTP services and user commands | existing engines reusable; Apron contracts required |
| 3 Discover solutions | managed APIs, self-hosted model/artifact candidates and compound role/routing graphs | attributed artifact/provider/planning sources plus explicit user candidates; source coverage disclosed and cheap deterministic pruning first | resolver expansion required |
| 4 Resolve deployment facts | source-neutral artifact identity, immutable source revision, manifest/component digests, bytes, attributed license/gating claims and hardware/provider targets | conforming artifact sources led by Hugging Face and local files, plus provider APIs and hardware detection; no weight download where manifests expose sufficient facts | core lineage materially reusable; permanent resolver contract required |
| 5 Evaluate task | execute accepted protocol; retain every score, failure, retry, trace and attributable cost | evaluation adapters return canonical `TaskAttemptRecord`s | adapter implementation required; engines exist |
| 6 Plan and select | calculate feasible endpoint configs; reconcile owned and external `PlanningClaim`s; enforce authority; qualify task and serving constraints; rank bounded comparable solutions | owned mechanism-aware calculator, conforming planning sources, evidence graph and deterministic optimizer; planners generate candidates but never verdicts | genuine core build |
| 7 Render and deploy | API selection or exact `vllm serve`, Compose, Helm/provider artifacts; health and lifecycle | endpoint `DeploymentPlan`s plus execution targets | llmcalc and existing renderers materially de-risk self-hosting |
| 8 Reproduce and measure | exact retained solution reruns task protocol and serving benchmark; outcome economics computed | evaluation adapter plus `vllm bench serve` or provider measurement | task and serving proof remain separate |
| 9 Diagnose and correct | traceback or regression → fingerprint → deterministic correction → re-render/redeploy | rule table seeded by failure injection and observed qualification regressions | differentiated build |
| 10 Record and requalify | append-only decision, attempt, deployment and evidence records; invalidate on material change | freshness clocks, fingerprint diffs, consented publication and replay | canonical evidence engine |

The self-hosted calculation/deployment path remains materially de-risked by llmcalc's RunPod implementation and deployment history. The genuine builds are the canonical decision/evidence graph, mechanism-aware calculator, bounded solution optimizer and proven diagnosis/requalification loop. Evaluation execution is integrated through conforming adapters; Apron does not need to recreate every task runner, sandbox, trace store or judge implementation to own the decision.

## 2a. The full product scope — every question the product answers (2026-09-07; expanded 2026-09-09 by ADR-011 and ADR-013)

**Source:** advocacy session — comprehensive feature verification against the production chain.

The product calculates on demand for capability signatures whose artifact/component execution mechanisms it recognizes, using its own GPU-free predictor and versioned engine constraints. It returns explicit unknowns for unsupported mechanisms, modality combinations, artifact transformations, kernels or topology effects. Quantized choices refer to real immutable checkpoint artifacts, not flags imagined over a base repository. GPU records verify and calibrate predictions over time; discovered engine capability is never mislabeled as calculator, endpoint or task evidence.

| user question | answer | mechanism |
|---|---|---|
| Which model or model system should I use for my task? | A scoped answer, never a universal ranking | Compare accepted candidates through the same task suite and application protocol; retain exact fingerprints, uncertainty and exclusions |
| Should I use a managed API or self-host? | Yes when the candidates have comparable task, serving and cost boundaries | Measure both against the accepted outcome; expose provider-opaque fields and utilization assumptions rather than inventing parity |
| Should different roles use different models? | Yes, as a compound `InferenceSolution` | `ApplicationSpec` defines endpoint-independent logical roles; the solution binds concrete endpoints and routing, with component identities and whole-system task attempts |
| Can one endpoint route across models or replicas? | Yes, without pretending the alias is one deployment | Distinguish application/model routing from replica selection; retain every declared backing identity and record the selected target per attempt or `routing_opaque` |
| Can several machines serve one model? | Yes only through a supported topology | Independent replicas add throughput but not one-request memory; capacity is combined only for a verified sharded/disaggregated execution group |
| Can the solution understand image, video or audio? | Yes when the exact input combination is exposed and verified | Match an accepted `CapabilitySignature` such as `{text,image} -> generated_text`; preserve checkpoint claim, engine support, endpoint exposure and request/task proof separately |
| Can it embed, classify or rerank rather than generate? | Yes | Treat vectors, labels and scores as result representations over their own pooling/scoring mechanisms and serving protocols, not as text-generation variants |
| Can it generate image, video or audio? | Yes through a conforming generated-media engine, never as an invented vLLM flag | Resolve the media pipeline component graph, calculate its denoising/decode workload, deploy and verify through a media adapter, then evaluate the accepted output task |
| Can a voice or multimodal application use several endpoints? | Yes, as an explicit cross-modal solution graph | Preserve transformations and evidence for ASR, language/vision, retrieval and TTS/media endpoints plus the end-to-end accepted outcome |
| What does one successful task cost? | Observed when attempts are measured; otherwise modeled with assumptions | Retain failures/retries/tools/judges and endpoint or allocated infrastructure cost; report attempt and accepted-outcome economics separately |
| Does this model appear to fit on my GPU? | Predicted before rental; verified only after execution | GPU-free Apron prediction against immutable metadata + declared GPU; GPU-executed vLLM measurement for verification |
| What flags do I use? | Yes | Config sweep produces exact `vllm serve` command and Compose file |
| What quantization should I use? | Ranked real candidates with exact evidence states; `Recommended` only after boot plus evaluation-derived task evidence and serving evidence for the exact endpoint/application | Traverse the artifact/transform/execution graph, calculate actual bytes, apply scheme/backend/version constraints, then reproduce the accepted outcome; never infer compatibility or quality equivalence from `FP8` or a repository name alone |
| What if I have 2/4/8 GPUs? | Yes | TP={1,2,4,8} swept, divisibility checked via vLLM validators, best TP recommended |
| What's the most efficient permitted deployment for my workload? | Ranked answer, with strength limited by evidence | Filter infeasible and unauthorized candidates; compare total workload cost, latency, throughput, time-to-ready, availability and declared trade-offs—not hourly GPU price or VRAM alone; verify promising candidates when authorized |
| What context length can I use? | Yes | KV formula computes max context for the available VRAM at the chosen config |
| How fast will it be? | Measured when a matching record exists; otherwise a labeled calibrated prediction or honest `unknown` | Benchmark records with TTFT/TPS/P50/P90 at stated workload; modeled values carry provenance, calibration scope and uncertainty and are never relabeled as measurements |
| It crashed, what do I do? | Yes | Run wrapper captures failure → diagnosis rule → corrected command → mechanism proof + accepted-request replay |
| vLLM updated, does my config still work? | Yes | Tier 0 per-tag CLI schema diff, version-aware display on all records |
| NVLink or PCIe — does it matter? | Yes | Topology-aware recommendation (interconnect type in HardwareSpec) |
| What are my alternatives? | Yes | Ranked configuration plus alternatives at different GPU count, quantization and cost, with objective, evidence state and trade-offs disclosed |
| Can I verify the answer myself? | Yes | `apron verify PLAN --target local` runs on the user's GPU and saves a local `VerificationReport`; publication requires separate `apron submit` consent |
| Can an agent use this? | Yes | MCP server (`apron mcp`), Claude calls it silently |
| Kubernetes / Helm deployment? | Yes (Phase 2) | Helm values rendering for vLLM's chart, production-stack, llm-d |
| Can the tool test on a rented GPU for me? | Yes (Phase 2) | User authorizes the user's own provider account; provider bills the user; report stays local unless the user separately runs `submit` |
| Does SGLang work too? | Yes under the independent engine-neutrality obligation | Second LLM-serving adapter through the same neutral core and exact shared capability signature |
| Does the architecture work beyond token-decoding engines? | Yes under the independent mechanism-neutrality obligation | A generated-media adapter proves different component, workload, unit and protocol contracts; this proof cannot be replaced by SGLang |
| Does the architecture work beyond one accelerator stack? | Yes under a separate accelerator/backend-neutrality obligation | A conforming hardware/runtime backend earns its own calculation, execution and evidence support; engine or modality support alone cannot imply it |

**Claims the product refuses to manufacture:**

- "Which model is generally best?" has no context-independent answer. Apron can identify a best-observed or measured-efficient solution only for an accepted task suite, application, serving workload, candidate set, policy, objective and evaluation protocol.
- An opaque managed API is not assigned an invented artifact, engine or hardware fingerprint. It can be qualified through observed behavior while its hidden fields remain `provider_opaque`.
- Hardware, mechanism or capability-signature support that has no conforming adapter is `unsupported` or `unknown`, not silently approximated. A publisher modality claim, engine-resolved signature and configured endpoint are separate scopes. Apple Silicon and CPU are valid architecture targets when their hardware/engine adapters and evidence exist; they are not excluded from the product model merely because the initial vLLM/CUDA execution path does not implement them.
- A remote image, audio or video URL is not fetched merely because a model accepts that modality. Retrieval needs accepted data-destination policy and an authorization envelope; adapters enforce domain/redirect and decoded-size limits, isolate media processing and keep private media and derived embeddings out of logs and published evidence by default.
- A Hugging Face id, model card, registry tag, provider mapping or external planner recommendation is not artifact, capability or qualification truth. The source observation remains visible; immutable content identity and the applicable deterministic or executed evidence decide what can be promoted.

**How strongly Apron may state a result:**

| epistemic status | meaning | permitted claim |
|---|---|---|
| `derived` | deterministic calculation replayable from identified inputs | the calculated value and derivation |
| `proven_constraint` | a scoped mathematical or pinned-source rule establishes compatibility or impossibility | the verdict only inside that rule's declared scope |
| `predicted` | a named model estimates a runtime outcome from qualified inputs or prior evidence | predicted value with provenance, calibration scope and uncertainty |
| `measured` | the exact execution produced the raw observation | observed value for that execution fingerprint only |

These statuses are independent from the authority level of the evidence source. A prior measurement—even from a very similar stack—remains prior evidence; when used for another proposed execution it contributes to a new `predicted` result and never becomes that execution's measurement.

**How results are produced — the qualification chain:**

The product does not pre-compute the Cartesian product of every model, API, role assignment, artifact, runtime and target. It discovers from the accepted request, performs cheap deterministic pruning, and spends evaluation or GPU budget only on candidates whose expected information can change the decision.

1. **Discovered and capability-eligible:** Candidate managed APIs, self-hosted artifacts and compound solutions are discovered through attributed sources and resolved far enough to match the accepted input combination, operation and output representation and to apply tool/API, context, license, privacy, policy and known compatibility constraints. Self-hosted identity comes from the immutable resolved manifest and content/component digests, not a registry name. Public benchmarks, model cards, planner outputs and prior records are attributed priors, not proof for the user's task.
2. **Task-evaluated:** Conforming evaluation adapters execute the accepted `EvaluationProtocol`. Every attempt, failure, retry, score, trace reference, time and attributable cost remains available. This evidence is scoped to the exact task suite, application and candidate solution.
3. **Calculated and deployment-verified:** For supported self-hosted mechanisms, Apron resolves immutable artifact facts and target facts, calculates endpoint candidates GPU-free, and may ingest conforming external `PlanningClaim`s before using actual GPU execution for engine conformance, boot and measurements. Planner disagreement remains visible; no planner label promotes the candidate. Managed APIs receive observed health and behavior evidence while hidden runtime fields remain opaque. Produced evidence may come from the maintainer pipeline, explicit local verification, or user-authorized provider resources billed directly to the user.
4. **Serving-verified and task-reproduced:** Native serving benchmarks measure latency, throughput and goodput at the accepted `ServingWorkloadSpec`; they do not measure correctness. The accepted task protocol is replayed against the exact retained solution. Only the combination can satisfy final qualification.
5. **Qualified, ranked and requalified:** The optimizer filters failed task, serving, policy and authorization constraints, then ranks the remaining comparable solutions by the accepted objective. Material changes to tasks, application, model/API, artifact, runtime, evaluator, provider, pricing or serving workload invalidate or trigger reassessment of the exact qualification.

Each stage returns what it has and labels both evidence authority and epistemic status honestly (§2a answer ladder). Publication remains a separate consented action.

Selection follows a permanent qualification-obligation graph: cheap capability and policy checks establish eligibility and authorization bounds; task evaluation and applicable deployment/serving checks may then be scheduled in the cheapest authorized order; exact retained-solution task reproduction and every applicable serving SLO are required before objective-based ranking can promote a qualified solution. Implementations may interleave measurements to reduce cost, but cannot skip a qualification obligation. Availability can remove an option but does not make the next option efficient. An unavailable preferred SKU may be substituted without new authorization only inside declared adaptation rules and hard cost/security/credential/data bounds. A request to measure a specific GPU makes that hardware part of the claim and forbids substitution.

The result vocabulary reflects the proof obtained: `discovered candidate`, `task-evaluated candidate`, `calculated deployment candidate`, `verified endpoint`, `qualified solution`, or `measured-efficient solution`. The last term means best observed among disclosed comparable candidates under one task, application, serving, economics and evaluation protocol. Measuring or qualifying one candidate proves that it works within that scope; it does not prove optimality.

**The dimensions the GPU-free candidate sweep represents:**

| dimension | values | resolved from |
|---|---|---|
| Capability signature | accepted input combination, operation and output representation | task requirement matched separately to publisher claim, pinned engine and configured endpoint |
| Component graph and mechanisms | decoder/cache, pooling, encoder-decoder, media encoder/projector, denoiser/media decoder or another typed mechanism | immutable artifact metadata plus pinned adapter resolver; architecture name is an index, not a formula selector |
| Model architecture | engine-registered or explicitly unsupported | config and pinned engine registry; does not itself prove capability |
| Accelerator/backend + usable memory | every target declared by a conforming hardware/execution adapter; initial support is narrower and explicit | backend-native detection or an accepted declared target; engine and mechanism support do not imply hardware support |
| GPU count (TP) | 1, 2, 4, 8 | swept, divisibility checked by vLLM validator |
| GPU topology | NVLink / PCIe | HardwareSpec interconnect field |
| Quantization candidate graph | checkpoint-native, publisher/third-party quantized artifacts, reproducible offline conversions, pinned-engine online transforms, mixed schemes | each node has immutable artifact, transform, execution, compatibility and quality identities; unsupported adapters remain explicit |
| Context length | user-requested | KV formula computes max that fits |
| Concurrency | user-requested | KV cache scaled to stated concurrency |
| Pipeline parallel | 1, 2, 4 | PP support flag from registry |
| KV cache quantization | bf16, fp8, nvfp4 | compatibility checked |
| Workload shape | token distributions, image count/dimensions, video frames/fps/duration, audio duration/sample rate/channels, vector shape, streaming and operation-specific parameters | accepted `ServingWorkloadSpec`; feeds the applicable mechanism calculation and benchmark |
| Prefix caching | on/off | model-specific recommendation |
| Engine version | pinned per container tag | Tier 0 tracks per-tag changes |

The candidate matrix can contain thousands of combinations, but the planner never manufactures them by crossing arbitrary flags. GPU-free calculation traverses only resolved artifact/transform nodes and applicable typed adapters, prunes provably incompatible choices with versioned constraints, and exposes `unsupported` or uncertain states. GPU execution validates selected runtime paths. Viable, rejected and unresolved candidates remain auditable rather than disappearing behind one answer.

## 2b. The answer ladder: what the user gets at every level of coverage

| level | when | what the user gets |
|---|---|---|
| measured-efficient solution | comparable candidates were measured under the same accepted task, application, serving, economics and evaluation protocols | the best observed solution in the disclosed evaluated set, plus alternatives and trade-offs; never a universal claim |
| qualified solution | the exact retained solution reproduced the accepted task outcome and passed every applicable serving constraint | exact solution graph, endpoint plans, task and serving evidence, outcome economics, freshness and limitations |
| task-evaluated candidate | the accepted task protocol ran, but deployment or serving qualification is incomplete | per-attempt outcomes, scores, failures, time/cost and complete task/application/evaluator/candidate fingerprints |
| verified record | the model, card, and engine tag were booted in the lab or independently reproduced | the exact command that booted, measured memory, any workload measurements actually executed, and the record permalink |
| related-artifact evidence | an artifact has an explicit structural, claimed-derivation, reproducible-derivation, tokenizer, or quality relation to a verified artifact | the source record and exact relation, visibly scoped; matching shapes alone never transfer a measurement or establish equivalence |
| prediction | no matching execution record | predicted fit, predicted incompatibility, or explicit unknown from Apron metadata calculations and version-pinned constraints; never labeled vLLM-validated |
| unknown | the engine itself does not recognize the architecture | an explicit "unrecognized" and a queue entry; never a guess |

Only the last level is an empty room. Coverage is cheaper than the record count suggests: infeasible combinations fail in seconds for cents and are still records; lineage inheritance lets one boot cover many derivative checkpoints; and the catalogue form factor lets others add records on cards the lab never rents.

### 2c. Why records publish at every evidence level, not only when fully benchmarked (2026-09-07)

**Source:** advocacy grilling — "prove that publishing without throughput data is not lazy."

The user's questions come in order: (1) "does it appear to fit?" (2) "what candidate flags should I try?" (3) "does it actually boot and how fast is it at my workload?" Questions 1-2 can receive a labeled prediction before GPU rental. Question 3 requires GPU execution. If every prediction required a benchmark before publication, the user would rent blind; if prediction were labeled validation, the product would manufacture certainty it does not have.

Precedents: club-3090 published boot-and-VRAM records before full benchmark suites; contributors who copied the Compose file then submitted full benchmarks later [W: github.com/noonghunna/club-3090]. NVIDIA's aiconfigurator produces plans without throughput data for cards outside its silicon database and says to verify results in real benchmarks [S: [aiconfigurator](https://github.com/ai-dynamo/aiconfigurator) README].

Budget math: 4 benchmark presets × 60 feasible model+GPU combinations = 240 runs = $14-41 in GPU time — 20-60% of Phase 1b budget on benchmarks alone. Publishing feasibility-only records for lower-priority combinations preserves budget for benchmarking the top-demanded ones.

Each evidence level answers a different question:

| level | what it answers | cost | useful alone? |
|---|---|---|---|
| Prediction | "Does it appear feasible enough to rent?" | $0 (CPU) | Yes — filters provably impossible candidates and exposes unknowns |
| Boot-verified | "Does it actually boot? What's the real VRAM?" | $0.17-$1 | Yes — gives working Compose file + measured memory |
| Workload-verified | "How fast at my workload?" | +$0.06-$0.17 per preset | Yes — decides whether perf meets the SLA |

Design rule: answer immediately at the cheapest level available, label the evidence level honestly (§2a ladder), and upgrade when budget, consented contributor submissions, or on-demand paid verification produces the next level. Local verification does not enter the public evidence corpus unless the user runs `apron submit`.

When a verified or related-artifact record's engine tag is more than two minor releases behind the user's detected or declared engine version, both the prediction and measured evidence display their engine/source version visibly, and the output invites the user to run `apron verify` against an explicit target. A prediction is correct only for its recorded rules and inputs; it is not execution evidence for the user's engine version. The resulting report stays local unless separately submitted.

## 3. First public product proof

The internal Phase 1a conformance fixture proves the deployment/diagnosis engine; it is not the complete public product claim. The first public proof begins with an accepted task suite, evaluates a disclosed candidate set, calculates and deploys at least one self-hosted endpoint, measures serving behavior, reproduces the accepted task outcome on the exact retained solution, reports outcome economics, and demonstrates request-preserving diagnosis or a qualification regression. Managed APIs and compound solutions use the same permanent contracts even when a particular proof does not exercise every solution class.

A web app, generated pages, broad engine coverage and community-scale evidence are not prerequisites for this proof, but neither are they defined out of the architecture. The usability bar is a reproducible `DecisionReport` and useful retained solution or honest no-qualifying-candidate result. If comparison coverage is insufficient, the endpoint may be qualified but is not called best or optimal.

## 3a. Where v1 is weak, and the three strengtheners

Weak: zero records until the first sitting; the first question people ask ("what GPU do I need") is answered for free by an LLM at roughly 80 percent accuracy, so the tool wins only on exact flags at a pinned tag, verification, and diagnosis; diagnosis starts at six proven classes; reaching the tool takes an install or an MCP configuration; the oracle value depends on others consuming it.

Strengtheners, in order of leverage:

1. **Diagnosis at the point of failure.** `apron run -- vllm serve ...` wraps the engine, captures the failure, and returns the corrected command in the same terminal with its proving record. No pasting, no page. This is the "I no longer think about deployment" experience and is Phase 1 scope.
2. **Repository qualification where changes happen.** A pinned `mondegreens/apron-action` converts model, artifact, engine, workload and deployment changes into the same canonical Apron check. The workflow reference keeps Apron present in the engineering loop; only completed checks and downstream actions count as use.
3. **MCP where agents already look.** Publish the same package to conforming MCP registries with a one-line install, so an agent uses it silently and the user never needs the tool's name.
4. **Upstream preflight.** If a vLLM dry-run lands upstream, Apron can consume it in the GPU execution path; CPU predictions and measured records retain their separate authorities (ADR-003, ADR-008).

### 3a-1. Retention versus acquisition (2026-09-07)

**Evidence:** Study of 115,466 GitHub repos (arxiv 2507.21678): user interaction weight is the #2 predictor of project survival. Solo planning-only tools (calculators, config generators) in this niche have zero survivors past 1 year — all are episodic (user asks once, gets answer, leaves). Solo serving tools (LocalAI 3.5yr, oobabooga 3.7yr) survived because users engage continuously.

The one-shot `apron deploy` is one **acquisition** surface. The GitHub Action is repository-native acquisition because a model, engine, application or deployment change can invoke Apron without a separate visit. The **retention** surfaces — how someone stays — are:

1. **`apron run -- vllm serve ...`** — wraps every deployment, captures every failure, returns corrections. The user runs this instead of bare `vllm serve`. Every deployment is a touch point.
2. **`apron mcp`** — an agent calls the tool on every deployment question. The user never types `apron`; the agent does. Every agent interaction is a touch point.
3. **`mondegreens/apron-action`** — a pinned workflow re-evaluates relevant repository changes and identifies invalidated evidence. Every completed material check is a touch point; installation, a badge view or a skipped workflow is not.

The distinction matters because the 90-day checkpoint must measure RETENTION, not just ACQUISITION. A user who planned once is the calculator pattern (death). A user whose agent calls the MCP tool on every deployment question is the serving pattern (survival).

### 3a-2. Community record submission (2026-09-07)

**Evidence:** club-3090 got 11 hardware profiles from 22 contributors at $0 to the maintainer — each contributor ran `report.sh --full` (~35 min) on their own GPU. Renting those 8 additional SKUs for Tier 1 would cost $800–1,600/month.

The catalogue's coverage is bottlenecked by the maintainer's rented hardware (Phase 1b: 3 SKUs). Users with uncovered cards (3090, L4, A6000, RTX 5090, DGX Spark) find nothing and leave.

**`apron verify PLAN --target TARGET`** executes the verification suite on an explicitly selected local, remote or rented GPU and saves a structured `VerificationReport` locally. `apron report RUN_ID` only displays or exports that result. `apron submit RUN_ID` sanitizes, previews and explicitly publishes it. The same verification pipeline serves Tier 1 and users; execution, inspection and publication are separate Phase 1a contracts.

Each record-submitting contributor has: used the tool, learned the format, owns hardware the maintainer doesn't, and has a reason to come back when vLLM updates.

## 3b-1. Public-facing feature presentation (2026-09-07)

The concept documents are written in design language. The repo must speak in two voices to two audiences, both in the same README, separated by writing style, not by labels.

**Top of README — the user who has a GPU and wants to run a model.** They may not know what vLLM is. They may not know what tensor parallelism means. They want: "type this command, copy this file, it works."

Content: the 2am pitch (§0a), the one-command example (§1), the result they get (the Compose file + "fits / doesn't fit"), and the three selling points: (1) answers before you commit — 10 seconds, $0; (2) fixes when it breaks — corrected command with proof; (3) stays current — tracks biweekly vLLM releases. No architecture terms. No invariant numbers. No jargon. The writing does the job. The user reads the top and has everything they need.

**Bottom of README — the developer building an inference stack.** They know vLLM. They want: the MCP tool schema, the Python API, the record format, the interop with recipes/aiconfigurator/InferenceX, how to consume the dataset, how to contribute records.

Content: MCP server configuration, API reference link, record schema overview, interop format examples, `verify`/`report`/`submit` contributor and consent guide, link to the full docs site.

**The transition between the two is natural, not labeled.** A heading like "## How it works" or "## API & MCP" shifts the tone. The user stops scrolling. The developer keeps going. Nobody is called a beginner.

**Pattern:** the same structure used by vLLM (pitch → architecture), club-3090 (Compose example → benchmark methodology), FastAPI (10-line example → full feature list).

**Additional surfaces:**

- **PyPI long_description** — mirrors the README.
- **MCP registry listing** — one-line: "Verified vLLM deployment planning — fits-or-not before download, diagnosis after failure, working Compose files with measured numbers."
- **GitHub Marketplace Action** — repository-native qualification and evidence invalidation from the same released core; the listing leads with the Check Summary rather than generic CI claims.
- **GitHub Pages docs site** (Zensical, §1c) — the full methodology, invariants, metrics, ADRs, catalogue pages, `architecture-dispatch-proof.md`. For contributors and upstream consumers.

The content exists in the concept. The packaging is an implementation deliverable — write it when the first record exists, not before. A README that says "0 verified records" is honest (§3b Honesty rule) but a README with a working example is better.

## 3b. CLI cautions (2026-09-06)

| area | rule |
|---|---|
| Rendering | Detect half-block glyph and 24-bit color support; fall back to 16 colors, then no mascot. Honor `NO_COLOR` and `TERM=dumb`. Drop the mascot below ~60 columns, one line below 40. Test on Windows Terminal, WSL, iTerm, Ghostty, plain Linux console. |
| Non-TTY | When stdout is not a TTY: no banner, no color, no prompts, JSON output. The shell is a view; the subcommand is the product (INV-11). |
| Exit codes | One enum and one exception hierarchy produce every exit code and every typed structured outcome; a test asserts the mapping is unchanged across versions, because CI pipelines, the Action and MCP consumers branch on it. Distinguish no qualifying solution, unsupported task/evaluator, does-not-fit, serving-SLO failure, task-outcome failure, unrecognized deployment failure, provider error and budget cap. Never block on a question when non-interactive; an unaccepted `DecisionRequest`, task suite or rubric exits with its own code and prints the immutable proposal to accept (INV-7). |
| Money | No paid evaluation, judge call or provider job without the applicable spend cap and lifecycle/deadline controls (INV-14, INV-27). Status separately shows the cap, gross attributable cost, market-equivalent/user-relevant cost, contributed credit and project out-of-pocket cost; private project subsidy never makes a user candidate appear free (INV-29). |
| Secrets and task data | Tokens come from environment or keychain only, never flags. Secrets are redacted before storage. Task data, prompts, traces and outputs stay local unless the accepted data policy and authorization explicitly permit the destination (INV-13, INV-26). |
| Run wrapper | Passes engine arguments through untouched; proposes a corrected command and waits, or applies only when explicitly asked. |
| Honesty | Every number carries its evidence level and freshness, by label not color alone. "0 verified records" is displayed when true. |
| Startup | Banner in well under a second; no torch, transformers, or engine import in the CLI process. No weight download or multi-gigabyte image pull without stating what and how much. |
| Docker | Preflight detects the socket and GPU runtime and names exactly what is missing. |
| Naming | Confirm the name is free on PyPI, Homebrew, and GitHub before it appears on a banner. |
| Schema | Version the JSON output schema from day one; print it in `--version`. |
| Telemetry | Off by default; opt-in with a one-line statement of what is sent; counting rules published. |

## 4. Usability risks and mitigations

| risk | mitigation |
|---|---|
| Deploying to the user's own machine needs a local executor | v1 deploys to Docker on the host where the CLI runs; provider deploy is the same script with a different backend; Kubernetes stays a render target until Phase 2 |
| A diagnosis rule that fires wrongly is worse than none | a rule without corrected-boot mechanism proof is a hypothesis; an application is `Fixed` only after the accepted request also passes, otherwise it is an explicit trade-off or unverified suggestion |
| Gated models (Llama, Gemma) in the lab | the user's own HF token for their deployments; the lab uses the maintainer's token and records the license accepted |
| Provider drift | thin backend interface; two backends from day one |
