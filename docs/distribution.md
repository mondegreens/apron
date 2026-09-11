# Distribution by solving — verified help as a publication and acquisition hypothesis

**Date:** 2026-09-07 · **Source:** advocacy session — verified against live GitHub issues and forum threads

## The mechanism

Verified answers to real deployment failures are a plausible distribution channel, not guaranteed acquisition. The pipeline may identify an open failure, test a mechanism-verified rule against the issue's accepted request, and prepare a useful response. A bootable change that alters the request is disclosed as an alternative with trade-offs, never as a fix. The evidence and diagnosis remain valuable even when publication is not authorized or produces no user. Detection, evidence, publication and acquisition are separate outcomes (ADR-010).

## Evidence: real issues the product could solve TODAY

Verified 2026-09-07 via GitHub API and web search. These are real people, stuck right now, with problems matching the product's diagnosis rules.

### vLLM issue tracker

| issue | posted | problem | diagnosis rule | zero replies? |
|---|---|---|---|---|
| #54318 | Aug 29, 2026 | Qwen3.8-Flash-Next-FP8 on 4× A100-80GB — Triton fused-MoE reports "fp8e4nv not supported" | Scoped external failure observation; correction unknown, rule remains `hypothesis` | No, and no verified fix posted |
| #55517 | Sep 6, 2026 (yesterday) | TP=3 on 3× A100 — "16 not divisible by 3" | `tp-divisibility` (vision heads=16, valid TP: 1,2,4,8) | Recent |
| #52029 | Aug 12, 2026 | opt-125m init failure on 8 GB GPU | Environment issue, not config — dry-run confirms model fits | **Yes — zero replies, ~1 month** |
| #54377 | Aug 29, 2026 | Cohere Transcribe on Turing GPU — dtype mismatch | `dtype-incompatibility` (Turing + fp16 → known attention bug) | **Yes — zero replies** |
| #55775 | Sep 7, 2026 (today) | RTX 5090 + Qwen3.8-27B-NVFP4 — CUDA crash at long context | MTP + FlashInfer engine bug → `--disable-mtp` workaround | **Yes — zero replies** |

**Scale:** 191 open issues matching known diagnosis patterns. 17 with zero replies. 53 "Engine core initialization failed" since June 2026.

Issue #54318 is a real example of why the product needs scoped failure records: the generic FP8 capability gate did not predict the selected runtime kernel failure, and the issue contains no corrected boot. It is not a verified diagnosis rule.

### vLLM forum (discuss.vllm.ai)

- "Running out of memory despite very low gpu-memory-utilization" (July 2026) — user confused about 16 GB GPU OOM. Apron: "weights exceed budget at any utilization; utilization controls the fraction, not the amount."
- "Does vllm inference work with Qwen3-VL-30B" (Nov 2025) — 2× RTX 5090 OOM. Apron: pre-calculate exact budget at TP=2.
- "torch.OutOfMemoryError" (March 2026) — RTX 5060-Ti 16 GB, multiple people with similar problems.

### Broader surface

- **3,645 open issues across all GitHub repos** matching vLLM deployment errors (not just vLLM's tracker — model repos, tooling repos, deployment repos)
- **Entire websites** exist to answer this question with static tables: willitrunai.com, sector88.co, markaicode.com, runaihome.com, spheron.network "GPU Requirements Cheat Sheet." All give generic advice. None run the engine's validators. None give exact flags for a specific model+GPU+version.
- **club-3090 issue tracker** demonstrates that users will contribute benchmark numbers through issue templates; Apron preserves that pattern while separating GPU execution (`verify`), local inspection (`report`), and consented publication (`submit`)

## How it works in the product

### Publication flow

After the product produces a mechanism-verified diagnosis and request-satisfying application:

1. Signal intake may create or update a deduplicated internal `AnomalyCase`; it never posts directly.
2. Diagnosis creates evidence and, only when useful to the affected user, a `ProposedExternalAction` containing the scoped cause, correction or disclosed alternative, proving record and relevant artifact.
3. A separate publisher checks the destination's standing owner policy, provenance, accepted-request proof, sanitization, duplicate state, confidence, rate limits and local norms.
4. The publisher may retain the proposal locally, create a draft for review, or publish automatically when standing policy authorizes that destination and action class.
5. A response names or links Apron only when that provenance is relevant and permitted; it is not forced advertising appended to every answer.

### Guardrails

| risk | guardrail |
|---|---|
| Spam perception | Publication destinations and action classes are allowlisted by standing owner policy. Visibility or star count cannot grant publication authority. |
| Wrong fix | Only `Fixed` applications are represented as fixes. Hypotheses remain unverified suggestions and request-breaking changes remain alternatives with explicit trade-offs. |
| Already-tried fix | The proposed action records the full issue state it evaluated and is invalidated or regenerated when that state changes. It never repeats a correction already reported as failed. |
| Terms and local norms | Automated posting requires destination-specific authorization and identity disclosure. Otherwise the output remains a local proposal or human-posted draft. |
| Promotion | The useful answer leads. Product naming, links and install instructions appear only when relevant to reproduction and allowed by destination policy. |
| Duplicate responses | Publisher checks the durable `ProposedExternalAction` and `PublicationAttempt` state by destination and deduplication key before posting. If an equivalent action exists, it updates or skips according to policy. |

### Precedent

- **Dependabot** — GitHub's own bot, opens PRs on millions of repos. Initially resisted, now standard. Self-distributes through usefulness.
- **RunLLM** — vLLM's own AI support bot on discuss.vllm.ai. Answers community questions automatically. Accepted because useful.
- **Renovate Bot** — opens dependency update PRs. Same pattern: automated, useful, accepted.

## What this means for distribution

The product definition notes that low distribution is the base case. This mechanism changes the distribution model from PASSIVE (publish and wait) to ACTIVE (solve problems where they're posted):

| distribution model | mechanism | controllable? |
|---|---|---|
| Passive (current concept) | Publish catalogue + MCP, wait for discovery | No — depends on search/word-of-mouth |
| Active publication hypothesis | Prepare verified help for authorized destinations; measure publication and subsequent use separately | Partly — useful preparation is controllable; permission, reception and acquisition are not |

Historical headline issue counts are not product claims under accepted D11; the preserved findings retain their scoped snapshots. A matching search result is not automatically a valid diagnosis, an authorized publication destination or a potential user. The mechanism is supported only by prospective measurements that distinguish prepared actions, authorized publications, useful responses and attributable subsequent product use.

### Repository-native distribution loop

ADR-012 adds an opt-in route that does not depend on publishing into somebody else's issue tracker. A model, inference-engine, application or deployment repository installs a pinned `mondegreens/apron-action` release; material changes invoke the same canonical Apron core and receive a GitHub Check that preserves calculated, transferred, proven and measured states, freshness, uncertainty and invalidation. The Action does not contain independent recommendation logic and does not call CPU analysis GPU validation. Authorized GPU verification uses the ordinary external authority and execution-target contracts.

This is both acquisition and retention: the first useful check introduces Apron where the change occurs, and subsequent material changes can re-evaluate the retained claim. A Marketplace view, workflow reference, badge impression or skipped job is not counted as product use. Completed checks, plans acted upon, invalidations resolved, authorized verifications and repeat material checks are counted separately. Public issue assistance remains valuable, but it is no longer the only active distribution hypothesis.

### Provider participation as a second evidence loop

Inference providers, GPU platforms and model labs may contribute credits, scoped credentials, quota or dedicated capacity when a conforming integration and reproducible records are useful to their users. The exchange is capacity for independent execution and attributable technical visibility, never capacity for placement or favorable conclusions. A provider may legitimately amplify a record in which it wins on the accepted task and economics; Apron remains credible because the same contract permits it to lose and does not grant pre-publication control.

Provider participation is additive distribution, not the cold-start plan. Phase 1a is budget-feasible through `MaintainerBaselineAllocation` with `ContributedResourcePool = 0`. Contributions can broaden hardware/provider coverage after passing the same adapter, authorization, evidence and publication rules. Each affected record discloses the contributor, terms relevant to interpretation, market-equivalent/user-relevant cost, subsidy and project out-of-pocket cost. `Supported provider` describes a tested adapter; `partner` or `sponsor` is used only after explicit agreement and brand approval.

This gives a provider four legitimate reasons to participate without compromising the product: a working integration, reproducible performance/compatibility evidence, lower support burden through diagnosis, and qualified demand from users whose accepted request the provider actually satisfies. No one of those benefits is promised, and none can promote evidence or authorize publication.

## What the product says in a solved issue

```
Observed failure `fp8e4nv not supported in this architecture` matches a
scoped hypothesis for this artifact, engine image, SM80 hardware, and Triton
fused-MoE path.

No verified correction exists for this exact execution fingerprint. apron
does not post a corrective command. Candidate backend, artifact, or engine
changes must first boot and pass the accepted workload on matching hardware.
```
