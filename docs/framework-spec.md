# Framework specification — extension points, invariants, metrics, governance

**Date:** 2026-09-06 · **Revised:** 2026-09-09 by ADR-013 routed/distributed serving topology · **Purpose:** state the system as a bounded, testable framework. This is the language the repository, documentation, and public pages use. Internal design principles are expressed here only as contracts, invariants, and metrics; metaphors do not appear in the repository.

## 1. Scope statement

The system is a deterministic qualification, planning, deployment, verification and diagnosis framework for inference solutions. It begins with an accepted task outcome and can compare managed APIs, self-hosted endpoints and compound role/routing systems without erasing their different evidence boundaries. It has ten typed extension points. A new capability enters through one of these contracts or changes the architecture through an explicit ADR; the extension-point list is not allowed to silently narrow the product.

| extension point | plugs in | contract (what a conforming implementation must provide) |
|---|---|---|
| **Engine adapter** | vLLM (initial), SGLang (engine-neutrality proof), generated-media and MLX/CPU engines | `accepts(capability_signature, component_mechanisms, serving_payload)`, `resolve_support(model_spec) → scoped_support_states`, `validate(plan) → verdicts` executed inside the pinned engine image, `render(plan) → artifacts`, `verify(artifact, backend) → report`, `classify(trace) → fingerprint`, `extract_schema(image) → cli_schema`; engine discovery never implies Apron calculation or verification support |
| **Artifact source/resolver** | Hugging Face Hub, other model registries, OCI/object storage, local filesystem and namespaced sources | `locate(query) → ArtifactLocator[]`, `resolve(locator) → ArtifactSourceObservation + ArtifactIdentity + attributed artifact claims + typed component mechanisms`; separates mutable source refs from resolved revisions and content/component digests; model-card fields remain publisher claims; golden fixtures preserve identical content across sources, changed content under one name, modality combinations, result shapes and unsupported stages |
| **Render target** | `vllm serve`, Compose, Helm values, provider formats | pure function of the neutral core plus the engine section; declares freshness status and owner |
| **Evidence source** | maintainer/user runs, Optima or other evaluation exports, Phoenix traces/experiments, InferenceX, recipes, aiconfigurator, community reports | `ingest() → records` with mandatory claim scope, evidence level, freshness clock and provenance; export mapping back to the source's shape where defined |
| **Planning source** | owned mechanism-aware calculator, AIConfigurator, provider recommenders, recipe planners | `propose(decision_request, resolved_candidates, target_facts) → planning_claims`; each claim preserves producer/version, exact inputs, proposed configuration, scope, epistemic status, provenance, calibration domain, uncertainty and unsupported/opaque fields; proposes candidates only and contains no qualification, ranking or recommendation authority |
| **Evaluation adapter** | Inspect AI, Harbor, HTTP/Optima-compatible agent, Phoenix experiment/trace replay, user command | `accepts(task_suite, application, solution, capability_signatures) → support_state`, `prepare(protocol, solution) → run_plan`, `execute(run_plan, authorization) → task_attempt_records`, `collect(run) → evaluation_evidence`, `replay(record, solution) → task_attempt_records`; preserves typed media references, every protocol and solution fingerprint, and contains no recommendation or authority logic |
| **Execution target/backend** | local container, remote container/SSH, RunPod, Modal; Kubernetes and external serving-fabric adapters | `prepare/provision`, `execute`, `observe`, `collect`, `teardown`; declares ownership/operator/source, spend policy, serving-topology support, observable route identity and exact execution fingerprint without changing plan/report schemas |
| **Signal source** | accepted task/application changes, local failures, model/API releases, evaluation regressions, engine issues/RFCs, recipes/planners, multiple registry trends, provider behavior/pricing/SKU availability | `poll() → queue_events` with source snapshot, scope, priority and dedup key; a signal can invalidate or queue evidence but cannot promote it, and no single registry metric defines eligibility or priority |
| **Authority source** | interactive owner approval, standing repository policy, provider/account controls, organization policy | `contribute(action_request, context) → authority_contribution`; immutable source/version; typed constraints over principal, action, resource and context; cannot be supplied or mutated by the executor for its own request |
| **Publisher** | portable evidence release/local archive, Hugging Face dataset mirror, GitHub issue/PR, upstream-specific adapter | `prepare(record_or_case) → proposed_action`, `publish(proposed_action, authorization) → publication_attempt`; record/release identity is host-neutral; sanitization, provenance, content manifest, deduplication, destination policy, rate handling and replay safety |

Every extension point has a conformance suite. A new implementation is accepted when the suite passes on the golden fixtures; there is no other acceptance path, and none is needed: the suites are published as a separate installable distribution, so an implementer runs the acceptance criteria against their own code, in their own repository, before proposing anything. Adapters register through packaging entry points, one group per extension point (`engineering-standards.md` §2, §7).

An external router, replica scheduler or distributed-serving control plane composes the existing planning-source, render-target, execution-target and evidence-source contracts. It does not create an eleventh extension point or become a core dependency merely because it performs several of those roles.

The authorization engine is fixed core, not an extension point. It evaluates an `ActionRequest` against every applicable `AuthorityContribution` and returns an immutable `AuthorizationDecision` plus its bounded `AuthorizationEnvelope`. Deny overrides permit; indeterminate, absent explicit permission or an unknown required obligation cannot authorize a side effect; independently supplied constraints intersect. `EvaluationAdapter`, `ExecutionTarget` and `Publisher` enforce the resulting envelope. Scheduling records why an authorized action was selected but cannot create authority.

## 2. Invariants (machine-checked; a violation fails CI or schema validation)

| id | invariant | enforced by |
|---|---|---|
| INV-1 | Every result assertion declares how it is known: `derived` from identified inputs and an inspectable derivation; `proven_constraint` from a scoped mathematical or pinned-source rule; `predicted` from a named model with provenance, calibration scope and uncertainty; or `measured` from the exact execution fingerprint and raw observation. A previous or cross-fingerprint measurement may inform a prediction but never becomes the proposed execution's measurement. Every numeric value carries unit and scope; a bare number is rejected. | schema validation, derivation replay and evidence-promotion tests |
| INV-2 | A diagnosis rule is `mechanism_verified` only with a scoped record proving that its correction removes the fingerprinted failure and restores engine health; otherwise it is `hypothesis`. Every concrete application separately records `mechanism_outcome` and `request_outcome` against an immutable accepted-request snapshot. `Fixed` requires `verified` plus `satisfied`; a healthy correction that violates any accepted constraint is `Alternative with trade-offs`; incomplete proof is `Unverified suggestion`. | CI over the rule table, remediation schema, corrected-boot test, and accepted-request replay |
| INV-3 | Adapter validation verdicts on the golden fixtures equal the pinned image's own verdicts. | conformance suite, executed in the image |
| INV-4 | Every record has an evidence level from the total order, a freshness clock, and provenance. Records from different levels that disagree are marked `contested`; nothing auto-resolves. | schema plus ingest test |
| INV-5 | `DecisionRequest`, `TaskSuiteSpec`, `ApplicationSpec`, `ServingWorkloadSpec`, `WorkloadSpec`, `InferenceSolution`, `CapabilitySignature`, `ArtifactLocator`, `ArtifactSourceObservation`, `ArtifactIdentity`, `ModelSpec`, `PlanningClaim`, `EvidenceReleaseManifest` and their typed payloads evolve append-only where possible; every historical fixture migrates forward without loss. | migration test suite |
| INV-6 | Values extracted from logs are typed numerics or enums validated against hardware facts. No extracted string is interpolated into a rendered command. | static check on renderers plus fuzz test on the extractor |
| INV-7 | An AI-proposed `DecisionRequest`, task suite, rubric, representativeness claim or workload cannot drive evaluation, rendering or deployment without the applicable recorded acceptance (principal/source, timestamp and accepted immutable revision). The proposer cannot accept its own proposal merely by acting as executor. | schema, API and authority-boundary tests |
| INV-8 | Every `DeploymentPlan` exports losslessly for the shared fields to the recipes YAML, aiconfigurator request, and InferenceX row formats, and re-imports to an equal neutral core. | round-trip test |
| INV-9 | Every workload and memory measurement carries an exact execution fingerprint. The planner refuses to present same-class, cross-GPU, cross-backend, cross-version or otherwise incomplete-fingerprint extrapolation as a measurement. | schema plus planner test |
| INV-10 | Unknown failure fingerprints are stored with trace, environment, and plan, and are reported as `unrecognized`, never as a guess. | classifier test |
| INV-11 | Interfaces (CLI, MCP, API, GitHub Action/CI, web) call the same core; no interface contains recommendation logic. | declared layer contracts checked as import boundaries in ordinary CI (`engineering-standards.md` §3) |
| INV-12 | Every published record is reproducible to the limit of its typed claim. A self-hosted deployment includes exact command, container/artifact revision, target and raw-log hash; a task outcome includes exact accepted task/application/evaluation/solution fingerprints and raw attempt provenance; a managed API exposes observed API/version/provider facts and marks hidden fields `provider_opaque`. Every record has lifecycle state `observed`, `superseded`, or `retracted`; correction keeps the original visible. Nothing is published as observed that the producing or declared external source did not observe. | kind-specific schema validation plus publish gates that re-parse raw deployment and task evidence |
| INV-13 | The lab executes model code only in ephemeral containers with no secrets mounted; tokens are injected for download only and never reach the run environment or logs; rendered artifacts default to loopback binding or authentication required; stored user logs are redacted before persistence. Remote media retrieval is deny-by-default and, when authorized, constrains domains and redirects, validates content, enforces byte/duration/dimension/frame/decompression limits, isolates decoders/processors, and prevents media or derived embeddings from entering evidence without permission. | backend/media contract tests, hostile-media fixtures and a secret scan on every log before storage |
| INV-14 | Every paid evaluation, judge call, model/API call or compute execution has an externally granted `AuthorizationDecision` bound to its action and accepted `DecisionRequest`, with hard per-job/period limits, maximum runtime, credential/provider/data-destination scope and applicable lifecycle/teardown rules. Adapters/backends enforce the envelope; scheduler estimates cannot replace or raise authorization. A paid job cannot start without immutable authority-source references. | authorization-engine, evaluation-adapter and backend contract tests; provider/account controls where available |
| INV-15 | Every record carries the model license and gating state; publication of measurements for a gated or restricted-license model requires a recorded permission check; provider catalogue data is never redistributed, only vendor specifications. | schema validation plus a publish gate |
| INV-16 | Every empirical or derived quantitative claim in public documentation has a reproducible scoped source/query and dated snapshot, an explicit `estimate` label with assumptions, or a test-generated derivation. Dates, versions, identifiers, ordered-list labels and schema enum/cardinality facts are not treated as empirical claims merely because they contain numerals. Unsupported headline totals fail the docs check. | docs lint over public quantitative claims; no runtime claim-ledger subsystem |
| INV-17 | Every quantization candidate references separate immutable artifact, optional offline/runtime transform, and resolved execution identities plus typed weight/activation/KV and linear/MoE schemes. Offline conversion produces a new artifact; repository names or matching tensor shapes cannot prove lineage; nominal bit width cannot replace actual tensor/scale bytes; unsupported methods return `unsupported` without changing the schema. | schema, resolver and calculator conformance tests |
| INV-18 | A self-hosted endpoint reaches `Recommended` only with boot evidence and accepted task and serving evidence for the exact endpoint/application fingerprint. An `InferenceSolution` reaches `Qualified` only after accepted task outcome is reproduced on the exact retained solution and every applicable serving SLO passes. Structural lineage, model name or prior API behavior never transfers these states. | promotion policy test over evaluation and verification records |
| INV-19 | `plan` performs no GPU execution; qualification and `verify` require accepted protocols plus applicable target/provider envelopes and save task/verification reports locally; bounded verification tears down; `deploy` explicitly creates or selects a continuing managed, self-hosted or compound solution; `run` wraps explicit execution; `report` is read-only; `submit` is the only evidence-publication command and requires sanitization, field preview and explicit consent. Paid actions require confirmation or standing authorization; every target has tested lifecycle semantics. | CLI side-effect tests, network-deny tests, consent test and adapter/backend conformance suites |
| INV-20 | Every produced or ingested execution record declares `claim_scope`, `production_mode`, and a machine-readable reason. Scheduling ranks evidence gaps across all supported targets using scope, missing/stale/contested coverage, demand, uncertainty/error, unresolved-failure value, freshness, adequate external coverage and cost; it has no class-based GPU inclusion or exclusion. External evidence suppresses a run only when it covers the required scope and fingerprint. Evidence cannot be promoted across claim scope or execution fingerprint. A hardware-specific claim makes the concrete target a hard constraint rather than a substitutable preference. | record schema, scheduler policy tests and evidence-promotion tests |
| INV-21 | An executor may act only inside an immutable `AuthorizationEnvelope` produced from external authority sources and cannot raise its own budget, widen credentials, extend its deadline or grant itself a new action class. A changed concrete target is reevaluated against the same envelope: it requires new authorization only if it violates a hard constraint, an allowed-adaptation rule, spend/security/credential scope or the accepted objective. Every authorization and orchestration decision has a stable id, state history and links to its inputs and outputs; every side-effecting attempt also has a deduplication key. | authorization-engine tests, privilege-boundary tests and crash/replay tests |
| INV-22 | Detection, evidence and publication are separate lifecycles. An anomaly first creates an internal case. Every external mutation requires a separately authorized, sanitized, provenance-checked, deduplicated and rate-controlled `PublicationAttempt`; failure or denial to publish cannot alter or discard the source evidence. | schema and publisher conformance tests, consent tests and duplicate-event replay tests |
| INV-23 | Selection is not "whatever is available." Cheap capability, policy and deployment-feasibility checks remove proven-invalid solution candidates; authorization removes actions outside the envelope; accepted task and serving evidence qualify candidates; and the deterministic optimizer ranks the remainder against the explicit objective. Every selection records the candidate set, exclusions, evidence states, evaluation coverage and trade-offs. `Measured-efficient` means best observed among the disclosed comparable solutions under recorded task, application, serving and economics protocols; one qualified solution is never `optimal`. | planner/optimizer golden tests, substitution tests and claim-language validation |
| INV-24 | Task semantics and serving load are independent. A throughput, latency, health or boot result cannot establish task success, and a task score cannot establish capacity, latency, availability or deployment compatibility. | schema separation, qualification-state and adversarial promotion tests |
| INV-25 | Every task outcome carries the exact `DecisionRequest`, `TaskSuiteSpec`, `ApplicationSpec`, `EvaluationProtocol` and complete `InferenceSolution` fingerprints. A changed prompt, tool, agent, routing policy, judge, artifact, API identity, inference parameter or evaluator produces new evidence; cross-fingerprint results may inform a prediction but cannot be emitted as measurements for the new solution. | task-attempt schema, fingerprint-diff and evidence-reuse tests |
| INV-26 | Task data, prompts, traces, outputs and evaluator inputs are private by default. An evaluation adapter cannot transmit them to an external candidate, judge, service or provider without a data-policy permission in the accepted request and a matching immutable `AuthorizationEnvelope`. Evaluation, telemetry and publication permissions never imply one another. | network-deny, data-label propagation, authorization and consent tests |
| INV-27 | Outcome economics retains every attempt, failure and retry and reports cost/time per attempt separately from observed or modeled cost/time per accepted outcome. Managed and self-hosted solutions cannot be ranked when workload horizon, quality acceptance rule or included cost boundary is incomparable or unknown. | aggregation property tests, failed-attempt fixtures and comparison-eligibility tests |
| INV-28 | `InferenceSolution` represents single managed APIs, single self-hosted endpoints and multi-role/routed endpoint graphs through one versioned contract. Each endpoint retains its own identity, evidence and deployment plan; no aggregate solution fingerprint may conceal a changed member or routing policy. | single/multi-endpoint golden fixtures, graph fingerprint and migration tests |
| INV-29 | Provider/model-lab credits, keys, quota or dedicated capacity are resource and authorization inputs only. They cannot alter evidence level, experiment priority absent a genuine cost/availability constraint, candidate ranking, publication visibility or result language. Records separately expose market-equivalent price, gross attributable cost, subsidy and project out-of-pocket cost; solution economics uses the entitlement available to the accepted user. Phase 1a remains executable under an owner-controlled `MaintainerBaselineAllocation` when `ContributedResourcePool = 0`. | scheduler metamorphic test with/without contribution, economics fixtures, publication-policy test and zero-contribution Phase 1a budget proof |
| INV-30 | The original preselected self-hosted path remains a first-class independently testable product flow: immutable model/artifact resolution → mechanism-aware calculation → plan/render → GPU deploy/verify → deterministic diagnosis → corrected boot → accepted-request replay. It cannot depend on a particular evaluation vendor, managed inference provider, sponsor or compound-solution router; ADR-011 composes this engine into solution qualification rather than replacing it. | permanent self-hosted end-to-end acceptance suite with local deterministic task scorer and all external evaluation/provider integrations disabled |
| INV-31 | Every capability assertion names an exact `CapabilitySignature` and its scope: `artifact_declared`, `engine_resolved`, `endpoint_exposed`, `boot_verified`, `request_verified`, or `task_qualified`. Required/optional input combinations, cardinalities, shape/streaming constraints, operation and output representation survive round-trip. A broader checkpoint claim, a different engine/endpoint, or one successful signature cannot promote another. | capability-resolution, forbidden-combination, endpoint-subset, promotion and migration tests |
| INV-32 | Resource prediction dispatches on typed component execution mechanisms plus the accepted workload shape, never on a semantic modality or broad architecture-family label. Missing calculator coverage is `unknown` even when the engine advertises the capability. Engine neutrality and non-token mechanism neutrality have independent conformance records; neither claim may be inferred from the other. | calculator dispatch/unknown-state tests and separate LLM-engine/media-engine conformance gates |
| INV-33 | Artifact location is not artifact identity. Every source ref resolves to an immutable source revision plus manifest and per-file/component content digests before calculation, execution or evidence reuse. Identical resolved contents may share one `ArtifactIdentity` while retaining every source observation; matching repository/model names cannot collapse different contents. Publisher metadata remains attributed and cannot promote an engine, endpoint, request or task capability. | resolver tests for mirrored content, mutable refs, changed manifests, conflicting metadata, gated/deleted sources and local-only artifacts |
| INV-34 | An owned or external planner emits `PlanningClaim`s and candidates, never product verdicts. Each claim carries exact producer/version, inputs, scope, epistemic status, provenance, calibration domain, uncertainty and unsupported/opaque fields. Qualification and bounded ranking are performed only by the common deterministic policy over the disclosed candidate/source set; disagreement is exposed rather than resolved by source preference. | planning-source conformance, conflicting-planner fixtures, provenance validation and forbidden-promotion tests |
| INV-35 | Every public evidence release is portable and content-addressed: its identity derives from a schema-versioned manifest and record digests, not a host URL. A publisher outage or ownership change cannot mutate record ids or prevent reconstruction from an exported release. Hugging Face and any other host are mirrors with separate publication attempts and freshness, never evidence authorities. | release reproducibility, cross-publisher identity and unavailable-publisher tests |
| INV-36 | Evidence-gap demand and coverage use attributed, dated signals from accepted requests/local failures and, where available, opt-in aggregate usage, multiple registries, providers, engines and external evidence systems. No single popularity counter is the eligibility set or silent tie-breaker; normalization, deduplication and contribution to priority are recorded. | scheduler metamorphic tests with missing/conflicting sources, signal provenance fixtures and coverage-metric validation |
| INV-37 | A repository integration is a versioned adapter over a pinned Apron core. Its check preserves canonical inputs and evidence states and cannot calculate, rank, qualify, promote evidence or invent remediation independently. GPU-free CI never emits engine/GPU validation; GPU execution requires the same external authorization and execution-target contracts as every other interface, and forked contributions receive neither secrets nor side-effect authority. | import-boundary test, CLI/Action golden-output equivalence, fork-permission fixture and execution-authorization tests |
| INV-38 | A logical endpoint cannot collapse its backing deployments into one evidence identity. `InferenceSolution` distinguishes application/model routing, replica selection and jointly executed sharding/disaggregation; every attempt records the selected route and backing target when observable or marks the applicable layer `routing_opaque`. Failover and route-policy changes alter the solution or attempt fingerprint as applicable. | direct, routed, failover, heterogeneous-alias and opaque-router fixtures; trace/header attribution tests |
| INV-39 | Capacity belonging to independent replicas is never summed to claim that one model or request fits. Cross-device or cross-node capacity is aggregated only for a typed distributed-execution group whose engine adapter verifies the supported mechanism, worker roles, placement and interconnect topology. | replica-versus-shard adversarial fixtures, topology validation and calculator dispatch tests |
| INV-40 | Authorization covers every possible dynamic route destination. A router alias does not grant data disclosure, provider/account access or spend authority to an undisclosed backend; unknown required destination scope is indeterminate and cannot authorize the external side effect. | route-expansion, failover, private-data and provider-account authorization tests |
| INV-41 | Engine neutrality, execution-mechanism neutrality and accelerator/backend neutrality are separate proof obligations. Support for a new engine, modality/mechanism or hardware backend is `unknown` or `unsupported` until its own adapter and conformance evidence exist; none is inferred from either of the other two. | independent engine, mechanism and accelerator/backend conformance gates plus forbidden-promotion tests |
| INV-42 | Wall-clock time, generated identifiers and randomness reach a deterministic component only through injected ports; there is no ambient source. A component described as deterministic names its seed sources and produces byte-identical output for identical inputs. No execution, solution, artifact or task fingerprint contains a value obtained from an uninjected source. | static import check on the deterministic core plus deterministic replay, crash and duplicate-event tests |
| INV-43 | Every record, manifest and evidence release serializes to RFC 8785 canonical form before hashing; identity is SHA-256 over that form carried with a multihash prefix. Published test vectors reproduce every digest from an independent implementation. A digest computed from an uncanonicalized encoding is rejected rather than stored. | digest reproducibility tests, cross-implementation test vectors and schema validation |

### 2a. What ends the project, and which invariant stands in the way

Development elsewhere cannot end it: a planner must validate against measured evidence, new engines/evaluators/providers enter through conformance-tested extension points, and a capability that does not fit the current contracts requires an explicit architecture decision rather than a silent scope cut. The things that can end it are internal:

| terminal failure | invariant |
|---|---|
| one false published record | INV-1, INV-12 |
| a security incident from the lab or a rendered artifact | INV-13 |
| disclosure of private task, prompt, trace or evaluator data | INV-26 |
| a cost runaway on rented hardware | INV-14 |
| a license violation in published measurements | INV-15 |
| provider funding changes or suppresses a conclusion | INV-29 |
| the task-to-solution layer hollows out the self-hosted calculator/deployment/diagnosis engine | INV-30 |
| a checkpoint, engine or endpoint is advertised for a modality combination it did not expose and verify | INV-31 |
| a resource formula is selected from a modality/family label instead of the actual mechanism | INV-32 |
| a registry name, model card or mutable ref impersonates immutable artifact or capability truth | INV-33 |
| an upstream or owned planner's recommendation becomes Apron's verdict without qualification | INV-34 |
| loss or ownership change of one evidence host erases or changes the public record | INV-35 |
| one registry's popularity metric silently determines the evidence queue | INV-36 |
| a GitHub check becomes a second decision engine or reports CPU analysis as GPU proof | INV-11, INV-19, INV-37 |
| a routed alias hides which deployment produced evidence or permits an undisclosed data/spend destination | INV-38, INV-40 |
| replica memory is presented as one sharded memory pool | INV-39 |
| support for one engine or mechanism is advertised as support for an unverified accelerator backend | INV-41 |
| a "deterministic" result that depends on an ambient clock, identifier or seed | INV-42 |
| a record digest that an independent implementation cannot reproduce | INV-35, INV-43 |
| an unsourced number in public text | INV-16 |
| silent drift of "verified" across engine versions | freshness clocks, ADR-006 |
| a wrong diagnosis rule breaking a working deployment | INV-2, INV-6 |

Maintainer absence is decay, not death; the open license, open formats, honest-unknown queue, SUCCESSION.md, and agentic pipeline make the dataset and the cycle outlive the maintainer. The agentic pipeline (Phase 3) continues Tier 0 diffs, Tier 1 dispatch within the spend cap, and diagnosis rule proposals autonomously; the maintainer's absence delays anomaly resolution and upstream contribution decisions, not the normal cycle.

## 3. System metrics (published with numerator, denominator, cohort, period, eligibility, exclusions)

| metric | definition |
|---|---|
| **Freshness lag** | hours from a vLLM tag publication to the completed Tier 0 record for that tag; and to the first Tier 1 boot record for the monitored cohort |
| **Prediction error** | per resource (weights, activation, KV budget, peak), absolute and relative error of `predicted` versus `measured`, by hardware class and engine version, with P50 and P90 |
| **Coverage** | share of a dated, declared eligible cohort assembled from accepted demand plus attributed registry/provider/engine/evidence signals, with source contribution, normalization, deduplication, eligibility and exclusions published; state is reported at each qualification-graph rung and segmented by exact capability signature and adapter stage rather than a blanket modality flag or one Hub ranking |
| **Time to promotion** | days from first resolution to boot-verified, and to workload-verified, per cohort |
| **First-try success rate** | share of all attempts in which the first rendered plan booted, over every attempt, not only successful combinations |
| **Diagnosis coverage** | share of failed attempts whose fingerprint matched a rule with a proving record; share resolved by the corrected plan on re-run; share left `unrecognized` |
| **Evidence composition** | share of records by evidence level and by source; share contested |
| **Interop consumption** | count of records exported to each external format; count of external tools or issues referencing a record id |
| **Interface completion** | share of CLI, MCP and repository-Action invocations that complete a plan, diagnosis, verification or evidence-invalidation check; installation, workflow reference and badge views are excluded |
| **Repository integration retention** | number of non-Mondegreens repositories completing a material Apron Action check; number completing another check after a later material change; fixture repositories, skipped jobs and repeated runs over unchanged inputs are excluded |
| **Cost per record** | provider dollars per boot record and per workload record, by hardware class |
| **Placement regret** | difference between the selected candidate and the best subsequently measured permitted candidate for the same accepted workload/objective, reported only over comparable benchmark scopes |
| **Qualification coverage** | share of accepted decision requests with at least one solution that reproduced the accepted task outcome and passed every applicable serving SLO, by task-suite segment and solution class |
| **Outcome economics** | total attributable cost and time per attempt and per accepted outcome, with failures, retries, utilization assumptions, cost boundary and observation/model status disclosed |
| **Qualification freshness** | time from a material task/application/model/runtime/provider change to invalidation and, where executed, completed requalification |
| **Task regression rate** | share of previously qualified exact solutions that fail the same accepted task protocol after a material version change, with denominator and changed fingerprint dimensions disclosed |

These are the only public performance claims. Any automation percentage stronger than "new well-formed checkpoints receive a calculated plan automatically" is published only after prospective measurement over the complete attempt database.

## 4. Governance

- **Architecture freeze**: components change only by ADR with a version bump (`adr/`).
- **RFC process**: any feature over roughly 500 lines or any new extension-point implementation opens an RFC referencing the conformance suite it will pass.
- **Versioning**: semantic versions on every public schema; explicit deprecation windows; checked migrations; compatibility tests on historical fixtures, generated from a declared schema-version by fixture-generation compatibility matrix rather than written per pair.
- **Data license**: CDLA-Permissive-2.0 for records, CC-BY-4.0 for documentation; machine-readable attribution in every export.
- **Signing and build provenance**: Sigstore keyless for community reports; environment attestation recorded where available; released executable/package artifacts carry source/build provenance and an applicable SBOM. An attestation proves origin and build linkage, not correctness or security.
- **Recognition**: `CITATION.cff` identifies the software and dataset citation; every evidence export preserves record-level contributors and resource provenance. Marketplace, workflow and badge visibility is never reported as product use.
- **Evidence language**: public text distinguishes calculated, inherited, boot-verified, workload-verified, `Fixed`, `Alternative with trade-offs`, `Unverified suggestion`, and contested; a hypothesis is labeled as such.
- **Corrections**: every public document carries a corrections section when a prior version was wrong; corrections are never silent.

## 5. Language rules for the repository and public pages

- Describe capabilities as contracts and invariants, never as intentions.
- Describe results as metrics with definitions, never as adjectives.
- Describe external projects as sources or targets with a mapping, never as competitors.
- Do not use: moat, disrupt, revolutionary, driving force, first-of-its-kind, "labs", "community will".
- Do use: conforms to, verified on, measured at, contested, unrecognized, hypothesis, superseded by.
- The words that carry the principles are the invariant ids and metric names above. A reader who traces INV-1, INV-3, and the prediction-error metric back to their consequences will find the strategy; nobody is told it.

## 6. Repository layout that expresses this

The directories below are subpackages of a `src/apron/` distribution, grouped into four layers with one permitted dependency direction: the deterministic core, then application logic, then adapters, then interfaces. `engineering-standards.md` §3 states the direction and the import contracts that enforce it (INV-11).

```
schemas/            versioned JSON schemas; discriminated unions; migrations/
adapters/vllm/      capability-aware engine adapter; conformance/ runs inside the pinned image
mechanisms/         autoregressive_decode/, pooling/, encoder_decoder/, media_encoder/, latent_denoising/, media_decode/
capabilities/       signature schema, combination/result-shape registry and promotion rules
artifacts/           identity/, resolvers/huggingface/, resolvers/local/, source observations and manifests
renderers/          serve/, compose/, helm_values/
evidence/sources/   lab/, inferencex/, recipes/, aiconfigurator/, community/
planning/sources/   owned/, aiconfigurator/, provider/, recipes/; candidate claims only, no verdict authority
evaluations/        adapters/, protocols/, task_suites/, applications/, scorers/; execution only, no recommendation authority
solutions/          managed/, self_hosted/, compound/; typed logical routes, replica pools and distributed-execution groups over referenced plans
backends/           runpod/, modal/, docker/
interfaces/         CLI, MCP, API and web adapters over the same core; `apron-action` is released from its separate distribution repository against these versioned contracts
signals/            accepted_demand/, local_failures/, registries/, engines/, providers/, evidence_sources/
orchestration/      durable jobs, decisions, policy references, deduplication and state transitions
publishers/         local draft, dataset and external-repository adapters; no recommendation logic
rules/              diagnosis rules as YAML-frontmatter markdown (see §6a below); each references proving record ids
records/            append-only; content-addressed portable releases exported through publisher mirrors
interim/            temporary implementations of features that belong upstream, each linked to its upstream issue or PR; deleted when upstream lands (ADR-008)
upstream/           index of open contributions (vLLM, recipes, aiconfigurator, llm-d-planner) with status and the record ids behind each
conformance/        one suite per extension point; golden fixtures; published from this
                    repository as the separate `apron-conformance` distribution so that a
                    third-party adapter author can execute the acceptance path (ADR-012)
metrics/            definitions and the publishing job
docs/               methodology, invariants, metrics, governance, ADRs
```

Each directory is one extension point or one invariant's home. The layout is the scope statement made physical.

### 6a. Diagnosis rule file format

Each rule lives at `rules/<fingerprint-slug>.md` as YAML-frontmatter markdown (its volatile popularity counts are not product evidence).

```yaml
---
name: qwen38-flash-next-fp8-sm80-kernel-failure
fingerprint: "fp8e4nv not supported in this architecture"
failure_class: kernel-compatibility
match_scope:
  artifact: Qwen3.8-Flash-Next-FP8
  gpu_compute_capability: "8.0"
  engine_image: vllm/vllm-openai:qwen38-flash-next
  kernel_path: triton-fused-moe
correction: null
source_observations:
  - https://github.com/vllm-project/vllm/issues/54318
proving_records: []
status: hypothesis
---

# Scoped Qwen3.8 FP8 fused-MoE failure on SM80

Issue #54318 observes this fingerprint during GPU initialization on four
A100 80 GB GPUs. It does not establish that all FP8 artifacts fail on Ampere:
vLLM's minimum capability and fallback behavior vary by quantization scheme
and backend.

## Correction

Unknown. A backend override, compatible artifact, or corrected engine image is
only promoted after the original accepted workload boots and passes its smoke
test on a matching execution fingerprint. The Marlin result in issue #55008 is
a separate incident, not a proving record for this rule.

## Proving records

None. The external issue proves the failure observation, not a correction.
```

Frontmatter fields:

| field | required | type | meaning |
|---|---|---|---|
| `name` | yes | string | human-readable rule name |
| `fingerprint` | yes | string | regex or exact match against error output |
| `match_scope` | yes | object | artifact, engine/container, backend or kernel path, and hardware dimensions required before the fingerprint applies |
| `correction` | yes | string or null | the tested correction; null while no correction is proven |
| `failure_class` | yes | enum | a versioned taxonomy including `oom`, `engine-core-init`, `max-model-len`, `dtype-incompatibility`, `tp-divisibility`, `quant-vs-compute-capability`, and `kernel-compatibility` |
| `source_observations` | yes | list | external or local records that prove the failure was observed; not proof that a correction works |
| `proving_records` | yes if mechanism-verified | list | scoped record IDs where the correction removed the identified failure and restored engine health |
| `status` | yes | `mechanism_verified` or `hypothesis` | proves only the correction mechanism; a concrete application still requires a separate request outcome (INV-2) |

The markdown body explains the mechanism and separates failure observations from proving records. CI rejects `status: mechanism_verified` unless a correction is non-null and at least one matching record proves corrected boot. A remediation application stores the accepted-request snapshot, corrected-plan diff, both outcomes, violated constraints, and its own proving records; only `verified` plus `satisfied` renders as `Fixed` (INV-2).

### 6b. Tool-neutral repository governance

Repository governance is expressed by enforceable outcomes, not mandatory filenames or a particular coding-agent harness (ADR-009):

- ordinary CI receives read-only repository permission unless a job demonstrably needs more;
- third-party actions are pinned to reviewed full commit SHAs; jobs have explicit timeout and concurrency behavior;
- provider execution, release publication, evidence publication and external-repository mutation use separate credential scopes and authority policies;
- pull-request checks apply the relevant product invariants, schema migrations, tests, lint and secret detection;
- issue or contribution templates exist only when their fields feed a real triage, ingestion or review path;
- authorized agents may perform Git operations under the same standing branch and safety policy as human contributors;
- commit messages describe project changes and contain no agent branding, prompts, sessions, generated-by markers or AI co-author trailers; a commit-message check enforces this rule;
- succession documentation identifies durable data and reproducible continuation paths without copying secrets or treating an automated executor as its own authority.

Contributor guidance may be rendered into tool-specific files when useful. Such files are replaceable views over these rules, never product dependencies, schema authorities or prerequisites for beginning truth-model work. The control required for a risky operation must exist before that operation, not before unrelated design work.

The retained smart configuration includes thin coding-agent entrypoints (`CLAUDE.md`, `AGENTS.md`, or the active harness equivalent), a compact principles view such as `SOUL.md`, an operational `WORKING-CONTEXT.md`, structured issue and pull-request templates, least-privilege CI/dependency automation, and contribution, security and succession guidance. Tool entrypoints point to canonical documents instead of copying decisions; the working-context file cannot override them; templates map to real ingestion/review fields; validation or generation is preferred where it can prevent drift. ADR-009 defines the complete roles and authority limits.

During maintainer absence, only operations already permitted by an external standing policy continue. Tier 0 can continue without GPU authority; Tier 1 can continue only within unexpired provider credentials, spend allocation and teardown controls; external publication follows its own destination policy. The system exposes stale evidence and blocked actions rather than silently widening authority.
