# ADR-011 — Qualify an inference solution from the user's task, not only deploy a preselected model

**Status:** Accepted, 2026-09-08 (owner-approved product zoom-out; D14); registry-neutral discovery clarified through ADR-002 and routed serving topology through ADR-013 on 2026-09-09
**Affects:** Product Definition; Target Architecture §§1, 3–8, 11–15; Framework Specification; Phase Plan Phases 0–5
**Evidence:** Artificial Analysis Optima product and launch documentation; UK AISI Inspect AI task/model documentation; Harbor task, trial and agent documentation; Arize Phoenix evaluation/experiment documentation; vLLM benchmark documentation; Baseten Model API and inference documentation; owner review on 2026-09-08

## Context

The existing product starts with a selected model or lineage and answers how to fit, configure, place, deploy, measure, diagnose and reverify it. It already requires accepted workload and quality evidence before a quantization candidate can be `Recommended`, but it has no contract that defines the user's real task, no active evaluation adapter that can produce that quality evidence, and no representation of a solution composed from multiple model roles. `WorkloadSpec` currently mixes a serving traffic shape with a coarse task label. `EvidenceSource` can ingest an evaluation result but cannot execute an evaluation protocol.

The GLM-5.3/GLM-5.3-Flash production briefing exposed the missing decision. A user may need to compare model capability on their own tasks, use different models for planning, execution or vision, compare managed APIs with self-hosted artifacts, and optimize the cost and time of an accepted result rather than a token or an attempted request. Artificial Analysis Optima already compares models and user-supplied agents on custom tasks by score, cost per task and time per task. Baseten already supplies optimized managed endpoints. Inspect AI, Harbor and Phoenix already supply reusable task, agent, scorer, sandbox, trace and experiment machinery. vLLM supplies serving-load measurements, not task correctness.

Reimplementing every benchmark harness, benchmark corpus, judge fleet, trace store and managed serving platform is not required to close the decision loop. The missing product is the provider- and harness-neutral evidence and decision layer that joins task outcome to the exact model, artifact, application, runtime and deployment that produced it, then deploys, diagnoses and requalifies that solution.

## Decision

### 1. The accepted request begins with an outcome

The top-level request is `DecisionRequest`. It records the user's task or application objective; accepted success criteria; minimum quality; serving SLOs; privacy, security, license and data-residency constraints; budget and time bounds; permitted providers, accounts and resources; and the optimization objective. AI may propose this request from natural language, but it cannot drive evaluation, provisioning or deployment until the user or authorized calling system accepts the immutable request snapshot.

The product question becomes:

> Given this accepted task, quality floor, serving SLO, policy and authorization envelope, which evidenced inference solution satisfies them with the best declared trade-off, and can it be deployed and kept qualified?

The product never substitutes a universal "best model" claim for this scoped question.

### 2. Task semantics and serving load are separate contracts

`TaskSuiteSpec` describes what must be accomplished: versioned cases or a sampling frame, required `CapabilitySignature`s with exact input combinations and expected result representations, tools, environment, task categories, success criteria, rubrics, segment weights, privacy classification and representativeness claims. Modality, operation and output shape remain separate under ADR-007.

`ServingWorkloadSpec` describes how the accepted application is served: request and concurrency distributions, burstiness, input/output shape distributions, prefix/cache behavior, LoRA usage, latency and throughput percentiles, availability, duration and capacity horizon.

The previous `WorkloadSpec` becomes an accepted envelope that references these independent specifications; task success is never inferred from serving throughput, and serving fitness is never inferred from task score.

### 3. The evaluated application is fingerprinted

`ApplicationSpec` records the endpoint-independent system that invokes models: prompt and template revisions, agent/scaffold and version, tool definitions and versions, retrieval or other application components, reasoning and sampling settings, turn/tool/retry limits, named logical role graph, invocation conditions and relevant environment identity. It does not bind a role to a concrete model or endpoint. A result without the application configuration that elicited it cannot qualify a different application.

### 4. The candidate is an inference solution, not necessarily one model

`InferenceSolution` is the object discovered, evaluated, selected, deployed and requalified. It can contain:

- a managed model API and its declared or observable version identity;
- one immutable self-hosted artifact, engine execution and deployment target;
- concrete model/endpoint bindings for the logical application roles;
- the resolved executable routing, fallback or escalation policy over those bindings.

Every endpoint in a solution retains its own `ModelSpec`, `ArtifactSpec`, `ExecutionSpec`, exposed `CapabilitySignature`s, provider/target identity and `DeploymentPlan` where applicable. Every cross-modal edge declares its representation and transform rather than hiding ASR, retrieval, vision, media generation or TTS inside an aggregate label. The solution is the only authority for concrete role bindings and executable routing; it references the logical roles from `ApplicationSpec`. `DeploymentPlan` remains the canonical executable plan for a deployable endpoint or coordinated set of endpoints; it is no longer the top-level decision object. Opaque or silently mutable managed APIs are representable, but their incomplete fingerprint limits evidence reuse and freshness.

ADR-013 makes three topology layers non-interchangeable inside this same object: an application/model router selects a capability or model binding; a replica scheduler selects one backing copy; and a distributed-execution group makes several workers jointly execute one request. A routed alias retains the identities of every declared backing deployment and the selected target for each attempt when observable. If that target is not observable, the layer is `routing_opaque`; the alias may be tested as a black box but its result cannot qualify an unidentified backend. Replica resources are not pooled into a fit calculation, while verified sharded/disaggregated resources are represented through their exact engine mechanism and topology.

A self-hosted artifact is located and resolved independently of any registry namespace. `ArtifactLocator` preserves where and at which requested revision it was found; `ArtifactIdentity` preserves the resolved manifest, component/file digests and structure. Hugging Face Hub, another registry, object storage and a local artifact are conforming locations, not competing identity systems. Publisher metadata and registry-derived capability, lineage, license or evaluation fields remain attributed source claims and cannot silently become engine, endpoint or task evidence.

### 5. Evaluation is a replayable protocol

`EvaluationProtocol` binds a `DecisionRequest`, `TaskSuiteSpec`, `ApplicationSpec` and candidate `InferenceSolution` to the evaluation harness and version, dataset snapshot, solver/agent, scorer and rubric, deterministic checks, judge model and prompt where used, sampling parameters, seeds, repetitions, concurrency, stopping rules, aggregation, uncertainty method and disclosure rules.

Deterministic verification is preferred where the task permits it. Model-graded, pairwise and human judgments are accepted only with their identities, prompts or instructions, sampling policy and uncertainty exposed. A generated task suite or rubric is a proposal until its representativeness and acceptance are recorded; generation does not establish validity.

### 6. Evaluation execution is a ninth typed extension point

Add **Evaluation adapter** to the framework. Its contract is:

`accepts(task_suite, application, solution, capability_signatures) -> support_state`

`prepare(protocol, solution) -> evaluation_run_plan`

`execute(evaluation_run_plan, authorization) -> task_attempt_records`

`collect(run) -> evaluation_evidence`

`replay(record, solution) -> task_attempt_records`

Initial conforming adapters may wrap Inspect AI, Harbor, an external HTTP/Optima agent interface, Phoenix datasets/experiments, or a user-supplied command. An adapter does not gain authority to disclose task data, call an external judge, spend provider funds or deploy resources. `EvidenceSource` remains the separate passive-ingest contract.

### 7. Task attempts preserve outcome, behavior and attributable cost

`TaskAttemptRecord` stores the exact decision, task-suite, application, evaluation-protocol and solution fingerprints; case and attempt identity; output and permitted artifacts; criterion scores and acceptance outcome; trace references; turns, retries and tool calls; input, cached-input and output usage; wall-clock and service latency; endpoint and judge charges; allocated infrastructure cost; failures; and raw-observation provenance.

Aggregates never erase attempts or failing segments. Published evaluation evidence declares sample count, repetitions, segment weights, exclusions, uncertainty and missing data. Cross-fingerprint evaluation results may generate a prediction or candidate prior; they never become measurements for a different solution.

### 8. Economics is measured at the accepted outcome

Economics reports, without conflating them:

- cost and time per attempt;
- observed total attributable cost divided by accepted outcomes;
- retry and failure contribution;
- expected cost and time to an accepted outcome with the model and assumptions identified;
- token/API cost;
- self-hosted startup, transfer, idle, replica, interruption and utilization cost;
- steady-state cost at the accepted traffic and availability envelope.

The optimizer may compare managed and self-hosted candidates only after normalizing the measurement boundary, workload horizon, quality acceptance rule and included cost components. Unknown or provider-opaque components remain visible.

### 9. Qualification and selection require both task and serving proof

Qualification is a dependency graph, not a requirement to spend in one chronological order:

`candidate_discovered -> capability_eligible -> solution_identity_resolved`

From the resolved identity, task evaluation and applicable deployment calculation/verification may be scheduled in whichever authorized order minimizes information cost. Final promotion requires:

`endpoint_available_or_boot_verified + task_evaluated + serving_slo_verified + task_outcome_reproduced_on_retained_solution -> solution_qualified`

A task result from another API, artifact, application or unresolved model name may prioritize a candidate but remains a prior, not measurement for the resolved or retained solution.

Not every candidate needs every self-hosted rung: a managed API has no Apron-observable boot or memory proof, so those fields are `not_applicable` or `provider_opaque`, never fabricated. It must still pass endpoint health, serving and exact-solution task reproduction before qualification.

Selection uses cheap deterministic pruning before paid evaluation, then records the disclosed candidate set and every exclusion. It rejects candidates that violate capability, policy, authorization, task-quality or serving constraints and ranks the remaining solutions against the accepted objective. `Qualified` means the solution met the accepted constraints under the recorded protocols. `Measured-efficient` or `best observed` is limited to the disclosed comparable set. A single successful solution is qualified, not optimal.

Candidate generation and verdict are separate. The owned calculator and conforming external planning sources may each produce `PlanningClaim`s and candidate configurations with exact producer/version, inputs, scope, provenance, calibration domain, uncertainty and unsupported fields. No planner recommendation is a product verdict. The deterministic selection policy reconciles compatible claims, exposes disagreement, commissions execution when needed, and promotes only from the evidence required above. Candidate-set disclosure names which registries, planners and providers were searched and which were unavailable or unsupported, so “best observed” cannot conceal source coverage.

### 10. Qualification is versioned, private by default and continuously invalidated

Task data, prompts, traces, outputs and evaluator inputs are private by default. Sending any of them to an external model, judge, evaluation service or provider must be permitted by the accepted data policy and `AuthorizationEnvelope`. Evaluation and publication remain separate actions.

Qualification freshness is invalidated or explicitly reassessed when any material part changes: task-suite distribution or revision; acceptance rubric; application prompt, agent, tools, role graph or routing; model/API identity; artifact, quantization or tokenizer; inference parameters; engine, kernel, container, driver or hardware; evaluator, judge or aggregation; provider behavior; pricing; or serving workload. Production traces may seed a new task-suite revision only under the same privacy, acceptance and provenance rules.

## Product spine

The permanent loop becomes:

`accept outcome -> discover solution candidates -> qualify task behavior -> calculate and authorize deployment candidates -> deploy -> verify serving and reproduce task outcome -> measure outcome economics -> publish consented evidence -> monitor invalidation -> diagnose -> correct -> requalify`

The existing resolve, calculator, artifact, deployment, verification, diagnosis and compatibility components remain required. This ADR places them inside the larger decision loop; it does not remove or weaken them.

The preselected self-hosted path remains a first-class independently testable product path: immutable model/artifact resolution, mechanism-aware prediction, plan/render, GPU deployment and verification, deterministic diagnosis, corrected boot and accepted-request replay. It must run with a local deterministic task scorer and without any particular evaluation vendor, managed inference provider, contributed resource or compound router. A task-to-solution implementation that merely wraps external leaderboards or managed APIs while hollowing out this path is non-conforming.

## Implementation and reuse decision

Phase 0 defines the permanent contracts and golden fixtures for a single-endpoint solution and a multi-role solution before schemas freeze. Phase 1a still proves the existing deployment/diagnosis engine end to end, but its accepted quality contract is represented by a real minimal `TaskSuiteSpec` and replayable `EvaluationProtocol`, not an unexplained `QualityEvidence` placeholder. The public showcase demonstrates the expanded task-to-qualified-solution loop.

The first evaluation implementations wrap established engines rather than duplicating them:

- Inspect AI for general task/dataset/solver/scorer and local or OpenAI-compatible model evaluation;
- Harbor for reproducible agentic tasks, sandboxes, trials and verifiers;
- Phoenix or equivalent trace/experiment systems for production-trace ingestion and replay;
- vLLM native benchmarks for serving behavior;
- managed providers, including Baseten when authorized, as candidates and execution targets rather than privileged answers.

These are initial integrations, not permanent exclusions on native capability. Their results enter Apron only through the canonical protocol, fingerprint, evidence and authority contracts.

## Required conformance and failure tests

- The same task score with a changed prompt, tool, judge, artifact, API identity or sampling configuration cannot be reused as an exact-solution measurement.
- High throughput with failed task outcomes cannot qualify a solution.
- High task score with a violated serving SLO cannot qualify a solution.
- An evaluation adapter cannot send private cases to an external candidate or judge without applicable authority.
- A generated rubric cannot become accepted merely because the generator produced it.
- Managed and self-hosted cost cannot be ranked when their cost boundary or workload horizon is incomparable.
- Failed and retried task attempts remain in outcome-economics aggregates.
- A one-model `InferenceSolution` and a multi-role solution pass through the same decision and evidence APIs without schema changes.
- A solution change that affects any material fingerprint creates new qualification evidence rather than mutating the old result.
- A single measured solution cannot be labeled best or optimal.
- The same resolved artifact obtained through two sources has one content identity with two source observations; equal names with different manifests remain different artifacts.
- A mutable registry ref is resolved before calculation or execution and cannot replace the recorded immutable revision on replay.
- Publisher metadata, a provider mapping or an external planner's `recommended` label cannot promote capability, qualification or selection state.
- Removing the Hugging Face adapter leaves local-artifact resolution, the owned calculator and prior portable evidence releases conforming.
- Two replicas cannot satisfy a one-request memory requirement that exceeds each replica; the equivalent capacity is eligible only when a conforming adapter verifies a joint distributed-execution topology.
- A heterogeneous router alias records the actual backing deployment per attempt or remains `routing_opaque`; an aggregate success cannot promote every declared backend.
- A failover target outside the accepted data, provider/account or spend envelope is denied even when the primary route was authorized.

## Consequences

- The explicit statement that model selection for a user's task is out of scope is superseded. Universal model rankings remain epistemically invalid; scoped solution qualification is the product.
- `QualityEvidence` becomes reproducible evidence derived from task attempts and an evaluation protocol rather than a free-standing attachment.
- `WorkloadSpec`, `DeploymentPlan` and `VerificationReport` remain useful but are no longer sufficient as the top-level product model.
- The target architecture receives a major version bump because its entry point, selected object, product spine and extension-point set change.
- The project can complement Artificial Analysis, Optima, Baseten and open evaluation frameworks while remaining useful when any one of them is absent.

## Primary sources reviewed

- <https://artificialanalysis.ai/optima>
- <https://artificialanalysis.ai/articles/optima>
- <https://inspect.aisi.org.uk/tasks.html>
- <https://inspect.aisi.org.uk/models.html>
- <https://www.harborframework.com/docs/core-concepts>
- <https://arize.com/docs/phoenix/>
- <https://docs.vllm.ai/en/latest/benchmarking/cli/>
- <https://www.baseten.co/products/model-apis/>
- <https://docs.baseten.co/inference/overview>
- <https://www.baseten.co/resources/webinar/executive-briefing-on-glm-5_3/>
