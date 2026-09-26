# ADR-013 — Represent routed and distributed serving topology without owning the serving fabric

**Status:** Accepted, 2026-09-09 (owner-approved transcript reconciliation and architecture stitching)
**Affects:** Target Architecture §§3–8, 14–15; Framework Specification; Phase Plan Phases 0, 2, 4–5; ADR-011 clarification
**Evidence:** source and documentation review of NVIDIA Personal AI Router, NVIDIA NeMo Switchyard, vLLM Semantic Router, Kubernetes Gateway API Inference Extension, llm-d Router, Ray Serve LLM, EXO, KServe LLMInferenceService and AIBrix; owner review on 2026-09-09

## Context

ADR-011 already allows a compound `InferenceSolution` with concrete role bindings and routing, fallback or escalation. Review of current serving systems exposed an ambiguity inside that correct abstraction. “Routing” can mean choosing a model for a request, choosing one replica of that model, or coordinating several workers that jointly execute one request. Treating those cases as equivalent would corrupt resource calculation, evidence attribution and authorization.

PAIR currently routes one request to one compatible machine and manages local Ollama/LM Studio lifecycles; it does not pool their memory. Switchyard selects a model or multi-model strategy per call or turn. vLLM Semantic Router explicitly separates logical model choice from later replica choice. Kubernetes Gateway API Inference Extension standardizes an `InferencePool` and Endpoint Picker boundary; llm-d Router implements load-, cache- and objective-aware endpoint selection and disaggregated execution around it. Ray Serve likewise separates ingress model routing from replica selection and can use distributed workers inside a replica. EXO is a materially different local case: it can shard one model across cooperating devices. AIBrix spans gateway routing, autoscaling, model/runtime lifecycle, distributed KV cache and disaggregated or multi-node execution, including documented non-NVIDIA accelerator work.

The convergence is therefore a stack, not one future product:

1. application/model routing chooses a logical capability, model, provider or bounded multi-model path;
2. replica/endpoint routing chooses a serving copy or pool member;
3. execution parallelism or disaggregation coordinates devices/workers that jointly serve one request.

Apron must be able to decide and prove a solution that contains any of these layers. It does not need to implement their schedulers, discovery protocols, gateways or distributed runtimes.

## Decision

### 1. Add typed serving topology to the existing `InferenceSolution`

An endpoint binding declares one of the following composable semantics:

- `direct_endpoint`: one declared managed or self-hosted endpoint;
- `logical_route`: an application/model policy over declared endpoint bindings;
- `replica_pool`: backing serving copies that share an explicitly declared equivalence scope and are selected per request;
- `distributed_execution_group`: workers that jointly execute one request through a named engine mechanism and topology.

Names are schema semantics, not required public CLI vocabulary. Heterogeneous model, provider or capability choices belong to `logical_route`, not to replica equivalence. A logical route may lead to a replica pool, whose selected member may itself be a distributed group. Every declared backing deployment retains its own `ModelSpec`, `ArtifactIdentity`, `ExecutionSpec`, provider/account or target identity, `DeploymentPlan` and evidence references.

### 2. Preserve route attribution and honest opacity

Every task or serving attempt records, for every applicable layer, the resolved logical route and physical backing target when observable, the policy/configuration fingerprint, failover sequence and source of the observation. Stable response headers, traces, gateway logs or engine telemetry may supply that observation through an adapter.

When a layer cannot reveal its selected target, it is `routing_opaque`. Apron may qualify the observable alias behaviour under the exact protocol and time window, but cannot attribute that result to or promote every possible backend. Evidence from a heterogeneous or mutable alias is not silently reused as evidence for one member.

### 3. Do not confuse replicas with sharding

Independent replica capacity improves concurrency, availability or aggregate throughput; it does not make a model fit when one replica lacks the required memory. The calculator never sums replica VRAM for one request.

Capacity across devices or nodes may be combined only within a `distributed_execution_group` whose engine adapter resolves and verifies the applicable tensor, pipeline, expert, encode/prefill/decode or other supported joint-execution mechanism; worker roles; model placement; usable memory; interconnect; and failure semantics. Ordinary data-parallel serving remains a replica pool and contributes aggregate throughput, not pooled per-request memory. An external fabric's assertion is an attributed planning claim until the required exact execution evidence exists.

### 4. Authorize the complete dynamic destination set

The accepted request and `AuthorizationEnvelope` cover every possible destination reachable through a logical route, replica policy or failover rule: task-data egress, provider/account and credentials, spend, location, security and lifecycle constraints. Authorization of an alias or primary endpoint does not authorize an undeclared fallback. If an externally operated router cannot constrain or disclose the relevant destination set, any required external disclosure or spend permission is indeterminate and the action is denied; a local black-box evaluation with no such side effect may still produce appropriately opaque evidence.

### 5. Compose existing extension points; add no serving-fabric core

A serving fabric integrates through some combination of:

- `PlanningSource`, for proposed topology, placement or route policy;
- `RenderTarget`, for its native resources or recipe;
- `ExecutionTarget`, for provision/attach, observe and lifecycle operations;
- `EvidenceSource`, for route, target, telemetry and benchmark observations;
- the existing engine and evaluation adapters where applicable.

No eleventh extension point is added. PAIR, Switchyard, GIE/llm-d, Semantic Router, Ray Serve, EXO, KServe, AIBrix and later systems are replaceable implementations or integration surfaces, not schema authorities or mandatory dependencies. Apron does not acquire generic service discovery, fleet membership, gateway proxying, autoscaling, request scheduling, model loading, KV transfer or distributed collective execution as core responsibilities.

### 6. Keep three neutrality claims independent

The architecture is open to accelerators and execution backends, but architecture openness is not implementation evidence. Engine neutrality, execution-mechanism/modality neutrality and accelerator/backend neutrality each require their own conforming adapter and record. A vLLM-on-CUDA result proves none of MLX, ROCm, XPU, Trainium, an NPU or a CPU path. A generated-media adapter does not prove a second accelerator. Conversely, one non-CUDA engine may contribute to multiple obligations only with separately attributable conformance results.

## Required conformance tests

- A direct endpoint and a one-member replica pool remain distinguishable and migrate without loss.
- Two identical replicas improve pool capacity but cannot satisfy a per-request memory requirement larger than either replica.
- A verified sharded group can use declared joint capacity only through its resolved engine mechanism and topology.
- A heterogeneous alias records the selected backing deployment for each attempt or remains `routing_opaque`; an alias-level success does not promote every member.
- A changed model policy, replica policy, backing deployment, failover sequence or distributed topology changes the applicable fingerprint and invalidates affected evidence.
- An authorized primary route with an unauthorized provider or data-region fallback is denied before disclosure or spend.
- Removing every external serving-fabric adapter leaves the permanent direct self-hosted Apron path conforming.
- Adding a new fabric needs no schema change when it can express its topology through the existing contracts; otherwise it requires a new ADR rather than vendor-specific core fields.

## Consequences

- The original product is preserved: Apron still owns task-to-solution decision, calculation, exact evidence, diagnosis and requalification; it does not become a router or orchestrator.
- Compound solutions become implementable without hiding deployment identity or fabricating pooled memory.
- Standard serving systems can become high-leverage distribution surfaces because their plans and observations can enter the same evidence loop.
- Opaque managed and routed systems remain comparable at their observable boundary, but receive weaker attribution and reuse than identified deployments.
- Hardware neutrality becomes a permanent proof obligation without promising universal hardware support before adapters and evidence exist.

## Primary sources reviewed

- <https://github.com/NVIDIA/Personal-AI-Router>
- <https://github.com/NVIDIA-NeMo/Switchyard>
- <https://github.com/vllm-project/semantic-router>
- <https://github.com/vllm-project/semantic-router/issues/2349>
- <https://github.com/kubernetes-sigs/gateway-api-inference-extension>
- <https://github.com/llm-d/llm-d-router>
- <https://docs.ray.io/en/latest/serve/llm/architecture/routing-policies.html>
- <https://github.com/exo-explore/exo>
- <https://kserve.github.io/website/docs/model-serving/generative-inference/llmisvc/llmisvc-overview>
- <https://github.com/vllm-project/aibrix> (source-reviewed at `5c77915f74d00a4dcdd6034615f0eba47b0af66f`, 2026-09-09)
