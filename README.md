<p align="center"><img src="docs/assets/apron-mark.svg" width="112" alt=""></p>

<h1 align="center">Apron</h1>

<p align="center">Which LLM to run for a given task, on which API or GPU, with what configuration, backed by evidence.</p>

Apron is a decision and evidence system for LLM inference. You give it a task the model has to perform, a quality floor, a latency or throughput target and a data policy. It returns the solution that reproduces that outcome: a managed model API, an open-weight model self-hosted on an inference engine such as vLLM, or a compound system that routes between several of either. The answer names the exact checkpoint and quantization, the engine version, the GPU or provider and the serving configuration. It carries the evidence that produced it and reports the cost per accepted result rather than per token. Every number states how it is known. The interfaces are a CLI, an MCP server in the same package so a coding agent gets a version-pinned answer instead of a guess, and a GitHub Action that re-checks a repository's deployment when a model or engine version changes.

> [!IMPORTANT]
> **Status.** Phase 0 (contracts and truth model) and Phase 1a (first complete vLLM product loop) implemented September 2026. One internal deployment evidence record with predicted-vs-measured memory, failure injection, correction and task reproduction on Qwen3-8B BF16 via RunPod RTX 4090/A6000. The PyPI names are reserved at 0.0.0 and contain nothing.

## In this repository

The specification, published before the code.

- [`product-definition.md`](docs/product-definition.md): what the system does and why. Read this one first.
- [`target-architecture.md`](docs/target-architecture.md), [`phase-plan.md`](docs/phase-plan.md) and [`framework-spec.md`](docs/framework-spec.md): the shape, the exit gates, and the invariants, extension-point contracts and metrics.
- [`adr/`](docs/adr/): thirteen architecture decision records, each with its context, decision and consequences.
- [`architecture-dispatch-proof.md`](docs/architecture-dispatch-proof.md): the derivation, from the pinned vLLM source, of how resource calculation must dispatch.

## What the specification fixes

**Evidence is joined and stays scoped.** Task quality, serving behaviour, compatibility, hardware fit and outcome economics are each measured by mature tools, and the measurements are not joined across the exact task, scaffold, model, artifact, runtime and hardware that produced them. Apron holds that join as a typed graph in which a record belongs to the exact combination that produced it and cannot be promoted to another by any sequence of legal operations. A stateful test of that property is a required CI check.

**How a claim is known is a schema field.** Every assertion is `derived` (follows from identified inputs), `proven_constraint` (a rule from the pinned engine source establishes it), `predicted` (the calculator modeled it) or `measured` (the exact execution observed it), with unit, scope, evidence level, freshness and provenance required. A CPU-side calculation is never presented as engine validation, and a single measured candidate is never called optimal.

**Calculation dispatches on mechanism, not model family.** Per-layer cache and attention mechanisms determine memory; a family label does not. An unknown mechanism returns `unknown`, never a fallback formula.

**Diagnosis is proven.** A failed boot yields a fingerprint, the fingerprint maps to a correction, and the correction is proven by a corrected boot with the accepted request re-checked separately. A rule without a proving record is labeled a `hypothesis`. [vLLM issue #54318](https://github.com/vllm-project/vllm/issues/54318), an FP8 artifact that passed configuration on an A100 and failed inside a fused-MoE kernel, is the class of failure this exists for.

**Authority is separate from execution.** A fixed authorization engine combines constraints from interactive approval, standing policy, provider controls and organization policy. Deny overrides permit, missing permission never authorizes a side effect, and the executor cannot widen its own budget, credentials or deadline. Provider credits are recorded as resource ownership and cannot reach the scheduler, optimizer or publisher as preference.

**Records outlive the tool.** Records use RFC 8785 canonical form, SHA-256 under a multihash prefix, a schema-versioned manifest and cross-language test vectors, so a release verifies without any host. Published records go to a Hugging Face dataset mirror under CDLA-Permissive-2.0. `verify` keeps a result on your machine; `submit` is the only path that shares it, and it asks first.

**Acceptance is a conformance suite.** There are ten extension points, each a `Protocol` with golden fixtures. The suites are a separate distribution with no dependency on `apron`, so an adapter is accepted in its author's repository before it is proposed here.

## What it builds on

Task evaluation runs through [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai), [Harbor](https://github.com/harbor-framework/harbor) and trace replay. Artifacts resolve from the [Hugging Face Hub](https://huggingface.co) and local files. [vLLM](https://github.com/vllm-project/vllm) is the first engine; [SGLang](https://github.com/sgl-project/sglang) is the second, and its adapter is the proof that the contracts are not shaped around one engine. Planning claims from [aiconfigurator](https://github.com/ai-dynamo/aiconfigurator) and provider recommenders enter as attributed candidates. Nothing is filed upstream without a reproducible record attached: verified configurations are rendered in the shape [vllm-project/recipes](https://github.com/vllm-project/recipes) accepts, prediction deltas go to the planner that produced them, and startup failures with proven corrections go to the engine's tracker. Anything with an upstream home is developed against that upstream; this repository keeps only what has none.

## Where a contribution lands now

- **A challenge to a decision.** Every ADR records its context and the reasoning behind the decision. Open a challenge in [Discussions](https://github.com/mondegreens/apron/discussions); amendments pass by lazy consensus.
- **A check of the dispatch proof** against the vLLM source at the commit it pins. It is a claim about code and can be wrong.
- **A deployment failure** for the knowledge base: the traceback, the exact artifact, the engine tag and the GPU. It enters the rule table as a hypothesis until a proving record exists.
- **Later, an adapter.** When the conformance suites ship, the reference adapter is the copy target and the suite is the acceptance test, run in your repository first.

AI-assisted work is welcome; say which tool, explain the change yourself, and answer reviewers yourself. Unattended agents do not open pull requests or issues here. [`AI_POLICY.md`](AI_POLICY.md) has the rest.

## Contact

Maintained by Vlad Ryzhkov. Design challenges and questions go to [Discussions](https://github.com/mondegreens/apron/discussions), not Issues.
