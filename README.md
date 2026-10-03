<p align="center"><img src="docs/assets/apron-mark.svg" width="112" alt=""></p>

<h1 align="center">Apron</h1>

<p align="center">Which LLM to run for a given task, on which API or GPU, with what configuration, backed by evidence.</p>

Apron is a decision and evidence system for LLM inference. You give it a task the model has to perform, a quality floor, a latency or throughput target and a data policy. It returns the solution that reproduces that outcome: a managed model API, an open-weight model self-hosted on an inference engine such as vLLM, or a compound system that routes between several of either. The answer names the exact checkpoint and quantization, the engine version, the GPU or provider and the serving configuration. It carries the evidence that produced it and reports the cost per accepted result rather than per token. Every number states how it is known. The interfaces are a CLI, an MCP server in the same package so a coding agent gets a version-pinned answer instead of a guess, and a GitHub Action that re-checks a repository's deployment when a model or engine version changes.

> [!IMPORTANT]
> **Status.** Phase 0 (contracts and truth model) and Phase 1a (first complete vLLM product loop) implemented September 2026. Phase 1b (the evidence cohort) is in progress: open models from 0.6B to current frontier releases, booted on rented RTX 4090, L4, A6000, A100, H100 and B200 GPUs on vLLM v0.29.0 and v0.30.0, each record keeping the prediction beside the measurement. The table below is generated from those records. The PyPI names are reserved at 0.0.0 and contain nothing.

## Measured records

<!-- records:start -->
Booted on rented GPUs. Memory: predicted / measured, GiB. Generated from the [stored records](records/phase-1b-cohort/records/verification-reports).

| Model | GPU | vLLM | Weights GiB | Startup peak GiB | KV pool tokens | p99 TTFT ms (SLO) | Tasks | Cold start s | Record |
|---|---|---|---|---|---|---|---|---|---|
| deepseek-ai/DeepSeek-V2-Lite | A100 80GB PCIe x1 | v0.29.0 | 29.26 / 29.32 | 0.27 / 0.41 | — | 930 (pass) | 0/3 fail | — | [`1220697d834da838`](records/phase-1b-cohort/records/verification-reports/1220697d834da838ea2b058265f8dc1159098f8868d943b13b0f887b05fbb10162b9.json) +7 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | A100 80GB PCIe x1 | v0.29.0 | 29.26 / 29.32 | 0.27 / 0.41 | — | 925 (pass) | 3/3 pass | — | [`1220c078c3a7d11e`](records/phase-1b-cohort/records/verification-reports/1220c078c3a7d11e65be0f1d182e4a5ee4bb2541d40445ea8477e0d5e47c5325dae4.json) +4 |
| google/gemma-2-2b-it | RTX 4090 x1 | v0.29.0 | 4.87 / 4.94 | 0.49 / 1.12 | — | 78 (pass) | 3/3 pass | — | [`12206b393c533d0a`](records/phase-1b-cohort/records/verification-reports/12206b393c533d0ae1e2e4c644df0af93e2bcc91115d04a0693e0f7494013333b586.json) +15 |
| google/gemma-4-31B-it | H100 80GB HBM3 x1 | v0.29.0 | 58.46 / 58.99 | 2.79 / 1.49 | 9,986 | 204 (pass) | 3/3 pass | 269 | [`1220ea39d74241fc`](records/phase-1b-cohort/records/verification-reports/1220ea39d74241fc727da331c646fcfa5c8d0338832b8295b8ec59d6594d44ce72fd.json) +4 |
| JunHowie/Qwen3-8B-GPTQ-Int4 | RTX 4090 x1 | v0.29.0 | 5.68 / 5.68 | 0.57 / 1.19 | — | 209 (pass) | 3/3 pass | — | [`1220b9e409435a57`](records/phase-1b-cohort/records/verification-reports/1220b9e409435a57a94ec162cf8e0d401bf53a8967bbb168a213dbeb6482f7596ca6.json) +7 |
| meta-llama/Llama-3.1-8B-Instruct | RTX 4090 x1 | v0.29.0 | 14.96 / 15.00 | 1.50 / 1.01 | — | 222 (pass) | 3/3 pass | — | [`1220636ca167b3f0`](records/phase-1b-cohort/records/verification-reports/1220636ca167b3f0c240df5c80fa2aec588bd2bec55f7a763fb5acc33fe1bcfb5cec.json) +7 |
| meta-models/Muse-Glimmer-30B | H100 80GB HBM3 x1 | v0.30.0 | 55.46 / 55.83 | 1.78 / 1.83 | 217,478 | 175 (pass) | 3/3 pass | 207 | [`1220bf6b162c4318`](records/phase-1b-cohort/records/verification-reports/1220bf6b162c43180f199a86f397319a5b06d51e8e2f7f8f8bd482b22e8258b3b1ed.json) +9 |
| mistralai/Mistral-7B-Instruct-v0.3 | RTX 4090 x1 | v0.29.0 | 13.50 / 13.51 | 1.35 / 0.28 | — | 203 (pass) | 3/3 pass | — | [`1220c4ad8130d1bb`](records/phase-1b-cohort/records/verification-reports/1220c4ad8130d1bb5d01e086aa11b4155e57e820ff67e02fed5be6c2b1111bc090bb.json) +17 |
| mistralai/Mistral-7B-Instruct-v0.3 | L4 x1 | v0.29.0 | 13.50 / 13.51 | 1.35 / 0.28 | — | 718 (pass) | 3/3 pass | — | [`1220ac02237cee07`](records/phase-1b-cohort/records/verification-reports/1220ac02237cee071400d8ff6a286d8975f85c8f18558aa041bdeaaac5f6a578a165.json) +7 |
| nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 | H100 80GB HBM3 x1 | v0.30.0 | 58.82 / 58.92 | 0.74 / 1.09 | 53,546 | 615 (pass) | 3/3 pass | 166 | [`12208228a1874270`](records/phase-1b-cohort/records/verification-reports/12208228a1874270fa2f9bdb214d951c18ce192291b5a384edc2fb7dc73a3f325225.json) +5 |
| openai/gpt-oss-120b | H100 80GB HBM3 x1 | v0.29.0 | 60.77 / 61.43 | 1.17 / 1.17 | 81,951 | 2,292 (fail) | 3/3 pass | 228 | [`1220ded0ff281432`](records/phase-1b-cohort/records/verification-reports/1220ded0ff28143274f8d56e3a39b7c807a3a742d7d03ef2729270470ba3794a57ef.json) +4 |
| openai/gpt-oss-20b | RTX 4090 x1 | v0.30.0 | 12.82 / 13.80 | 1.10 / 1.10 | 93,400 | 120 (pass) | 3/3 pass | 184 | [`1220cda095dbab4a`](records/phase-1b-cohort/records/verification-reports/1220cda095dbab4a2973d3d00253628fd43d042f82c09246a6a0c877c3af8bf96a2b.json) +4 |
| Qwen/Qwen3-0.6B | RTX 4090 x1 | v0.29.0 | 1.11 / 1.12 | 0.30 / 0.30 | — | 62 (pass) | 1/3 fail | — | [`1220c6220073e292`](records/phase-1b-cohort/records/verification-reports/1220c6220073e29214264a963891ea6b80eef4c94d99856874270159411ceb31e476.json) +4 |
| Qwen/Qwen3-0.6B-FP8 | RTX 4090 x1 | v0.29.0 | 0.70 / 0.74 | 0.30 / 0.30 | — | 65 (pass) | 1/3 fail | — | [`1220e319388876e8`](records/phase-1b-cohort/records/verification-reports/1220e319388876e8d22bc59a7cd1235262ecd2a9ccae4df7b3fe259b4fd45a260f90.json) +4 |
| Qwen/Qwen3-0.6B-FP8 | H100 80GB HBM3 x1 | v0.29.0 | 0.58 / 0.74 | 0.06 / 0.34 | — | 69 (pass) | 1/3 fail | — | [`12200078d8e3999e`](records/phase-1b-cohort/records/verification-reports/12200078d8e3999ec13fe5d4ac6bfa6d2fdd69b297b01e1763a3ed34dcbee1c96dd0.json) +7 |
| Qwen/Qwen3-1.7B | RTX 4090 x1 | v0.29.0 | 3.78 / 3.22 | 0.38 / 0.60 | — | 53 (pass) | 3/3 pass | — | [`122056e563edf84b`](records/phase-1b-cohort/records/verification-reports/122056e563edf84b102233abdb30ce90733de29eaa8ba8e23b06bbf302188c3ca8e6.json) +18 |
| Qwen/Qwen3-1.7B | L4 x1 | v0.29.0 | 3.78 / 3.22 | 0.38 / 0.60 | — | 173 (pass) | 3/3 pass | — | [`1220205b397c672e`](records/phase-1b-cohort/records/verification-reports/1220205b397c672ee4ad674fc589d167da7dda4c181ee1603eafcd7c66235184198f.json) +7 |
| Qwen/Qwen3-14B | RTX A6000 x1 | v0.29.0 | 27.51 / 27.52 | 2.75 / 1.49 | — | 585 (pass) | 3/3 pass | — | [`12203903ded913c6`](records/phase-1b-cohort/records/verification-reports/12203903ded913c699483a8f5a7902faf7cb5059209d1455fdfbe17c5ec95f0895b3.json) +7 |
| Qwen/Qwen3-32B | A100 80GB PCIe x1 | v0.29.0 | 61.02 / 61.03 | 6.10 / 1.49 | — | 716 (pass) | 3/3 pass | — | [`1220374d2a3c7d3a`](records/phase-1b-cohort/records/verification-reports/1220374d2a3c7d3a33d8fe3102211051c5dc5185751d49fbd620a90eb1d34897a479.json) +8 |
| Qwen/Qwen3-32B | H100 80GB HBM3 x1 | v0.29.0 | 61.02 / 61.03 | 6.10 / 1.61 | — | 223 (pass) | 3/3 pass | — | [`1220eef514a1a275`](records/phase-1b-cohort/records/verification-reports/1220eef514a1a275cd13bd98c041db9bda93e42c9df0b5126d4f6ede6c0811582fca.json) +8 |
| Qwen/Qwen3-8B | A100-SXM4-80GB x4 | v0.29.0 | 15.26 / 7.64 | 1.53 / 0.21 | — | 92 (pass) | 3/3 pass | — | [`12209ac3de7807d5`](records/phase-1b-cohort/records/verification-reports/12209ac3de7807d51809acd76e127d26d804a5ec59e024150da7d0c9536599571ea5.json) +7 |
| Qwen/Qwen3-8B | RTX 4090 x1 | v0.29.0 | 15.26 / 15.27 | 1.53 / 1.19 | — | 212 (pass) | 3/3 pass | — | [`122046447ffc1d89`](records/phase-1b-cohort/records/verification-reports/122046447ffc1d89490228e1a78bf46bfc0f154518d1ee3fc69b76ea755703418513.json) +8 |
| Qwen/Qwen3-8B-AWQ | H100 80GB HBM3 x1 | v0.29.0 | 5.68 / 5.82 | 1.28 / 1.29 | 464,992 | 86 (pass) | 3/3 pass | — | [`1220df650fa3167d`](records/phase-1b-cohort/records/verification-reports/1220df650fa3167d293bc9cbd0e14aaa9de6c018c051dcc36f2849ce113a5f0d1269.json) +4 |
| Qwen/Qwen3.6-35B-A3B-FP8 | H100 80GB HBM3 x1 | v0.30.0 | 34.88 / 34.23 | 1.01 / 1.92 | 147,108 | 141 (pass) | 3/3 pass | 509 | [`12208654b66bdeae`](records/phase-1b-cohort/records/verification-reports/12208654b66bdeae281a67714f40e69e681f90b81a7aa320361dd639fe8151aa3daa.json) +4 |
| Qwen/Qwen3.8-27B | H100 80GB HBM3 x1 | v0.30.0 | 50.96 / 51.10 | 1.88 / 1.92 | 31,085 | 191 (pass) | 3/3 pass | 228 | [`1220f20079fa79b1`](records/phase-1b-cohort/records/verification-reports/1220f20079fa79b1a0a51f6320c035b4a744f80896d23157ac9754195fac1dd1df30.json) +4 |
| state-spaces/mamba-2.8b-hf | RTX 4090 x1 | v0.29.0 | 5.16 / 5.23 | 0.26 / 0.26 | 818,560 | 195 (pass) | 0/3 fail | — | [`12206810a074c3de`](records/phase-1b-cohort/records/verification-reports/12206810a074c3de321e4b0f9b559394be33816d5161d21d68bdd6b58d995db0779b.json) +12 |
| zai-org/GLM-4.7-Flash | H100 80GB HBM3 x1 | v0.29.0 | 58.15 / 55.87 | 0.65 / 2.04 | 169,216 | 1,324 (pass) | 3/3 pass | 105 | [`122089055d7f1339`](records/phase-1b-cohort/records/verification-reports/122089055d7f13395c37eac33178d384e3e999c6c15145469b473a6d6848be0202eb.json) +4 |
| zai-org/GLM-5.3 | H200 x8 | v0.30.0 | 88.20 / 88.91 | 2.77 / 2.62 | 358,592 | 1,535 (pass) | 4/5 pass | 1778 | [`12208ef8aa4a3de7`](records/phase-1b-cohort/records/verification-reports/12208ef8aa4a3de74e08ddb9ad562482d2dee791404bfe4fb3c484a796406c979607.json) +8 |
| zai-org/GLM-5.3-Flash | B200 x2 | v0.30.0 | 149.40 / 152.10 | 6.28 / 4.38 | 70,778 | 0 (fail) | 0/5 fail | 1522 | [`12208ef6f21ef008`](records/phase-1b-cohort/records/verification-reports/12208ef6f21ef008c1cf64b94c292e15ca883f05d636ba7d857dfe8e1206ab28f395.json) +13 |
| zai-org/GLM-5.3-Flash | H200 x4 | v0.30.0 | 76.26 / 76.37 | 2.83 / 3.10 | 3,148,314 | 377 (fail) | 4/5 pass | 2073 | [`1220106dee467e37`](records/phase-1b-cohort/records/verification-reports/1220106dee467e37adfbd9dc70016adaf0e36982c51536fe50ece890b7e14a0e55ad.json) +7 |
<!-- records:end -->

## In this repository

The specification, published before the code.

- [`product-definition.md`](docs/product-definition.md): what the system does and why. Read this one first.
- [`target-architecture.md`](docs/target-architecture.md), [`phase-plan.md`](docs/phase-plan.md) and [`framework-spec.md`](docs/framework-spec.md): the shape, the exit gates, and the invariants, extension-point contracts and metrics.
- [`adr/`](docs/adr/): thirteen architecture decision records, each with its context, decision and consequences.
- [`architecture-dispatch-proof.md`](docs/architecture-dispatch-proof.md): the derivation, from the pinned vLLM source, of how resource calculation must dispatch.
- [Findings](docs/blog/index.md): where results of the evidence runs are published; every number in a post is generated from, and cites, a stored record.

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
