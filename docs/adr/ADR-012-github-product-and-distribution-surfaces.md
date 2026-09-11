# ADR-012 — GitHub organization, product repository and repository-native distribution

**Status:** Accepted, 2026-09-09; conformance-distribution clarification added 2026-09-10; repository names updated 2026-09-10 when the product was renamed from Olive to Apron (`naming-architecture.md` §1a). The topology, the boundary rule and the activation conditions are unchanged; only the names are. The private research repository was renamed accordingly on 2026-09-10.
**Affects:** repository topology, Target Architecture §12–14, Phase 1b, public release and contribution surfaces
**Evidence:** [GitHub Marketplace Actions](https://docs.github.com/en/actions/how-tos/create-and-publish-actions/publish-in-github-marketplace), [artifact attestations](https://docs.github.com/en/actions/concepts/security/artifact-attestations), [GitHub Discussions](https://docs.github.com/en/discussions/quickstart), [citation files](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files), [GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site), [MCP Registry](https://github.com/modelcontextprotocol/registry)

## Context

Apron must be installable and callable through CLI and MCP, but those surfaces alone require a user or agent to discover and configure the product before it can help. Model, inference-engine, deployment and application work already happens in repositories. A repository-native check can meet users at the model or deployment change that invalidates a prior claim and can make Apron's result visible without copying its decision logic into another project.

The private research history contains valuable investigations, rejected hypotheses, vendor snapshots and internal material. Making that repository public would expose material that has not passed publication, licensing, sanitization and claim-language review. Conversely, developing the public product inside a private research history would deny contributors a clean project record.

GitHub also offers many unrelated surfaces. Creating a repository, service or package for each one would fragment the product and create additional release and security obligations. The architecture therefore needs a deliberate organization topology and activation rule.

## Decision

### 1. Organization and repository topology

`github.com/mondegreens` is the public organization and institutional namespace. The initial topology is:

```text
mondegreens/
├── .github          public organization profile and shared community policy
├── apron            public canonical product, CLI, MCP, schemas and documentation
├── apron-research   private research laboratory with its own history
└── apron-action     public repository integration and Marketplace package
```

The current research history is never made public merely to bootstrap `apron`. Approved, licensed and sanitized conclusions are deliberately re-expressed or promoted into the clean public product history. Secrets, private task material, vendor working trees, transcripts and unpublished research do not cross that boundary.

`apron` owns the canonical core. Documentation, adapters, schemas, MCP mode and evidence-release tooling remain in that repository. A separate repository is justified only for an independently installable artifact whose host requires its own root metadata, version and release lifecycle. This rule admits `apron-action`; it rejects speculative repositories for documentation, MCP, each engine/provider adapter or each evidence mirror.

The rule governs repositories, not distributions. `apron` publishes a second package, `apron-conformance`, containing the extension-point conformance suites and golden fixtures. `framework-spec.md` §1 admits no acceptance path for an adapter other than passing those suites, so the party that must execute them is the adapter's author, in their own repository, without installing the product. A suite reachable only from inside this tree would make engine, registry, mechanism and accelerator neutrality unprovable by anyone but the maintainer. The package adds no repository, no separate history and no additional security boundary; it is versioned against the same released core as any other consumer.

### 2. GitHub Action as a first-class product adapter

`mondegreens/apron-action` is a thin, separately versioned adapter over a pinned released Apron core. It contains no calculator, qualification, ranking, evidence-promotion or remediation authority. Its inputs and outputs map losslessly to the same versioned contracts used by CLI, MCP, API, CI and web.

The Action may evaluate repository changes to model/artifact references, engine or container versions, workload and deployment specifications, rendered artifacts and retained qualification fingerprints. Its Check Summary exposes at minimum:

- the exact changed and evaluated inputs;
- calculated, inherited/transferred, proven-constraint and measured states without collapsing them;
- source coverage, freshness, uncertainty and explicit unknowns;
- invalidated prior qualifications and the evidence required to restore them;
- stable record or report identifiers when a publishable record exists.

GPU-free resolution, calculation and version-pinned compatibility analysis may run on an ordinary hosted runner. They never become engine validation or GPU measurement. GPU execution occurs only through an explicit local or provider target and the same accepted protocol, external `AuthorizationDecision`, credential scope, cost/runtime/teardown limits and publication separation used by every other Apron interface. Pull requests from forks receive no provider secrets or side-effect authority.

The first Marketplace release is gated by a stable structured-output schema and one external fixture repository that consumes a SHA- or immutable-release-pinned Action and receives the expected Check Summary. GPU dispatch is not required for the first Action release; it becomes an Action capability only when the normal execution-target and authorization conformance suites pass through that adapter.

### 3. GitHub trust, recognition and community surfaces

The public product uses repository-native facilities where each has a concrete consumer:

- GitHub Releases are the human and machine release ledger for source, changelog and installable artifacts.
- PyPI remains the primary Python/CLI package channel; the MCP server is the same released Apron package and is published to conforming MCP registries.
- GitHub Pages publishes methodology, contracts, metrics and the generated evidence catalogue from public records.
- GitHub Discussions holds Q&A, exploratory integration questions and community knowledge; Issues remain actionable bugs, accepted features and structured evidence/failure intake.
- A public organization Project may expose evidence gaps and contribution-ready work without exposing private research planning.
- `CITATION.cff` supplies canonical software and dataset citation; record-level machine-readable attribution remains mandatory.
- Releases expected to be executed or redistributed carry build provenance and, where applicable, an SBOM through artifact attestations. Attestation proves origin and build linkage, not correctness or security.
- Topics, a social preview, a compact badge set and the organization profile aid discovery but never count as product use or evidence.
- Rulesets, protected environments, `CODEOWNERS`, private security advisories, dependency/security automation and least-privilege workflow permissions enforce ADR-009 and ADR-010.

Mondegreens does not enable `FUNDING.yml` or GitHub Sponsors while the owner policy is to accept no money. Providers, model labs and users may contribute scoped credentials, credits, quota or capacity under INV-29. Public acknowledgement names the contributor and applicable terms on the affected records or a resource-contribution page; it never implies endorsement, partnership, favorable placement or authority over conclusions.

### 4. Conditional surfaces

- A GitHub App is introduced only when a required workflow needs installation-scoped asynchronous webhooks or cross-repository operation that a repository-owned Action cannot provide. The App uses the same core and least-privilege permissions.
- An Apron image is published to GHCR only when a supported hermetic Apron runner or remote MCP deployment requires an OCI artifact. Apron does not republish upstream engine images; engine images remain upstream-owned and pinned by digest.
- A separate evidence, documentation, MCP or adapter repository requires an independently versioned consumer contract and lifecycle that cannot be served by `apron`. Repository count is not a distribution metric.
- A Discord, Slack or other off-GitHub community channel requires demonstrated moderation and support demand. GitHub Discussions is the default public knowledge surface.

## Consequences

- CLI and MCP remain the primary direct-use interfaces from ADR-004; the GitHub Action becomes the primary repository-native acquisition and continuous-qualification adapter.
- Every consuming workflow visibly identifies a versioned Apron integration without turning a badge, install or workflow reference into evidence of successful use.
- Research remains private while the product receives a clean, inspectable and contributor-friendly public history. The public development strategy (`public-development-strategy.md`) governs how the public repository operates, what crosses the private/public boundary, community health standards, security norms, governance, AGENTS.md, and protection mechanisms. It must be read alongside this ADR before any agent sets up or configures the public repository.
- GitHub improves engineering, trust, recognition and distribution; it does not itself improve inference runtime performance. Runtime gains still come from engine/provider implementations, measured configuration selection and qualification against the accepted workload.
- Marketplace, MCP registry, PyPI, Pages, Hugging Face evidence mirrors and upstream contributions form complementary distribution routes around one core rather than competing product implementations.
