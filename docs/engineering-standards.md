# Engineering standards — object model, boundaries, determinism, testing method and dependency policy

**Date:** 2026-09-10 · **Revised:** 2026-09-11 (§10a, §10b and §11 amended for the lockfile drift check, the advisory type lane and the fork-workflow settings) · **Status:** policy, not product architecture (ADR-009); this document changes without an architecture version bump · **Authority:** ADR-003, ADR-006, ADR-009, ADR-012; `framework-spec.md` §2, §4, §6

## 0. Standing, and how to read this

Most of this document is the house default and the reason behind it, not a rule you can fail. Three things here are binding, and each is binding because a claim the project publishes is false without it: `Protocol` ports (§1, because conformance must be executable without depending on the product), determinism seams (§4, INV-42) and record canonicalization (§5, INV-43). Everything else is a default. A contribution that departs from a default and states why is a normal contribution.

The distinction is deliberate and matches `public-development-strategy.md` §4b: the project is strict about what it claims and generous about how the code is written. A check that would only make the repository tidier is a comment, not a gate.

Read this when you are about to write code, not before deciding whether to. `public-development-strategy.md` §4a names the two or three documents that actually apply to a given contribution.

The architecture states what the system is and what must never be true. This document states how the code that realizes it is built, bounded and verified. Where it disagrees with an ADR, `framework-spec.md` or a schema, those win. §12 records the practices deliberately left ungated, and why.

## 1. Object model

Two of the entries below are binding because a claim depends on them; the rest are the house default and the reason for it. A contribution that departs from a default and says why is a normal contribution, not a violation. Nothing in this section is a merge gate on its own.

| decision | standing |
|---|---|
| Extension-point ports are `typing.Protocol`, not abstract base classes | **Binding.** `framework-spec.md` §1 admits no acceptance path other than the conformance suite, so a conforming implementation must be able to exist without importing Apron. A nominal base class would make conformance require a dependency on the product and would contradict registry, engine and accelerator neutrality. An abstract base class remains available where adapters genuinely share executable behaviour, but never as the conformance contract |
| Records, envelopes, plans and reports are frozen models (`ConfigDict(frozen=True)`) | **Binding.** Append-only evolution and immutable authorization envelopes are asserted by INV-5, INV-21 and INV-35. Without enforced immutability those are descriptions rather than properties |
| Alternatives within one contract use discriminated unions with `Literal` tags and `Field(discriminator=...)` | Default. Applies at least to `InferenceSolution`, `ArtifactRelation`, `ActionAttempt`, the four epistemic statuses of INV-1, `AuthorityContribution` and the component-mechanism registry. Tag lookup rather than trial validation keeps error shapes deterministic, which matters because those errors are part of the CLI and Action contract |
| Deep inheritance hierarchies are avoided in favour of discriminated unions of frozen models | Default. The reason is migration: INV-5 requires every historical fixture to move forward, and a union of explicit variants migrates predictably where a base-class tree does not |
| Dependency wiring is constructor injection with one composition root per interface | Default. A framework is not excluded on principle; the reason for the default is that the composition root is where INV-11's interface boundary is checked, and an implicit container makes that boundary harder to read |
| The calculator, the authorization engine's combination logic, the renderers and the evidence-promotion policy are functions rather than objects | Default, and close to binding in effect: these are the components whose determinism INV-42 asserts, and a function is verifiable without construction or lifecycle |

Patterns that earn their place and the reason each is required rather than preferred:

- **Adapter/port** at all ten extension points — the conformance contract.
- **Strategy over a mechanism registry** for resource calculation, whose registry default is `unknown`. INV-32 forbids selecting a formula from a modality or family label; a registry whose miss returns a fallback would violate it silently.
- **Policy combination** in the authorization engine: deny-overrides, intersection of independently supplied constraints, indeterminate never permitting. This is a rule-combining algorithm and is tested as one, with an explicit truth table, not as a sequence of conditionals.
- **Frozen value objects** for every record and envelope.

## 2. Adapter discovery

Adapters register through packaging metadata, one entry-point group per extension point:

```toml
[project.entry-points."apron.engines"]
vllm = "apron_vllm:VllmEngineAdapter"

[project.entry-points."apron.artifact_sources"]
huggingface = "apron.adapters.huggingface:HubResolver"
```

`importlib.metadata.entry_points(group=...)` is the loader. Entry points are a PyPA interoperability specification and are available in the standard library from the declared Python floor. Group names are `apron.engines`, `apron.artifact_sources`, `apron.render_targets`, `apron.evidence_sources`, `apron.planning_sources`, `apron.evaluation_adapters`, `apron.execution_targets`, `apron.signal_sources`, `apron.authority_sources` and `apron.publishers`. Discovery never implies support: a discovered adapter still resolves to `unknown` or `unsupported` per stage until its conformance record exists.

## 3. Module boundaries

Distribution layout is `src/apron/`. The twenty-two directories named in `framework-spec.md` §6 are subpackages within it, assigned to four layers with one permitted dependency direction. The assignment is recorded here so that it is not re-derived by each reader:

| layer | directories from `framework-spec.md` §6 | also holds |
|---|---|---|
| `domain/` | `schemas/` with its `migrations/`, `mechanisms/`, `capabilities/`, artifact identity | the calculator, fingerprints, authorization combination, evidence-promotion policy, `solutions/` — no I/O, no adapters |
| `application/` | `orchestration/`, `signals/`, `metrics/` | the qualification graph, scheduling, use cases |
| `adapters/` | engine adapters, artifact resolvers, `renderers/`, `evidence/sources/`, `planning/sources/`, `evaluations/`, `backends/`, `publishers/` | execution targets, authority sources |
| `interfaces/` | CLI, MCP, API | composition roots only |

Five of the §6 directories are not source packages and sit at the repository root instead: `conformance/`, which is published as its own distribution (§7); `rules/`, holding diagnosis rules as YAML-frontmatter markdown; `records/`, append-only; `interim/`, for implementations awaiting an upstream home (ADR-008 §3); and `upstream/`, the index of open contributions. `docs/` and `tests/` join them.

`domain` imports nothing from the other three. `application` imports `domain`. `adapters` import `domain` and `application`. `interfaces` import all three and contain no recommendation, calculation, ranking or promotion logic.

Contracts are declared in `pyproject.toml` and checked by import-linter in ordinary CI. This is the mechanism INV-11 requires; the invariant previously named an architectural test without naming how it is executed.

## 4. Determinism (INV-42)

The architecture describes the optimizer, the authorization engine, corrections, generated endpoint plans and decision reports as deterministic. That claim is unverifiable while any component can read an ambient clock, generate an identifier or draw randomness.

- Wall-clock time, identifier generation and randomness reach `domain/` only through injected `Clock`, `IdGenerator` and `Rng` ports.
- A component described as deterministic names its seed sources and produces byte-identical output for identical inputs.
- No execution, solution, artifact or task fingerprint contains a value obtained from an uninjected source.
- The same ports are the seam through which orchestration replay, crash and duplicate-event tests (ADR-010, INV-21, INV-22) become executable without a live provider.

These ports are cheap while `domain/` is small and are not retrofittable once record writers and the orchestrator exist. They are a start condition for schema implementation.

## 5. Record serialization and identity (INV-43)

INV-35 derives an evidence release's identity from a schema-versioned manifest and record digests rather than a host URL, and INV-12 requires reproducibility to the limit of a typed claim. Both require a defined mapping from a record to bytes; without it, mapping order, floating-point representation, Unicode normalization and omitted optional fields all change the digest.

- Canonical form is RFC 8785 (JSON Canonicalization Scheme).
- Identity is SHA-256 over that form, carried with a multihash prefix so the algorithm is recorded rather than assumed.
- Test vectors are published with the schemas. A digest is correct when an independent implementation, in another language, reproduces it from the same record.
- A digest computed from an uncanonicalized encoding is rejected rather than stored.

The vectors matter more than the algorithm choice. The portability claim in INV-35 is that another tool can reconstruct and verify a release; an implementation that cannot recompute a digest falsifies it.

## 6. Concurrency

The default is a synchronous `domain/`, with asynchrony inside the adapters that perform network or subprocess I/O, and an interface owning the event loop at its composition root.

The reason is testability rather than performance: the deterministic core is where properties, replay and simulation execute, and asynchrony there would make every determinism check depend on scheduling. Where an extension point genuinely needs an asynchronous contract — a streaming evaluation adapter is the likely first case — that is a contract change proposed with its reason, not a rule to work around.

## 7. Conformance suites and third-party execution

`framework-spec.md` §1 states that a new implementation is accepted when the conformance suite passes on the golden fixtures and that there is no other acceptance path. That acceptance path must be executable by whoever wrote the adapter.

- The suites are published from `apron` as a separate distribution, `apron-conformance`, installable without the product. ADR-012 §1 restricts speculative repositories, not distributions; this adds no repository.
- The distribution exposes a pytest plugin. An adapter author runs the suite against their own implementation in their own repository, against fixtures they did not write.
- The suite is written before the first adapter of a given extension point, with a fake adapter as its first client.
- Fixtures recorded from a pinned engine carry the engine digest and capture date. A recording older than the currently pinned image fails the suite (ADR-003 §9 amendment). Without expiry, GPU-free tests drift from the engine and pass against a fiction.

## 8. Test kinds and where each applies

The corpus already specifies verification obligations in four places: the `enforced by` column of `framework-spec.md` §2, the required conformance and failure tests of ADR-011, the required fixtures of ADR-007, and the injected failure classes of ADR-005 §3, plus each phase exit gate. This section states what kind of test satisfies an obligation, not which obligations exist.

| kind | applies to | note |
|---|---|---|
| Unit | Pure functions in `domain/` whose correctness is arithmetic or structural: exact byte computation, canonical serialization, digest, fingerprint construction, authorization combination | Irreplaceable and small. An error here corrupts an append-only record permanently, and a failing acceptance test cannot localize it |
| Property-based | Round-trip and migration (INV-5, INV-8); calculator laws — monotonicity in context length and concurrency, additivity across components, unknown-preservation, unit and scope always present (INV-1, INV-32) | Example-based tests cannot express these; they are the actual content of the invariants |
| Stateful / model-based | The qualification graph (INV-18, INV-24, INV-25, INV-31). The property under attack is that no sequence of legal operations promotes evidence across a claim scope or execution fingerprint | The graph is already specified as a state machine; this executes it |
| Conformance | The ten extension points, per §7 | The only acceptance path |
| Adversarial / negative | Every obligation phrased as a prohibition. The test attempts the violation and asserts refusal | Already enumerated across ADR-005, ADR-007 and ADR-011 |
| Simulation | Orchestration durability: crash, timeout, target loss, replay, deduplication, teardown (ADR-010, INV-21, INV-22) | Executed through the §4 ports with fault-injecting fakes. Integration tests do not reach these paths |
| Recorded contract | Artifact sources, provider backends and publication destinations | Tier 0 covers the engine; nothing covers registry or provider drift |
| Snapshot | Rendered artifacts and interface output: `vllm serve` commands, Compose, Helm values, CLI output, Check Summary | A missing snapshot fails, not only a differing one |
| Mutation | The pure core only | See §12 |

## 9. Where test-first does not apply

Test-first is the default for `domain/` and for every contract in §7. It is explicitly not the method for code whose specification is an external system's observed behaviour: engine execution output, artifact-source responses, provider APIs.

Those are **recorded-fixture-first**. Capture the real response, commit it with provenance and capture date, then write the code that parses it. A test written before observing the system asserts an assumption about it — the same distinction the architecture already draws between `predicted` and `measured`, applied to the test suite.

## 10. CI tiers

| tier | contents | trigger |
|---|---|---|
| pure | `domain/` unit and property tests, canonicalization, digest, import contracts | every push |
| contract | conformance suites on golden fixtures, round-trip, migration over historical fixtures, docs lint | every pull request |
| integration | recorded-contract adapters, snapshot output, orchestration simulation | pull requests touching those paths; scheduled in full |
| execution | GPU conformance, boot, profile and remediation (ADR-003) | release and explicit dispatch, inside the authorization envelope |
| periodic | long property runs, mutation runs, simulation soak | scheduled |

Budgets are set from measurement rather than declared in advance. GPU-free CI never emits engine or GPU validation (INV-37).

## 10a. What guards a pull request

A change can deviate from the product in three ways, and each has its own automated guard. Nothing else is gated. The point of naming them is that a contributor can predict, before opening a pull request, exactly which class of objection is possible.

**Shape** — where code may live and what it may import. **Contract** — the typed surfaces an implementation must satisfy. **Claim** — what the produced data is allowed to assert.

Most of the `enforced by` column of `framework-spec.md` §2 is GPU-free and therefore runs at pull-request time. The exception is stated in ADR-003: engine compatibility, boot, kernel and profile verdicts require GPU execution and are a release gate, not a pull-request gate. GPU-free CI never presents its own result as engine validation (INV-37).

| job | guards | principally enforces | on failure |
|---|---|---|---|
| `format` | style | — | auto-fixed and reported, never a rejection |
| `types` | shape | schema contracts as the public surface. pyright is the blocking checker; Astral's `ty` runs in the same job as a separate advisory lane whose failure is reported and never rejects (`product-definition.md` §1c) | blocks on pyright; advises on `ty` |
| `boundaries` | shape | INV-11 layer contracts; the deterministic core imports no adapter and no ambient clock, identifier or randomness source (INV-42, static half) | blocks |
| `schemas` | contract | INV-1 unit/scope and epistemic status on every assertion; INV-4 evidence level, freshness and provenance; INV-17, INV-20, INV-25 required identities and fingerprints; discriminated-union completeness | blocks |
| `migrations` | contract | INV-5 over every historical fixture, generated from the compatibility matrix; a new or changed public field without a migration entry and at least one fixture fails here | blocks |
| `identity` | claim | INV-43 canonical form and digest reproducibility against the published vectors; INV-35 release identity independent of any host | blocks |
| `determinism` | claim | INV-42 executable half: the deterministic paths run twice under different injected clock, identifier and randomness and must produce byte-identical output | blocks |
| `conformance` | contract | the `apron-conformance` suites against every in-tree adapter on recorded fixtures; INV-31 capability scope, INV-33 locator versus identity, INV-41 separate neutrality proofs. Adapter logic is checked here; engine verdicts are not (ADR-003) | blocks |
| `interop` | contract | INV-8 lossless round-trip for shared fields to the three pinned ecosystem shapes | blocks |
| `policy` | claim | the authorization engine's combining truth table (INV-14, INV-21, INV-40); promotion policy (INV-18, INV-24); optimizer and substitution golden tests (INV-23, INV-27); metamorphic scheduler with and without contributed resources (INV-29, INV-36); planner claims never becoming verdicts (INV-34); routing and replica-versus-shard fixtures (INV-38, INV-39) | blocks |
| `isolation` | claim | network-deny and data-label propagation (INV-13, INV-26); CLI side-effect and consent tests (INV-19, INV-22); hostile-media fixtures; renderer static check and extractor fuzzing (INV-6) | blocks |
| `rules` | claim | the diagnosis rule table: `status: mechanism_verified` requires a non-null correction and at least one matching proving record, otherwise the rule is `hypothesis` (INV-2) | blocks |
| `secrets` | claim | secret detection over the diff and over any stored log, alongside repository push protection (INV-13) | blocks |
| `supply-chain` | shape | every third-party action resolves to a full commit SHA; dependency audit over the committed `pylock.toml`; regeneration of that export from the resolver's native lockfile, failing on any diff, because the two are updated by different mechanisms (§11); code scanning | blocks on an unpinned action, a stale `pylock.toml` or a known vulnerability; advises on an available update |
| `docs` | claim | INV-16 over meaningful empirical or derived quantitative claims only, quoting the sentence and naming the missing source. Dates, versions, identifiers, list numbering and test-derived structural facts are out of scope | blocks on an unsourced claim in public text; advises elsewhere |
| `spec-drift` | claim | a change to a frozen architecture document without a referenced ADR, and an ADR whose affected sections no longer exist | advises, and requests the ADR reference |
| `commit-grammar` | repository tidiness | ADR-009 §5 subject grammar and absence of agent branding or generated-by trailers | advises; the history is corrected on merge (`public-development-strategy.md` §4b) |

Two properties of this set matter more than its contents. Every blocking job answers a question about the product rather than about the contributor, so a rejection is always traceable to a claim that would otherwise become false. And every blocking job is runnable locally by the person writing the change — the conformance suites are installable, the fixtures are in the tree, and the same commands run in CI — so a contributor never discovers an objection they could not have anticipated.

## 10b. The checks run before the commit, not after the pull request

The job graph above is the backstop. Its primary use is local, and it is written to be usable that way: every job in §10a runs from one command in the working tree, against fixtures that are in the tree, using suites that are installable.

That command is the last step of authoring rather than the first step of review. An agent or contributor runs it before committing, so what reaches a reviewer is a question of intent rather than of correctness. This is the project's answer to the measured increase in review burden that AI-assisted authoring produces — a 19% productivity decline for experienced developers and a 6.5% rise in review load (arXiv 2510.10165). The burden is removed by moving correctness into the loop that writes the code, not by adding a second reviewer after the code exists.

Two consequences follow:

- **No check may exist that cannot run locally.** A gate reachable only through continuous integration moves a discoverable error to the latest and most expensive moment, and teaches a contributor that the rules are found by failing rather than by reading.
- **Automated review of a proposed change belongs in the authoring session**, which already holds the full context of the change, rather than in a workflow holding a credential. A continuous-integration reviewer sees less than the session that wrote the code, and costs a credential boundary to operate: pull requests from forks carry no secrets by design (INV-37), and the workflow trigger that circumvents that restriction is precisely the one that exposes them.

An external automated reviewer is introduced when external pull requests exist to calibrate it against, not before. When introduced it is advisory under `public-development-strategy.md` §4b: it comments, and the deterministic checks gate. It is also **triggered by a maintainer on a specific pull request and never posts automatically on every pull request** — LLVM's AI tool policy states that "automated review tools that publish comments without human review are not allowed", and Node.js's AI guidelines (merged 2026-08-12) forbid automated review responses, so a reviewer that posts on arrival is a contribution this project could not itself make upstream. Its subject is the judgment the checks cannot make — whether an abstraction suits the extension point, whether evidence language claims more than its fingerprint supports, whether an `unknown` is honest — never a re-derivation of a result §10a already produced deterministically and for free.

**Workflow security follows ADR-009 and is not negotiable per job.** The default token permission is read-only and widened only where a job demonstrably needs it. Every job declares a timeout and the workflow declares a concurrency group that cancels superseded runs. Third-party actions are pinned to reviewed full commit SHAs. Pull requests from forks receive no secrets, no protected environment and no side-effect authority (INV-37); provider execution, release publication, evidence publication and external-repository mutation each use a separate credential scope and a protected environment with a required reviewer, and are reachable only by explicit dispatch, never by a pull-request event. No workflow triggered by a fork's pull request runs with elevated context. Two concrete consequences: **no workflow uses `pull_request_target`**, which is the trigger that grants a fork's code the elevated context this rule forbids, and the repository setting for fork pull request workflow approval is **all external contributors**, the strictest of the three values GitHub offers.

## 11. Dependency and version policy

| axis | rule |
|---|---|
| Python | Floor is the declared minimum; the matrix tests the floor and the newest stable release, not the versions between them. A newly released version is added within a stated window; a version is dropped at upstream end of life. The floor is stated on its own terms and is not described as matching another project's floor |
| Direct dependencies | Grouped minor and patch updates, separate security updates, and a cooldown period before an update is eligible for automatic merge |
| Pinned actions | ADR-009 §2 requires full-commit-SHA pins. SHA pins are kept current by automated update proposals; a pin policy without an update mechanism produces stale dependencies rather than reviewed ones |
| Lockfile | The resolver's native lockfile is primary; a PEP 751 `pylock.toml` export is committed beside it as the tool-neutral auditable record and is the input to dependency vulnerability scanning. The two are maintained by different mechanisms and therefore drift: the automated update service edits the native lockfile and does not read the export, while the audit tool reads the export and not the native lockfile. The `supply-chain` job (§10a) regenerates the export and fails on a diff, so a dependency update that skipped the export is caught rather than silently audited against stale content |
| Engine support window | The number of simultaneously supported engine minor versions is stated explicitly. The engine releases on a two-week cadence with backward compatibility guaranteed for a limited number of minors, so an unstated window is either an unaffordable re-verification obligation or an unsupported currency claim |

## 12. Recommended practice, deliberately ungated

The following improve the suite and are recorded here as practice. None is an invariant, an exit gate or a checklist item, because each requires recurring human attention or recurring cost and would therefore be dropped. An obligation the project states and stops honouring is a false claim under INV-1 and §2a, which is a worse outcome than not stating it.

- **Mutation testing** over the pure core, scoped to changed lines rather than a global score.
- **A unified index** mapping every written verification obligation across `framework-spec.md`, ADR-005, ADR-007, ADR-011 and the phase exit gates to a named test with a stable identifier. Valuable, and only durable if the placeholder for an unwritten test expires rather than remaining indefinitely skipped.
- **Scheduled re-recording** of external contract fixtures.
- **Ownership controls on golden fixtures.** The mechanical half — surfacing the byte difference of a regenerated fixture in the change description — costs nothing and is retained; a review gate on the same files is not.
- **Tolerance bands and repetition policy** for profiled memory observations, which are not reproducible between executions.
- **Deterministic simulation of the full orchestrator**, as distinct from the §4 ports, which are required.
- **Instrumentation tests** proving that telemetry events fire, are counted by the published rules, and that the absence of consent suppresses transmission.

Any of these may be promoted to an invariant later, on the evidence that it was actually held.
