# Contributing to Apron

Read [how development works](docs/development-process.md) first — it explains phases, briefs, tasks and where contributions land.

## Before you commit

Run the full check suite from the working tree:

```
prek run --all-files
```

This runs every check that CI will run. It is expected to pass before a commit exists, not after a pull request.

## What to read for each contribution type

Each path names at most three documents and one command.

### Adding an adapter

Engine, artifact source, render target, evaluation, execution target, planning source, evidence source, signal source, authority source or publisher adapter:

1. [`framework-spec.md` §1](docs/framework-spec.md) — the extension-point contracts
2. [`engineering-standards.md`](docs/engineering-standards.md) — layout, boundaries, test method
3. The ADR governing that extension point (linked from the framework spec)
4. Run the conformance suite from the `apron-conformance` distribution against your implementation (the suite ships as a Phase 1a prerequisite; Phase 0 produces the `typing.Protocol` definitions with fake-adapter test clients)

### Adding a calculator mechanism

1. [ADR-003](docs/adr/ADR-003-engine-native-computation.md) — Apron prediction vs engine evidence
2. [ADR-007](docs/adr/ADR-007-modality-agnostic-specs.md) — capability signatures and mechanisms
3. [`architecture-dispatch-proof.md`](docs/architecture-dispatch-proof.md) — why dispatch is mechanism-specific

### Adding a diagnosis rule

1. [`framework-spec.md` §6a](docs/framework-spec.md) — rule file format and fields
2. INV-2: a rule is `mechanism_verified` only with a proving record; otherwise it is `hypothesis`

### Challenging a design decision

Read the full [ADR set](docs/adr/). Open a challenge in [Discussions](https://github.com/mondegreens/apron/discussions). Amendments pass by lazy consensus with a stated window.

## How acceptance works

Acceptance criteria are mechanical. Layer contracts (import-linter), conformance suites, schema migration over historical fixtures, digest reproducibility and the invariant checks decide what merges. These are not matters of reviewer taste — they are deterministic, runnable, and the same for every contributor.

A gate blocks only when merging would let the project assert something untrue or unsafe. A gate that protects tidiness or style reports what to change and never rejects a contribution.

## AI-assisted contributions

See [`AI_POLICY.md`](AI_POLICY.md) for the full contract. The short version: disclose the tool, explain the change yourself, answer reviewers yourself.

## Pull requests

- The default branch takes squash merges only. The commit message that lands is written by the maintainer, so the commit-message grammar never touches your own commits.
- When checks are green, review begins. When they are not, a comment carries the local command that reproduces the failure.
- An idle pull request becomes a draft rather than a closed one.
