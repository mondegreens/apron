# How development works

This page explains how the project moves from decisions to running code.
Read it once before your first task. It is short because the process is
simple; every reference is a link you can follow if you need more.

## The document chain

Six kinds of document exist, each feeding the next:

1. **Product definition** ([`product-definition.md`](product-definition.md))
   — what the system does and for whom.
2. **Target architecture** ([`target-architecture.md`](target-architecture.md))
   — the permanent shape: sections, invariants, extension points.
3. **Architecture decision records** ([`adr/`](adr/))
   — each decision with its context, evidence and consequences. Thirteen
   exist. They are numbered, not ordered by importance.
4. **Phase plan** ([`phase-plan.md`](phase-plan.md))
   — the build sequence. Five phases, each with a start condition, an
   objective and an exit gate.
5. **Phase brief** (e.g. [`phase-0-brief.md`](phase-0-brief.md))
   — the ordered task list for one phase. Every decision an implementer
   would otherwise have to invent is made here, with a source.
6. **Code and tests** — the implementation, checked by `prek run
   --all-files` before every commit.

Each document type answers a different question:

| document | question it answers |
|---|---|
| product definition | what and why |
| target architecture | what shape, what must never be true |
| ADR | why this choice and not another |
| phase plan | what to build, in what order, with what exit gate |
| phase brief | which files, which fixtures, which tests, in what sequence |
| code | does it work |

A contributor reads downward to the level they need. Someone adding an
adapter reads the framework spec and the conformance suite. Someone
implementing a phase reads the brief. Someone challenging a design
decision reads the ADRs.

## Phases

A phase is a bounded set of work with three properties:

- **Start conditions** that must be true before implementation begins.
  These are checked, not assumed.
- **An objective** stated in one sentence.
- **An exit gate** that is mechanical: when the tests pass and the
  fixtures round-trip, the phase is done. No subjective review decides
  whether an exit gate is met.

Phases are sequential. Each phase's exit gate produces the start
conditions for the next. The phase plan names five phases plus a 90-day
external validation protocol. Phase boundaries can move; the target
architecture cannot be narrowed through the plan.

## Phase briefs

A phase brief turns one phase of the plan into tasks. It exists because
the phase plan says *what*, not *in which order or with which fixtures*.
An implementer should not start a phase from the plan alone.

A brief contains:

- **Goal** — why the phase exists and what its tasks produce, in three
  to five sentences. A contributor reads this to decide whether the
  brief is relevant to them.
- **Ordered tasks** — each with its inputs, outputs and acceptance
  criteria. Dependencies between tasks are explicit.
- **Fixture specifications** — what each test fixture looks like, where
  it goes, what it proves.
- **Exit gate as tests** — the exact test categories that together
  satisfy the phase plan's exit gate.

The first contributor through a phase writes both the code and the
trail that others will follow. The brief is that trail: once the phase
is done, the conformance suites, golden fixtures and test patterns it
produced are the surface that later contributors build against.

### Naming, location and lifecycle

Briefs live in `docs/`, named by phase: `phase-0-brief.md`,
`phase-1a-brief.md`, `phase-1b-brief.md`. No subdirectory — there
will be at most five briefs, one per phase.

Every brief carries a status line at the top, matching the ADR
convention:

| status | meaning |
|---|---|
| `active` | implementation is in progress |
| `implemented, <date>` | the exit gate passed; the phase is done |
| `superseded by <link>` | a revised brief replaced this one |

A completed brief stays in `docs/`. It is not moved, archived or
deleted — it is the record of what was built and in what order. The
status line is how you tell whether a brief is current work or
history.

## Tasks

A task is one item in a brief. It has:

- **A clear input** — what must exist before this task starts (a prior
  task's output, an owner decision, a pinned source artifact).
- **A clear output** — a file, a test, a fixture, a schema module.
- **Acceptance criteria** — what "done" means. Always mechanical: a test
  passes, a fixture round-trips, a digest matches.

Tasks are small enough that a single session can complete one. They are
ordered so that each task's output is the next task's input. When a task
is done, it is done — there is no revisit unless a later task finds a
defect.

## Where contributions land

Not every part of the project is open to every kind of contribution at
every time. The surface grows as phases complete.

### Open now (no prior approval needed)

- **Challenge a decision.** Every ADR records its reasoning. Open a
  challenge in [Discussions](../../discussions); amendments pass by lazy
  consensus.
- **Check the dispatch proof** against the pinned vLLM source. It is a
  claim about code and can be wrong.
- **Report a deployment failure** for the knowledge base: traceback,
  exact artifact, engine tag, GPU. It enters `rules/` as a `hypothesis`.
- **Documentation correction.**

### Open after Phase 0 (conformance suites ship)

- **Adapters.** Engine, artifact source, evaluation, execution target
  and every other extension point. Write the adapter in your own
  repository, run the conformance suite there, propose it here only
  after it passes. See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for the
  reading list.
- **Diagnosis rules with a proving record.** A hypothesis promoted to
  `mechanism_verified` by a corrected-boot record.
- **Bug fix with a failing test.**

### Requires an approved issue or RFC

- New extension-point implementation
- Schema change
- Any change over roughly 500 lines
- Architecture decision amendment

The scope boundaries are enforced by convention and by the acceptance
criteria in [`AGENTS.md`](../AGENTS.md), not by access controls.

## How the trail gets cleared

The project is built by one person walking the full path for the first
time. Each phase produces artifacts that make the next contributor's
path shorter:

| phase | what it opens for others |
|---|---|
| Phase 0 | Frozen schemas, golden fixtures, conformance suites. A contributor can now write an adapter and test it independently. |
| Phase 1a | A working CLI and MCP server. A contributor can run `apron plan` and file a bug against real output. |
| Phase 1b | Evidence breadth, diagnosis rules, the GitHub Action. External repositories can consume the Action. |
| Phase 2 | Structured API, remote MCP, evaluation adapters. External tools can integrate. |
| Phase 3+ | The automated pipeline runs; contributors extend coverage rather than building infrastructure. |

The first walk is the hardest because it produces everything at once:
the contracts, the fixtures, the suites, the CLI, and the patterns
that make later contributions mechanical. That is deliberate. The
project is strict about what it claims and generous about how the code
is written — a contribution that departs from a default and says why
is a normal contribution, not a violation.

## AI-assisted development

AI tools are welcome at every phase. The contract is in
[`AI_POLICY.md`](../AI_POLICY.md): disclose the tool, explain the
change yourself, answer reviewers yourself. Unattended agents do not
open pull requests or issues.

A phase brief is written so that an AI agent can implement a task from
it without re-deriving the plan from the ADRs. The brief is the context
boundary: everything above it (product definition, architecture, ADRs)
is decisions already made; everything below it (code, tests) is
execution.
