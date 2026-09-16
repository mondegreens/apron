# Phase 0 brief — assessment portions

Each portion below is a self-contained prompt for `/assess`. Run them in
order. Each references the brief (`docs/phase-0-brief.md`) by section
so the skill reads only what it needs.

---

## Portion 1 of 6 — External format pinning and provenance

Assess the external format pinning plan in `docs/phase-0-brief.md`
Part 1 (§1.1–§1.3, lines 47–340).

The task: pin representative source artifacts from 5 ecosystem sources
(vllm-project/recipes, NVIDIA aiconfigurator, SemiAnalysis InferenceX,
vLLM configuration surface, Hugging Face Hub) plus the llmcalc legacy
catalogue. Each pin records URL, commit, date, digest, license and
the exact schema or example used. Two additional sources (Inspect AI,
Harbor) are informational references, not exit-gate.

Key questions:
- Are the 5 exit-gate sources the right ones per ADR-002 §4?
- Is the provenance schema (§1.1) sufficient to reconstruct a pin
  without the original repository?
- Are the pinned files from each source the right ones to ground the
  schemas against? Would a different file from the same repo be more
  representative?
- Is the llmcalc import (§1.3 step 7) correctly scoped — 101 entries,
  no fallback formulas, explicit unknowns?
- Does the ordering (§1.3) have dependency issues?

Reference documents: ADR-002, ADR-003 §10 (llmcalc), ADR-006,
phase-plan.md §Phase 0 lines 11–17.

Cloned sources are in `.sources/` (recipes, aiconfigurator, InferenceX,
vllm, inspect_ai, harbor).

---

## Portion 2 of 6 — Implementation conventions and infrastructure

Assess the implementation conventions in `docs/phase-0-brief.md` §2.0
(lines 344–412).

The task: these conventions are load-bearing infrastructure that every
schema depends on. They were added after adversarial review found that
the original brief had no canonicalization convention, no
fingerprint-vs-digest distinction, no cross-layer reference rule, and
no migration-vs-identity convention.

Key questions:
- Does `model_dump(mode="json")` actually solve the datetime/canonical
  problem? Verify against `src/apron/domain/canonical.py`.
- Is the `fingerprint_fields()` classmethod the right pattern for
  identity-subset fingerprinting? What are the alternatives (decorator,
  metaclass, config field)?
- Is "cross-layer references are fingerprint strings" sound? Does it
  create problems for queries, joins, or human readability?
- Does the migration convention (new version, new digest, old refs
  valid) actually satisfy INV-5 and INV-12?
- Are the module locations (framework-spec.md §6 directories) correctly
  mapped?

Reference documents: engineering-standards.md §1, §4, §5;
framework-spec.md §6; ADR-006 §Record-serialization; existing code in
`src/apron/domain/canonical.py` and `src/apron/domain/ports.py`.

---

## Portion 3 of 6 — Schema layers 0–3 (primitives through tasks)

Assess the schema design in `docs/phase-0-brief.md` §2.1 Layers 0–3
(lines 413–480).

The task: these are the foundational schemas everything else builds on.
Layer 0 has primitives + ExecutionTarget Protocol. Layer 1 has artifact
identity and lineage. Layer 2 has models, execution, evidence and the
calculator contract. Layer 3 has task, application and serving specs.

Key questions:
- Is the layer ordering actually dependency-free downward? Can Layer 2
  be built without importing anything from Layer 3 or higher?
- Is `ExecutionTarget` correctly placed in Layer 0, or should it be a
  separate extension-point alongside the others in §2.8 step 11?
- Is the calculator Protocol in Layer 2 the right abstraction? It
  dispatches on ComponentMechanism + HardwareSpec → PlanningClaim.
  Does the phase plan require a different signature?
- Does `QualityEvidence` "carries fingerprints" (not embeds) actually
  work when the evidence needs to display what it references?
- Is `WorkloadSpec` as a binding envelope (Layer 3) the right pattern,
  or should `TaskSuiteSpec` and `ServingWorkloadSpec` just be
  independent fields on `DecisionRequest`?

Reference documents: ADR-001 §4 (ExecutionTarget), ADR-006 (artifacts,
quantization, calculator), ADR-007 (capabilities, mechanisms), ADR-011
§1–3 (task/application/serving separation), phase-plan.md §Phase 0.

---

## Portion 4 of 6 — Schema layers 4–7 (solutions through reports)

Assess the schema design in `docs/phase-0-brief.md` §2.1 Layers 4–7
(lines 480–527).

The task: these are the higher-level schemas that consume the
foundations. Layer 4 has solutions, plans and evaluation protocol.
Layer 5 has records (including RemediationRecord and DiagnosisRule,
added after adversarial review). Layer 6 has authority and automation.
Layer 7 has decision reports and resource pools.

Key questions:
- Does `EvaluationProtocol` carrying DecisionRequest by fingerprint
  (not embedding) actually eliminate the circular dependency? What
  happens when you need to display the evaluation protocol alongside
  the request it binds?
- Are `RemediationRecord` and `DiagnosisRule` (Layer 5) the right
  shapes? Verify against ADR-006 §Remediation-proof exact fields.
- Is the authority layer (Layer 6) complete? ADR-010 has 15 decision
  items — does the schema set cover all of them?
- Is `DecisionReport` (Layer 7) the right top-level output? Does it
  carry everything the phase-plan exit gate requires?
- Does the `PlanningClaim` schema correctly handle the aiconfigurator
  mapping (database_mode as producer metadata, NOT as EpistemicStatus)?

Reference documents: ADR-010 (all 15 decisions), ADR-011 §4–9
(solutions, evaluation, economics, qualification), ADR-013 (topology),
phase-plan.md §Phase 0 exit gate.

---

## Portion 5 of 6 — Fixtures (golden, negative, capability, all categories)

Assess the fixture design in `docs/phase-0-brief.md` §2.2–§2.7 plus
the negative/rejection, authority, and evidence-state fixture sections
(lines 528–780).

The task: ~88 fixture files across 9 categories. The golden fixtures
(3 success + 5 negative) are the most important — they exercise the
full document chain from DecisionRequest through DecisionReport. The
category fixtures (capabilities, mechanisms, topology, evaluation,
quantization, authority, evidence-states) each prove specific
invariants.

Key questions:
- Do the 3 success golden fixtures actually exercise different code
  paths (self-hosted vs managed vs compound), or are they just the
  same fixture with different labels?
- Do the 5 negative golden fixtures cover the exact scenarios
  phase-plan.md line 29 mandates?
- Are the authority fixtures sufficient for ADR-010's 15 decisions and
  the restart/dedup requirement?
- Do the evidence-state fixtures cover contested (INV-4), lifecycle
  (INV-12), and promotion gates (INV-18)?
- Is ~88 fixtures too many for Phase 0? Could some be deferred to
  Phase 1 without weakening the exit gate?
- Are the three ecosystem export fixtures (recipes YAML, aiconfigurator
  JSON, InferenceX row) in the golden self-hosted set sufficient to
  prove round-trip?

Reference documents: phase-plan.md §Phase 0 exit gate (lines 43–45),
ADR-006 (evidence levels, conflict rule, remediation proof), ADR-010
(restart/dedup), ADR-011 (qualification graph, conformance tests),
ADR-013 (topology conformance tests), framework-spec.md invariants.

---

## Portion 6 of 6 — Tests, exit gate, and implementation ordering

Assess the test design and exit gate in `docs/phase-0-brief.md` Part 3
(§3.1–§3.16, lines 781–1063) and the exit gate summary (lines
1065–1108), plus the implementation ordering (§2.8, lines 716–780).

The task: 12 test files, ~90 test functions, 18 exit-gate items, and
14 implementation steps. This is where the brief either holds together
or falls apart — if the tests can't be written in the stated order, or
the exit gate doesn't match the phase plan's, the brief fails.

Key questions:
- Can step 1 (migration infrastructure) actually be built and tested
  before any schema exists? What does the synthetic v1→v2 migration
  test against?
- Does the implementation order (steps 1–14) have hidden dependencies
  that would force backtracking?
- Do the 18 exit-gate items cover everything phase-plan.md §Phase 0
  exit gate requires? Read the exit gate text line by line and
  cross-check.
- Are the shared-field tables (§3.5a) buildable from the pinned
  source files? Do the external formats have enough documentation to
  define "shared" unambiguously?
- Is the extension-point Protocol step (step 11) correctly sequenced
  after the data schemas? Or do some Protocols (like EvaluationAdapter)
  need data schemas from layers that come after them?
- Is the conformance suite boundary correct — Protocols in Phase 0,
  full suites as Phase 1a prerequisite?

Reference documents: phase-plan.md §Phase 0 exit gate, framework-spec.md
§1 (extension points), engineering-standards.md §1 (Protocol ports,
binding) and §7 (conformance suites), CONTRIBUTING.md (contributor
surface promise).
