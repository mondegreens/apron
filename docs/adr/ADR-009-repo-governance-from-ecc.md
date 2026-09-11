# ADR-009 — Repository governance is policy, not product architecture

**Status:** Accepted 2026-09-07; replaced in full 2026-09-08 by D2
**Affects:** repository policy, CI, contribution workflow, release and provider workflows
**Evidence:** [GitHub Actions secure-use reference](https://docs.github.com/en/actions/reference/security/secure-use), [deployment environments and required reviewers](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments)

## Context

The earlier decision promoted the file layout of one AI-oriented repository into mandatory Apron architecture. It required named agent-instruction and live-context files before schemas could be drafted. Those filenames and workflows are implementation preferences, not product requirements, and they create a second governance surface beside the actual invariants and tests.

The repository still needs strong controls. Some workflows execute untrusted artifacts, use provider credentials, spend money, publish records, or mutate external systems. Those controls must be enforceable independently of which human or coding agent performs the work.

## Decision

1. **The binding rules are tool-neutral.** Product invariants, architecture boundaries, tests, schemas, branch rules and authorization policies are authoritative. Contributor guidance may be rendered into whatever instruction files active development tools consume, but no vendor- or agent-specific file is a product dependency, schema authority or phase gate.
2. **CI uses least privilege.** Workflow permissions are explicit and minimal; third-party actions are pinned to reviewed full commit SHAs; jobs have timeouts and concurrency controls; secrets are available only to the jobs and protected environments that need them.
3. **Risk-bearing jobs have separate authority.** Provider execution, release publication, evidence publication and external repository mutation use distinct credential scopes and protected policies. A job does not gain broader authority merely because it passed ordinary CI.
4. **Repository changes may be automated.** Humans and authorized agents may commit, amend, rebase, tag, push and merge when the active task and standing repository policy allow it. Destructive history rewrites remain subject to the repository's safety policy. This ADR does not itself authorize any Git operation.
5. **Commit history is project-native.** Commit subjects and bodies describe the project change. They contain no Codex, Claude, AI-generated, prompt, session or similar agent branding, and no AI `Co-authored-by` or generated-by trailers. A commit-message check enforces the repository grammar and forbidden metadata.
6. **Templates are justified by an actual intake path.** PR and issue templates may be added when their structured fields are consumed by review, ingestion or triage. Their filenames and number are not architectural requirements.
7. **Governance never blocks truth-model work merely because optional scaffolding is absent.** The specific controls needed for a risky operation must exist before that operation: schema and invariant checks before accepting records, provider spend/teardown controls before a provider run, and license/sanitization/authorization checks before publication.
8. **The organization boundary is explicit.** Shared defaults in `mondegreens/.github` are policy views, not product authority. `apron-research` remains private with its own history; public `apron` and `apron-action` accept only deliberately reviewed, licensed and sanitized material. The Action cannot read the research repository or inherit its credentials merely because both repositories belong to the same organization (ADR-012).

## Retained smart configuration

The useful repository artifacts from the original exploration remain planned. They are views and adapters over canonical rules, not independent authorities:

| artifact | retained role | drift control |
|---|---|---|
| `CLAUDE.md`, `AGENTS.md`, or another harness entrypoint | Short tool-specific instructions and commands for the active coding agent | Points to canonical architecture, invariants and tests; contains no duplicate product decisions; replaceable without ADR |
| `SOUL.md` | Compact human-readable statement of project principles | References invariant ids and governing documents rather than restating mutable rules |
| `WORKING-CONTEXT.md` | Current phase, active work, constraints and last handoff | Operational snapshot only; cannot override an ADR, schema, test or owner decision |
| Structured issue templates | Deployment-failure intake, feature outcomes and concise feedback | Fields map to a real triage or ingestion schema; unused fields/templates are not activated |
| Pull-request template | Applicable invariant, evidence, migration and test checklist | Checked against current invariant ids so stale checklist entries fail review |
| CI and dependency automation | Tests, lint, schema/migration checks, secret detection, pinned actions and dependency updates | Least-privilege permissions, protected credentials, explicit timeouts and reviewed SHA updates |
| `CONTRIBUTING.md`, `SECURITY.md`, succession guidance | Contribution path, private vulnerability reporting and continuation instructions | Links to current commands/data locations; stores no credentials and grants no execution authority |

These artifacts may be created when their canonical input exists and kept from drifting through generation or validation where practical. Their absence never changes product semantics, while their content must never contradict the binding sources.

## Consequences

- Development can use different coding agents without changing product architecture.
- CI and external side effects remain protected by enforceable permissions rather than prose alone.
- Git automation is allowed under standing policy while commit history remains free of agent advertising and session residue.
- Useful coordination files remain explicitly planned but can be added, replaced or removed without an architecture ADR because their authority comes from the canonical sources they reference.
- Phase 0 is not gated on a prescribed collection of governance documents.
