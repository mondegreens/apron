# ADR-008 — Upstream-first: the repository is a lab that produces contributions, not a platform

**Status:** Accepted 2026-09-06; amended 2026-09-08 by D2
**Affects:** Phase Plan (all phases), `framework-spec.md` §6 repository layout

## Context

The question "why not just contribute to existing projects" has a partial yes. Several high-value artifacts have an upstream home: a stable no-weight spec/conformance API and startup error-message improvements in vLLM, GPU hardware policies where `tools/recipes` accepts them, verified consumer and professional configurations in vllm-project/recipes, and planner accuracy fixes in aiconfigurator or llm-d-planner. Apron still must own its mechanism-aware GPU-free calculator because vLLM does not expose a universal config-only resource calculation path. Its value must come from joining that calculator to the llmcalc legacy, rented-GPU records, prediction deltas, diagnosis rules with separate mechanism and accepted-request proof, per-tag Tier 0 diffs, and interop—not from presenting another unverified generic estimator. Every upstream contribution is downstream of that loop.

## Decision

1. **This repository holds only what has no upstream home:** the lab harness and backends, the records and their publishing job, the schemas and interop mappings, the diagnosis rule table with proving records, Tier 0 extraction and diffs, the conformance suites, and a thin CLI and MCP over them.
2. **Anything with an upstream home is developed against that upstream from the start**, in a fork or branch of the upstream repository, with this repository holding only the evidence that motivates and validates it. The dry-run, hardware policies, and error messages target vLLM; verified configurations target recipes; accuracy deltas target aiconfigurator and llm-d-planner as issues with reproducible records.
3. **Duplication is temporary and labeled.** Where the tool needs an upstream feature before it lands (the dry-run being the main case), the interim implementation lives under a clearly named `interim/` path with the upstream issue or PR linked, and is deleted when the upstream lands.
4. **The measure of the project is external consumption, not its own user count:** records consumed by external tools, patches landed upstream, issues resolved with our records attached, and agent invocations through MCP. These are the metrics in `framework-spec.md` §3 and the 90-day checkpoint.
5. **The dataset is the durable artifact.** Publication begins with the first publication-eligible record: one that has passed provenance, license, sanitization and consent checks. The internal engineering record used to prove the Phase 1a loop is not automatically public. Published records live on the Hugging Face Hub under CDLA-Permissive-2.0 and outlive the tool and its maintainer regardless of adoption.

## Consequences

- The repository layout in `framework-spec.md` §6 loses nothing but gains `interim/` and a `upstream/` index of open contributions with their status.
- Phase 1 exit gates are unchanged; neither public dataset publication nor an upstream contribution is required to prove the internal conformance loop. A later checkpoint may measure upstream consumption only after a publication-eligible artifact exists.
- The 90-day decision rule counts patches landed and records consumed alongside the existing product-pull targets.
- If after the first hundred records no upstream consumes and nothing lands, the records remain public and reusable, which is more than a merged pull request leaves behind.
