# ADR-001 — CPU control plane is location-neutral; maintainer GPU execution uses rented providers

**Status:** Accepted, 2026-09-06; revised by owner decision, 2026-09-07
**Affects:** Target Architecture §7 (verification lab), Phase Plan Phase 1 and Phase 3

## Context

Phase Plan v1.2 assumed maintainer-operated DGX Spark hardware. v1.3 moved to RunPod burst verification but then overcorrected by saying nothing in planning or verification could run on the maintainer's laptop. The maintainer has no GPU and will not purchase one, but that fact constrains maintainer-operated GPU execution—not metadata resolution, calculation, rendering, orchestration or the product's ability to verify on a user's voluntarily supplied GPU.

## Decision

1. The GPU-free control plane—resolve, calculate, plan, render, query evidence, diagnose hypotheses, orchestrate and validate schemas—may run wherever the CLI/core runs: a user's machine, the maintainer's development machine, CI or a hosted CPU service. Its outputs remain predictions or artifacts, never runtime measurements.
2. Maintainer-operated GPU execution runs on rented providers. RunPod is the initial burst backend; Modal or another conforming provider prevents one provider from becoming load-bearing. The maintainer owns no GPU and the architecture never requires one.
3. User-operated verification may run, by explicit choice, on a compatible GPU the user owns, rents or reaches remotely. User hardware supplements the maintainer lab; it does not become maintainer-lab evidence merely because the same binary produced the record.
4. `ExecutionTarget` is a permanent backend-neutral contract with at least `local-container`, `remote-container`, and `rented-provider` kinds. Every backend supplies prepare/provision, execute, observe, collect and teardown semantics plus the exact execution fingerprint. Transport and ownership do not change the plan, verification or record schemas.
5. CLI verbs reveal side effects: `apron plan` is GPU-free; `apron verify PLAN --target TARGET` performs a bounded execution, saves a local `VerificationReport`, and tears down; `apron deploy ... --target TARGET` composes planning and verification but intentionally leaves a managed endpoint whose lifecycle and continuing cost are explicit; `apron run -- ...` wraps an explicitly supplied execution to observe and diagnose it; `apron report RUN_ID` is read-only display/export; `apron submit RUN_ID` is a separate explicit, consented publication of a sanitized report. Execution never implies submission.
6. A paid target displays artifact size, maximum estimated cost and hard timeout and requires explicit confirmation or a standing `AuthorizationEnvelope` before provisioning. Every backend enforces teardown independently of job success.
7. A no-GPU or unavailable target returns a typed non-measurement outcome such as `hardware_unavailable`. It may still return the CPU prediction separately but cannot create compatibility, boot or workload measurements.
8. Evidence authority follows provenance, execution fingerprint, signature/attestation and reproduction—not GPU ownership. A maintainer run on rented hardware can be maintainer-lab evidence; a user run remains community evidence until independently reproduced under the evidence policy.
9. Owned maintainer hardware is reconsidered only when recorded revalidation spend makes rental more expensive than ownership; ownership never changes the backend or evidence contracts.

## Consequences

- The same public verification workflow is conformance-tested positively on a rented GPU and negatively without a GPU; no external contributor is required to satisfy a software exit gate.
- The Spark/unified-memory class remains a coverage track acquired through a rented/remote target or consented user reports; it is not assumed to be maintainer-owned.
- Every evidence record carries target kind, operator/source, provider when applicable, detected hardware and complete execution fingerprint.
- A CPU-only host can build images, plan and orchestrate remote verification, but cannot itself produce vLLM compatibility or boot evidence.
- Reports remain local until `submit`; sanitization and a preview of submitted fields precede every publication.
