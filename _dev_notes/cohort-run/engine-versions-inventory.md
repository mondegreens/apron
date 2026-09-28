# Engine versions in Apron: inventory and plan (2026-09-27)

Owner: Apron must follow vLLM releases, not stay on one; nothing from the
concept may be missed.  A read-only agent listed every place the engine
version or image is held (code + concept, file:line); summary below.

## Where the engine lives today

- One image (`runner_image.py`: tag v0.29.0-rc8, one digest) used by every
  pod, the compose renderer, `RequestedExecutionSpec.image_digest` (solution
  identity) and the execution fingerprint (hash includes the image).
- One facts set loaded at import (`vllm_quantization.py`): quantization
  capabilities, parameters, hybrid list, tokenizer modes, names, batch
  defaults; `hf_lineage` uses it.  (Now per version: `vllm_facts/<tag>.json`.)
- Rules per minor directory (`rules/vllm-v0.29/`), `DiagnosisRule.engine_version`
  in the rule's identity; `load_rules(..., "v0.29.0")` hard-wired; a missing
  directory silently yields no rules.
- `VllmEngineAdapter(engine_version="v0.29.0")` default;
  `resolve_support`/`extract_schema(image_tag)` ignore the tag and read a
  2026-09-13 fixture.
- Records: no readable engine version on `VerificationReport`,
  `CompatibilityEvidence`, `PlanningClaim` or `RemediationRecord`; only the
  image digest inside hashes (ADR-003 §5 wants engine version, driver, CUDA,
  PyTorch on measurements and the rules' engine version on predictions).
- The calculator's layered/KV paging, async x2, activation shape are traced
  from v0.29.0.
- The publish workflow moves `:latest` on every build.

## Concept requirements (quoted in the agent's report)

ADR-003 §1, §4, §5, §9, §11; ADR-006 (version-pinned tiers, newer engine
wins within a level, evidence expires at the next vLLM minor); ADR-007 §8;
ADR-011 (invalidation on engine change); engineering-standards (number of
supported minors stated); product-definition (per-tag diffs, version shown on
records, records >2 minors behind flagged); phase plan Phase 3 Tier 0 (tag
diff, recompute, invalidate, rule lifecycle per version).

## Needed now (to run v0.30.0 models in this phase)

1. Engine registry: version -> image repository, tag, digest, CUDA; stop
   moving `:latest`.
2. Planner picks the newest pinned version whose facts register the model's
   architecture (unknown when none); `modern-models.json` "engine" is a check.
3. Thread the version through `image_digest` -> target -> pod image ->
   compose -> execution fingerprint.
4. Facts read per version everywhere (load check, capability, hybrids,
   tokenizer modes, KV scales, batch defaults, lineage); no silent default.
5. Readable engine version on PlanningClaim and VerificationReport (DISPLAY;
   identity already via the digest), migration.
6. Rules for v0.30 (all hypothesis until proven on v0.30); loader errors on a
   missing directory; diagnosis selects rules by the plan's version.
7. `extract_schema` / `resolve_support` from the version's facts.
8. Guard tests per version (done for facts and requirements).
9. Calculator mechanisms re-checked on v0.30 source, or v0.30-only
   architectures stay unknown.
10. Findings/article show engine version per row; the support window stated.

## Later (Phase 3)

Release detection and Tier 0 diff; freshness clocks and invalidation;
per-version rule promotion; "newer wins" display; ADR amendment for the
Apron runner image instead of the official vllm/vllm-openai image.
