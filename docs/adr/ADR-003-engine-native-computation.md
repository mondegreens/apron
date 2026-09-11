# ADR-003 — Apron owns prediction; GPU-executed engines own runtime evidence

**Status:** Accepted, 2026-09-06; revised by owner decisions, 2026-09-07; fixture-freshness rule added 2026-09-10
**Affects:** Target Architecture §1, §4, §10; Phase Plan Phase 1, Phase 3 Tier 0
**Evidence:** vLLM source at pinned commit a1541f57

## Context

An early proposal extracted vLLM's config parsing and KV-cache formulas into the planner's own code and listed hardware compatibility gates and pre-boot feasibility as "must build". A later proposal replaced that with a universal config-only or meta-device call into vLLM and claimed that Apron did not need a resource calculator. Source verification disproved that replacement: vLLM obtains the physical KV-cache specification from instantiated attention layers (`gpu_model_runner.py:7633-7664`) and profiles non-KV memory with a GPU forward pass (`gpu_worker.py:520-617`). A CUDA-targeted vLLM runtime without a visible GPU is also not a test of vLLM on a declared GPU: platform detection may resolve no CUDA platform, full configuration construction may fail, and capability checks may be skipped.

The llmcalc lineage adds material evidence. Its primary catalogue path uses per-model empirical VRAM totals and deployment-derived vLLM overrides, not the generic fallback formulas. The owner confirms that every catalogue model except Gemma was successfully deployed on RunPod after extensive live debugging and spend. That experience is a seed evidence set, while the fallback's single KV formula, fixed activation percentage, generic quantization reductions, and noisy GPU table are not a sufficient universal calculator.

## Decision

1. The CPU control plane produces a `prediction` from immutable model metadata, Apron calculations, version-pinned engine constraints, and a declared hardware target. It does not claim to test, dry-run, or validate vLLM.
2. A vLLM compatibility, boot, or kernel verdict requires execution in a GPU-enabled container. Its output is a `measurement` tied to the actual hardware and software fingerprint. A matching prior record may be presented as prior evidence, never as a new execution.
3. Building or inspecting the pinned CUDA image without a GPU is allowed, but produces neither a vLLM compatibility verdict nor boot evidence.
4. Tier 0 may extract versioned schemas and source constraints and recompute Apron predictions without a GPU. Tier 1 executes vLLM on GPU hardware. If an upstream vLLM dry-run appears, it belongs to the GPU execution path unless its hardware authority is separately proven.
5. Prediction records carry the declared hardware and the engine/source version used by their rules. Measurement records additionally carry detected hardware, container digest, driver, CUDA, PyTorch, vLLM, executed command, and raw outcome. The words `validated`, `verified`, and `measured` are reserved for executed evidence.
6. The planner builds the prediction terms the engine runtime does not provide before execution: weight bytes from the safetensors index, memory-budget terms, versioned compatibility constraints, candidate configuration, artifact rendering, and diagnosis hypotheses. Prediction uncertainty and unknowns remain explicit.
7. The engine adapter contract (§4) is defined so that the neutral core of `DeploymentPlan` holds model/component identity, exact capability signature, hardware, workload, parallelism (TP/PP/EP/DP/DCP), precision, memory budget partition, and the expected Pareto point; engine-specific flags, endpoint protocol and environment live in typed or opaque per-engine sections. Phase 4 separately tests this neutral core across a second LLM-serving engine and across a generated-media mechanism (ADR-005, ADR-007).
8. Apron owns a mechanism-aware CPU resource calculator. It implements reusable component mechanisms rather than one function per model, architecture label or modality: exact artifact/component weights; MHA/GQA/MQA, MLA, sliding-window/hybrid and recurrent-state caches; media encoder/cache and media-token expansion; pooling and encoder-decoder/streaming state; latent denoising/media decode; TP/PP partitioning; host/disk/loading terms; and explicitly fingerprinted activation/runtime-overhead predictions. Missing metadata or an unsupported mechanism returns `unknown`; it is never replaced by a plausible constant.
9. vLLM source is a versioned specification input and GPU conformance oracle, not a universally callable CPU calculator. Tier 0 extracts constraints and regression fixtures from the pinned source. Tier 1 compares Apron predictions with the KV specifications, profile results, boot result, and workload result produced by the instantiated engine.
10. The llmcalc catalogue is imported as `owner_attested_legacy_boot` evidence, preserving the tested model id, configuration, provider and every surviving environment field. Unknown model revisions, image digests, engine versions, dates, hardware or workloads remain explicitly unknown. Replays graduate records to the normal measured evidence levels; legacy values may seed candidates and priors but do not silently become current-version measurements.
11. **Fixture freshness (added 2026-09-10).** Every extracted constraint and regression fixture records the engine image digest and source revision it was captured from, together with its capture date. A fixture older than the currently pinned image fails the suite rather than passing silently. GPU-free tests are recorded from the engine; without expiry they drift from it and keep passing against a stale reference, which reintroduces through the test tier the failure INV-1 exists to prevent.

## Consequences

- A CPU prediction can prevent obviously impossible rentals, but cannot promote a plan to vLLM-validated status.
- A disagreement between prediction and GPU execution is a first-class delta to diagnose; it is not automatically a vLLM defect.
- Ordinary CI can build images and test prediction logic without a GPU. Release conformance for vLLM compatibility rules requires GPU execution.
- The prediction calculator must have explicit coverage tests for each supported memory mechanism and fail closed for unsupported ones; it does not claim automatic coverage merely because vLLM contains a class.
- The llmcalc catalogue is retained instead of discarded. Its deployment-tested configurations are more valuable than its fallback formulas, and prediction-versus-replay deltas become the initial calibration corpus.
- No universal meta-device probe is a Phase 0 dependency. A meta-device path may be evaluated as a conformance aid, but it cannot replace the owned predictor until it is proven across the supported architecture cohort.
