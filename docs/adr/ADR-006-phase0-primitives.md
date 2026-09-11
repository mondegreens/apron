# ADR-006 — Definitions of the Phase 0 primitives that were named but undefined

**Status:** Accepted, 2026-09-06; execution-fingerprint, quantization-identity, and remediation-proof amendments accepted 2026-09-07; task/solution qualification amendments accepted through ADR-011 on 2026-09-08; queue source-selection amendment accepted through ADR-002 on 2026-09-09; record-serialization and determinism-boundary primitives added 2026-09-10
**Affects:** Target Architecture §1, §4, §5, §8, §9, §14, §15; Phase Plan Phase 0

## Decisions

**Output tiers (§4).**
- *Candidate*: prediction only, from Apron calculations and version-pinned constraints; no vLLM validation or boot evidence.
- *Conservative*: fits by calculation with a stated margin, and boot-verified on the same hardware class at any engine version.
- *Recommended endpoint*: task- and serving-verified for the exact endpoint/application fingerprint at the current applicable engine or API identity, expressed against the accepted SLO and objective. ADR-011 reserves *Qualified solution* for exact-solution task reproduction plus every applicable serving constraint and limits comparative efficiency claims to a disclosed comparable set.
Phase 1 emits Recommended only where such evidence exists.

**Evidence levels, total order and conflict rule (§8).**
maintainer-lab measured > independently reproduced > authoritative-external (recipes `verified`, InferenceX row, vendor documentation) > community-reported > calculated-only.
Within a level, the record at the newer engine version wins. Across levels, a disagreement is marked *contested* and both records are shown; nothing auto-resolves. Independently-reproduced is a level, not a multiplier.

**Freshness clocks (§8, Phase 3).**
Compatibility and boot evidence expire at the next vLLM minor release (biweekly cadence per vLLM `RELEASE.md`). Workload evidence expires per engine minor. Price evidence expires after 30 days. Security evidence expires at the next engine minor or on any dependency advisory. Metadata at a pinned revision never expires.

**Lineage fingerprint (§1, §8).**
Hash of the safetensors tensor-name, shape and dtype set at the pinned revision; `config.json` structural hash as tie-break; tokenizer hash recorded but not used for lineage. Aligned in spirit with vLLM's startup-plan fingerprint so records can be joined to the engine's own cache.

The lineage fingerprint is structural similarity, not content identity or derivation proof. Under the ADR-002 amendment, `ArtifactLocator` carries the source kind, URI and requested revision; `ArtifactSourceObservation` carries the resolved immutable revision, manifest, digests, license/gating observations and attributed publisher claims; `ArtifactIdentity` derives from the resolved file/component contents and structure. A registry repository id is one location, not identity. Two artifacts with the same tensor schema may contain different values; an AWQ/GPTQ conversion normally changes the artifact and may also change the tensor schema.

**Quantization identities and candidate graph (§1, §4, §5, §8).**
Quantization is represented by a complete candidate graph from the first schema version, not by a `quantization` string or a flag sweep over one repository. Its typed objects are:

- `ModelLineage`: the logical model family and its attributable publishers/derivatives;
- `ArtifactSpec`: one `ArtifactIdentity` plus every applicable `ArtifactLocator`/`ArtifactSourceObservation`, exact weight/index manifest and content digests, configuration and quantization-config digests, tokenizer/chat-template identity, attributed license/gating observations and claimed lineage;
- `ArtifactRelation`: typed and evidenced `published_by_owner`, `claimed_derived_from`, `reproducibly_derived_from`, `structurally_compatible_with`, `quality_compared_with`, and `tokenizer_compatible_with` edges;
- `OfflineTransformSpec`: reproducible producer, version, recipe, calibration data identity and output artifact; its result is a new `ArtifactSpec`;
- `RuntimeTransformSpec`: `none` or exact pinned-engine online conversion, target layer rules, exclusions and producer/version; it does not invent a checkpoint;
- `ExecutionSpec`: engine image digest, resolved checkpoint method, implementation overrides and selected linear/MoE/attention kernels;
- `QuantizationSpec`: distinct weight, activation and KV-cache formats; linear/MoE schemes; bits, group/block shape, scale/zero-point representation, excluded modules, calibration recipe/dataset and producer version;
- `QualityEvidence` and `CompatibilityEvidence`: separately scoped records attached to a candidate and execution fingerprint. ADR-011 further requires `QualityEvidence` to reference the accepted task attempts, application and evaluation protocol that produced it; it cannot exist as an unexplained score attachment.

Checkpoint-native artifacts, publisher quantizations, third-party siblings, reproducible offline conversions and supported runtime transforms are candidate classes in the same contract. Specific methods are typed adapters and may return `unsupported`; adding a method must not require a schema, record or public-API redesign. Repository naming and matching tensor shapes never establish lineage, tokenizer compatibility, license compatibility or quality equivalence.

Endpoint evidence retains the states `artifact_resolved`, `format_detected`, `engine_config_validated`, `boot_verified` and `serving_verified`; these need not be acquired in one universal chronological order for every solution class. Memory prediction uses the selected artifact's actual tensor headers, auxiliary scales and transform/runtime workspace, never a nominal bit-width division of base-model bytes. ADR-011 places applicable endpoint evidence inside the broader `InferenceSolution` qualification graph and requires task reproduction on the exact retained solution before `Qualified`; earlier states remain visible with their precise evidence.

**Execution fingerprint and measurement boundary (§4, §8).**
A runtime measurement belongs only to the execution that produced it. Its fingerprint includes the immutable artifact and quantization identity; normalized engine configuration and container digest; GPU SKU, total memory and compute capability; selected attention/linear/MoE backends; PyTorch, CUDA/runtime and driver versions; graph mode and capture sizes; allocator-relevant environment; TP/PP/DP/EP topology, world size and rank; and the profiling/workload shape. vLLM's startup-plan fingerprint is the minimum reuse boundary, not the complete public provenance record. A prior record with an exact matching fingerprint may be shown as matching prior measurement evidence subject to freshness; it is never presented as a new execution. Same-class, cross-GPU, cross-backend or incomplete-stack reuse produces a prediction with source lineage, uncertainty and safety margin, never a measurement.

Memory profiling stores non-overlapping raw observations: initial total/free and requested memory; loaded-model/weight memory; persistent consumption; transient peak headroom; non-PyTorch increase; CUDA-graph estimate, whether it was applied, and actual captured-graph memory; resulting available KV-cache memory; and any explicit safety buffer. vLLM's `peak_activation_memory` is not normalized as pure activation memory because at the inspected revision it includes transient headroom plus an optionally applied CUDA-graph estimate.

**Failure fingerprint (§9).**
Tuple of exception class, engine call-site module, and error family, after stripping numbers, byte sizes, paths, PIDs, addresses and timestamps. The seed taxonomy comes from the eighteen symptom sections of vLLM's troubleshooting document and the 2026 issue clusters; it is extended only by records.

**Decision and specification acceptance (§15, amended by ADR-011).**
An AI-proposed `DecisionRequest`, task suite, rubric, representativeness claim or serving workload cannot drive evaluation or rendering until the applicable human or authorized calling system explicitly accepts its immutable revision; principal/source and timestamp are recorded. The executor cannot accept its own proposal merely by executing it.

**Remediation proof (§9).**
A remediation application records two independent outcomes. `mechanism_outcome` is `verified`, `failed`, or `not_evaluated` and answers whether the correction removed the fingerprinted failure and restored engine health. `request_outcome` is `satisfied`, `violated`, or `not_evaluated` and answers whether the corrected exact solution still satisfies the accepted `DecisionRequest`, task outcome, serving workload, artifact/lineage policy and smoke semantics. The accepted request snapshot, task/application/evaluation fingerprints, corrected-solution and plan diff, violated constraints, and proving records are preserved.

The public result is derived, not guessed: `Fixed` requires `mechanism_outcome: verified` and `request_outcome: satisfied`; `Alternative with trade-offs` means the mechanism works but one or more accepted constraints are violated; `Unverified suggestion` covers any correction without complete execution proof. A rule may be promoted from `hypothesis` to `mechanism_verified` by a scoped corrected-boot record, but that rule status never proves that a particular application satisfies a different user's request.

**Log sanitization contract (§9, §14).**
Extraction from logs yields typed numerics and enums only, validated against hardware facts and schema ranges. No extracted string is ever interpolated into a rendered command. This is how §14's "cannot flow directly" and §9's "corrected plan from parsed error values" are both satisfied.

**Record serialization and digest (§8, added 2026-09-10).**
Record, manifest and release identity requires a defined mapping from a structure to bytes. INV-35 makes a release's identity its schema-versioned manifest and record digests rather than a host URL, and INV-12 requires reproducibility to the limit of a typed claim; neither is checkable while mapping order, floating-point representation, Unicode normalization or omitted optional fields can change the digest.

Canonical form is RFC 8785 (JSON Canonicalization Scheme). Identity is SHA-256 over that canonical form, carried with a multihash prefix so the algorithm is recorded rather than assumed. Test vectors are published with the schemas, and a digest is correct when an independent implementation in another language reproduces it from the same record. A digest computed from an uncanonicalized encoding is rejected rather than stored. This is INV-43.

The vectors, not the algorithm, carry the portability claim: a conforming publisher or external consumer that cannot recompute a digest falsifies INV-35. Because record identity is derived from this mapping, it is a start condition for schema implementation rather than a later refinement.

**Determinism boundary (§4, §7, §15, added 2026-09-10).**
The architecture describes the placement optimizer, the authorization engine, diagnosis corrections, generated endpoint plans and decision reports as deterministic. That property is unverifiable while any component can read an ambient clock, generate an identifier or draw randomness.

Wall-clock time, identifier generation and randomness reach a deterministic component only through injected `Clock`, `IdGenerator` and `Rng` ports. A component described as deterministic names its seed sources and produces byte-identical output for identical inputs. No execution, solution, artifact or task fingerprint contains a value obtained from an uninjected source. This is INV-42.

The same ports are the seam through which ADR-010's durability obligations — restart, replay, deduplication and teardown after crash, timeout or target loss — become executable without a live provider. They are inexpensive while the deterministic core is small and are not retrofittable once record writers and orchestration exist, so they are also a start condition for schema implementation.

**Engine support states (Phase 0).**
native vLLM / Transformers backend / plugin / recognized-but-failed / unsupported, each with a GPU-architecture and CPU-architecture dimension.

**Data license.** CDLA-Permissive-2.0 for evidence records, CC-BY-4.0 for documentation (see ADR-002).

**Signing.** Sigstore (keyless, OIDC-bound) for community reports; environment attestation recorded where the provider exposes it.

**Day-one queue policy (Phase 3; superseded in source selection by the 2026-09-09 ADR-002 amendment).** Accepted requests and every locally observed failed plan enter directly. The remaining dated cohort is assembled from attributed signals across multiple artifact registries, provider catalogues, engine issues/releases, recipes, external planners and evidence systems, then normalized and deduplicated by resolved lineage/artifact fingerprints. Hugging Face 30-day downloads may contribute one explicitly labeled signal; because its counting varies by library/query file and does not publicly deduplicate users, it never defines eligibility or priority alone. Claim scope, missing/stale/contested evidence, uncertainty/error, unresolved-failure value, freshness and adequate existing coverage remain binding scheduler inputs.

## Consequences

Each definition above is implementable without further decisions. Where the first ten evidence records show a definition to be wrong, a superseding ADR revises it; none is silently changed. The 2026-09-07 amendments reject architecture-family-wide measurement portability, a native-checkpoint-only data model, and boot-only remediation claims before any schema is frozen.
