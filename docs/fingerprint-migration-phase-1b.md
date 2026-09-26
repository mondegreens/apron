# Fingerprint migration — Phase 1b

Phase 1b Part 4 changes how fingerprints are computed (F1) and makes every
schema strict (F2). Both changes were approved by the owner in
[issue #41](https://github.com/mondegreens/apron/issues/41). This note lists
every schema whose digest changes, and why.

INV-43 is unchanged. A record is projected, serialized to RFC 8785 canonical
form, and hashed with SHA-256 under the multihash prefix `0x1220`. Only the
projection changed.

## F1 — IDENTITY fields at every depth

`fingerprint_hex` used to keep the IDENTITY fields of the top-level model only.
Nested models were dumped whole, so their DISPLAY fields (for example
`schema_version`) leaked into the hash. The new `identity_projection` keeps
IDENTITY fields at every depth: nested models, tuples, lists and dict values.
A model with no markers at all is plain data, and all of its fields are kept.

The digest changes for these schemas because a nested model has DISPLAY fields:

| Schema | Nested DISPLAY now dropped |
|---|---|
| `ArtifactSpec` | `identity` → `ArtifactIdentity.schema_version` |
| `ModelSpec` | `components` → `ComponentMechanism.schema_version` |
| `TaskSuiteSpec` | `required_capabilities` → `CapabilitySignature.schema_version` |
| `DecisionReport` | `candidates` → `CandidateEntry` DISPLAY fields |

These schemas had no field markers. They used to hash an empty projection, so
every instance shared one digest. They now hash all their fields:

| Schema | Before | After |
|---|---|---|
| `CalculatorInput` | digest of `{}` | all fields |
| `RenderContext` | digest of `{}` | all fields |
| `ExternalFormatProvenance`, `PinnedFileEntry`, `SourceLocation` | digest of `{}` | all fields |

Records that carry one of these fingerprints *by value* also change digest. For
example, `TaskAttemptRecord.task_suite_fingerprint`, the `EvaluationProtocol`
digests and `DirectEndpoint.model_spec_fingerprint`.

A flat model hashes exactly as before
(`tests/unit/test_fingerprint_nested.py::test_flat_model_digest_matches_previous_definition`).

## F2 and Phase 1b fields — new IDENTITY fields

A new IDENTITY field adds a key to the projection even when it is empty, so
these digests change too:

| Schema | New IDENTITY fields |
|---|---|
| `ServingWorkloadSpec` | `input_sequence_length`, `output_sequence_length`, `p99_ttft_ms`, `p99_tpot_ms` |
| `VerificationReport` | `solution_fingerprint`, `deployment_plan_digest`, `boot_outcome` |
| `RemediationRecord` | `classifier_model_id`, `classifier_input_digest`, `diagnosed_failure_class` |
| `DiagnosisRule` (v4) | `engine`, `engine_version`, `rule_version`, `supersedes`, `correction_strategy`, `extraction_schema`, `source_sites`, `float16_blocklist`, `kernel_architectures` |
| `WorkloadSpec` | none, but its `serving_workload_fingerprint` value changes with `ServingWorkloadSpec` |

New schemas: `RequestedExecutionSpec` and `SolutionIdentity` (the solution
fingerprint), and `SourceSite` and `ExtractionField` (inside `DiagnosisRule`).

`PlanningClaim.solution_fingerprint` and the serving and cost fields on
`VerificationReport` are DISPLAY. They do not change digests.

## Stored records

Stored records load through the migration path and are validated strictly
(`apron.domain.schemas.migrations.load_record`). A record with a field the
schema does not know fails loudly with the field and its path; nothing is
silently stripped. Rule files at schema version 3 migrate to version 4
automatically.

## Test vectors

`tests/vectors/fingerprint/vectors.json` publishes, for every domain model, a
minimal valid input, its identity projection in canonical UTF-8, and the
fingerprint. An independent implementation can reproduce every digest from the
projection bytes. Regenerate the vectors after a schema change with
`APRON_REGENERATE_VECTORS=1 uv run pytest tests/unit/test_fingerprint_vectors.py`.
