"""Step 11 tests: imported boot observations — seed candidates, no verification, no formulas."""

from apron.domain.canonical import canonicalize
from apron.domain.fingerprints import assert_fully_classified
from apron.domain.protocols import EvidenceSource
from apron.domain.schemas.imports import (
    CatalogueImportSource,
    ImportedBootObservation,
)

_REGISTRY = {
    "meta-llama/Meta-Llama-3-8B": {
        "n_params": 8_000_000_000,
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "vram_fp16_gb": 14.5,
        "architecture": "LlamaForCausalLM",
        "vllm_params": {
            "gpu_memory_utilization": 0.85,
            "max_num_seqs": 8,
        },
    },
    "deepseek-ai/DeepSeek-V3": {
        "n_params": 685_000_000_000,
        "hidden_size": 7168,
        "num_hidden_layers": 61,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "vram_fp16_gb": 1100.0,
        "architecture": "DeepSeekForCausalLM",
        "vllm_params": {
            "tensor_parallel_size": 8,
            "gpu_memory_utilization": 0.95,
            "enforce_eager": True,
            "max_num_seqs": 8,
            "trust_remote_code": True,
        },
    },
    "Qwen/Qwen3-8B": {
        "n_params": 8_000_000_000,
        "hidden_size": 4096,
        "num_hidden_layers": 36,
        "vram_fp16_gb": 16.5,
        "architecture": "Qwen2ForCausalLM",
        "vllm_params": {
            "gpu_memory_utilization": 0.85,
        },
    },
    "google/gemma-2-9b": {
        "n_params": 9_000_000_000,
        "hidden_size": 3584,
        "num_hidden_layers": 42,
        "vram_fp16_gb": 18.0,
        "architecture": "Gemma2ForCausalLM",
        "vllm_params": {
            "enforce_eager": True,
        },
    },
}

_SOURCE = CatalogueImportSource(
    _REGISTRY,
    source_name="test_catalogue",
    provenance="test catalogue, owner attestation 2026-09-07",
)


def _round_trip(model_cls, instance):
    canonical = canonicalize(instance.model_dump(mode="json"))
    restored = model_cls.model_validate_json(canonical)
    assert canonicalize(restored.model_dump(mode="json")) == canonical


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


def test_imported_boot_observation_classified():
    assert_fully_classified(ImportedBootObservation)


def test_imported_boot_observation_round_trip():
    obs = ImportedBootObservation(
        model_id="Qwen/Qwen3-8B",
        empirical_vram_gb=16.5,
        observed_configuration={"gpu_memory_utilization": 0.85},
        provenance="test catalogue, owner attestation 2026-09-07",
    )
    _round_trip(ImportedBootObservation, obs)


# ---------------------------------------------------------------------------
# adapter satisfies EvidenceSource Protocol
# ---------------------------------------------------------------------------


def test_catalogue_import_satisfies_protocol():
    assert isinstance(_SOURCE, EvidenceSource)


# ---------------------------------------------------------------------------
# entries seed candidates — Gemma excluded
# ---------------------------------------------------------------------------


def test_entries_seed_candidates_gemma_excluded():
    entries = _SOURCE.collect()
    assert len(entries) == 3
    model_ids = {e["model_id"] for e in entries}
    assert "google/gemma-2-9b" not in model_ids
    assert "Qwen/Qwen3-8B" in model_ids
    assert "deepseek-ai/DeepSeek-V3" in model_ids


def test_empty_registry_produces_nothing():
    source = CatalogueImportSource({}, source_name="empty", provenance="none")
    assert source.collect() == []


# ---------------------------------------------------------------------------
# entries do NOT satisfy verification gates
# ---------------------------------------------------------------------------


def test_import_status_is_owner_attested():
    for entry in _SOURCE.collect():
        assert entry["import_status"] == "owner_attested_boot"


def test_outcome_is_booted_healthy():
    for entry in _SOURCE.collect():
        assert entry["outcome"] == "booted_healthy"


def test_missing_verification_fields_are_none():
    for entry in _SOURCE.collect():
        assert entry["artifact_revision"] is None
        assert entry["engine_image_digest"] is None
        assert entry["execution_fingerprint"] is None
        assert entry["hardware_fingerprint"] is None
        assert entry["workload"] is None
        assert entry["date"] is None


# ---------------------------------------------------------------------------
# preserves empirical VRAM and working configuration
# ---------------------------------------------------------------------------


def test_preserves_empirical_vram():
    entries = {e["model_id"]: e for e in _SOURCE.collect()}
    assert entries["Qwen/Qwen3-8B"]["empirical_vram_gb"] == 16.5
    assert entries["deepseek-ai/DeepSeek-V3"]["empirical_vram_gb"] == 1100.0


def test_preserves_working_configuration():
    entries = {e["model_id"]: e for e in _SOURCE.collect()}

    ds = entries["deepseek-ai/DeepSeek-V3"]["observed_configuration"]
    assert ds["tensor_parallel_size"] == 8
    assert ds["trust_remote_code"] is True
    assert ds["enforce_eager"] is True

    llama = entries["meta-llama/Meta-Llama-3-8B"]["observed_configuration"]
    assert llama["gpu_memory_utilization"] == 0.85


# ---------------------------------------------------------------------------
# does NOT carry forward stale metadata or formulas
# ---------------------------------------------------------------------------


def test_does_not_carry_stale_metadata():
    for entry in _SOURCE.collect():
        assert "n_params" not in entry
        assert "hidden_size" not in entry
        assert "num_hidden_layers" not in entry
        assert "architecture" not in entry


def test_provenance_carried():
    for entry in _SOURCE.collect():
        assert entry["provenance"] != ""
