"""F1: published model-fingerprint vectors (INV-43).

``tests/vectors/fingerprint/vectors.json`` holds, for every domain model, a
minimal valid input, its IDENTITY projection in RFC 8785 canonical form, and
the resulting multihash SHA-256 fingerprint.  An independent implementation
can reproduce every digest from ``identity_projection_utf8_hex`` alone.

Set ``APRON_REGENERATE_VECTORS=1`` to rewrite the file after a schema change.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from _domain_models import all_domain_models, minimal_input

from apron.domain.canonical import canonicalize, digest_hex
from apron.domain.fingerprints import fingerprint_hex, identity_projection

VECTOR_FILE = Path(__file__).parents[1] / "vectors" / "fingerprint" / "vectors.json"

_PROVENANCE = (
    "Apron model-fingerprint vectors. For each domain model: a minimal valid "
    "input, the IDENTITY projection at every depth (F1) serialized to RFC 8785 "
    "canonical UTF-8, and SHA-256 with multihash prefix 0x1220 over those bytes."
)


def generate_vectors() -> dict[str, Any]:
    vectors = []
    for model in all_domain_models():
        instance = model.model_validate(minimal_input(model))
        projection = canonicalize(identity_projection(instance))
        vectors.append(
            {
                "schema": f"{model.__module__}.{model.__qualname__}",
                "input": instance.model_dump(mode="json"),
                "identity_projection_utf8_hex": projection.hex(),
                "fingerprint_hex": fingerprint_hex(instance),
            }
        )
    return {"_provenance": _PROVENANCE, "vectors": vectors}


def test_vectors_are_stable_across_two_runs() -> None:
    assert canonicalize(generate_vectors()) == canonicalize(generate_vectors())


def test_vector_file_matches_current_schemas() -> None:
    generated = generate_vectors()
    if os.environ.get("APRON_REGENERATE_VECTORS") == "1":
        VECTOR_FILE.parent.mkdir(parents=True, exist_ok=True)
        VECTOR_FILE.write_text(json.dumps(generated, indent=2, sort_keys=True) + "\n")
    stored = json.loads(VECTOR_FILE.read_text())
    assert stored == generated


def test_each_vector_digest_reproduces_from_its_projection_bytes() -> None:
    for vector in json.loads(VECTOR_FILE.read_text())["vectors"]:
        projection = bytes.fromhex(vector["identity_projection_utf8_hex"])
        assert digest_hex(projection) == vector["fingerprint_hex"], vector["schema"]
