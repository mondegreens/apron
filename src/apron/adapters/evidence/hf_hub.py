"""HF Hub artifact resolver — implements ArtifactSourceResolver Protocol.

Resolves a HuggingFace Hub model locator to an ArtifactSourceObservation
with content digests, revision pin, and file metadata.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from apron.domain.artifacts import ArtifactSourceObservation, FileDigest

if TYPE_CHECKING:
    from apron.domain.schemas.primitives import ArtifactLocator


@dataclass(frozen=True)
class ResolutionError:
    step: str
    reason: str
    model_id: str | None = None
    source: str = "huggingface"


class HFHubResolver:
    """Resolves HuggingFace Hub model references to ArtifactSourceObservations.

    Uses ``huggingface_hub`` for API access. All network calls are isolated
    behind methods that can be overridden for testing.
    """

    def supports(self, source_kind: str) -> bool:
        return source_kind == "huggingface"

    def resolve(self, locator: ArtifactLocator) -> ArtifactSourceObservation:
        model_id = locator.uri
        revision = locator.requested_revision

        info = self._get_model_info(model_id, revision)
        resolved_rev = info["sha"]

        files_to_fetch = ["config.json", "model.safetensors.index.json"]
        file_digests: list[FileDigest] = []
        for filename in files_to_fetch:
            content = self._download_file(model_id, filename, resolved_rev)
            if content is not None:
                sha256 = hashlib.sha256(content).hexdigest()
                file_digests.append(
                    FileDigest(path=filename, sha256=sha256, size_bytes=len(content))
                )

        readme = self._download_file(model_id, "README.md", resolved_rev)
        if readme is not None:
            sha256 = hashlib.sha256(readme).hexdigest()
            file_digests.append(
                FileDigest(path="README.md", sha256=sha256, size_bytes=len(readme))
            )

        config_content = self._download_file(model_id, "config.json", resolved_rev)
        manifest_digest = None
        if config_content is not None:
            manifest_digest = hashlib.sha256(config_content).hexdigest()

        gating = None
        if info.get("gated"):
            gating = str(info["gated"])

        license_observed = None
        tags = info.get("tags", [])
        for tag in tags:
            if tag.startswith("license:"):
                license_observed = tag.split(":", 1)[1]
                break

        safetensors_info = info.get("safetensors", {})
        publisher_metadata: dict[str, str] = {}
        if safetensors_info:
            params = safetensors_info.get("parameters", {})
            for dtype, count in params.items():
                publisher_metadata[f"parameters_{dtype}"] = str(count)

        return ArtifactSourceObservation(
            source_kind="huggingface",
            resolved_revision=resolved_rev,
            manifest_digest=manifest_digest,
            file_digests=tuple(file_digests),
            license_observed=license_observed,
            gating_observed=gating,
            publisher_metadata=publisher_metadata or None,
        )

    def _get_model_info(self, model_id: str, revision: str | None) -> dict[str, Any]:
        """Fetch model info from HF Hub API. Override for testing."""
        from huggingface_hub import model_info  # type: ignore[import-not-found]

        info = model_info(model_id, revision=revision)
        return {
            "sha": info.sha,
            "gated": info.gated,
            "tags": info.tags or [],
            "safetensors": info.safetensors or {},
        }

    def _download_file(self, model_id: str, filename: str, revision: str) -> bytes | None:
        """Download a single file from HF Hub. Override for testing."""
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]

        try:
            path = hf_hub_download(repo_id=model_id, filename=filename, revision=revision)
            return Path(path).read_bytes()
        except Exception:
            return None


class FixtureHFHubResolver(HFHubResolver):
    """HF Hub resolver backed by local fixture files. No network calls."""

    def __init__(self, fixture_dir: Path) -> None:
        self._fixture_dir = fixture_dir
        self._model_info: dict[str, Any] | None = None

    def _get_model_info(self, model_id: str, revision: str | None) -> dict[str, Any]:
        if self._model_info is None:
            info_path = self._fixture_dir / "model_info.json"
            self._model_info = json.loads(info_path.read_bytes())
        assert self._model_info is not None
        return self._model_info

    def _download_file(self, model_id: str, filename: str, revision: str) -> bytes | None:
        path = self._fixture_dir / filename
        if path.exists():
            return path.read_bytes()
        return None
