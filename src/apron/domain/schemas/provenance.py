"""External format provenance — records what was pinned and where it came from."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class PinnedFileEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_path: str
    local_path: str
    sha256: str
    size_bytes: int
    role: str


class SourceLocation(BaseModel):
    model_config = ConfigDict(frozen=True)

    repository: str | None
    revision: str | None
    retrieval_method: Literal[
        "git_clone",
        "git_fetch_shallow",
        "huggingface_hub_api",
        "http_download",
        "owner_provided",
    ]


class ExternalFormatProvenance(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    source_name: str
    locations: list[SourceLocation]
    retrieval_date: str
    license: str
    has_formal_schema: bool
    formal_schema_file: str | None = None
    schema_description: str
    pinned_files: list[PinnedFileEntry]
    notes: str
