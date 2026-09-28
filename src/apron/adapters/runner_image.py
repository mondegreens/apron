"""The pinned Apron runner images, one per vLLM version, by digest (F11).

A tag can be moved; a digest cannot.  Every pod, rendered compose file and
execution fingerprint uses ``RUNNER_IMAGE`` (repository@sha256:…).  The tag is
kept only as a human-readable label.  Resolved with the registry manifest
API and ``docker manifest inspect`` (L0-A2); see
``_dev_notes/cohort-run/image-digest.md``.
"""

from typing import NamedTuple

RUNNER_IMAGE_REPOSITORY = "ghcr.io/mondegreens/apron-runner"
RUNNER_IMAGE_TAG = "v0.29.0-rc8"
RUNNER_IMAGE_DIGEST = "sha256:faed210cbc55187206ce07533652780ba090762eaac82a5d0d27020131e7f336"
RUNNER_IMAGE = f"{RUNNER_IMAGE_REPOSITORY}@{RUNNER_IMAGE_DIGEST}"

# The CUDA toolkit the image is built on (docker/Dockerfile ``ARG CUDA_VERSION``).
# Pods are placed only on hosts whose driver supports it: the image sets
# NVIDIA_DISABLE_REQUIRE, so on an older driver it would start and then fail
# with CUDA error 804 (forward compatibility is not supported on GeForce).
RUNNER_IMAGE_CUDA = "13.0"
# Host CUDA versions whose driver runs a 13.0 toolkit.  RunPod's filter matches
# the host's version exactly: "13.0" alone excluded newer drivers — every B200
# host (13.2) among them (stock map, 2026-09-27).  The API accepts values past
# its documented list (13.0).
RUNNER_HOST_CUDA_VERSIONS = ("13.0", "13.1", "13.2", "13.3", "13.4")


# ---------------------------------------------------------------------------
# One image per vLLM version (owner, 2026-09-27: Apron follows vLLM releases).
# A version is runnable once its image is built and its digest pinned here;
# its facts live in adapters/backends/vllm_facts/<version>.json.
# ---------------------------------------------------------------------------


class RunnerImage(NamedTuple):
    version: str  # the vLLM release inside
    tag: str
    digest: str
    cuda: str = RUNNER_IMAGE_CUDA
    host_cuda_versions: tuple[str, ...] = RUNNER_HOST_CUDA_VERSIONS

    @property
    def image(self) -> str:
        return f"{RUNNER_IMAGE_REPOSITORY}@{self.digest}"


RUNNER_IMAGES: dict[str, RunnerImage] = {
    "v0.29.0": RunnerImage("v0.29.0", RUNNER_IMAGE_TAG, RUNNER_IMAGE_DIGEST),
    # Built by docker-publish run 36361024237 (tag runner-v0.30.0-rc1) from
    # docker/requirements-v0.30.0.txt; digest read from the registry manifest
    # (2026-09-28).  Same CUDA 13.0 base as v0.29.0.
    "v0.30.0": RunnerImage(
        "v0.30.0",
        "v0.30.0-rc1",
        "sha256:36dde61da5524d9b2d8d8e931d6765d841d833dc19e44d5c572c45215bce84ba",
    ),
}


def runner_image(version: str) -> RunnerImage:
    try:
        return RUNNER_IMAGES[version]
    except KeyError:
        raise KeyError(f"no pinned runner image for vLLM {version}") from None


def image_for_digest(digest: str) -> str:
    """The image reference for a digest a plan requested (identity lives in the digest)."""
    if not any(i.digest == digest for i in RUNNER_IMAGES.values()):
        raise KeyError(f"{digest} is not a pinned runner image")
    return f"{RUNNER_IMAGE_REPOSITORY}@{digest}"


def newest_runner_image() -> RunnerImage:
    def key(tag: str) -> tuple[int, ...]:
        return tuple(int(x) for x in tag.lstrip("v").split("."))

    return RUNNER_IMAGES[max(RUNNER_IMAGES, key=key)]
