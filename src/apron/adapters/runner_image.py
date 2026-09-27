"""The pinned Apron runner image, referenced by digest everywhere (F11).

A tag can be moved; a digest cannot.  Every pod, rendered compose file and
execution fingerprint uses ``RUNNER_IMAGE`` (repository@sha256:…).  The tag is
kept only as a human-readable label.  Resolved with the registry manifest
API and ``docker manifest inspect`` (L0-A2); see
``_dev_notes/cohort-run/image-digest.md``.
"""

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
