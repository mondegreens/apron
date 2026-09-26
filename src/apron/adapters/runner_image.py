"""The pinned Apron runner image, referenced by digest everywhere (F11).

A tag can be moved; a digest cannot.  Every pod, rendered compose file and
execution fingerprint uses ``RUNNER_IMAGE`` (repository@sha256:…).  The tag is
kept only as a human-readable label.  Resolved with the registry manifest
API and ``docker manifest inspect`` (L0-A2); see
``_dev_notes/cohort-run/image-digest.md``.
"""

RUNNER_IMAGE_REPOSITORY = "ghcr.io/mondegreens/apron-runner"
RUNNER_IMAGE_TAG = "v0.29.0-rc7"
RUNNER_IMAGE_DIGEST = "sha256:a53c7b6d3a883669a24c8a62ee4770235257a85e0be1560dc39ebf892d4f311c"
RUNNER_IMAGE = f"{RUNNER_IMAGE_REPOSITORY}@{RUNNER_IMAGE_DIGEST}"
