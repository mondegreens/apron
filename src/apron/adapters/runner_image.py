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
