# Runner image digest (L0-A2)

## v0.29.0-rc7 — currently pinned in `src/apron/adapters/runner_image.py`

Resolved 2026-09-26 from the local machine, no GPU:

1. Registry manifest API (anonymous pull token from `ghcr.io/token`):
   `HEAD https://ghcr.io/v2/mondegreens/apron-runner/manifests/v0.29.0-rc7`
   returned `200`, `content-type: application/vnd.oci.image.index.v1+json`,
   `docker-content-digest: sha256:a53c7b6d3a883669a24c8a62ee4770235257a85e0be1560dc39ebf892d4f311c`.
2. `docker manifest inspect ghcr.io/mondegreens/apron-runner:v0.29.0-rc7` shows
   an OCI index with one `linux/amd64` manifest
   (`sha256:ef8ba97f8a593866a5fdc16376bf2da697fae6b6f516349a3b7baaf79dc1c381`)
   and one attestation manifest (`unknown/unknown`).

The pin is the index digest `sha256:a53c7b…`.

## Status: superseded before any GPU boot

This image predates two changes in this branch:
- the F7 token isolation (`docker/start.sh`, `docker/apron-download`);
- `10.0` in `TORCH_CUDA_ARCH_LIST` (class 6 needs B200).

On this image a gated download gets no token, and the token that `start.sh`
exports reaches SSH sessions.  The F7 `/proc/<pid>/environ` check would fail.
The image must be rebuilt, pushed, re-pinned here and pulled by digest before
L0-A3 (the first GPU boot).  Tracked as task #29.  A pull by digest has not
been verified yet; it is part of that task.
