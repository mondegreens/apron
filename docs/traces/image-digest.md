# Runner image digest (L0-A2)

## v0.29.0-rc8 — pinned in `src/apron/adapters/runner_image.py`

Built by `.github/workflows/docker-publish.yml` from tag `runner-v0.29.0-rc8`
at commit `4ab0bb6` (run 36269230503, 2026-09-26, success).  The workflow
also moved `:latest`, which resolves to the same index.

Resolved 2026-09-26 from the local machine, no GPU, no layer download:

1. `HEAD https://ghcr.io/v2/mondegreens/apron-runner/manifests/v0.29.0-rc8`
   (anonymous pull token) → `docker-content-digest:
   sha256:faed210cbc55187206ce07533652780ba090762eaac82a5d0d27020131e7f336`
   (OCI index: `linux/amd64`
   `sha256:d48640446ed1c8395c634b276fb05c62c7cc5dff49e62accc5a8208ba4aedecf`
   plus one attestation manifest).
2. The amd64 manifest's content hashes to its digest; its config and all 20
   layer blobs answer `200` by digest (9.1 GiB compressed).
3. `docker manifest inspect ghcr.io/mondegreens/apron-runner@sha256:faed210c…` succeeds.
4. Image config: `TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0;10.0` (class 6 on
   B200) and history `COPY … /usr/local/bin/apron-download` (F7).

Pull by digest was verified at the registry (every blob reachable by
digest), not by downloading 9 GiB to a laptop; the first pod (L0-A3) pulls
it by digest for real.

Note: `docker/start.sh:139` mentions a non-existent `apron-serve` in a
comment.  Left as built so `docker/` equals the pinned image's source; fix it
with the next image build.

## v0.29.0-rc7 — superseded by rc8

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

### Status: superseded before any GPU boot

This image predates two changes in this branch:
- the F7 token isolation (`docker/start.sh`, `docker/apron-download`);
- `10.0` in `TORCH_CUDA_ARCH_LIST` (class 6 needs B200).

On this image a gated download gets no token, and the token that `start.sh`
exports reaches SSH sessions.  The F7 `/proc/<pid>/environ` check would fail.
The image must be rebuilt, pushed, re-pinned here and pulled by digest before
L0-A3 (the first GPU boot).  Tracked as task #29.  A pull by digest has not
been verified yet; it is part of that task.

## v0.30.0-rc1 (2026-09-28)

- Built by `docker-publish.yml` run 36361024237 on tag `runner-v0.30.0-rc1`,
  requirements `docker/requirements-v0.30.0.txt` (vllm==0.30.0, torch==2.13.0),
  CUDA 13.0 base, linux/amd64, on a GitHub-hosted runner (no GPU needed to build).
- Registry manifest `ghcr.io/mondegreens/apron-runner:v0.30.0-rc1` →
  `sha256:36dde61da5524d9b2d8d8e931d6765d841d833dc19e44d5c572c45215bce84ba`
  (docker-content-digest, read anonymously from ghcr.io).
- v0.29.0-rc8 re-read the same way: still
  `sha256:faed210cbc55187206ce07533652780ba090762eaac82a5d0d27020131e7f336`.
- The workflow also moved `:latest` to this image.  Nothing in Apron reads
  `:latest` (pods use the digest); stopping the move is task V.8.
