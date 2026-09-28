"""Model weights staged on a RunPod network volume before a GPU pod starts.

Phase plan, GPU dollar protection rule 1: never download on GPU-billed time.
A network volume lives in one datacenter and outlives pods.  A CPU pod in
that datacenter downloads and verifies the weights onto it; the GPU pod then
attaches the same volume at the same path and only loads them.

- Volumes: RunPod REST API (``/v1/networkvolumes``; the Python SDK has no
  volume calls).
- Where: a datacenter that supports network storage and has Secure stock for
  the requested GPU on hosts that can run the runner image
  (``dataCenters.storageSupport`` plus ``lowestPrice`` filtered by
  ``dataCenterId``, read-only GraphQL).
- The CPU pod: REST ``computeType: CPU`` with the volume attached (the SDK's
  ``create_pod`` only makes GPU pods).  It runs the runner image, so the
  download is the same F7 step as on a GPU pod: the token reaches only the
  download process.

Docs read 2026-09-27: docs.runpod.io/storage/network-volumes (200-400 MB/s,
up to 10 GB/s peak; attach at deploy time only), /api-reference/pods/POST/pods
(CPU pods accept ``networkVolumeId``), /storage/s3-api.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from apron.adapters.backends.runpod import (
    CLOUD_TYPE,
    GRAPHQL_URL,
    POD_NAME_PREFIX,
    VOLUME_MOUNT,
    RunPodTarget,
)
from apron.adapters.runner_image import RUNNER_HOST_CUDA_VERSIONS
from apron.application.sanitization import mask_secrets

logger = logging.getLogger(__name__)

REST_URL = "https://rest.runpod.io/v1"
VOLUME_NAME_PREFIX = "apron-weights"
# Standard tier, docs.runpod.io/storage/network-volumes (2026-09-27).
VOLUME_USD_PER_GB_MONTH = 0.07
HOURS_PER_MONTH = 730.0
# The stager only downloads: a few vCPUs for parallel transfers.  Sizes are
# tried largest first; a datacenter listing a flavor in stock still refused
# 8 vCPUs and took 4 (AP-JP-1, 2026-09-27).
STAGER_CPU_FLAVORS = ("cpu3c", "cpu5c", "cpu3g", "cpu5g")
STAGER_VCPUS = (8, 4, 2)
STAGER_CONTAINER_GB = 20
NO_CAPACITY = "no longer any instances available"

_DATACENTERS_QUERY = "query { dataCenters { id storageSupport } }"
_CPU_STOCK_QUERY = "query { dataCenters { id cpuAvailability { cpuFlavorId stockStatus } } }"
_DC_STOCK_QUERY = """query Stock {{
  gpuTypes(input: {{id: "{gpu}"}}) {{
    lowestPrice(input: {{
      gpuCount: {count}, secureCloud: true, dataCenterId: "{dc}",
      allowedCudaVersions: [{cuda}]
    }}) {{ stockStatus }}
  }}
}}"""
_STOCK_RANK = {"High": 3, "Medium": 2, "Low": 1}
# Owner's experience (2026-09-27): RunPod's US datacenters run pods faster than
# the European ones, and the Asian ones slower still.  Staging records the
# measured transfer speed per datacenter (``staged`` events) to replace this.
REGION_ORDER = ("US-", "CA-", "EU-", "EUR-", "OC-", "AP-")


def region_rank(data_center_id: str) -> int:
    """Lower is preferred: US first, Asia last, unknown prefixes after all."""
    return next(
        (i for i, prefix in enumerate(REGION_ORDER) if data_center_id.startswith(prefix)),
        len(REGION_ORDER),
    )


# High-performance storage is the default for a volume created in a datacenter
# that offers it (docs.runpod.io/storage/high-performance-storage: "enabled by
# default"; "Exact pricing varies by data center").  Rates as the console bills
# them: US-CA-2's 480 GB volume shows $67.20/mo (owner's screenshot,
# 2026-09-28), $0.14 per GB-month, twice the standard rate the ledger had used.
HIGH_PERFORMANCE_USD_PER_GB_MONTH: dict[str, float] = {"US-CA-2": 0.14}


def volume_usd_per_gb_month(data_center_id: str | None) -> float:
    """The per-GB monthly rate a new volume in *data_center_id* is billed at."""
    return HIGH_PERFORMANCE_USD_PER_GB_MONTH.get(data_center_id or "", VOLUME_USD_PER_GB_MONTH)


def storage_cost(size_gb: int, hours: float, data_center_id: str | None = None) -> float:
    """What a volume of *size_gb* in *data_center_id* costs for *hours*."""
    rate = volume_usd_per_gb_month(data_center_id)
    return round(size_gb * rate * hours / HOURS_PER_MONTH, 6)


class RunPodStorage:
    """Network volumes and the datacenter choice (REST + read-only GraphQL)."""

    def __init__(self, api_key: str, *, timeout: float = 30.0) -> None:
        self._key = api_key
        self._timeout = timeout

    # -- REST: volumes ------------------------------------------------------

    def _rest(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        resp = httpx.request(
            method,
            f"{REST_URL}{path}",
            json=body,
            headers={"Authorization": f"Bearer {self._key}"},
            timeout=self._timeout,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"RunPod {method} {path}: HTTP {resp.status_code} {mask_secrets(resp.text[:300])}"
            )
        return resp.json() if resp.content else None

    def list_volumes(self) -> list[dict[str, Any]]:
        rows = self._rest("GET", "/networkvolumes")
        return rows if isinstance(rows, list) else []

    def create_volume(self, name: str, size_gb: int, data_center_id: str) -> dict[str, Any]:
        volume = self._rest(
            "POST",
            "/networkvolumes",
            {"name": name, "size": int(size_gb), "dataCenterId": data_center_id},
        )
        logger.info("network volume %s created in %s", volume.get("id"), data_center_id)
        return volume

    def resize_volume(self, volume_id: str, size_gb: int) -> dict[str, Any]:
        """Grow a volume (RunPod cannot shrink one)."""
        return self._rest("PATCH", f"/networkvolumes/{volume_id}", {"size": int(size_gb)})

    def delete_volume(self, volume_id: str) -> None:
        self._rest("DELETE", f"/networkvolumes/{volume_id}")

    def ensure_volume(self, data_center_id: str, size_gb: int) -> dict[str, Any]:
        """Apron's weights volume in *data_center_id*, created or grown to *size_gb*."""
        name = f"{VOLUME_NAME_PREFIX}-{data_center_id.lower()}"
        for volume in self.list_volumes():
            if volume.get("name") == name and volume.get("dataCenterId") == data_center_id:
                if int(volume.get("size") or 0) < size_gb:
                    volume = {**volume, **(self.resize_volume(volume["id"], size_gb) or {})}
                return volume
        return self.create_volume(name, size_gb, data_center_id)

    # -- GraphQL: where ------------------------------------------------------

    def _gql(self, query: str) -> dict[str, Any]:
        resp = httpx.post(
            f"{GRAPHQL_URL}?api_key={self._key}",
            json={"query": query},
            headers={"Content-Type": "application/json"},
            timeout=self._timeout,
        )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(mask_secrets(str(exc))) from None
        data = resp.json()
        if "errors" in data:
            raise RuntimeError(f"GraphQL errors: {mask_secrets(str(data['errors']))}")
        return data.get("data") or {}

    def storage_datacenters(self) -> list[str]:
        """Datacenters with network storage, preferred region first."""
        rows = self._gql(_DATACENTERS_QUERY).get("dataCenters") or []
        ids = [str(r["id"]) for r in rows if r.get("storageSupport")]
        return sorted(ids, key=lambda dc: (region_rank(dc), dc))

    def cpu_stock(self, data_center_id: str) -> list[str]:
        """CPU flavors with Secure stock in one datacenter, best stocked first."""
        rows = self._gql(_CPU_STOCK_QUERY).get("dataCenters") or []
        row = next((r for r in rows if r.get("id") == data_center_id), {})
        stocked = [
            (_STOCK_RANK.get(str(c.get("stockStatus")), 0), str(c["cpuFlavorId"]))
            for c in row.get("cpuAvailability") or []
            if c.get("stockStatus")
        ]
        return [flavor for _, flavor in sorted(stocked, key=lambda x: -x[0])]

    def stock_in(self, data_center_id: str, gpu_type: str, gpu_count: int) -> str | None:
        """Secure stock for *gpu_type* x *gpu_count* in one datacenter, on hosts
        that can run the runner image; ``None`` means none."""
        cuda = ", ".join(f'"{v}"' for v in RUNNER_HOST_CUDA_VERSIONS)
        query = _DC_STOCK_QUERY.format(
            gpu=gpu_type, count=int(gpu_count), dc=data_center_id, cuda=cuda
        )
        types = self._gql(query).get("gpuTypes") or []
        lowest = (types[0].get("lowestPrice") or {}) if types else {}
        status = lowest.get("stockStatus")
        return str(status) if status else None

    def choose_datacenter(
        self, gpu_type: str, gpu_count: int, *, prefer: tuple[str, ...] = ()
    ) -> str | None:
        """A storage datacenter with stock for the GPU: a preferred one (an
        existing volume's) when it has stock, else the best-stocked; ``None``
        when there is none right now."""
        stocked: list[tuple[int, str]] = []
        for dc in self.storage_datacenters():
            status = self.stock_in(dc, gpu_type, gpu_count)
            # The weights are staged from a CPU pod in the same datacenter.
            if status and self.cpu_stock(dc):
                stocked.append((_STOCK_RANK.get(status, 0), dc))
        for dc in prefer:
            if any(d == dc for _, d in stocked):
                return dc
        # Region first (speed), then stock.
        stocked.sort(key=lambda sd: (region_rank(sd[1]), -sd[0], sd[1]))
        return stocked[0][1] if stocked else None


class RunPodStagerPod(RunPodTarget):
    """A CPU pod with the weights volume attached: downloads, never serves.

    Same SSH, command, cost and teardown paths as a GPU pod (the safety net
    and orphan cleanup cover it: its name starts with ``apron-run``).  No
    GPU, so no hardware detection and no execution fingerprint.
    """

    def __init__(self, *, network_volume_id: str, data_center_id: str, **kwargs: Any) -> None:
        kwargs.setdefault("gpu_type", "cpu")
        super().__init__(
            network_volume_id=network_volume_id, data_center_id=data_center_id, **kwargs
        )

    @property
    def execution_fingerprint(self) -> str:
        raise RuntimeError("a stager pod does not execute models")

    @property
    def location(self) -> str | None:
        return self._data_center_id

    def provision(
        self, wait_timeout: int = 0, env: dict[str, str] | None = None
    ) -> dict[str, Any]:
        if not self._api_key:
            raise RuntimeError("Cannot provision without RUNPOD_API_KEY")
        if self._pod_id is not None:
            self.teardown()
        storage = RunPodStorage(str(self._api_key))
        flavors = storage.cpu_stock(str(self._data_center_id)) or list(STAGER_CPU_FLAVORS)
        pod: dict[str, Any] | None = None
        refusals: list[str] = []
        for flavor in flavors:
            for vcpus in STAGER_VCPUS:
                body = {
                    "name": f"{POD_NAME_PREFIX}-stager",
                    "imageName": self._image,
                    "computeType": "CPU",
                    "cpuFlavorIds": [flavor],
                    "vcpuCount": vcpus,
                    "cloudType": CLOUD_TYPE,
                    "dataCenterIds": [self._data_center_id],
                    "networkVolumeId": self._network_volume_id,
                    "volumeMountPath": VOLUME_MOUNT,
                    "containerDiskInGb": STAGER_CONTAINER_GB,
                    "ports": ["22/tcp"],
                    "env": env or {},
                }
                try:
                    pod = storage._rest("POST", "/pods", body)
                    break
                except RuntimeError as exc:
                    if NO_CAPACITY not in str(exc):
                        raise
                    refusals.append(f"{flavor}x{vcpus}")
            if pod is not None:
                break
        if pod is None:
            raise RuntimeError(
                f"{NO_CAPACITY} for a CPU stager in {self._data_center_id}: {refusals}"
            )
        self._pod_id = pod["id"]
        logger.info("stager pod %s created in %s", self._pod_id, self._data_center_id)
        self._register_teardown_guard()
        started = time.monotonic()
        pod_info = self._wait_for_running(wait_timeout)
        self._establish_ssh(pod_info)
        return {
            "status": "provisioned",
            "pod_id": self._pod_id,
            "seconds_to_ssh": round(time.monotonic() - started, 1),
        }
