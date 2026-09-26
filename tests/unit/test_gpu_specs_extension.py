"""§6.2 GPU_SPECS additions and gpu_count plumbing."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from apron.adapters.backends.runpod import CLOUD_TYPE, GPU_SPECS, RunPodTarget


@pytest.mark.parametrize(
    ("gpu_id", "cc"),
    [("NVIDIA GeForce RTX 3090", "8.6"), ("NVIDIA L40", "8.9"), ("NVIDIA B200", "10.0")],
)
def test_phase_1b_skus_present(gpu_id: str, cc: str) -> None:
    assert GPU_SPECS[gpu_id]["compute_capability"] == cc
    assert GPU_SPECS[gpu_id]["total_memory_bytes"] > 20 * (1 << 30)


def test_original_seven_skus_kept() -> None:
    for gpu_id in (
        "NVIDIA GeForce RTX 4090",
        "NVIDIA RTX A5000",
        "NVIDIA L4",
        "NVIDIA RTX A6000",
        "NVIDIA A100 80GB PCIe",
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA H100 80GB HBM3",
    ):
        assert gpu_id in GPU_SPECS


def test_gpu_count_reaches_create_pod_and_secure_cloud_only() -> None:
    created: list[dict[str, Any]] = []
    sdk = SimpleNamespace(api_key=None, create_pod=lambda **kw: created.append(kw) or {"id": "p"})
    target = RunPodTarget(
        api_key="k", ssh_key_path="/tmp/k", gpu_type="NVIDIA GeForce RTX 4090", gpu_count=4
    )
    with (
        patch.dict(sys.modules, {"runpod": sdk}),
        patch.object(target, "_register_teardown_guard"),
        patch.object(target, "_wait_for_running", side_effect=RuntimeError("stop")),
        pytest.raises(RuntimeError),
    ):
        target.provision()
    assert created[0]["gpu_count"] == 4
    assert created[0]["cloud_type"] == CLOUD_TYPE == "SECURE"
    assert created[0]["name"] == "apron-run"


def test_gpu_count_in_execution_fingerprint() -> None:
    hw = {
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "total_memory_bytes": 1,
        "compute_capability": "8.9",
    }
    one = RunPodTarget(api_key="k", ssh_key_path="/tmp/k", gpu_count=1)
    four = RunPodTarget(api_key="k", ssh_key_path="/tmp/k", gpu_count=4)
    one._build_execution_fingerprint({**hw, "gpu_count": 1})
    four._build_execution_fingerprint({**hw, "gpu_count": 4})
    assert one.execution_fingerprint != four.execution_fingerprint


def test_execution_fingerprint_carries_image_digest() -> None:
    from apron.adapters.runner_image import RUNNER_IMAGE_DIGEST

    a = RunPodTarget(api_key="k", ssh_key_path="/tmp/k")
    b = RunPodTarget(api_key="k", ssh_key_path="/tmp/k", image="repo@sha256:" + "f" * 64)
    hw = {"gpu_name": "g", "total_memory_bytes": 1, "compute_capability": "8.9"}
    a._build_execution_fingerprint(hw)
    b._build_execution_fingerprint(hw)
    assert a._image.endswith(RUNNER_IMAGE_DIGEST)
    assert a.execution_fingerprint != b.execution_fingerprint


def test_invalid_gpu_count_rejected() -> None:
    with pytest.raises(ValueError):
        RunPodTarget(api_key="k", gpu_count=0)


def test_discovery_uses_secure_price_only() -> None:
    prices = {
        "NVIDIA GeForce RTX 4090": {"securePrice": 0.74, "communityPrice": 0.34},
        "NVIDIA L4": {"securePrice": None, "communityPrice": 0.20},
    }
    sdk = SimpleNamespace(api_key=None, get_gpu=lambda gid: prices.get(gid, {}))
    with patch.dict(sys.modules, {"runpod": sdk}):
        found = RunPodTarget(api_key="k", ssh_key_path="/tmp/k").discover_gpus()
    assert [g["gpu_type_id"] for g in found] == ["NVIDIA GeForce RTX 4090"]
    assert found[0]["hourly_rate_usd"] == 0.74
    assert "community_price" not in found[0]
