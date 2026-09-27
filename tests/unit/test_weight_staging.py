"""Staged weights: a network volume, a CPU stager, and GPU pods that only load.

Phase plan, GPU dollar protection rule 1: never download on GPU-billed time.
RunPod facts used here were read on 2026-09-27: network volumes via
``rest.runpod.io/v1/networkvolumes``; CPU pods accept ``networkVolumeId``
(docs.runpod.io/api-reference/pods/POST/pods); ``dataCenters.storageSupport``
and ``lowestPrice(dataCenterId)`` answer read-only GraphQL (probed live).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from apron.adapters.backends.runpod import VOLUME_MOUNT, RunPodTarget
from apron.adapters.backends.runpod_storage import (
    RunPodStagerPod,
    RunPodStorage,
    storage_cost,
)
from apron.adapters.backends.vllm_engine import MODELS_DIR, VllmEngineAdapter
from apron.application.orchestration.budget import BudgetTracker, PhaseTiming
from apron.application.orchestration.staging import stage_weights

if TYPE_CHECKING:
    from pathlib import Path


class _Clock:
    def __init__(self) -> None:
        self.t = datetime(2026, 9, 27, tzinfo=UTC)

    def now(self) -> datetime:
        return self.t


class _Ledger:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def append(self, entry: dict[str, Any]) -> None:
        self.entries.append(dict(entry))

    def read_all(self) -> list[dict[str, Any]]:
        return [dict(e) for e in self.entries]


def _budget(clock: _Clock | None = None) -> BudgetTracker:
    return BudgetTracker(authorized=500.0, ledger=_Ledger(), clock=clock or _Clock())


# ---------------------------------------------------------------------------
# The GPU pod attaches the volume where the engine reads weights
# ---------------------------------------------------------------------------


def test_gpu_pod_attaches_the_volume_in_its_datacenter() -> None:
    created: list[dict[str, Any]] = []
    sdk = SimpleNamespace(api_key=None, create_pod=lambda **kw: created.append(kw) or {"id": "p"})
    target = RunPodTarget(
        api_key="k",
        ssh_key_path="/tmp/k",
        gpu_type="NVIDIA H100 80GB HBM3",
        network_volume_id="vol1",
        data_center_id="US-NE-1",
    )
    with (
        patch.dict(sys.modules, {"runpod": sdk}),
        patch.object(target, "_register_teardown_guard"),
        patch.object(target, "_wait_for_running", side_effect=RuntimeError("stop")),
        pytest.raises(RuntimeError),
    ):
        target.provision()
    assert created[0]["network_volume_id"] == "vol1"
    assert created[0]["data_center_id"] == "US-NE-1"
    assert created[0]["volume_in_gb"] == 0  # the network volume is the pod volume
    # same path as a downloaded model: the engine finds staged weights unchanged
    assert created[0]["volume_mount_path"] == VOLUME_MOUNT
    assert MODELS_DIR.startswith(VOLUME_MOUNT + "/")
    assert target.weights_persist
    assert "vol1" in target.weights_source and "US-NE-1" in target.weights_source


def test_a_volume_without_its_datacenter_is_refused() -> None:
    with pytest.raises(ValueError, match="data_center_id"):
        RunPodTarget(api_key="k", ssh_key_path="/tmp/k", network_volume_id="vol1")


def test_plain_pod_keeps_its_own_volume() -> None:
    target = RunPodTarget(api_key="k", ssh_key_path="/tmp/k")
    assert not target.weights_persist
    assert target.weights_source == "downloaded to the pod volume"


def test_stock_is_asked_in_the_volumes_datacenter() -> None:
    target = RunPodTarget(
        api_key="k",
        ssh_key_path="/tmp/k",
        gpu_type="NVIDIA H100 80GB HBM3",
        network_volume_id="vol1",
        data_center_id="EU-RO-1",
    )
    queries: list[str] = []

    def gql(query: str) -> dict[str, Any]:
        queries.append(query)
        return {"gpuTypes": [{"lowestPrice": {"stockStatus": "Low"}}]}

    with patch.object(target, "_gql_status", side_effect=gql):
        assert target.stock_status() == "Low"
    assert 'dataCenterId: "EU-RO-1"' in queries[0]
    assert "allowedCudaVersions" in queries[0]


# ---------------------------------------------------------------------------
# The CPU stager
# ---------------------------------------------------------------------------


def test_stager_is_a_cpu_pod_on_the_volume() -> None:
    bodies: list[dict[str, Any]] = []

    def rest(self: RunPodStorage, method: str, path: str, body: Any = None) -> Any:
        bodies.append({"method": method, "path": path, "body": body})
        return {"id": "stager1"}

    stager = RunPodStagerPod(
        api_key="k", ssh_key_path="/tmp/k", network_volume_id="vol1", data_center_id="US-NE-1"
    )
    with (
        patch.object(RunPodStorage, "_rest", rest),
        patch.object(RunPodStorage, "cpu_stock", return_value=["cpu3c"]),
        patch.object(stager, "_register_teardown_guard"),
        patch.object(stager, "_wait_for_running", side_effect=RuntimeError("stop")),
        pytest.raises(RuntimeError),
    ):
        stager.provision()
    body = bodies[0]["body"]
    assert (bodies[0]["method"], bodies[0]["path"]) == ("POST", "/pods")
    assert body["computeType"] == "CPU"
    assert body["cpuFlavorIds"] == ["cpu3c"] and body["vcpuCount"] == 8
    assert body["networkVolumeId"] == "vol1"
    assert body["dataCenterIds"] == ["US-NE-1"]
    assert body["volumeMountPath"] == VOLUME_MOUNT
    assert body["cloudType"] == "SECURE"
    assert body["name"].startswith("apron-run")  # the orphan cleanup covers it
    assert stager.pod_id == "stager1"
    with pytest.raises(RuntimeError, match="does not execute"):
        _ = stager.execution_fingerprint


# ---------------------------------------------------------------------------
# Choosing the datacenter and the volume
# ---------------------------------------------------------------------------


def test_datacenter_prefers_an_existing_volume_with_stock() -> None:
    storage = RunPodStorage("k")
    stock = {"EU-RO-1": "Low", "US-NE-1": "High", "EU-FR-1": None}
    with (
        patch.object(storage, "storage_datacenters", return_value=sorted(stock)),
        patch.object(storage, "stock_in", side_effect=lambda dc, g, n: stock[dc]),
        patch.object(storage, "cpu_stock", return_value=["cpu3c"]),
    ):
        assert storage.choose_datacenter("H100", 1) == "US-NE-1"  # best stocked
        assert storage.choose_datacenter("H100", 1, prefer=("EU-RO-1",)) == "EU-RO-1"
        assert storage.choose_datacenter("H100", 1, prefer=("EU-FR-1",)) == "US-NE-1"
    with (
        patch.object(storage, "storage_datacenters", return_value=["EU-FR-1"]),
        patch.object(storage, "stock_in", return_value=None),
    ):
        assert storage.choose_datacenter("H100", 1) is None


def test_ensure_volume_reuses_grows_or_creates() -> None:
    storage = RunPodStorage("k")
    calls: list[tuple[str, str, Any]] = []
    existing = [
        {"id": "v1", "name": "apron-weights-us-ne-1", "dataCenterId": "US-NE-1", "size": 50}
    ]

    def rest(method: str, path: str, body: Any = None) -> Any:
        calls.append((method, path, body))
        if method == "GET":
            return existing
        return {"id": "v1" if method == "PATCH" else "v2", **(body or {})}

    with patch.object(storage, "_rest", side_effect=rest):
        assert storage.ensure_volume("US-NE-1", 40)["id"] == "v1"
        assert [c[0] for c in calls] == ["GET"]
        calls.clear()
        grown = storage.ensure_volume("US-NE-1", 120)
        assert grown["size"] == 120 and calls[-1][:2] == ("PATCH", "/networkvolumes/v1")
        calls.clear()
        made = storage.ensure_volume("EU-RO-1", 80)
        assert made["id"] == "v2"
        assert calls[-1] == (
            "POST",
            "/networkvolumes",
            {"name": "apron-weights-eu-ro-1", "size": 80, "dataCenterId": "EU-RO-1"},
        )


def test_storage_cost_is_the_standard_tier_rate() -> None:
    assert storage_cost(730, 730.0) == pytest.approx(730 * 0.07)
    assert storage_cost(100, 1.0) == pytest.approx(100 * 0.07 / 730, abs=1e-6)  # micro-dollars


# ---------------------------------------------------------------------------
# The engine on staged weights
# ---------------------------------------------------------------------------


class _Target:
    def __init__(self, persist: bool, stdout: str = "") -> None:
        self.weights_persist = persist
        self.commands: list[str] = []
        self._stdout = stdout

    def execute(self, command: str, **_: Any) -> dict[str, Any]:
        self.commands.append(command)
        return {"stdout": self._stdout, "exit_code": 0}


def test_staged_weights_are_never_evicted() -> None:
    staged = _Target(persist=True)
    VllmEngineAdapter().evict_models(staged, keep="Qwen/Qwen3-32B")
    assert staged.commands == []
    plain = _Target(persist=False)
    VllmEngineAdapter().evict_models(plain, keep="Qwen/Qwen3-32B")
    assert any("rm -rf" in c for c in plain.commands)


def test_prefetch_when_the_checkpoint_fits_host_memory() -> None:
    engine = VllmEngineAdapter()
    fits = _Target(persist=True, stdout=f"{60 * 10**9}\n{200 * 10**9}\n")
    assert engine.load_args(fits, "Qwen/Qwen3-32B") == [
        "--safetensors-load-strategy",
        "prefetch",
    ]
    too_big = _Target(persist=True, stdout=f"{700 * 10**9}\n{200 * 10**9}\n")
    assert engine.load_args(too_big, "zai-org/GLM-5.3") == []
    unreadable = _Target(persist=True, stdout="du: cannot access\n")
    assert engine.load_args(unreadable, "x/y") == []
    local = _Target(persist=False, stdout=f"{1}\n{10**12}\n")
    assert engine.load_args(local, "x/y") == []
    assert local.commands == []  # nothing probed for weights on the pod's own disk


# ---------------------------------------------------------------------------
# Staging: paid through the budget, torn down on every path
# ---------------------------------------------------------------------------


class _Stager:
    def __init__(self, clock: _Clock, fail: bool = False) -> None:
        self.pod_id: str | None = None
        self.torn_down = False
        self._clock, self._fail = clock, fail

    def provision(self, env: dict[str, str] | None = None) -> dict[str, Any]:
        if self._fail:
            raise RuntimeError("no CPU capacity")
        self.pod_id = "cpu1"
        return {"status": "provisioned"}

    def pod_reported_cost(self) -> float | None:
        return 0.12 if self.pod_id else None

    def teardown(self) -> None:
        self.torn_down = True


class _Engine:
    def __init__(self, clock: _Clock) -> None:
        self._clock = clock
        self.downloaded: list[str] = []

    def runner_supports_token_isolation(self, target: Any) -> bool:
        return True

    def download_weights(self, target: Any, model_id: str, timeout: int = 0) -> dict[str, Any]:
        self._clock.t += timedelta(minutes=5)
        self.downloaded.append(model_id)
        ok = model_id != "broken/model"
        return {"ok": ok, "seconds": 300.0, "output_tail": "DOWNLOAD_VERIFIED" if ok else "short"}


def test_staging_downloads_settles_and_tears_down() -> None:
    clock = _Clock()
    budget = _budget(clock)
    stager, engine = _Stager(clock), _Engine(clock)
    events: list[dict[str, Any]] = []
    result = stage_weights(
        ["Qwen/Qwen3-32B", "broken/model"],
        stager=stager,
        engine=engine,
        budget=budget,
        clock=clock,
        hourly_rate=0.3,
        estimate=0.5,
        event=events.append,
    )
    assert engine.downloaded == ["Qwen/Qwen3-32B", "broken/model"]
    assert [m.ok for m in result.models] == [True, False]
    assert not result.ok  # a failed download is reported, not hidden
    assert stager.torn_down
    assert result.cost == 0.12  # the pod's reported cost, not the estimate
    assert budget.spent == pytest.approx(0.12)
    assert not budget.holds
    assert [e["model_id"] for e in events] == ["Qwen/Qwen3-32B", "broken/model"]
    assert result.seconds == 600.0


def test_staging_failure_still_settles_and_tears_down() -> None:
    clock = _Clock()
    budget = _budget(clock)
    stager = _Stager(clock, fail=True)
    result = stage_weights(
        ["Qwen/Qwen3-32B"],
        stager=stager,
        engine=_Engine(clock),
        budget=budget,
        clock=clock,
        hourly_rate=0.3,
        estimate=0.5,
    )
    assert result.error and "no CPU capacity" in result.error
    assert stager.torn_down
    assert not budget.holds  # the hold was settled (at the rate: no reported cost)


# ---------------------------------------------------------------------------
# Records: where pod time went, and where weights came from
# ---------------------------------------------------------------------------


def test_phase_seconds_split_weights_and_engine_start() -> None:
    timing = PhaseTiming(
        provision_start=0.0, teardown_end=1000.0, weights=120.0, engine_start=90.5
    )
    seconds = timing.seconds()
    assert seconds["weights"] == 120.0 and seconds["engine_start"] == 90.5
    assert seconds["total"] == 1000.0  # parts of provision_boot_teardown, not extra time
    assert "weights" not in PhaseTiming(provision_start=0.0, teardown_end=1.0).seconds()


def test_storage_accrues_once_per_window(tmp_path: Path) -> None:
    from apron.interfaces.cohort_root import WeightsSite, accrue_storage

    clock = _Clock()
    budget = _budget(clock)
    site = WeightsSite(volume_id="vol1", data_center_id="US-NE-1", size_gb=100)
    assert accrue_storage(site, budget, tmp_path) == 0.0  # created now
    clock.t += timedelta(hours=10)
    assert accrue_storage(site, budget, tmp_path) == pytest.approx(storage_cost(100, 10.0))
    assert accrue_storage(site, budget, tmp_path) == 0.0  # nothing new since
    state = json.loads((tmp_path / "volumes.json").read_text())
    assert state["vol1"]["data_center_id"] == "US-NE-1"
    assert budget.spent == pytest.approx(storage_cost(100, 10.0))


def test_stager_steps_down_until_a_cpu_pod_is_free() -> None:
    tried: list[tuple[str, int]] = []

    def rest(self: RunPodStorage, method: str, path: str, body: Any = None) -> Any:
        tried.append((body["cpuFlavorIds"][0], body["vcpuCount"]))
        if (body["cpuFlavorIds"][0], body["vcpuCount"]) != ("cpu3g", 4):
            raise RuntimeError("HTTP 500 There are no longer any instances available with ...")
        return {"id": "stager2"}

    stager = RunPodStagerPod(
        api_key="k", ssh_key_path="/tmp/k", network_volume_id="vol1", data_center_id="AP-JP-1"
    )
    with (
        patch.object(RunPodStorage, "_rest", rest),
        patch.object(RunPodStorage, "cpu_stock", return_value=["cpu3c", "cpu3g"]),
        patch.object(stager, "_register_teardown_guard"),
        patch.object(stager, "_wait_for_running", side_effect=RuntimeError("stop")),
        pytest.raises(RuntimeError, match="stop"),
    ):
        stager.provision()
    assert tried == [("cpu3c", 8), ("cpu3c", 4), ("cpu3c", 2), ("cpu3g", 8), ("cpu3g", 4)]
    assert stager.pod_id == "stager2"


def test_stager_reports_every_refusal_and_other_errors_stop() -> None:
    def full(self: RunPodStorage, method: str, path: str, body: Any = None) -> Any:
        raise RuntimeError("There are no longer any instances available")

    stager = RunPodStagerPod(
        api_key="k", ssh_key_path="/tmp/k", network_volume_id="vol1", data_center_id="AP-JP-1"
    )
    with (
        patch.object(RunPodStorage, "_rest", full),
        patch.object(RunPodStorage, "cpu_stock", return_value=["cpu3c"]),
        pytest.raises(RuntimeError, match=r"cpu3cx8.*cpu3cx2"),
    ):
        stager.provision()

    def bad(self: RunPodStorage, method: str, path: str, body: Any = None) -> Any:
        raise RuntimeError("HTTP 401 unauthorized")

    with (
        patch.object(RunPodStorage, "_rest", bad),
        patch.object(RunPodStorage, "cpu_stock", return_value=["cpu3c", "cpu5c"]),
        pytest.raises(RuntimeError, match="401"),
    ):
        stager.provision()


def test_us_datacenters_come_first_then_europe_then_asia() -> None:
    from apron.adapters.backends.runpod_storage import region_rank

    ids = ["AP-JP-1", "EUR-IS-1", "US-NE-1", "EU-RO-1", "CA-MTL-3", "XX-1"]
    assert sorted(ids, key=region_rank)[:2] == ["US-NE-1", "CA-MTL-3"]
    assert sorted(ids, key=region_rank)[-2:] == ["AP-JP-1", "XX-1"]
    storage = RunPodStorage("k")
    stock = {"AP-JP-1": "High", "EUR-IS-1": "High", "US-NE-1": "Low"}
    with (
        patch.object(storage, "storage_datacenters", return_value=sorted(stock)),
        patch.object(storage, "stock_in", side_effect=lambda dc, g, n: stock[dc]),
        patch.object(storage, "cpu_stock", return_value=["cpu3c"]),
    ):
        # a US datacenter with low stock beats a better-stocked European or Asian one
        assert storage.choose_datacenter("H100", 1) == "US-NE-1"


def test_a_stager_stopped_mid_provision_is_still_named_and_paid() -> None:
    clock = _Clock()
    budget = _budget(clock)

    class _Half(_Stager):
        def provision(self, env: dict[str, str] | None = None) -> dict[str, Any]:
            self.pod_id = "cpu9"  # created, then the wait for SSH is interrupted
            raise RuntimeError("interrupted while the image pulled")

    stager = _Half(clock)
    result = stage_weights(
        ["Qwen/Qwen3-32B"],
        stager=stager,
        engine=_Engine(clock),
        budget=budget,
        clock=clock,
        hourly_rate=0.3,
        estimate=0.5,
    )
    assert stager.torn_down and result.pod_id == "cpu9"
    annotated = [e for e in budget.ledger.read_all() if e["op"] == "annotate"]
    assert annotated and annotated[0]["pod_id"] == "cpu9"  # reconcile can find its bill
    assert not budget.holds


def test_hybrid_architectures_are_an_explicit_unknown() -> None:
    from apron.adapters.backends.vllm_quantization import HYBRID_ARCHITECTURES
    from apron.domain.mechanisms.model_spec_builder import UNKNOWN_MECHANISM, build_model_spec

    # generated from the pinned source's IsHybrid classes
    assert {"NemotronHForCausalLM", "Qwen3_5MoeForConditionalGeneration"} <= HYBRID_ARCHITECTURES
    config = {
        "architectures": ["NemotronHForCausalLM"],
        "num_hidden_layers": 52,
        "num_attention_heads": 32,
        "hidden_size": 2688,
    }
    plain = build_model_spec(config)
    assert plain.components[0].mechanism == "autoregressive_decode"  # the old, wrong reading
    hybrid = build_model_spec(config, unmodelled_architectures=HYBRID_ARCHITECTURES)
    assert hybrid.components[0].mechanism == UNKNOWN_MECHANISM
