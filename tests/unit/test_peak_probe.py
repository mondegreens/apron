"""The startup peak probe (scripts/peak_probe/, scripts/peak_probe.py) — no GPU, no network.

- the trace replay finds the peak of ``+alloc - free_requested`` and lists
  what was alive then, grouped by allocation site;
- the ``sitecustomize`` does nothing without ``APRON_PEAK_PROBE=1`` and, with
  it, wraps ``profile_run`` of the runner module on import, writes its JSON and
  never breaks the boot (a fake ``torch`` and a fake ``vllm`` package stand in);
- the driver boots exactly the command the cohort's engine adapter renders.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any, ClassVar

import pytest

REPO = Path(__file__).resolve().parents[2]
PROBE_DIR = REPO / "scripts" / "peak_probe"
RUN_DIR = REPO / "_dev_notes" / "cohort-run"
RUNNER = "vllm.v1.worker.gpu.model_runner"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve string annotations through it
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def analyze() -> Any:
    return _load("peak_probe_analyze_under_test", PROBE_DIR / "analyze.py")


def _frame(filename: str, line: int, name: str) -> dict[str, Any]:
    return {"filename": filename, "line": line, "name": name}


SITE = "/opt/venv/lib/python3.12/site-packages/"
FORWARD = [
    _frame(SITE + "torch/nn/functional.py", 100, "silu"),
    _frame(SITE + "vllm/model_executor/layers/activation.py", 80, "forward_native"),
    _frame(SITE + "vllm/model_executor/models/glm4_moe.py", 210, "forward"),
    _frame(SITE + "vllm/v1/worker/gpu/model_runner.py", 760, "_dummy_run"),
    _frame(SITE + "vllm/v1/worker/gpu/model_runner.py", 868, "profile_run"),
]
COMPILE = [
    _frame(SITE + "torch/_inductor/runtime/triton_heuristics.py", 900, "clone_args"),
    _frame(SITE + "torch/_inductor/compile_fx.py", 700, "compile_fx_inner"),
    _frame(SITE + "vllm/compilation/backends.py", 300, "__call__"),
    _frame(SITE + "vllm/v1/worker/gpu/model_runner.py", 760, "_dummy_run"),
]
SAMPLER = [
    _frame(SITE + "vllm/v1/worker/gpu/sample/sampler.py", 50, "sample"),
    _frame(SITE + "vllm/v1/worker/gpu/model_runner.py", 840, "_dummy_sampler_run"),
]


def _ev(action: str, addr: int, size: int, frames: list[dict[str, Any]] | None = None) -> dict:
    return {"action": action, "addr": addr, "size": size, "stream": 0, "frames": frames or []}


def synthetic_snapshot() -> dict[str, Any]:
    """Rounded to 512: A 1024, B 4096, C 10240, D 512, E 512, pre-recording 2048.

    running: 1024, 5120, 1024, -1024, 9216, 9728 (peak, index 7), -512, 0.
    """
    events = [
        _ev("alloc", 1, 1000, FORWARD),  # 0: A
        {"action": "segment_alloc", "addr": 0, "size": 2 << 20, "stream": 0, "frames": []},
        _ev("alloc", 2, 4096, COMPILE),  # 2: B (autotune clone)
        _ev("free_requested", 2, 4096, COMPILE),  # 3
        _ev("free_completed", 2, 4096, COMPILE),  # 4: ignored
        _ev("free_requested", 99, 2048, FORWARD),  # 5: a block from before recording
        _ev("alloc", 3, 10_000, FORWARD),  # 6: C, same site as A
        _ev("alloc", 4, 512, SAMPLER),  # 7: D -> peak
        _ev("free_requested", 3, 10_000, FORWARD),  # 8
        _ev("alloc", 2, 100, SAMPLER),  # 9: E, address reused
    ]
    return {"segments": [], "device_traces": [events]}


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------


def test_replay_finds_the_peak_and_the_live_set(analyze: Any) -> None:
    result = analyze.analyze(synthetic_snapshot(), device=0, max_entries=1000)
    assert result["peak_increase_bytes"] == 9728
    assert result["peak_increase_requested_bytes"] == 1000 + 10_000 + 512 - 2048
    assert result["peak_event_index"] == 7
    assert result["peak_event"]["phase"] == "sampler"
    # A, C and D alive; the pre-recording free explains live - peak.
    assert result["peak_live_count"] == 3
    assert result["peak_live_bytes"] == 1024 + 10240 + 512
    assert result["freed_pre_recording_bytes"] == 2048
    assert result["peak_live_bytes"] - result["peak_increase_bytes"] == 2048
    assert result["end_increase_bytes"] == 0
    assert result["live_at_end_bytes"] == 1024 + 512 + 512  # A, D, E
    assert result["truncated"] is False
    assert result["actions"]["free_completed"] == 1


def test_live_blocks_are_grouped_by_site_largest_first(analyze: Any) -> None:
    result = analyze.analyze(synthetic_snapshot())
    sites = result["sites_at_peak"]
    assert [g["bytes"] for g in sites] == [1024 + 10240, 512]
    top = sites[0]
    assert top["count"] == 2
    assert top["requested_bytes"] == 11_000
    assert top["phases"] == {"forward": 11264}
    # innermost frames under vllm/ first; torch/nn is not a site frame
    assert top["site"][0] == "vllm/model_executor/layers/activation.py:80:forward_native"
    assert all("torch/nn" not in f for f in top["site"])
    assert result["phase_at_peak"] == {"forward": 11264, "sampler": 512}
    assert result["dominant_phase_at_peak"] == "forward"
    assert result["largest_blocks_at_peak"][0]["size"] == 10240


def test_phases_and_rounding(analyze: Any) -> None:
    assert analyze.phase_of(COMPILE) == "compile"
    assert analyze.phase_of(SAMPLER) == "sampler"
    assert analyze.phase_of(FORWARD) == "forward"
    enc = [_frame(SITE + "vllm/v1/worker/gpu/mm/encoder_runner.py", 1, "profile_encoder_cache")]
    assert analyze.phase_of(enc) == "encoder"
    assert analyze.phase_of([_frame("/x.py", 1, "f")]) == "other"
    assert [analyze.rounded(n) for n in (1, 512, 513, 1 << 20)] == [512, 512, 1024, 1 << 20]


def test_a_full_ring_buffer_is_flagged(analyze: Any) -> None:
    snap = synthetic_snapshot()
    assert analyze.analyze(snap, max_entries=len(snap["device_traces"][0]))["truncated"] is True


def test_an_empty_trace_reports_no_peak(analyze: Any) -> None:
    result = analyze.analyze({"segments": [], "device_traces": []})
    assert result["peak_increase_bytes"] == 0
    assert result["peak_event"] is None
    assert result["sites_at_peak"] == []


# ---------------------------------------------------------------------------
# sitecustomize
# ---------------------------------------------------------------------------


def _load_sitecustomize(tag: str) -> Any:
    return _load(f"peak_probe_sitecustomize_{tag}", PROBE_DIR / "sitecustomize.py")


def test_sitecustomize_is_a_no_op_without_the_switch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("APRON_PEAK_PROBE", raising=False)
    monkeypatch.setenv("APRON_PEAK_PROBE_OUT", str(tmp_path / "probe.json"))
    before = list(sys.meta_path)
    module = _load_sitecustomize("off")
    assert sys.meta_path == before
    assert module.install() is None
    assert sys.meta_path == before
    assert not (tmp_path / "probe.json").exists()
    monkeypatch.setenv("APRON_PEAK_PROBE", "0")
    assert module.install() is None


class FakeCuda:
    """What the wrapper reads from torch.cuda, with a scripted peak."""

    def __init__(self, snapshot: Any) -> None:
        self.allocated = 60 << 30
        self.max_allocated = self.allocated
        self.history: list[dict[str, Any]] = []
        self._snapshot_value = snapshot
        self.memory = types.SimpleNamespace(
            _record_memory_history=self._record, _snapshot=self._snapshot
        )

    def _record(self, **kwargs: Any) -> None:
        self.history.append(kwargs)

    def _snapshot(self) -> Any:
        if isinstance(self._snapshot_value, Exception):
            raise self._snapshot_value
        return self._snapshot_value

    def current_device(self) -> int:
        return 0

    def synchronize(self, device: int = 0) -> None:
        return None

    def get_device_name(self, device: int = 0) -> str:
        return "FAKE H100"

    def memory_allocated(self, device: int = 0) -> int:
        return self.allocated

    def max_memory_allocated(self, device: int = 0) -> int:
        return self.max_allocated

    def memory_reserved(self, device: int = 0) -> int:
        return self.allocated + (1 << 30)


RUNNER_SOURCE = """
import torch


class GPUModelRunner:
    def profile_run(self):
        torch.cuda.max_allocated += 9728  # the scripted peak of synthetic_snapshot()
        if getattr(torch, "fail", False):
            raise RuntimeError("vLLM's own failure")
        return "ran"
"""


@pytest.fixture
def fake_vllm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A fake ``vllm`` package with the V2 runner module, first on sys.path."""
    root = tmp_path / "pkgs"
    pkg = root / "vllm" / "v1" / "worker" / "gpu"
    pkg.mkdir(parents=True)
    for d in (root / "vllm", root / "vllm" / "v1", root / "vllm" / "v1" / "worker", pkg):
        (d / "__init__.py").write_text("")
    (root / "vllm" / "__init__.py").write_text('__version__ = "0.29.0-fake"\n')
    (pkg / "model_runner.py").write_text(RUNNER_SOURCE)
    monkeypatch.syspath_prepend(str(root))
    saved_meta = list(sys.meta_path)
    for name in [m for m in sys.modules if m == "vllm" or m.startswith("vllm.")]:
        monkeypatch.delitem(sys.modules, name)
    yield root
    sys.meta_path[:] = saved_meta
    for name in [m for m in sys.modules if m == "vllm" or m.startswith("vllm.")]:
        del sys.modules[name]


def _enable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, snapshot: Any) -> tuple[Any, Path]:
    torch = types.ModuleType("torch")
    torch.__version__ = "2.13.0-fake"  # type: ignore[attr-defined]
    torch.cuda = FakeCuda(snapshot)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    out = tmp_path / "out" / "probe.json"
    monkeypatch.setenv("APRON_PEAK_PROBE", "1")
    monkeypatch.setenv("APRON_PEAK_PROBE_OUT", str(out))
    monkeypatch.setenv("APRON_PEAK_PROBE_MAX_ENTRIES", "5000")
    return torch, out


def test_the_hook_wraps_profile_run_and_writes_the_analysis(
    fake_vllm: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    torch, out = _enable(monkeypatch, tmp_path, synthetic_snapshot())
    probe = _load_sitecustomize("on")
    runner_module = importlib.import_module(RUNNER)
    assert getattr(runner_module.GPUModelRunner.profile_run, "__apron_peak_probe__", False)

    assert runner_module.GPUModelRunner().profile_run() == "ran"

    doc = json.loads(out.read_text())
    run = doc["runs"][0]
    assert run["runner"] == f"{RUNNER}.GPUModelRunner"
    assert run["errors"] == [] and doc["errors"] == []
    assert run["torch_peak_increase"] == 9728
    assert run["analysis"]["peak_increase_bytes"] == 9728
    assert run["replay_minus_torch_peak_bytes"] == 0
    assert run["vllm_version"] == "0.29.0-fake"
    assert torch.cuda.history == [
        {
            "enabled": "all",
            "context": "all",
            "stacks": "python",
            "max_entries": 5000,
            "clear_history": True,
        },
        {"enabled": None},
    ]
    assert probe.patch_module(runner_module, "GPUModelRunner") is False  # never wrapped twice


def test_a_failing_snapshot_never_breaks_the_boot(
    fake_vllm: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    torch, out = _enable(monkeypatch, tmp_path, RuntimeError("snapshot exploded"))
    _load_sitecustomize("snapfail")
    runner_module = importlib.import_module(RUNNER)
    assert runner_module.GPUModelRunner().profile_run() == "ran"
    run = json.loads(out.read_text())["runs"][0]
    assert [e["where"] for e in run["errors"]] == ["snapshot"]
    assert "snapshot exploded" in run["errors"][0]["error"]
    assert torch.cuda.history[-1] == {"enabled": None}  # recording switched off anyway
    assert run["torch_peak_increase"] == 9728


def test_vllms_own_failure_is_recorded_and_re_raised(
    fake_vllm: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    torch, out = _enable(monkeypatch, tmp_path, synthetic_snapshot())
    torch.fail = True
    _load_sitecustomize("vllmfail")
    runner_module = importlib.import_module(RUNNER)
    with pytest.raises(RuntimeError, match="vLLM's own failure"):
        runner_module.GPUModelRunner().profile_run()
    run = json.loads(out.read_text())["runs"][0]
    assert run["profile_run_error"] == "RuntimeError: vLLM's own failure"
    assert run["analysis"]["peak_increase_bytes"] == 9728


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


@pytest.fixture
def driver() -> Any:
    return _load("peak_probe_driver_under_test", REPO / "scripts" / "peak_probe.py")


needs_manifest = pytest.mark.skipif(
    not (RUN_DIR / "solutions.jsonl").exists(), reason="no cohort manifest on record"
)

# The recorded boots: the reports' plan digests and vLLM's logged peak.
RECORDED = {
    "zai-org/GLM-4.7-Flash": 2_190_433_320,
    "google/gemma-4-31B-it": 1_599_875_317,
}


def _expected_serve(model_id: str) -> str:
    return (
        f"vllm serve /runpod-volume/models/{model_id} --served-model-name {model_id} "
        "--dtype bfloat16 --gpu-memory-utilization 0.9 --max-model-len 640"
    )


@needs_manifest
@pytest.mark.parametrize("model_id", sorted(RECORDED))
def test_the_plan_is_the_one_the_recorded_boot_ran(driver: Any, model_id: str) -> None:
    from apron.adapters.backends.vllm_engine import VllmEngineAdapter
    from apron.domain.fingerprints import fingerprint_hex

    solution = driver.recorded_solution(model_id)
    engine = VllmEngineAdapter()
    assert driver.serve_command(engine, solution.plan, [], ()) == _expected_serve(model_id)
    recorded = driver.recorded_measurements(solution.fingerprint)
    assert recorded, "the H100 memory report is on record"
    assert {r["deployment_plan_digest"] for r in recorded} == {fingerprint_hex(solution.plan)}
    assert RECORDED[model_id] in {r["torch_peak_increase"] for r in recorded}


class FakePod:
    """Answers the engine adapter's SSH commands like a staged-weights pod."""

    weights_persist = True

    def __init__(self) -> None:
        self.commands: list[str] = []

    def execute(self, command: str, **_: Any) -> dict[str, Any]:
        self.commands.append(command)
        if command.startswith("du -sb"):
            return {"stdout": "62442977152\n270000000000\n", "exit_code": 0}
        return {"stdout": "", "exit_code": 0}  # /health passes at once


@needs_manifest
@pytest.mark.parametrize("model_id", sorted(RECORDED))
def test_the_probe_launch_is_the_cohort_launch_plus_the_probe_env(
    driver: Any, model_id: str
) -> None:
    from apron.adapters.backends.vllm_engine import VllmEngineAdapter

    engine = VllmEngineAdapter()
    solution = driver.recorded_solution(model_id)
    cohort_pod = FakePod()
    boot = engine.boot(solution.plan, cohort_pod, health_timeout=5)  # the cohort's path
    cohort_launch = cohort_pod.commands[1]  # after load_args' du probe

    probe_pod = FakePod()
    load = engine.load_args(probe_pod, model_id)
    assert load == ["--safetensors-load-strategy", "prefetch"]
    serve = driver.serve_command(engine, solution.plan, load, ())
    assert serve == boot.command
    env = driver.probe_env("/workspace/peak-probe-0.json", "/workspace/probe-cache-0")
    launch = driver.probe_launch(serve, env)
    assigns = " ".join(f"{k}={v}" for k, v in env.items())
    assert launch.replace(assigns + " ", "") == cohort_launch
    # The token removal and offline mode still come first.
    assert (
        "exec env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN HF_HUB_OFFLINE=1 APRON_PEAK_PROBE=1"
        in (launch)
    )
    assert "PYTHONPATH=/workspace/peak_probe" in launch


def test_variants_follow_their_model_and_keep_json_values(driver: Any) -> None:
    boots = driver.parse_boots(
        ["zai-org/GLM-4.7-Flash", "google/gemma-4-31B-it"],
        [
            'google/gemma-4-31B-it=--limit-mm-per-prompt \'{"image":0,"video":0,"audio":0}\'',
            "google/gemma-4-31B-it=--language-model-only",
        ],
    )
    assert [(b.model_id, b.extra) for b in boots] == [
        ("zai-org/GLM-4.7-Flash", ()),
        ("google/gemma-4-31B-it", ()),
        ("google/gemma-4-31B-it", ("--limit-mm-per-prompt", '{"image":0,"video":0,"audio":0}')),
        ("google/gemma-4-31B-it", ("--language-model-only",)),
    ]
    assert len({b.name for b in boots}) == 4
    with pytest.raises(SystemExit, match="not JSON"):
        driver.parse_boots(["m/x"], ['m/x=--limit-mm-per-prompt {"image":0}'])
    with pytest.raises(SystemExit, match="listed model"):
        driver.parse_boots(["m/x"], ["m/y=--language-model-only"])


def test_open_holds_and_the_estimate(driver: Any, tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    rows = [
        {"op": "hold", "label": "a", "amount": 1.0},
        {"op": "settle", "label": "a", "amount": 0.5},
        {"op": "hold", "label": "stage:b", "amount": 0.2},
    ]
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert driver.open_holds(ledger) == ["stage:b"]
    # 12 min provisioning + 10 min per boot, x1.5
    assert driver.estimate_usd(3.0, 2) == round(3.0 * 32 / 60 * 1.5, 4)


PROFILE_LOG = (
    "INFO vLLM API server version 0.29.0\n"
    "DEBUG Memory profiling takes 31.20 seconds. Total non KV cache memory: 63.53GiB; "
    "torch peak memory increase: 2.04GiB; total consumed (from mem_get_info): 61.49GiB; "
    "weights memory: 55.87GiB.\n"
    "INFO Available KV cache memory: 8.57 GiB\n"
)


class FakeRunPodTarget:
    """Stands in for RunPodTarget in ``main --run``: no API, no SSH, no money."""

    instances: ClassVar[list[FakeRunPodTarget]] = []
    weights_persist = True

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.pod_id: str | None = None
        self.commands: list[str] = []
        self.uploaded: dict[str, bytes] = {}
        self.torn_down = False
        FakeRunPodTarget.instances.append(self)

    @staticmethod
    def build_env(**kwargs: Any) -> dict[str, str]:
        return {"VLLM_LOGGING_LEVEL": "DEBUG"}

    def stock_status(self) -> str:
        return "High"

    def provision(self, env: dict[str, str], wait_timeout: int = 0) -> dict[str, Any]:
        self.pod_id = "pod-fake"
        return {"status": "provisioned"}

    def pod_reported_cost(self, pod_id: str | None = None) -> float | None:
        return 0.5 if (pod_id or self.pod_id) else None

    def teardown(self) -> None:
        self.torn_down = True
        self.pod_id = None

    def execute(self, command: str, **_: Any) -> dict[str, Any]:
        import base64

        self.commands.append(command)
        if "test -x /usr/local/bin/apron-download" in command:
            return {"stdout": "yes\n", "exit_code": 0}
        if "base64 -d" in command:
            payload, target = command.split("echo ", 1)[1].split(" | base64 -d > ")
            self.uploaded[target.strip()] = base64.b64decode(payload)
            return {"stdout": "", "exit_code": 0}
        if "/dev/tcp/" in command:
            return {"stdout": "0\n", "exit_code": 0}
        if command.startswith("nvidia-smi"):
            return {"stdout": "5\n", "exit_code": 0}
        if command.startswith("du -sb"):
            return {"stdout": "62442977152\n270000000000\n", "exit_code": 0}
        if command.startswith("test -s"):
            return {"stdout": "yes\n", "exit_code": 0}
        if command.startswith("grep -c"):
            return {"stdout": "1\n", "exit_code": 0}
        if command.startswith("for p in"):
            return {"stdout": "123 0\n", "exit_code": 0}
        if command.startswith("cat /var/log/vllm.log"):
            return {"stdout": PROFILE_LOG, "exit_code": 0}
        return {"stdout": "", "exit_code": 0}

    def collect(self, paths: list[str]) -> dict[str, Any]:
        doc = {
            "runs": [{"torch_peak_increase": 2190433320, "analysis": {"peak_increase_bytes": 1}}]
        }
        return {p: json.dumps(doc) for p in paths}


@needs_manifest
def test_a_run_holds_boots_collects_tears_down_and_settles(
    driver: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"op": "spend", "label": "x", "amount": 1.0, "at": "t"}) + "\n")
    FakeRunPodTarget.instances = []
    monkeypatch.setattr(driver, "RunPodTarget", FakeRunPodTarget)
    monkeypatch.setattr(driver, "LEDGER", ledger)
    monkeypatch.setattr(driver, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(driver, "live_rates", lambda key: {driver.GPU: 3.0})
    monkeypatch.setattr(driver, "_ssh_public_key", lambda: None)
    monkeypatch.setattr(driver, "install_log_masking", lambda: None)
    monkeypatch.setenv("RUNPOD_API_KEY", "not-a-real-key")
    monkeypatch.delenv("HF_TOKEN", raising=False)

    assert driver.main(["--run", "zai-org/GLM-4.7-Flash"]) == 0

    (pod,) = FakeRunPodTarget.instances
    assert pod.kwargs["network_volume_id"] == "7jiuum2nk6"
    assert pod.kwargs["data_center_id"] == "US-CA-2"
    from apron.adapters.runner_image import runner_image

    # the image the models were recorded on (v0.29.0 for GLM and Gemma)
    assert pod.kwargs["image"].endswith(runner_image("v0.29.0").digest)
    assert pod.torn_down
    for name in ("sitecustomize.py", "analyze.py"):
        assert pod.uploaded[f"/workspace/peak_probe/{name}"] == (PROBE_DIR / name).read_bytes()
    launches = [c for c in pod.commands if "APRON_PEAK_PROBE=1" in c]
    assert len(launches) == 1 and "--safetensors-load-strategy prefetch" in launches[0]

    ops = [json.loads(line) for line in ledger.read_text().splitlines()][1:]
    label = "probe:peak:zai-org/GLM-4.7-Flash"
    assert [(e["op"], e["label"]) for e in ops] == [
        ("hold", label),
        ("annotate", label),
        ("settle", label),
    ]
    assert ops[1]["pod_id"] == "pod-fake"
    assert ops[2]["amount"] == 0.5  # the provider-reported cost

    result = json.loads((tmp_path / "out" / "zai-org--GLM-4.7-Flash.json").read_text())
    assert result["status"] == "profiled"
    assert result["vllm_logged"]["torch_peak_increase"] == int(2.04 * 2**30)
    assert result["probe"]["runs"][0]["torch_peak_increase"] == 2190433320
    assert result["token_check"]["token_free"] is True
    assert 2190433320 in {r["torch_peak_increase"] for r in result["recorded"]}


class NoPodTarget(FakeRunPodTarget):
    def provision(self, env: dict[str, str], wait_timeout: int = 0) -> dict[str, Any]:
        raise RuntimeError("no capacity")


@needs_manifest
def test_a_pod_never_created_releases_the_hold(
    driver: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(driver, "RunPodTarget", NoPodTarget)
    monkeypatch.setattr(driver, "LEDGER", ledger)
    monkeypatch.setattr(driver, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(driver, "live_rates", lambda key: {driver.GPU: 3.0})
    monkeypatch.setattr(driver, "_ssh_public_key", lambda: None)
    monkeypatch.setattr(driver, "install_log_masking", lambda: None)
    monkeypatch.setenv("RUNPOD_API_KEY", "not-a-real-key")
    with pytest.raises(RuntimeError, match="no capacity"):
        driver.main(["--run", "google/gemma-4-31B-it"])
    ops = [json.loads(line)["op"] for line in ledger.read_text().splitlines()]
    assert ops == ["hold", "release"]


def test_each_model_keeps_the_image_it_was_recorded_on(
    driver: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One pod runs one image: the probe boots the engine a model was recorded
    on (GLM and Gemma on v0.29.0; Qwen3.6 and Muse-Glimmer on v0.30.0) and
    refuses to mix them on one pod."""
    from apron.adapters.runner_image import runner_image

    real = next(
        json.loads(line)
        for line in (REPO / "_dev_notes" / "cohort-run" / "solutions.jsonl").read_text().splitlines()
        if line.strip()
        and json.loads(line).get("model_id") == "zai-org/GLM-4.7-Flash"
        and json.loads(line)["requested_execution"]["gpu_sku"] == driver.GPU
    )
    rows = []
    for model, version in (("a/one", "v0.29.0"), ("b/two", "v0.30.0")):
        row = json.loads(json.dumps(real))
        row["model_id"] = model
        row["requested_execution"]["image_digest"] = runner_image(version).digest
        rows.append(json.dumps(row))
    manifest = tmp_path / "solutions.jsonl"
    manifest.write_text("\n".join(rows) + "\n")
    one = driver.recorded_solution("a/one", manifest=manifest)
    two = driver.recorded_solution("b/two", manifest=manifest)
    assert one.image_digest == runner_image("v0.29.0").digest
    assert two.image_digest == runner_image("v0.30.0").digest
    # Mixed images are refused before any pod exists (exit 5).
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    monkeypatch.setattr(driver, "open_holds", lambda _ledger: [])
    real_lookup = driver.recorded_solution
    monkeypatch.setattr(
        driver, "recorded_solution", lambda m, **_: real_lookup(m, manifest=manifest)
    )
    monkeypatch.setattr(driver, "RunPodTarget", _refuse_pod)
    assert driver.main(["--run", "a/one", "b/two"]) == 5


def _refuse_pod(*_a: Any, **_k: Any) -> Any:
    raise AssertionError("a pod was created for models recorded on different images")
