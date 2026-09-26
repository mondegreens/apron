"""F7 / INV-13: the HF token reaches the download step only, never vLLM.

GPU-free checks of the commands and the runner image script; the live
``/proc/<pid>/environ`` check runs on the first GPU boot (L0) and its output
is kept as evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from apron.adapters.backends.runpod import RunPodTarget
from apron.adapters.backends.vllm_engine import VllmEngineAdapter
from apron.domain.schemas.solutions import DeploymentPlan

START_SH = Path(__file__).parents[2] / "docker" / "start.sh"
DOWNLOAD = Path(__file__).parents[2] / "docker" / "apron-download"
_TOKEN = "hf_" + "Q" * 34


class _Recorder:
    def __init__(self, replies: dict[str, dict[str, Any]] | None = None) -> None:
        self.commands: list[str] = []
        self._replies = replies or {}

    def execute(self, command: str) -> dict[str, Any]:
        self.commands.append(command)
        for needle, reply in self._replies.items():
            if needle in command:
                return reply
        return {"stdout": "", "stderr": "", "exit_code": 0}


_PLAN = DeploymentPlan(
    dtype="bfloat16",
    engine_configuration={"max_model_len": "640"},
    resource_allocation={"model_id": "meta-llama/Llama-3.1-8B-Instruct"},
)


def test_boot_serves_local_path_with_tokens_unset() -> None:
    target = _Recorder({"curl -sf": {"exit_code": 0}})
    result = VllmEngineAdapter().boot(_PLAN, target, health_timeout=5)
    launch = target.commands[0]
    assert result.healthy
    assert "env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN HF_HUB_OFFLINE=1" in launch
    assert "vllm serve /workspace/models/meta-llama/Llama-3.1-8B-Instruct" in launch
    assert "--served-model-name meta-llama/Llama-3.1-8B-Instruct" in launch
    assert "hf_" not in launch


def test_download_reads_token_file_and_never_carries_the_value() -> None:
    target = _Recorder()
    VllmEngineAdapter().download_weights(target, "google/gemma-2-2b-it")
    command = target.commands[0]
    assert "/run/apron/hf_token" in command
    assert "hf_" not in command.replace("hf_token", "")
    assert "env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN" in command


def test_failed_boot_returns_log_tail() -> None:
    target = _Recorder(
        {
            "curl -sf": {"exit_code": 7},
            "pgrep": {"stdout": "down\n"},
            "tail -200": {"stdout": "ValueError: Numerical instability. Please use bfloat16"},
        }
    )
    result = VllmEngineAdapter().boot(_PLAN, target, health_timeout=30)
    assert not result.healthy
    assert "Numerical instability" in result.log_tail


def test_token_environ_check_reports_names_not_values() -> None:
    clean = VllmEngineAdapter().token_environ_check(
        _Recorder({"environ": {"stdout": "11 0\n12 0\n"}})
    )
    leaked = VllmEngineAdapter().token_environ_check(_Recorder({"environ": {"stdout": "11 1\n"}}))
    missing = VllmEngineAdapter().token_environ_check(_Recorder())
    assert clean["token_free"] is True and clean["processes"] == {11: 0, 12: 0}
    assert leaked["token_free"] is False
    assert missing["token_free"] is False  # no vLLM process is not evidence of absence
    assert "cut -d= -f1" in clean["command"]  # only variable names are read


def test_process_pattern_cannot_match_its_own_shell() -> None:
    import re

    from apron.adapters.backends.vllm_engine import VLLM_PROCESS_PATTERN

    own_command = f"pgrep -f '{VLLM_PROCESS_PATTERN}'"
    assert not re.search(VLLM_PROCESS_PATTERN, own_command)
    assert re.search(VLLM_PROCESS_PATTERN, "/opt/venv/bin/python3 /opt/venv/bin/vllm serve m")
    assert re.search(VLLM_PROCESS_PATTERN, "VLLM::EngineCore")


def test_build_env_ssh_mode_carries_token_only_for_download() -> None:
    env = RunPodTarget.build_env(hf_token=_TOKEN)
    assert env["HF_TOKEN"] == _TOKEN
    assert "VLLM_MODEL" not in env  # SSH-driven boot: the image waits
    assert "HF_TOKEN" not in RunPodTarget.build_env()


def test_start_sh_moves_token_to_file_and_out_of_ssh_environment() -> None:
    script = START_SH.read_text()
    to_file = script.index("/run/apron/hf_token")
    unset = script.index("unset HF_TOKEN HUGGING_FACE_HUB_TOKEN")
    export = script.index("/etc/apron_environment")
    assert to_file < unset < export
    assert "grep -v -E '^(HF_TOKEN|HUGGING_FACE_HUB_TOKEN)='" in script
    assert "env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN HF_HUB_OFFLINE=1" in script
    assert 'export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"' not in script


def test_apron_download_is_the_only_token_reader() -> None:
    import re

    script = DOWNLOAD.read_text()
    assert "/run/apron/hf_token" in script
    assert not re.search(r"^\s*export\s", script, re.MULTILINE)


def test_runner_image_capability_check() -> None:
    yes = _Recorder({"apron-download": {"stdout": "yes\n"}})
    no = _Recorder({"apron-download": {"stdout": "no\n"}})
    assert VllmEngineAdapter.runner_supports_token_isolation(yes) is True
    assert VllmEngineAdapter.runner_supports_token_isolation(no) is False


def test_download_skips_native_duplicates_in_both_paths() -> None:
    from apron.adapters.backends.vllm_engine import DOWNLOAD_IGNORE

    target = _Recorder()
    VllmEngineAdapter().download_weights(target, "mistralai/Mistral-7B-Instruct-v0.3")
    for pattern in DOWNLOAD_IGNORE:
        assert pattern in target.commands[0]
        assert pattern in DOWNLOAD.read_text()
