"""Tests for the CLI commands."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import apron.domain.mechanisms.calculator  # noqa: F401
from apron.interfaces.cli import app

runner = CliRunner()
FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "external-formats" / "huggingface-hub"


@pytest.fixture(autouse=True)
def _ensure_calculator_registered():
    from apron.domain.mechanisms import _CALCULATOR_REGISTRY
    from apron.domain.mechanisms.calculator import calculate_autoregressive_decode

    _CALCULATOR_REGISTRY["autoregressive_decode"] = calculate_autoregressive_decode
    yield


@pytest.fixture()
def request_file(tmp_path: Path) -> Path:
    req = {
        "objective": "Deploy Qwen3-8B for chat",
        "quality_floor": 0.7,
    }
    p = tmp_path / "request.json"
    p.write_text(json.dumps(req))
    return p


def test_plan_with_fixture(request_file: Path, tmp_path: Path):
    output_dir = tmp_path / "output"
    result = runner.invoke(
        app,
        [
            "plan",
            str(request_file),
            "--model",
            "Qwen/Qwen3-8B",
            "--output-dir",
            str(output_dir),
            "--fixture-dir",
            str(FIXTURE_DIR),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (output_dir / "deployment-plan.json").exists()
    plan = json.loads((output_dir / "deployment-plan.json").read_text())
    assert plan["tensor_parallel"] >= 1
    assert plan["dtype"] is not None


def test_plan_verbose(request_file: Path, tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "plan",
            str(request_file),
            "--model",
            "Qwen/Qwen3-8B",
            "--output-dir",
            str(tmp_path),
            "--fixture-dir",
            str(FIXTURE_DIR),
            "--verbose",
        ],
    )
    assert result.exit_code == 0
    assert "weight_memory_bytes" in result.output


def test_plan_missing_request_file(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "plan",
            str(tmp_path / "nonexistent.json"),
            "--model",
            "Qwen/Qwen3-8B",
        ],
    )
    assert result.exit_code != 0


def test_verify_missing_plan_file(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "verify",
            str(tmp_path / "nonexistent.json"),
        ],
    )
    assert result.exit_code != 0


def test_run_echo():
    result = runner.invoke(app, ["run", "echo", "hello"])
    assert result.exit_code == 0
    assert "hello" in result.output


def test_plan_produces_rendered_outputs(request_file: Path, tmp_path: Path):
    output_dir = tmp_path / "output"
    runner.invoke(
        app,
        [
            "plan",
            str(request_file),
            "--model",
            "Qwen/Qwen3-8B",
            "--output-dir",
            str(output_dir),
            "--fixture-dir",
            str(FIXTURE_DIR),
        ],
    )
    renders_path = output_dir / "rendered-outputs.json"
    assert renders_path.exists()
    rendered = json.loads(renders_path.read_text())
    assert "vllm_serve" in rendered
    assert "docker_compose" in rendered


def test_verify_displays_cost_and_deadline(tmp_path: Path):
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(
        json.dumps(
            {
                "tensor_parallel": 1,
                "dtype": "bfloat16",
                "batch_size": 8,
                "resource_allocation": {"gpu_sku": "RTX 4090", "weight_bytes": "16381470720"},
            }
        )
    )
    result = runner.invoke(app, ["verify", str(plan_file), "--yes"])
    assert result.exit_code == 0, result.output
    assert "Estimated cost" in result.output
    assert "Hard deadline" in result.output
    assert "Data destination" in result.output


def test_verify_prompts_without_yes(tmp_path: Path):
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps({"tensor_parallel": 1}))
    result = runner.invoke(app, ["verify", str(plan_file)], input="n\n")
    assert result.exit_code != 0


def test_submit_requires_confirmation():
    result = runner.invoke(
        app,
        [
            "submit",
            "nonexistent",
            "--destination",
            "test",
        ],
    )
    assert result.exit_code != 0


def test_report_missing_record():
    result = runner.invoke(app, ["report", "nonexistent-digest"])
    assert result.exit_code != 0
