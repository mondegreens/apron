"""§9.1 step 4: chat_template_kwargs only for templates that accept them."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from apron.adapters.evidence.hf_hub import FixtureHFHubResolver
from apron.application.orchestration.evidence import chat_template_kwargs
from apron.application.orchestration.plan_pipeline import _download_chat_template

# Excerpts of the published templates.
QWEN3_TEMPLATE = (
    "{%- if enable_thinking is defined and enable_thinking is false %}"
    "{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}"
)
MISTRAL_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "[INST] {{ message['content'] }} [/INST]{% endfor %}"
)

HF = Path(__file__).parents[1] / "fixtures" / "external-formats" / "huggingface-hub"


def test_qwen3_template_gets_enable_thinking_false() -> None:
    assert chat_template_kwargs(QWEN3_TEMPLATE) == {"enable_thinking": False}


def test_template_without_the_variable_gets_nothing() -> None:
    assert chat_template_kwargs(MISTRAL_TEMPLATE) is None
    assert chat_template_kwargs(None) is None


@pytest.fixture()
def hub(tmp_path: Path) -> Path:
    for name in ("config.json", "model_info.json", "model.safetensors.index.json"):
        shutil.copy(HF / name, tmp_path / name)
    return tmp_path


def test_template_from_tokenizer_config(hub: Path) -> None:
    (hub / "tokenizer_config.json").write_text(json.dumps({"chat_template": QWEN3_TEMPLATE}))
    template = _download_chat_template(FixtureHFHubResolver(hub), "Qwen/Qwen3-8B", "rev")
    assert template == QWEN3_TEMPLATE


def test_jinja_file_wins_and_named_list_picks_default(hub: Path) -> None:
    (hub / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": [
                    {"name": "tool_use", "template": "x"},
                    {"name": "default", "template": MISTRAL_TEMPLATE},
                ]
            }
        )
    )
    resolver = FixtureHFHubResolver(hub)
    assert _download_chat_template(resolver, "m", "rev") == MISTRAL_TEMPLATE
    (hub / "chat_template.jinja").write_text(QWEN3_TEMPLATE)
    assert _download_chat_template(resolver, "m", "rev") == QWEN3_TEMPLATE


def test_no_template_is_none(hub: Path) -> None:
    assert _download_chat_template(FixtureHFHubResolver(hub), "m", "rev") is None


def test_cli_sends_kwargs_only_for_accepting_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apron.adapters.backends.local_store import LocalRecordStore
    from apron.adapters.evaluations import deterministic_scorer
    from apron.domain.schemas.authority import DecisionRequest
    from apron.interfaces import cli

    seen: list[dict] = []

    def fake_execute(self, protocol, endpoint):  # type: ignore[no-untyped-def]
        seen.append(protocol)
        return []

    monkeypatch.setattr(deterministic_scorer.DeterministicScorer, "execute", fake_execute)

    class _Target:
        proxy_url = "https://p-8000.proxy.runpod.net"

    run = Path(__file__).parents[1] / "fixtures" / "phase-1a-run"
    for template in (QWEN3_TEMPLATE, MISTRAL_TEMPLATE):
        cli._run_task_suite(
            run / "task-suite-spec.json",
            request=DecisionRequest(objective="o"),
            model_id="m",
            solution_fp="1220" + "ab" * 32,
            target=_Target(),
            store=LocalRecordStore(tmp_path),
            report_digest="1220" + "cd" * 32,
            chat_template=template,
        )
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template_kwargs" not in seen[1]
