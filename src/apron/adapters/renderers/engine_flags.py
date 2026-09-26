"""Render a DeploymentPlan's engine_configuration as vLLM command-line flags.

vLLM parses boolean config fields with ``argparse.BooleanOptionalAction``
(``vllm/engine/arg_utils.py:387-389``): ``--flag`` / ``--no-flag``, never
``--flag true``.  The plan stores every value as a string, so ``"true"`` and
``"false"`` become the bare and negated flags; anything else is ``--flag value``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


def engine_flag_args(configuration: Mapping[str, str]) -> list[str]:
    args: list[str] = []
    for key, value in configuration.items():
        flag = key.replace("_", "-")
        text = str(value).strip().lower()
        if text == "true":
            args.append(f"--{flag}")
        elif text == "false":
            args.append(f"--no-{flag}")
        else:
            args.extend([f"--{flag}", str(value)])
    return args
