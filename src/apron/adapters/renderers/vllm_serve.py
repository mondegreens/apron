"""VllmServe renderer — renders a RenderContext to a vllm serve command."""

from typing import Any

from apron.adapters.renderers.engine_flags import engine_flag_args
from apron.domain.schemas.solutions import RenderContext


class VllmServeRenderer:
    @property
    def target_format(self) -> str:
        return "vllm_serve"

    def render(self, context: RenderContext) -> dict[str, Any]:
        args = [context.locator.uri]
        args.extend(["--dtype", context.plan.dtype or "auto"])
        if context.plan.tensor_parallel > 1:
            args.extend(["--tensor-parallel-size", str(context.plan.tensor_parallel)])
        args.extend(engine_flag_args(context.plan.engine_configuration))
        return {
            "format": "vllm_serve",
            "command": "vllm serve " + " ".join(args),
            "args": args,
        }

    def parse(self, data: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        args = data.get("args", [])
        i = 0
        while i < len(args):
            if args[i] == "--dtype" and i + 1 < len(args):
                result["dtype"] = args[i + 1]
                i += 2
            elif args[i] == "--tensor-parallel-size" and i + 1 < len(args):
                result["tensor_parallel"] = int(args[i + 1])
                i += 2
            else:
                i += 1
        if "tensor_parallel" not in result:
            result["tensor_parallel"] = 1
        return result
