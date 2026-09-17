"""DockerCompose renderer — renders a RenderContext to a docker-compose dict."""

from typing import Any

from apron.domain.schemas.solutions import RenderContext


class DockerComposeRenderer:
    @property
    def target_format(self) -> str:
        return "docker_compose"

    def render(self, context: RenderContext) -> dict[str, Any]:
        vllm_args = [context.locator.uri]
        vllm_args.extend(["--dtype", context.plan.dtype or "auto"])
        if context.plan.tensor_parallel > 1:
            vllm_args.extend(["--tensor-parallel-size", str(context.plan.tensor_parallel)])
        for key, value in context.plan.engine_configuration.items():
            vllm_args.extend([f"--{key.replace('_', '-')}", str(value)])

        gpu_count = context.plan.tensor_parallel
        compose: dict[str, Any] = {
            "services": {
                "vllm": {
                    "image": "vllm/vllm-openai:latest",
                    "command": ["vllm", "serve", *vllm_args],
                    "volumes": ["/models:/models:ro"],
                    "ports": ["8000:8000"],
                    "deploy": {
                        "resources": {
                            "reservations": {
                                "devices": [
                                    {
                                        "driver": "nvidia",
                                        "count": gpu_count,
                                        "capabilities": ["gpu"],
                                    }
                                ]
                            }
                        }
                    },
                }
            }
        }
        return {"format": "docker_compose", "compose": compose}

    def parse(self, data: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        compose = data.get("compose", {})
        vllm_svc = compose.get("services", {}).get("vllm", {})
        cmd = vllm_svc.get("command", [])
        i = 0
        while i < len(cmd):
            if cmd[i] == "--dtype" and i + 1 < len(cmd):
                result["dtype"] = cmd[i + 1]
                i += 2
            elif cmd[i] == "--tensor-parallel-size" and i + 1 < len(cmd):
                result["tensor_parallel"] = int(cmd[i + 1])
                i += 2
            else:
                i += 1
        if "tensor_parallel" not in result:
            result["tensor_parallel"] = 1
        return result
