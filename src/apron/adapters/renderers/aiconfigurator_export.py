"""AIConfigurator export renderer — RenderTarget Protocol implementation."""

from typing import Any

from apron.adapters.renderers import (
    AICONFIGURATOR_SHARED_FIELDS,
    _extract_shared,
    import_from_aiconfigurator,
)
from apron.domain.schemas.solutions import RenderContext


class AIConfiguratorExportRenderer:
    @property
    def target_format(self) -> str:
        return "aiconfigurator"

    def render(self, context: RenderContext) -> dict[str, Any]:
        exported = _extract_shared(context.plan, AICONFIGURATOR_SHARED_FIELDS)
        exported["model.path"] = context.locator.uri
        if context.execution_spec is not None:
            exported["backend.name"] = "vllm"
        if context.hardware is not None:
            exported["systems.prefill"] = context.hardware.gpu_sku
        return exported

    def parse(self, data: dict[str, Any]) -> dict[str, Any]:
        return import_from_aiconfigurator(data)
