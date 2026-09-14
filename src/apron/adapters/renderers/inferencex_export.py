"""InferenceX export renderer — RenderTarget Protocol implementation."""

from typing import Any

from apron.adapters.renderers import (
    INFERENCEX_SHARED_FIELDS,
    _extract_shared,
    import_from_inferencex,
)
from apron.domain.schemas.solutions import RenderContext


class InferenceXExportRenderer:
    @property
    def target_format(self) -> str:
        return "inferencex"

    def render(self, context: RenderContext) -> dict[str, Any]:
        exported = _extract_shared(context.plan, INFERENCEX_SHARED_FIELDS)
        exported["model"] = context.locator.uri
        if context.hardware is not None:
            exported["hw"] = context.hardware.gpu_sku
        if context.execution_spec is not None:
            exported["framework"] = "vllm"
        return exported

    def parse(self, data: dict[str, Any]) -> dict[str, Any]:
        return import_from_inferencex(data)
