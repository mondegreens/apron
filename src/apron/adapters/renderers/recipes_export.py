"""Recipes YAML export renderer — RenderTarget Protocol implementation."""

from typing import Any

from apron.adapters.renderers import (
    RECIPES_SHARED_FIELDS,
    _extract_shared,
    import_from_recipes,
)
from apron.domain.schemas.solutions import RenderContext


class RecipesExportRenderer:
    @property
    def target_format(self) -> str:
        return "recipes_yaml"

    def render(self, context: RenderContext) -> dict[str, Any]:
        exported = _extract_shared(context.plan, RECIPES_SHARED_FIELDS)
        exported["model_id"] = context.locator.uri
        return exported

    def parse(self, data: dict[str, Any]) -> dict[str, Any]:
        return import_from_recipes(data)
