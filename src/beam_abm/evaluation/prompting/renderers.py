"""Prompt renderer composition mixin."""

from __future__ import annotations

from .renderers_narrative import PromptNarrativeRendererMixin
from .renderers_primitives import PromptRendererPrimitivesMixin


class PromptRendererMixin(PromptNarrativeRendererMixin, PromptRendererPrimitivesMixin):
    """Composed prompt renderer that groups primitives and narrative synthesis."""


__all__ = ["PromptRendererMixin"]
