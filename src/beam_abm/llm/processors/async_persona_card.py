"""Async persona-card survey processor.

Implements the plan's narrative persona-card prompting strategy for the main
survey processing pipeline.
"""

from __future__ import annotations

import pandas as pd

from beam_abm.llm.processors.async_base import AsyncSurveyProcessor
from beam_abm.llm.schemas import ModelConfig


class AsyncPersonaCardProcessor(AsyncSurveyProcessor):
    """Narrative persona-card prompting strategy."""

    def __init__(
        self,
        model_name: str | None = None,
        model: ModelConfig | None = None,
        conversation_logger=None,
        normalized_data_manager=None,
        start_time: float | None = None,
    ):
        if model is None and model_name is None:
            raise ValueError("Provide either model_name or model")
        model_config = model or ModelConfig(name=str(model_name))

        super().__init__(
            model=model_config,
            use_structured_output=True,
            conversation_logger=conversation_logger,
            normalized_data_manager=normalized_data_manager,
            start_time=start_time,
        )
        self.processor_name = "async_persona_card"

    async def create_prompt(self, row: pd.Series) -> str:
        # Turn key/value survey attributes into a compact narrative persona card.
        items = [f"{k}={v}" for k, v in row.items()]
        persona = (
            "Persona card (compact):\n" + "; ".join(items[:50]) + ("; ..." if len(items) > 50 else "") + "\n\n"
            "Use the persona card to infer likely survey answers."
        )

        specific = (
            "Based on the persona card below, predict the respondent's likely answers to various survey questions "
            "about vaccine trust, opinions, and protective behaviors. Provide numeric predictions for each category "
            "using the appropriate scales."
        )

        return self._build_base_prompt(f"{specific}\n\n{persona}", survey_data="")
