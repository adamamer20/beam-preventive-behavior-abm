"""Async perspective-framed survey processor.

Implements the plan's perspective framing (first-person vs third-person) prompting
strategy for the main survey processing pipeline.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from beam_abm.llm.processors.async_base import AsyncSurveyProcessor
from beam_abm.llm.schemas import ModelConfig

Perspective = Literal["first_person", "third_person"]


class AsyncFramedProcessor(AsyncSurveyProcessor):
    """Perspective framing strategy (first/third person)."""

    def __init__(
        self,
        *,
        perspective: Perspective,
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
        self.processor_name = f"async_framed::{perspective}"
        self.perspective: Perspective = perspective

    async def create_prompt(self, row: pd.Series) -> str:
        formatted_data = self._format_survey_data(row)

        if self.perspective == "first_person":
            framing = "Write predictions as if you are the respondent (first-person perspective)."
        else:
            framing = "Write predictions as an outside observer describing the respondent (third-person perspective)."

        specific = (
            "Based on the survey response data below, predict the respondent's likely answers to various survey questions "
            "about vaccine trust, opinions, and protective behaviors. Provide numeric predictions for each category using "
            "the appropriate scales.\n\n" + framing
        )

        return self._build_base_prompt(specific, formatted_data)
