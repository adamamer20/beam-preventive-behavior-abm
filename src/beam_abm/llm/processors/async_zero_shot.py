"""Async zero-shot survey processor for concurrent vLLM processing."""

import re

import pandas as pd

from beam_abm.common.logging import get_logger
from beam_abm.llm.processors.async_base import AsyncSurveyProcessor
from beam_abm.llm.schemas import ModelConfig
from beam_abm.llm.schemas.survey import SurveyPrediction

logger = get_logger(__name__)


class AsyncZeroShotProcessor(AsyncSurveyProcessor):
    """Async zero-shot processor that uses concurrent vLLM processing."""

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
        self.processor_name = "async_zero_shot"

    async def create_prompt(self, row: pd.Series) -> str:
        """Create zero-shot prompt for survey response prediction."""
        # Format the row data using base method
        formatted_data = self._format_survey_data(row)

        # Adapt prompt based on thinking mode
        if self.enable_thinking:
            specific_content = """Based on the following survey response data, predict the respondent's likely answers to various survey questions about vaccine trust, opinions, and protective behaviors.

Provide numeric predictions for each category using the appropriate scales."""
        else:
            specific_content = """Based on the following survey response data, predict the respondent's likely answers to various survey questions about vaccine trust, opinions, and protective behaviors.

Please provide numeric predictions for each category using the appropriate scales."""

        return self._build_base_prompt(specific_content, formatted_data)

    async def parse_response(self, response: str, row: pd.Series) -> SurveyPrediction:
        """Parse the model response into a SurveyPrediction."""
        # Initialize result dictionary - if parsing fails, this will cause the method to fail
        # and the base class will handle it by returning pd.NA in the results
        result = {}

        # Try to extract numeric predictions from the response
        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            if ":" in line:
                key_part, value_part = line.split(":", 1)
                key = key_part.strip().lower()
                value = value_part.strip()

                # Try to extract numeric values
                try:
                    # Look for numbers in the value
                    numbers = re.findall(r"\d+", value)
                    if numbers:
                        numeric_value = int(numbers[0])

                        # Map to field names
                        for field_name in SurveyPrediction.model_fields.keys():
                            if field_name.lower() in key or any(part in key for part in field_name.lower().split("_")):
                                result[field_name] = numeric_value
                                break
                except ValueError:
                    continue

        # If we couldn't extract all required fields, this will fail validation
        # and the base class will handle it by returning pd.NA
        return SurveyPrediction(**result)
