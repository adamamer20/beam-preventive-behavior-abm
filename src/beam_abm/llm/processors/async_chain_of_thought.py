"""Async chain-of-thought survey processor for concurrent vLLM processing."""

import pandas as pd

from beam_abm.common.logging import get_logger
from beam_abm.llm.processors.async_base import AsyncSurveyProcessor
from beam_abm.llm.schemas import ModelConfig

logger = get_logger(__name__)


class AsyncChainOfThoughtProcessor(AsyncSurveyProcessor):
    """Async chain-of-thought processor that uses concurrent vLLM processing."""

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
            use_structured_output=True,  # Enable guided decoding with thinking field
            conversation_logger=conversation_logger,
            enable_thinking=True,  # CoT always uses thinking
            normalized_data_manager=normalized_data_manager,
            start_time=start_time,
        )
        self.processor_name = "async_chain_of_thought"

    def get_response_model(self):
        """CoT always uses thinking model."""
        return self._get_thinking_response_model()

    async def create_prompt(self, row: pd.Series) -> str:
        """Create chain-of-thought prompt for detailed reasoning using thinking field."""
        # Format the row data using base method
        formatted_data = self._format_survey_data(row)

        # CoT always uses thinking field for reasoning
        cot_prompt = (
            "Respond in JSON format. FIRST write your reasoning in the 'thinking' field (max 500 words), THEN provide numeric predictions.\n\n"
            "In the 'thinking' field, construct a concise psychological and behavioral profile by considering:\n"
            " - Key beliefs, attitudes, and biases regarding vaccination and epidemic-related behavior\n"
            " - How responses reflect personal experiences, social influences, or risk perceptions\n"
            " - Important contextual factors that influenced their answers\n\n"
            "Keep reasoning focused and concise (4-5 sentences). Start with reasoning in the 'thinking' field, then fill all other prediction fields."
        )

        return self._build_base_prompt(cot_prompt, formatted_data)
