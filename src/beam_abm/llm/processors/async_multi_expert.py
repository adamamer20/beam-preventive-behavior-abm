"""Async multi-expert delta processor for concurrent vLLM processing."""

import json

import pandas as pd
from pydantic import Field, create_model

from beam_abm.common.logging import get_logger
from beam_abm.llm.processors.async_base import AsyncSurveyProcessor
from beam_abm.llm.schemas import ModelConfig
from beam_abm.llm.schemas.survey import SurveyPrediction

logger = get_logger(__name__)


class AsyncMultiExpertProcessor(AsyncSurveyProcessor):
    """Async multi-expert processor that returns delta-only experts and aggregated predictions."""

    def __init__(
        self,
        model_name: str | None = None,
        model: ModelConfig | None = None,
        conversation_logger=None,
        enable_thinking: bool = True,
        normalized_data_manager=None,
        start_time: float | None = None,
    ):
        # Prefer an explicit ModelConfig so callers can select backend.
        if model is not None:
            model_config = model
        elif model_name is not None:
            model_config = ModelConfig(name=model_name)
        else:
            raise ValueError("Either model_name or model must be provided")

        super().__init__(
            model=model_config,
            use_structured_output=True,  # Enable guided decoding with thinking field
            conversation_logger=conversation_logger,
            enable_thinking=enable_thinking,
            normalized_data_manager=normalized_data_manager,
            start_time=start_time,
        )
        self.processor_name = "async_multi_expert"

        # Initialize expert prompts - delta-only experts
        self.expert_prompts = {
            "behavioral_psychologist": (
                "You are a Behavioral Psychologist specializing in health behavior. "
                "Focus on cognition, risk perception, and decision heuristics."
            ),
            "epidemiologist": (
                "You are an Epidemiologist and Public Health Expert. "
                "Focus on disease-risk context and population health signals."
            ),
            "social_influence": (
                "You are a Social Influence and Communication Specialist. "
                "Focus on social norms, trust, and information channels."
            ),
        }

    def get_response_model(self):
        """Multi-expert always uses thinking model."""
        return self._get_thinking_response_model()

    async def create_prompt(self, row: pd.Series) -> str:
        """Create prompt for delta-only experts."""
        formatted_data = self._format_survey_data(row)
        return self._build_base_prompt(
            "Provide ONLY a delta adjustment to the baseline prediction. Return JSON only.",
            formatted_data,
        )

    def _delta_response_model(self):
        return create_model(
            "ExpertDelta",
            delta_raw=(float, Field(..., description="Additive adjustment to baseline")),
            mechanism=(str, Field(..., description="Short mechanism summary")),
            fields_used=(list[str], Field(..., description="Attribute keys used for the delta")),
        )

    async def _run_delta_experts(self, row: pd.Series) -> dict[str, object] | None:
        formatted_data = self._format_survey_data(row)
        baseline_y0 = row.get("baseline_y0")
        try:
            baseline_y0 = float(baseline_y0)
        except (TypeError, ValueError):
            logger.error("baseline_y0 missing or invalid for multi-expert delta aggregation")
            return None

        expert_requests = []
        delta_model = self._delta_response_model()
        for name, expert_prompt in self.expert_prompts.items():
            system_prompt = self._build_base_prompt(
                f"{expert_prompt}\n\nYou are one of several experts. Provide ONLY the delta adjustment to the baseline prediction.",
                formatted_data,
            )
            human_prompt = "Return ONLY JSON with keys: delta_raw, mechanism, fields_used."
            expert_requests.append(
                {
                    "model": self.model,
                    "messages": [
                        {"from": "system", "value": system_prompt},
                        {"from": "human", "value": human_prompt},
                    ],
                    "temperature": 0.1,
                    "use_structured_output": True,
                    "response_model": delta_model,
                    "enable_thinking": self.enable_thinking,
                    "method": f"multi_expert::expert::{name}",
                }
            )

        expert_responses = await self.backend.generate_responses_concurrent(expert_requests)
        expert_deltas: dict[str, dict[str, object] | None] = {}
        deltas: list[float] = []
        for name, text in zip(self.expert_prompts.keys(), expert_responses, strict=False):
            try:
                parsed = json.loads(text)
                expert_deltas[name] = parsed
                delta_raw = parsed.get("delta_raw") if isinstance(parsed, dict) else None
                deltas.append(float(delta_raw))
            except (json.JSONDecodeError, TypeError, ValueError):
                expert_deltas[name] = None

        if len(deltas) != len(self.expert_prompts):
            return {
                "baseline_y0": baseline_y0,
                "expert_deltas": expert_deltas,
                "y_pred": None,
            }

        return {
            "baseline_y0": baseline_y0,
            "expert_deltas": expert_deltas,
            "y_pred": max(0.0, min(1.0, baseline_y0 + sum(deltas))),
        }

    async def _process_single_record_async(
        self, record: dict[str, object], _survey_prompt: str, record_index: int
    ) -> dict[str, object]:
        """Process a single record using delta-only experts.

        Returns a placeholder prediction to avoid SurveyPrediction parsing.
        """
        row = pd.Series(record)
        result = await self._run_delta_experts(row)
        return {
            "record_index": record_index,
            "prediction": None,
            "baseline_y0": None if result is None else result.get("baseline_y0"),
            "expert_deltas": None if result is None else result.get("expert_deltas"),
            "y_pred": None if result is None else result.get("y_pred"),
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    async def parse_response(self, _response: str, _row: pd.Series) -> SurveyPrediction:  # pragma: no cover
        raise NotImplementedError("AsyncMultiExpertProcessor does not parse SurveyPrediction outputs.")

    async def process_batch_concurrent(self, data: pd.DataFrame) -> list[dict[str, object] | None]:
        """Process a batch of data with multi-expert delta analysis."""
        if data.empty:
            return []

        results = []

        for _, row in data.iterrows():
            try:
                result = await self._run_delta_experts(row)
                results.append(result)
            except (RuntimeError, TypeError, ValueError) as e:
                logger.error(f"Error processing row in multi-expert delta analysis: {e}")
                results.append(None)

        return results
