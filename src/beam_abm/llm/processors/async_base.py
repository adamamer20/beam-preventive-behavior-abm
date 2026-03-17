"""Async base survey processor class for concurrent vLLM processing."""

import asyncio
import json
import time
from typing import Any

import pandas as pd

from beam_abm.common.logging import get_logger
from beam_abm.llm.backends.factory import get_backend
from beam_abm.llm.config import get_llm_runtime_settings
from beam_abm.llm.schemas import ModelConfig, SurveyPrediction

logger = get_logger(__name__)


class AsyncSurveyProcessor:
    """Async base class for survey processing strategies using vLLM AsyncLLMEngine."""

    def __init__(
        self,
        model: ModelConfig,
        use_structured_output: bool = False,
        conversation_logger=None,
        enable_thinking: bool = False,
        normalized_data_manager=None,
        start_time: float | None = None,
    ):
        """Initialize the async processor.

        Parameters
        ----------
        model : ModelConfig
            The model configuration
        conversation_logger : ConversationLogger, optional
            Logger for storing full conversations
        enable_thinking : bool, optional
            Whether to enable thinking mode for Qwen3 models
        normalized_data_manager : NormalizedDataManager, optional
            Data manager for normalized data organization
        start_time : float, optional
            Processing start time (defaults to current time)
        """
        self.model = model
        logger.info(f"Initializing {self.__class__.__name__} with model: {self.model.name}")
        self.use_structured_output = use_structured_output
        self.conversation_logger = conversation_logger
        self.enable_thinking = enable_thinking
        self.normalized_data_manager = normalized_data_manager
        self.start_time = start_time if start_time is not None else time.time()
        # Will be set when we get engine info
        self._vllm_max_seqs = None

        self.backend = get_backend(self.model)
        # Keep existing ConversationLogger behavior across backends.
        if hasattr(self.backend, "set_conversation_logger"):
            self.backend.set_conversation_logger(self.conversation_logger)

    def get_response_model(self):
        """Get the Pydantic model for structured output.

        Can be overridden by subclasses to use different models.
        """
        return SurveyPrediction

    def _format_survey_data(self, row: pd.Series) -> str:
        """Format survey row data into a standardized string format.

        NOTE: This function is NOT redundant with load_and_process_survey_data().
        - _format_survey_data: Formats individual rows (pd.Series) into strings for LLM consumption
        - load_and_process_survey_data: Loads and preprocesses entire datasets (pd.DataFrame) from CSV files
        These serve different purposes in the pipeline.

        Parameters
        ----------
        row : pd.Series
            Survey row data

        Returns
        -------
        str
            Formatted survey data string
        """
        return "\n".join([f"{key}: {value}" for key, value in row.items()])

    def _build_base_prompt(self, specific_content: str, survey_data: str) -> str:
        """Build a complete prompt combining base prompt with specific content.

        Parameters
        ----------
        specific_content : str
            Processor-specific prompt content
        survey_data : str
            Formatted survey data

        Returns
        -------
        str
            Complete prompt
        """
        base_prompt = get_llm_runtime_settings().base_prompt
        return f"{base_prompt}\n\n{specific_content}\n\nSurvey Response Data:\n{survey_data}"

    def _get_thinking_response_model(self):
        """Get the response model with thinking field.

        Returns
        -------
        type
            SurveyPredictionWithThinking class
        """
        from beam_abm.llm.schemas.survey import SurveyPredictionWithThinking

        return SurveyPredictionWithThinking

    async def process_records_async(
        self, survey_prompt: str, survey_answers: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Process multiple survey records concurrently using optimal vLLM batching.

        Parameters
        ----------
        survey_prompt : str
            The survey prompt template
        survey_answers : pd.DataFrame
            The survey answers to process

        Returns
        -------
        tuple[pd.DataFrame, dict[str, int]]
            Processed results and token usage statistics
        """
        start_time = time.time()
        total_records = len(survey_answers)
        logger.info(f"Starting optimized async processing of {total_records} records")

        # Convert dataframe to list of records
        records = survey_answers.to_dict(orient="records")

        # TODO: to be set from model_config!
        batch_size = 48

        # Get optimal batch size based on vLLM capacity
        logger.info(f"Using batch size: {batch_size} (aligns with vLLM capacity)")

        results = []
        token_stats = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        # Process in chunks that match vLLM's optimal capacity
        num_batches = (total_records + batch_size - 1) // batch_size
        for i in range(0, total_records, batch_size):
            batch = records[i : i + batch_size]
            batch_num = i // batch_size + 1

            logger.info(f"Processing batch {batch_num}/{num_batches} with {len(batch)} records")

            # Create all tasks for this batch - no artificial semaphore constraint
            tasks = [
                self._process_single_record_async(record, survey_prompt, idx + i) for idx, record in enumerate(batch)
            ]

            # Process entire batch concurrently - let vLLM optimize internally
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect results and handle exceptions
            for idx, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error processing record {i + idx}: {result}")
                    # Create error placeholder
                    error_result = {
                        "record_index": i + idx,
                        "prediction": None,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "error": str(result),
                    }
                    results.append(error_result)
                else:
                    results.append(result)
                    # Accumulate token stats
                    if "input_tokens" in result:
                        token_stats["input_tokens"] += result["input_tokens"]
                        token_stats["output_tokens"] += result["output_tokens"]
                        token_stats["total_tokens"] += result["total_tokens"]

            logger.info(f"Batch {batch_num}/{num_batches} completed")

        processing_time = time.time() - start_time
        avg_time_per_record = processing_time / total_records if total_records > 0 else 0
        logger.info(f"Optimized async processing completed in {processing_time:.2f}s")
        logger.info(f"Average time per record: {avg_time_per_record:.3f}s")
        logger.info(
            f"Token usage - Input: {token_stats['input_tokens']}, Output: {token_stats['output_tokens']}, Total: {token_stats['total_tokens']}"
        )

        # Convert results to DataFrame
        predictions_df = self._results_to_dataframe(results, survey_answers)

        return predictions_df, token_stats

    async def _process_single_record_async(
        self, record: dict[str, Any], survey_prompt: str, record_index: int
    ) -> dict[str, Any]:
        """Process a single survey record asynchronously.

        This method should be implemented by subclasses.

        Parameters
        ----------
        record : dict[str, Any]
            Individual survey record
        survey_prompt : str
            The survey prompt template
        record_index : int
            Index of the record being processed

        Returns
        -------
        dict[str, Any]
            Processing result with prediction and token usage
        """
        try:
            # Convert record to Series for compatibility with create_prompt
            row = pd.Series(record)

            # Create prompt using the subclass implementation
            prompt = await self.create_prompt(row)

            # Log prompt length for token analysis
            prompt_length = len(prompt)
            if prompt_length > 10000:  # Long prompts
                logger.warning(f"Very long prompt for record {record_index}: {prompt_length} characters")
            elif prompt_length > 5000:
                logger.info(f"Long prompt for record {record_index}: {prompt_length} characters")

            # Determine if we're using structured thinking (JSON thinking field)
            response_model = self.get_response_model()
            has_thinking_field = hasattr(response_model, "model_fields") and "thinking" in getattr(
                response_model, "model_fields", {}
            )

            # When using structured output with JSON thinking field, disable chat template thinking
            # We'll handle any <think></think> tags in post-processing
            if self.use_structured_output and has_thinking_field:
                chat_template_thinking = False  # Disable chat template thinking, use JSON thinking
            else:
                chat_template_thinking = self.enable_thinking

            # Generate response using LLM engine manager with optional structured output
            response = await self.backend.generate_response_async(
                model=self.model,
                messages=[{"from": "human", "value": prompt}],
                temperature=0.1,
                use_structured_output=self.use_structured_output,
                response_model=response_model,
                max_retries=get_llm_runtime_settings().service.llm_max_retries,
                record_index=record_index,
                method=self.__class__.__name__,
                enable_thinking=chat_template_thinking,
            )

            # Parse response - use structured parsing if enabled
            if self.use_structured_output:
                try:
                    # Direct JSON parsing for structured output
                    json_data = json.loads(response.strip())
                    response_model = self.get_response_model()

                    # If response_model expects thinking but json_data doesn't have it,
                    # use the base model instead (thinking was already extracted)
                    validation_model = response_model
                    if (
                        hasattr(response_model, "__name__")
                        and "WithThinking" in response_model.__name__
                        and "thinking" not in json_data
                    ):
                        validation_model = SurveyPrediction
                        logger.debug("Using SurveyPrediction for validation since thinking was already extracted")

                    parsed_response = validation_model(**json_data)

                    # Convert to SurveyPrediction if needed for compatibility
                    if isinstance(parsed_response, SurveyPrediction):
                        prediction = parsed_response
                    else:
                        # Handle case where response_model is a subclass with extra fields
                        prediction_dict = parsed_response.model_dump(exclude={"thinking"})
                        prediction = SurveyPrediction(**prediction_dict)

                    logger.debug(f"Successfully parsed structured response for record {record_index}")
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    logger.error(f"Failed to parse structured response: {e}")
                    # No fallback - let it fail
                    raise
            else:
                # Use traditional parsing method for non-structured output
                prediction = await self.parse_response(response, row)

            # Estimate token usage
            input_tokens = self.backend.estimate_tokens(prompt, self.model)
            output_tokens = self.backend.estimate_tokens(response, self.model)

            return {
                "record_index": record_index,
                "prediction": prediction,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }

        except Exception as e:
            logger.error(f"Error in async single record processing: {e}")
            # Return pd.NA values - no fallback defaults
            return {
                "record_index": record_index,
                "prediction": pd.NA,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "error": str(e),
            }

    async def create_prompt(self, _row: pd.Series) -> str:
        """Create a prompt for the given row - to be implemented by subclasses.

        Parameters
        ----------
        _row : pd.Series
            Survey row data (unused in base class)
        """
        raise NotImplementedError("Subclasses must implement create_prompt")

    async def parse_response(self, _response: str, _row: pd.Series) -> SurveyPrediction:
        """Parse the response into a SurveyPrediction - to be implemented by subclasses.

        Parameters
        ----------
        _response : str
            LLM response text (unused in base class)
        _row : pd.Series
            Survey row data (unused in base class)
        """
        raise NotImplementedError("Subclasses must implement parse_response")

    def _results_to_dataframe(self, results: list[dict[str, Any]], _original_df: pd.DataFrame) -> pd.DataFrame:
        """Convert processing results to a DataFrame.

        Parameters
        ----------
        results : list[dict[str, Any]]
            List of processing results
        _original_df : pd.DataFrame
            Original survey dataframe for structure reference (unused in current implementation)

        Returns
        -------
        pd.DataFrame
            Processed results as DataFrame
        """
        # Extract predictions and prepare dataframe
        predictions_data = []

        for result in results:
            record_index = result["record_index"]
            prediction = result.get("prediction")

            if prediction is not pd.NA and prediction is not None and isinstance(prediction, SurveyPrediction):
                # Convert prediction to dict
                pred_dict = prediction.model_dump()
                pred_dict["record_index"] = record_index
                pred_dict["input_tokens"] = result.get("input_tokens", 0)
                pred_dict["output_tokens"] = result.get("output_tokens", 0)
                pred_dict["total_tokens"] = result.get("total_tokens", 0)
                predictions_data.append(pred_dict)
            else:
                # Handle failed predictions - use pd.NA for all values
                logger.warning(f"No valid prediction for record {record_index}")
                placeholder = {"record_index": record_index}
                # Add pd.NA values for expected columns
                for col in [
                    "institut_trust_vax_1",
                    "institut_trust_vax_2",
                    "institut_trust_vax_3",
                    "institut_trust_vax_4",
                    "institut_trust_vax_5",
                    "institut_trust_vax_6",
                    "institut_trust_vax_7",
                    "institut_trust_vax_8",
                    "opinion_1",
                    "fiveC_1",
                    "fiveC_2",
                    "fiveC_3",
                    "fiveC_4",
                    "fiveC_5",
                    "protective_behaviour_1",
                    "protective_behaviour_2",
                    "protective_behaviour_3",
                    "protective_behaviour_4",
                ]:
                    placeholder[col] = pd.NA
                # Add thinking field if this processor uses it
                if hasattr(self, "get_response_model"):
                    response_model = self.get_response_model()
                    if hasattr(response_model, "model_fields") and "thinking" in response_model.model_fields:
                        placeholder["thinking"] = pd.NA
                placeholder["input_tokens"] = result.get("input_tokens", 0)
                placeholder["output_tokens"] = result.get("output_tokens", 0)
                placeholder["total_tokens"] = result.get("total_tokens", 0)
                predictions_data.append(placeholder)

        return pd.DataFrame(predictions_data)

    # Sync wrapper for backward compatibility
    def process_records(self, survey_prompt: str, survey_answers: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
        """Sync wrapper for async processing (backward compatibility).

        Parameters
        ----------
        survey_prompt : str
            The survey prompt template
        survey_answers : pd.DataFrame
            The survey answers to process

        Returns
        -------
        tuple[pd.DataFrame, dict[str, int]]
            Processed results and token usage statistics
        """
        logger.info("Using sync wrapper for async processing")
        return asyncio.run(self.process_records_async(survey_prompt, survey_answers))
