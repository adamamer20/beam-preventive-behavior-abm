# This file was renamed from async_vllm_manager.py as part of the vLLM LLMEngine refactor.
# All logic will be updated in-place.

"""LLMEngineManager: offline vLLM LLMEngine with chunked prefill and Qwen-3 support."""

import asyncio
import json
import time
import uuid
from threading import Lock

from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import StructuredOutputsParams

from beam_abm.common.logging import get_logger
from beam_abm.llm.config import get_llm_runtime_settings
from beam_abm.llm.schemas.survey import SurveyPrediction
from beam_abm.llm.utils.chat_template_sanitizer import strip_leading_chatgpt_boilerplate
from beam_abm.llm.utils.conversation_logger import ConversationLogger
from beam_abm.llm.utils.vllm_optimizer import calculate_optimal_config, get_performance_estimate

logger = get_logger(__name__)


class AsyncVLLMManager:
    """
    Manages offline vLLM AsyncLLMEngine instances for direct inference with chunked prefill support.
    Supports Qwen-3 sampling parameters and chat template logic.
    Fully async for perfect concurrency.
    """

    def __init__(self, conversation_logger: ConversationLogger = None):
        self.engines = {}
        self.tokenizers = {}
        self._engine_locks = {}  # Per-engine thread locks
        self.conversation_logger = conversation_logger
        logger.info("LLMEngineManager initialized for AsyncLLMEngine")

    def _get_engine_cache_key(
        self,
        *,
        model_name: str,
        max_model_len: int,
        quantization: str | None,
        load_format: str,
        gpu_memory_utilization: float,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        enable_chunked_prefill: bool,
        enable_prefix_caching: bool,
        tensor_parallel_size: int,
        trust_remote_code: bool,
        enable_lora: bool,
    ) -> str:
        return (
            f"{model_name}|len={max_model_len}|quant={quantization}|format={load_format}"
            f"|gpu={gpu_memory_utilization}|seqs={max_num_seqs}|batch={max_num_batched_tokens}"
            f"|chunked={enable_chunked_prefill}|prefix={enable_prefix_caching}"
            f"|tp={tensor_parallel_size}|trust={trust_remote_code}|lora={enable_lora}"
        )

    def _get_tokenizer_key(self, model_name: str, trust_remote_code: bool) -> str:
        return f"{model_name}|trust={trust_remote_code}"

    async def shutdown_all_engines(self, *, reason: str | None = None, wait_seconds: float = 0.25) -> None:
        """Shut down all cached engines and wait for GPU cleanup.

        Parameters
        ----------
        reason : str | None
            Optional reason for logging context.
        wait_seconds : float
            Seconds to wait after teardown to allow CUDA cleanup.
        """
        if not self.engines:
            return
        suffix = f" ({reason})" if reason else ""
        logger.info(f"Forcing teardown of all vLLM engines{suffix}")
        for key, old_engine in list(self.engines.items()):
            if hasattr(old_engine, "shutdown_background_loop"):
                try:
                    old_engine.shutdown_background_loop()
                    logger.info(f"Gracefully shut down engine for {key}")
                except Exception as e:
                    logger.warning(f"Graceful shutdown failed for {key}, forcing delete: {e}")
            else:
                logger.warning(f"Engine for {key} does not support shutdown_background_loop; forcing delete")
            del self.engines[key]
            if key in self._engine_locks:
                del self._engine_locks[key]

        # Aggressive CUDA memory cleanup to prevent allocator state corruption
        import gc

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

        if wait_seconds > 0:
            await asyncio.sleep(float(wait_seconds))

    async def _load_engine(
        self,
        model_name: str,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.92,
        quantization: str | None = None,
        load_format: str = "auto",
        trust_remote_code: bool = True,
        enable_lora: bool = False,
        max_num_seqs: int = 64,
        max_num_batched_tokens: int = 8192,
        enable_chunked_prefill: bool = True,
        enable_prefix_caching: bool = False,
        tensor_parallel_size: int = 1,
        use_optimization: bool = False,
        **kwargs,
    ) -> AsyncLLMEngine:
        """Load or reuse a vLLM AsyncLLMEngine instance with chunked prefill."""
        cache_key = self._get_engine_cache_key(
            model_name=model_name,
            max_model_len=max_model_len,
            quantization=quantization,
            load_format=load_format,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_prefix_caching=enable_prefix_caching,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            enable_lora=enable_lora,
        )

        # Get or create thread lock for this engine (thread-safe across event loops)
        if cache_key not in self._engine_locks:
            self._engine_locks[cache_key] = Lock()

        with self._engine_locks[cache_key]:
            if cache_key in self.engines:
                logger.info(f"Reusing cached async engine for {model_name}")
                return self.engines[cache_key]

            logger.info(f"Loading vLLM AsyncLLMEngine for {model_name}")

            # Aggressive engine eviction: unload all OTHER engines before loading a new one
            # This prevents CUDA allocator corruption when switching between model sizes
            for key, old_engine in list(self.engines.items()):
                if key != cache_key:  # any engine that is not the one we're about to (re)load
                    if hasattr(old_engine, "shutdown_background_loop"):
                        try:
                            old_engine.shutdown_background_loop()
                            logger.info(f"Gracefully shut down engine for {key}")
                        except Exception as e:
                            logger.warning(f"Graceful shutdown failed for {key}, forcing delete: {e}")
                    else:
                        logger.warning(f"Engine for {key} does not support shutdown_background_loop; forcing delete")
                    del self.engines[key]  # drop the handle so ref-count reaches zero
                    if key in self._engine_locks:
                        del self._engine_locks[key]

            # Aggressive CUDA memory cleanup to prevent allocator state corruption
            import gc

            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # give blocks back to the caching allocator
                torch.cuda.ipc_collect()  # clear any lingering CUDA IPC handles
            gc.collect()  # Python objects holding DevicePtrs

            # Apply optimization if enabled
            if use_optimization:
                # Estimate average prompt length (could be improved with actual data)
                avg_prompt_length = min(max_model_len, max(3000, max_model_len // 2))

                # Convert quantization to standard format
                quant_type = "4bit" if quantization in ["awq", "gptq", "bitsandbytes"] else "fp16"

                # Get optimal configuration (auto-detect GPU memory)
                optimal_config = calculate_optimal_config(
                    model_name=model_name,
                    avg_prompt_length=avg_prompt_length,
                    quantization=quant_type,
                    # gpu_memory_gb auto-detected
                )

                # Get performance estimate
                perf_estimate = get_performance_estimate(model_name, optimal_config)

                # Apply optimal settings (user can still override)
                max_num_seqs = optimal_config.max_num_seqs
                max_num_batched_tokens = optimal_config.max_num_batched_tokens
                gpu_memory_utilization = optimal_config.gpu_memory_utilization

                logger.info(f"Applied optimization for {model_name}:")
                logger.info(f"  max_num_seqs: {max_num_seqs}")
                logger.info(f"  max_num_batched_tokens: {max_num_batched_tokens}")
                logger.info(f"  gpu_memory_utilization: {gpu_memory_utilization}")
                logger.info(f"  Expected throughput: {perf_estimate.estimated_tokens_per_sec:.1f} tokens/s")

                # Add optimization info to kwargs for potential logging
                kwargs.update(
                    {
                        "block_size": optimal_config.block_size,
                        "swap_space": optimal_config.swap_space,
                        "_optimization_applied": True,
                        "_optimization_info": optimal_config.optimization_info.model_dump(),
                    }
                )

            # Check if this is a Qwen3 model for reasoning support
            is_qwen3_model = "qwen3" in model_name.lower() or "qwen-3" in model_name.lower()

            # Only forward engine-level kwargs to vLLM. If a request-level kwarg
            # (e.g. OpenAI's reasoning controls) leaks into here, raise loudly.
            extra_engine_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
            try:
                import inspect

                allowed = set(inspect.signature(AsyncEngineArgs.__init__).parameters)
                allowed.discard("self")
                unknown = sorted(k for k in extra_engine_kwargs if k not in allowed)
                if unknown:
                    raise TypeError(
                        "Unexpected vLLM AsyncEngineArgs kwargs: "
                        + ", ".join(repr(k) for k in unknown)
                        + ". These should be handled at the request layer (not engine init)."
                    )
            except Exception:
                # Fall back to letting vLLM raise a TypeError if needed.
                pass

            engine_args = AsyncEngineArgs(
                model=model_name,
                quantization=quantization,
                load_format=load_format,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                enable_chunked_prefill=enable_chunked_prefill,
                max_num_batched_tokens=max_num_batched_tokens,
                enable_prefix_caching=enable_prefix_caching,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=trust_remote_code,
                enable_lora=enable_lora,
                # Enable reasoning parser for Qwen3 models.
                reasoning_parser="qwen3" if is_qwen3_model else None,
                **extra_engine_kwargs,
            )
            engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.engines[cache_key] = engine

            # Cache tokenizer
            tokenizer_key = self._get_tokenizer_key(model_name, trust_remote_code)
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
            self.tokenizers[tokenizer_key] = tokenizer

            logger.info(f"Successfully loaded vLLM AsyncLLMEngine for {model_name}")
            return engine

    def _apply_chat_template(
        self,
        model_name: str,
        messages: list[dict[str, str]],
        enable_thinking: bool = False,
    ) -> str:
        tokenizer_key = self._get_tokenizer_key(model_name, True)
        tokenizer = self.tokenizers.get(tokenizer_key)
        if not tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizers[tokenizer_key] = tokenizer
        openai_messages = []
        for message in messages:
            role = message.get("role", message.get("from", "user"))
            content = message.get("content", message.get("value", ""))
            if role == "human":
                role = "user"
            elif role == "gpt":
                role = "assistant"
            openai_messages.append({"role": role, "content": content})
        prompt_text: str | None = None
        if "qwen3" in model_name.lower() or "qwen-3" in model_name.lower():
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    prompt_text = tokenizer.apply_chat_template(
                        openai_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=enable_thinking,
                    )
                except TypeError:
                    prompt_text = tokenizer.apply_chat_template(
                        openai_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
        if prompt_text is None and hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(
                openai_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        if isinstance(prompt_text, str) and prompt_text:
            return strip_leading_chatgpt_boilerplate(prompt_text)
        chat_text = ""
        for message in openai_messages:
            role = message["role"]
            content = message["content"]
            if role == "user":
                chat_text += f"<|user|>\n{content}\n"
            elif role == "assistant":
                chat_text += f"<|assistant|>\n{content}\n"
            elif role == "system":
                chat_text += f"<|system|>\n{content}\n"
        chat_text += "<|assistant|>\n"
        return chat_text

    # _run_until_finished is no longer needed for async engine

    async def generate_response_async(
        self,
        model_name: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        enable_thinking: bool = False,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.92,
        quantization: str | None = None,
        load_format: str = "auto",
        enable_lora: bool = False,
        max_num_seqs: int = 64,
        max_num_batched_tokens: int = 8192,
        enable_chunked_prefill: bool = True,
        enable_prefix_caching: bool = False,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        use_optimization: bool = True,
        use_structured_output: bool = False,
        response_model: type | None = None,
        max_retries: int | None = None,
        record_index: int = None,
        record_id: str | None = None,
        prompt_strategy: str | None = None,
        perturbation: str | None = None,
        method: str = "unknown",
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        text_verbosity: str | None = None,
        **kwargs,
    ) -> str:
        """
        Generate a response using the offline vLLM AsyncLLMEngine (fully async, non-blocking).

        This method is safe to call concurrently from multiple async tasks.
        For true concurrency, use asyncio.gather or similar patterns:

            responses = await asyncio.gather(*[
                llm_engine_manager.generate_response_async(model, messages1),
                llm_engine_manager.generate_response_async(model, messages2),
                ...
            ])
        """
        if max_retries is None:
            max_retries = get_llm_runtime_settings().service.llm_max_retries

        # These are request-level controls used by OpenAI-compatible backends.
        # They are intentionally ignored by vLLM.
        _ = (reasoning_effort, reasoning_summary, text_verbosity)

        engine_cache_key = self._get_engine_cache_key(
            model_name=model_name,
            max_model_len=max_model_len,
            quantization=quantization,
            load_format=load_format,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_prefix_caching=enable_prefix_caching,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            enable_lora=enable_lora,
        )

        # Load engine (reuses cached if available)
        engine = await self._load_engine(
            model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            quantization=quantization,
            load_format=load_format,
            enable_lora=enable_lora,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_prefix_caching=enable_prefix_caching,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            use_optimization=use_optimization,
            **kwargs,
        )

        # Apply chat template
        prompt = self._apply_chat_template(model_name, messages, enable_thinking)

        # Check token limits BEFORE generation
        prompt_tokens = self.estimate_tokens(prompt, model_name)
        total_available_tokens = max_model_len
        available_for_output = total_available_tokens - prompt_tokens
        output_max_tokens = max(1, available_for_output)

        meta_bits = []
        if record_id:
            meta_bits.append(f"id={record_id}")
        if prompt_strategy:
            meta_bits.append(f"strategy={prompt_strategy}")
        if perturbation:
            meta_bits.append(f"perturbation={perturbation}")
        meta_prefix = f" ({', '.join(meta_bits)})" if meta_bits else ""
        logger.info(
            "Token analysis{}: prompt={}, max_model_len={}, available_for_output={}, output_max_tokens={}",
            meta_prefix,
            prompt_tokens,
            max_model_len,
            available_for_output,
            output_max_tokens,
        )

        if prompt_tokens >= max_model_len:
            raise ValueError(
                f"Input prompt ({prompt_tokens} tokens) exceeds model context window ({max_model_len} tokens)"
            )

        # Create sampling parameters with optional guided decoding
        sampling_params_kwargs = {
            "max_tokens": output_max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "skip_special_tokens": True,
        }

        # Add structured outputs for guided decoding - MANDATORY when requested
        if use_structured_output and response_model:
            try:
                # Create schema from Pydantic model
                schema = response_model.model_json_schema()
                structured_params = StructuredOutputsParams(json=schema, disable_any_whitespace=True)
                sampling_params_kwargs["structured_outputs"] = structured_params
                logger.info(f"Using structured outputs with {response_model.__name__} schema")
            except Exception as e:
                logger.error(f"Failed to create structured outputs params: {e}")
                logger.error("Structured outputs is mandatory but failed to initialize")
                raise RuntimeError(f"Structured outputs failed: {e}") from e

        sampling_params = SamplingParams(**sampling_params_kwargs)

        # Retry logic for structured output
        for attempt in range(max_retries):
            try:
                # Generate unique request ID
                request_id = str(uuid.uuid4())

                # Track generation start time
                self._generation_start_time = time.time()

                # Process async generation (generate method handles adding request automatically)
                generated_text = ""
                finish_reason: str | None = None
                async for request_output in engine.generate(
                    prompt=prompt, sampling_params=sampling_params, request_id=request_id
                ):
                    if request_output.finished:
                        if request_output.outputs:
                            generated_text = request_output.outputs[0].text
                            finish_reason = getattr(request_output.outputs[0], "finish_reason", None)
                        break

                if generated_text:
                    # Extract reasoning content for Qwen3 models with thinking enabled
                    reasoning_content = None
                    final_answer = generated_text

                    # Clean up unwanted <think></think> tags when using structured output with JSON thinking field
                    if use_structured_output and response_model and hasattr(response_model, "model_fields"):
                        has_thinking_field = "thinking" in getattr(response_model, "model_fields", {})
                        if has_thinking_field and "<think>" in generated_text:
                            # Remove empty or unwanted <think></think> tags that interfere with JSON parsing
                            import re

                            # Remove <think>...</think> tags (including empty ones)
                            cleaned_text = re.sub(r"<think>.*?</think>", "", generated_text, flags=re.DOTALL).strip()
                            if cleaned_text and cleaned_text != generated_text:
                                logger.debug("Removed <think> tags from structured output response")
                                final_answer = cleaned_text

                    # Extract thinking content from JSON structured output
                    if use_structured_output and response_model and hasattr(response_model, "model_fields"):
                        has_thinking_field = "thinking" in getattr(response_model, "model_fields", {})
                        if has_thinking_field:
                            try:
                                json_data = json.loads(final_answer)  # Use cleaned final_answer
                                if "thinking" in json_data:
                                    reasoning_content = json_data.get("thinking", "")
                                    # Create a version without thinking for final answer
                                    json_data_no_thinking = {k: v for k, v in json_data.items() if k != "thinking"}
                                    final_answer = json.dumps(json_data_no_thinking)
                                    if reasoning_content:
                                        logger.debug(f"Extracted thinking content: {len(reasoning_content)} chars")
                            except (json.JSONDecodeError, Exception) as e:
                                logger.warning(f"Failed to extract thinking content from structured output: {e}")

                    # Use final answer for token estimation and validation
                    # display_text = final_answer if reasoning_content else generated_text

                    # Monitor actual token usage
                    output_tokens = self.estimate_tokens(generated_text, model_name)
                    total_tokens = prompt_tokens + output_tokens
                    token_utilization = total_tokens / total_available_tokens

                    logger.info(
                        f"Token usage: output={output_tokens}, total={total_tokens}/{total_available_tokens} ({token_utilization:.1%})"
                    )

                    if reasoning_content:
                        logger.info(
                            f"Reasoning extracted: {len(reasoning_content)} chars reasoning, {len(final_answer)} chars answer"
                        )

                    # Check if we're hitting token limits
                    if token_utilization > 0.95:
                        logger.warning(f"Very high token utilization ({token_utilization:.1%}) - possible truncation")
                    elif token_utilization > 0.85:
                        logger.warning(f"High token utilization ({token_utilization:.1%}) - approaching limit")

                    # Proactive retry when we are effectively at the context limit.
                    # This often indicates the model was forced to stop early.
                    truncation_utilization_threshold = 0.999
                    suspected_truncation = token_utilization >= truncation_utilization_threshold or (
                        finish_reason is not None and str(finish_reason).lower() == "length"
                    )
                    if suspected_truncation and attempt < max_retries - 1:
                        logger.warning(
                            f"Suspected truncation on attempt {attempt + 1}/{max_retries} "
                            f"(finish_reason={finish_reason}, utilization={token_utilization:.1%}). Retrying..."
                        )
                        continue

                    # Check for potential truncation indicators
                    if generated_text.endswith(('",', '":"', '":', "{")):
                        logger.warning(f"Generated text appears truncated: ends with '{generated_text[-10:]}'")

                    # If using structured output, validate the response
                    if use_structured_output and response_model:
                        try:
                            json_data = json.loads(final_answer.strip())

                            # If we extracted thinking content, validate against the base model without thinking
                            validation_model = response_model
                            if (
                                reasoning_content
                                and hasattr(response_model, "__name__")
                                and "WithThinking" in response_model.__name__
                            ):
                                validation_model = SurveyPrediction
                                logger.debug("Using SurveyPrediction for validation since thinking was extracted")

                            # Validate by creating the model instance
                            validation_model(**json_data)
                            logger.debug(f"Structured response validated on attempt {attempt + 1}")
                        except (json.JSONDecodeError, ValueError, TypeError) as e:
                            logger.warning(f"Generated text that failed validation: {generated_text[:200]}...")
                            logger.warning(
                                f"Full generated text length: {len(generated_text)}, ends with: '{generated_text[-50:]}'"
                            )

                            # Check if failure might be due to truncation
                            if token_utilization > 0.9 or generated_text.endswith(('",', '":"', '":', "{")):
                                logger.error(
                                    f"Validation failure likely due to token truncation (utilization: {token_utilization:.1%})"
                                )

                            # Log failed validation conversation
                            if self.conversation_logger and record_index is not None:
                                self.conversation_logger.log_conversation(
                                    record_index=record_index,
                                    record_id=record_id,
                                    prompt_strategy=prompt_strategy,
                                    perturbation=perturbation,
                                    model_name=model_name,
                                    method=method,
                                    prompt_text=prompt,
                                    response_text=generated_text,
                                    thinking_content=reasoning_content or "",
                                    input_tokens=prompt_tokens,
                                    output_tokens=output_tokens,
                                    total_tokens=total_tokens,
                                    validation_success=False,
                                    validation_error=str(e),
                                    attempt_number=attempt + 1,
                                    request_id=request_id,
                                )

                            if attempt < max_retries - 1:
                                logger.warning(f"Structured output validation failed on attempt {attempt + 1}: {e}")
                                logger.info(f"Retrying... ({attempt + 2}/{max_retries})")
                                continue
                            else:
                                logger.error(f"Structured output validation failed after {max_retries} attempts: {e}")
                                raise

                    logger.debug(f"Generated response length: {len(generated_text)} for model {model_name}")

                    # Log conversation if logger is available
                    if self.conversation_logger and record_index is not None:
                        # Calculate basic performance metrics
                        generation_time = time.time() - self._generation_start_time
                        tokens_per_sec = total_tokens / generation_time if generation_time > 0 else 0

                        # Extract optimization info if available
                        optimization_info = kwargs.get("_optimization_info", {})
                        optimization_applied = kwargs.get("_optimization_applied", False)

                        self.conversation_logger.log_conversation(
                            record_index=record_index,
                            record_id=record_id,
                            prompt_strategy=prompt_strategy,
                            perturbation=perturbation,
                            model_name=model_name,
                            method=method,
                            prompt_text=prompt,
                            response_text=generated_text,
                            thinking_content=reasoning_content or "",
                            input_tokens=prompt_tokens,
                            output_tokens=output_tokens,
                            total_tokens=total_tokens,
                            validation_success=True,
                            attempt_number=attempt + 1,
                            request_id=request_id,
                            # Basic performance metrics
                            inference_time_seconds=generation_time,
                            tokens_per_second=tokens_per_sec,
                            optimization_applied=optimization_applied,
                            estimated_throughput=optimization_info.get("estimated_tokens_per_sec", 0),
                            max_num_seqs=max_num_seqs,
                            max_num_batched_tokens=max_num_batched_tokens,
                            gpu_memory_utilization=gpu_memory_utilization,
                        )

                    # Return final answer (reasoning content is logged but not returned)
                    return final_answer.strip() if reasoning_content else generated_text.strip()
                else:
                    if attempt < max_retries - 1:
                        logger.warning(f"No response generated on attempt {attempt + 1}, retrying...")
                        continue
                    else:
                        logger.warning(f"No response generated for {model_name} after {max_retries} attempts")
                        return ""

            except Exception as e:
                # Log the failed conversation attempt
                if self.conversation_logger and record_index is not None:
                    self.conversation_logger.log_conversation(
                        record_index=record_index,
                        record_id=record_id,
                        prompt_strategy=prompt_strategy,
                        perturbation=perturbation,
                        model_name=model_name,
                        method=method or "unknown",
                        prompt_text=prompt,
                        response_text="",  # No response due to exception
                        thinking_content="",
                        input_tokens=prompt_tokens,
                        output_tokens=0,
                        total_tokens=prompt_tokens,
                        validation_success=False,
                        validation_error=f"Generation exception: {str(e)}",
                        attempt_number=attempt + 1,
                        request_id=request_id or "",
                    )

                msg = str(e).lower()
                engine_dead = "engine dead" in msg or "enginecore" in msg or "enginedeaderror" in msg
                if engine_dead:
                    logger.warning("Engine appears to be dead, evicting cached engine...")
                    if engine_cache_key in self.engines:
                        old_engine = self.engines.pop(engine_cache_key)
                        shutdown_fn = getattr(old_engine, "shutdown_background_loop", None)
                        if callable(shutdown_fn):
                            try:
                                shutdown_fn()
                                logger.info("Gracefully shut down dead engine")
                            except Exception as shutdown_error:
                                logger.warning(f"Failed to shut down dead engine: {shutdown_error}")
                    logger.info("Cleared dead engine from cache")

                if attempt < max_retries - 1:
                    logger.warning(f"Generation failed on attempt {attempt + 1}: {e}")

                    if engine_dead:
                        logger.warning("Reloading engine after failure...")
                        engine = await self._load_engine(
                            model_name,
                            max_model_len=max_model_len,
                            gpu_memory_utilization=gpu_memory_utilization,
                            quantization=quantization,
                            load_format=load_format,
                            enable_lora=enable_lora,
                            max_num_seqs=max_num_seqs,
                            max_num_batched_tokens=max_num_batched_tokens,
                            trust_remote_code=trust_remote_code,
                            **kwargs,
                        )

                    logger.info(f"Retrying... ({attempt + 2}/{max_retries})")
                    continue
                else:
                    logger.error(f"Generation failed after {max_retries} attempts: {e}")
                    raise

        return ""

    async def generate_responses_concurrent(self, requests: list[dict]) -> list[str]:
        """
        Generate multiple responses concurrently. Each request dict should have keys:
            - model_name
            - messages
            - (other kwargs for generate_response_async)
        Returns a list of responses in the same order as requests.
        """
        coros = [self.generate_response_async(**req) for req in requests]
        return await asyncio.gather(*coros)

    def generate_response_sync(
        self,
        model_name: str,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> str:
        """
        Synchronous wrapper for generate_response_async.
        Runs the async method in a new event loop.
        """
        return asyncio.run(self.generate_response_async(model_name, messages, **kwargs))

    def generate_response(
        self,
        model_name: str,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> str:
        """
        Synchronous wrapper for generate_response_async.
        Runs the async method in a new event loop.
        """
        return asyncio.run(self.generate_response_async(model_name, messages, **kwargs))

    def estimate_tokens(self, text: str, model_name: str) -> int:
        tokenizer_key = self._get_tokenizer_key(model_name, True)
        if tokenizer_key not in self.tokenizers:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizers[tokenizer_key] = tokenizer
        tokenizer = self.tokenizers[tokenizer_key]
        return len(tokenizer.encode(text))

    async def reset_engines(self) -> None:
        """
        Hard-reset the manager: shut down every live engine, clear caches,
        and fully flush the CUDA allocator.
        Useful between benchmark phases (1.7B → 4B → 8B) to prevent
        CUDA allocator corruption.
        """
        await self.cleanup()  # existing logic
        import gc

        import torch

        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
        gc.collect()
        logger.info("Engine reset completed - CUDA allocator fully flushed")

    async def cleanup(self):
        """Async cleanup of all engines and resources."""
        import gc

        import torch

        logger.info("Starting LLMEngineManager cleanup")

        # Cleanup all engines
        for engine in self.engines.values():
            if hasattr(engine, "shutdown_background_loop"):
                try:
                    engine.shutdown_background_loop()
                    logger.info("Gracefully shut down engine during cleanup")
                except Exception as e:
                    logger.warning(f"Failed to shutdown engine during cleanup: {e}")
            else:
                logger.warning("Engine does not support shutdown_background_loop; forcing delete")

        # Clear tokenizer cache
        for tokenizer in self.tokenizers.values():
            del tokenizer

        self.engines.clear()
        self.tokenizers.clear()
        self._engine_locks.clear()

        # Force GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Force garbage collection
        gc.collect()

        logger.info("LLMEngineManager cleanup completed with GPU memory cleanup")


"""Example usage:

from beam_abm.llm.clients.vllm import AsyncVLLMManager
import asyncio

llm_engine_manager = AsyncVLLMManager()

# Synchronous API (for legacy compatibility)
response = llm_engine_manager.generate_response(
    model_name="Qwen/Qwen1.5-7B-Chat",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.1,
    enable_thinking=True,
)

# Async API (recommended for new code)
response = await llm_engine_manager.generate_response_async(
    model_name="Qwen/Qwen1.5-7B-Chat",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.1,
    enable_thinking=True,
)

# High-performance concurrent generation
responses = await asyncio.gather(*[
    llm_engine_manager.generate_response_async(
        "Qwen/Qwen1.5-7B-Chat",
        [{"role": "user", "content": "Hi 1"}]
    ),
    llm_engine_manager.generate_response_async(
        "Qwen/Qwen1.5-7B-Chat",
        [{"role": "user", "content": "Hi 2"}]
    ),
])

# Reset engines between different model sizes to prevent CUDA allocator corruption:
# await llm_engine_manager.generate_response_async(model_name="mistral-1.7b", ...)
# await llm_engine_manager.reset_engines()       # frees GPU completely
# await llm_engine_manager.generate_response_async(model_name="mistral-4b",  ...)
# await llm_engine_manager.reset_engines()
# await llm_engine_manager.generate_response_async(model_name="mistral-8b",  ...)

# Concurrent helper method
responses = await llm_engine_manager.generate_responses_concurrent([
    {"model_name": "Qwen/Qwen1.5-7B-Chat", "messages": [{"role": "user", "content": "Hi 1"}]},
    {"model_name": "Qwen/Qwen1.5-7B-Chat", "messages": [{"role": "user", "content": "Hi 2"}]},
])
"""
