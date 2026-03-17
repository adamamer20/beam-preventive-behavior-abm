"""vLLM backend adapter.

This is a thin adapter around the existing `AsyncVLLMManager` implementation.
"""

from __future__ import annotations

from typing import Any

from beam_abm.llm.backends.base import AsyncLLMBackend, LLMRequest, gather_concurrent
from beam_abm.llm.clients.vllm import AsyncVLLMManager
from beam_abm.llm.config import get_llm_runtime_settings
from beam_abm.llm.model_defaults import get_generation_defaults
from beam_abm.llm.schemas.model_config import ModelConfig
from beam_abm.llm.utils.conversation_logger import ConversationLogger


class VLLMBackend(AsyncLLMBackend):
    """Backend that uses the offline vLLM `AsyncLLMEngine` manager."""

    def __init__(self, manager: AsyncVLLMManager | None = None):
        self._manager = manager or AsyncVLLMManager()

    def set_conversation_logger(self, conversation_logger: ConversationLogger | None) -> None:
        self._manager.conversation_logger = conversation_logger

    @property
    def manager(self) -> AsyncVLLMManager:
        return self._manager

    async def shutdown_all_engines(self, *, reason: str | None = None, wait_seconds: float = 0.25) -> None:
        await self._manager.shutdown_all_engines(reason=reason, wait_seconds=wait_seconds)

    async def generate_response_async(
        self,
        *,
        model: ModelConfig,
        messages: list[dict[str, str]],
        max_model_len: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repetition_penalty: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        enable_thinking: bool = False,
        use_structured_output: bool = False,
        response_model: type | None = None,
        max_retries: int | None = None,
        record_index: int | None = None,
        record_id: str | None = None,
        prompt_strategy: str | None = None,
        perturbation: str | None = None,
        method: str = "unknown",
        **kwargs: Any,
    ) -> str:
        options = model.options
        defaults = get_generation_defaults(model.name)
        resolved_max_model_len = (
            int(max_model_len)
            if max_model_len is not None
            else (options.max_model_len if options and options.max_model_len else 8192)
        )
        if max_retries is None:
            max_retries = get_llm_runtime_settings().service.llm_max_retries
        return await self._manager.generate_response_async(
            model_name=model.name,
            messages=messages,
            temperature=temperature
            if temperature is not None
            else (options.temperature if options and options.temperature is not None else defaults["temperature"]),
            top_p=top_p
            if top_p is not None
            else (options.top_p if options and options.top_p is not None else defaults["top_p"]),
            top_k=top_k
            if top_k is not None
            else (options.top_k if options and options.top_k is not None else defaults["top_k"]),
            min_p=min_p
            if min_p is not None
            else (options.min_p if options and options.min_p is not None else defaults["min_p"]),
            repetition_penalty=repetition_penalty
            if repetition_penalty is not None
            else (
                options.repetition_penalty
                if options and options.repetition_penalty is not None
                else defaults["repetition_penalty"]
            ),
            presence_penalty=presence_penalty if presence_penalty is not None else defaults["presence_penalty"],
            frequency_penalty=frequency_penalty if frequency_penalty is not None else defaults["frequency_penalty"],
            enable_thinking=enable_thinking,
            max_model_len=resolved_max_model_len,
            gpu_memory_utilization=(
                options.gpu_memory_utilization if options and options.gpu_memory_utilization else 0.92
            ),
            quantization=(options.quantization if options else None),
            load_format=(options.load_format if options and options.load_format else "auto"),
            max_num_seqs=(options.max_num_seqs if options and options.max_num_seqs else 64),
            max_num_batched_tokens=(
                options.max_num_batched_tokens if options and options.max_num_batched_tokens else 8192
            ),
            enable_chunked_prefill=(
                options.enable_chunked_prefill if options and options.enable_chunked_prefill is not None else True
            ),
            enable_prefix_caching=(
                options.enable_prefix_caching if options and options.enable_prefix_caching is not None else False
            ),
            tensor_parallel_size=(options.tensor_parallel_size if options and options.tensor_parallel_size else 1),
            use_optimization=(options.use_optimization if options and options.use_optimization is not None else False),
            trust_remote_code=(
                options.trust_remote_code if options and options.trust_remote_code is not None else True
            ),
            use_structured_output=use_structured_output,
            response_model=response_model,
            max_retries=max_retries,
            record_index=record_index,
            record_id=record_id,
            prompt_strategy=prompt_strategy,
            perturbation=perturbation,
            method=method,
            **kwargs,
        )

    async def generate_responses_concurrent(self, requests: list[LLMRequest]) -> list[str]:
        # Use manager's optimized method when available.
        # It expects vLLM-style dicts with keys similar to its generate_response_async.
        try:
            mapped: list[dict[str, Any]] = []
            for req in requests:
                model = req["model"]
                messages = req["messages"]
                options = model.options
                defaults = get_generation_defaults(model.name)
                mapped.append(
                    {
                        "model_name": model.name,
                        "messages": messages,
                        "temperature": req.get(
                            "temperature",
                            options.temperature
                            if options and options.temperature is not None
                            else defaults["temperature"],
                        ),
                        "top_p": req.get(
                            "top_p",
                            options.top_p if options and options.top_p is not None else defaults["top_p"],
                        ),
                        "top_k": req.get(
                            "top_k",
                            options.top_k if options and options.top_k is not None else defaults["top_k"],
                        ),
                        "min_p": req.get(
                            "min_p",
                            options.min_p if options and options.min_p is not None else defaults["min_p"],
                        ),
                        "repetition_penalty": req.get(
                            "repetition_penalty",
                            options.repetition_penalty
                            if options and options.repetition_penalty is not None
                            else defaults["repetition_penalty"],
                        ),
                        "presence_penalty": req.get("presence_penalty", defaults["presence_penalty"]),
                        "frequency_penalty": req.get("frequency_penalty", defaults["frequency_penalty"]),
                        "enable_thinking": req.get("enable_thinking", False),
                        "use_structured_output": req.get("use_structured_output", False),
                        "response_model": req.get("response_model"),
                        "max_retries": req.get("max_retries")
                        if req.get("max_retries") is not None
                        else get_llm_runtime_settings().service.llm_max_retries,
                        "record_index": req.get("record_index"),
                        "record_id": req.get("record_id"),
                        "prompt_strategy": req.get("prompt_strategy"),
                        "perturbation": req.get("perturbation"),
                        "method": req.get("method", "unknown"),
                        "max_model_len": req.get("max_model_len")
                        if req.get("max_model_len") is not None
                        else (options.max_model_len if options and options.max_model_len else 8192),
                        "gpu_memory_utilization": options.gpu_memory_utilization
                        if options and options.gpu_memory_utilization
                        else 0.92,
                        "max_num_seqs": options.max_num_seqs if options and options.max_num_seqs else 64,
                        "max_num_batched_tokens": options.max_num_batched_tokens
                        if options and options.max_num_batched_tokens
                        else 8192,
                        "enable_chunked_prefill": options.enable_chunked_prefill
                        if options and options.enable_chunked_prefill is not None
                        else True,
                        "enable_prefix_caching": options.enable_prefix_caching
                        if options and options.enable_prefix_caching is not None
                        else False,
                        "tensor_parallel_size": options.tensor_parallel_size
                        if options and options.tensor_parallel_size
                        else 1,
                        "use_optimization": options.use_optimization
                        if options and options.use_optimization is not None
                        else False,
                        "quantization": options.quantization if options else None,
                        "load_format": options.load_format if options and options.load_format else "auto",
                        "trust_remote_code": options.trust_remote_code
                        if options and options.trust_remote_code is not None
                        else True,
                    }
                )
            return await self._manager.generate_responses_concurrent(mapped)
        except Exception:
            # Fallback to generic gather helper.
            return await gather_concurrent(self, requests)

    def estimate_tokens(self, text: str, model: ModelConfig) -> int:
        return self._manager.estimate_tokens(text, model.name)
