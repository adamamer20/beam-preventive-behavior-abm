"""Azure OpenAI backend (async).

Implements the same async surface as the vLLM backend so processors can be backend-agnostic.

Structured output strategy:
- Prefer the OpenAI Python SDK beta parsed API when `use_structured_output=True` and a
  Pydantic `response_model` is supplied.

Notes:
- Azure deployments are identified by deployment name, used as the `model` parameter.
- Endpoint, key, and API version are read from env via `beam_abm.llm.config.get_llm_runtime_settings()` by default.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from typing import Any

from beam_abm.common.logging import get_logger
from beam_abm.llm.backends.base import AsyncLLMBackend, LLMRequest, gather_concurrent
from beam_abm.llm.config import get_llm_runtime_settings
from beam_abm.llm.schemas.model_config import ModelConfig
from beam_abm.llm.utils.conversation_logger import ConversationLogger
from beam_abm.llm.utils.tokens import count_tokens


def _supports_temperature_top_p(model_name: str | None) -> bool:
    """Return True if Azure model supports temperature/top_p.

    Azure OpenAI rejects sampling parameters for some model families (notably `gpt-5*`).
    We treat those as "defaults-only" and omit the parameters entirely.
    """

    if not model_name:
        return True
    s = str(model_name).strip().lower()
    # Be tolerant of naming differences between deployment names and model hints.
    s = s.replace("_", "-")
    return not s.startswith("gpt-5")


logger = get_logger(__name__)


def _is_retryable_error(exc: Exception) -> bool:
    """Best-effort classification of transient Azure/OpenAI errors."""

    msg = str(exc).lower()
    if "timed out" in msg or "timeout" in msg:
        return True
    if "rate limit" in msg or "too many requests" in msg or "429" in msg or "498" in msg:
        return True
    if "service unavailable" in msg or "503" in msg:
        return True
    if "gateway" in msg and ("502" in msg or "504" in msg):
        return True
    return False


def _retry_sleep_seconds(exc: Exception, attempt_index: int) -> float:
    """Compute exponential backoff with jitter.

    Parameters
    ----------
    exc:
        The exception that triggered the retry.
    attempt_index:
        Zero-based retry attempt index (0 for the first failure).

    Returns
    -------
    float
        Seconds to sleep before the next retry.
    """

    # Faster for generic transient failures, slower for throttling.
    base = 0.5
    cap = 30.0
    msg = str(exc).lower()
    if "rate limit" in msg or "too many requests" in msg or "429" in msg or "498" in msg:
        base = 2.0
        cap = 60.0

    delay = min(cap, base * (2.0**attempt_index))
    # Full jitter in [0.5x, 1.5x].
    return delay * (0.5 + random.random())


def _messages_to_openai(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for msg in messages:
        frm = msg.get("from") or msg.get("role") or "human"
        value = msg.get("value") or msg.get("content") or ""

        if frm in {"human", "user"}:
            role = "user"
        elif frm in {"assistant", "ai"}:
            role = "assistant"
        elif frm in {"system"}:
            role = "system"
        else:
            role = "user"
        out.append({"role": role, "content": value})
    return out


def _format_prompt_for_logging(openai_messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for m in openai_messages:
        parts.append(f"[{m['role']}]\n{m['content']}")
    return "\n\n".join(parts)


class AzureOpenAIBackend(AsyncLLMBackend):
    """Backend using Azure OpenAI's async chat completions API."""

    def __init__(self, conversation_logger: ConversationLogger | None = None):
        self._client = None
        self.conversation_logger = conversation_logger

    def set_conversation_logger(self, conversation_logger: ConversationLogger | None) -> None:
        self.conversation_logger = conversation_logger

    def _get_client(self, model: ModelConfig):
        if self._client is not None:
            return self._client

        # Lazy import so the repo remains usable without Azure deps installed at import time.
        from openai import AsyncAzureOpenAI

        endpoint = (
            model.azure.endpoint
            if model.azure and model.azure.endpoint
            else get_llm_runtime_settings().service.azure_openai_endpoint
        )
        api_key = get_llm_runtime_settings().service.azure_openai_api_key
        api_version = (
            model.azure.api_version
            if model.azure and model.azure.api_version
            else get_llm_runtime_settings().service.azure_openai_api_version
        )

        if not endpoint or not api_key or not api_version:
            raise RuntimeError(
                "Azure OpenAI backend selected but AZURE_OPENAI_* env vars are missing. "
                "Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION."
            )

        timeout_seconds = get_llm_runtime_settings().service.azure_openai_timeout_seconds
        sdk_max_retries = get_llm_runtime_settings().service.azure_openai_sdk_max_retries

        self._client = AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            timeout=timeout_seconds,
            max_retries=sdk_max_retries,
        )
        return self._client

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
        if enable_thinking:
            # Azure/OpenAI doesn't have a chat-template "thinking" toggle like Qwen3.
            # We keep the flag for interface parity but it has no direct effect here.
            pass

        if max_retries is None:
            max_retries = get_llm_runtime_settings().service.llm_max_retries

        deployment = None
        if model.azure and model.azure.deployment:
            deployment = model.azure.deployment
        elif get_llm_runtime_settings().service.azure_openai_deployment:
            deployment = get_llm_runtime_settings().service.azure_openai_deployment
        if not deployment:
            raise RuntimeError(
                "Azure OpenAI backend requires deployment name. "
                "Set ModelConfig.azure.deployment or AZURE_OPENAI_DEPLOYMENT."
            )
        openai_messages = _messages_to_openai(messages)
        prompt_for_logging = _format_prompt_for_logging(openai_messages)

        resolved_max_model_len = (
            int(max_model_len)
            if max_model_len is not None
            else (model.options.max_model_len if model.options and model.options.max_model_len else 8192)
        )
        prompt_tokens = self.estimate_tokens(prompt_for_logging, model)
        available_for_output = max(1, resolved_max_model_len - prompt_tokens)

        request_id = str(uuid.uuid4())

        # Avoid sending temperature/top_p for models that only support defaults.
        # Prefer model_hint/name for capability detection; fall back to deployment.
        hinted = None
        if model.azure and model.azure.model_hint:
            hinted = model.azure.model_hint
        else:
            hinted = model.name

        temp = temperature
        tp = top_p
        if not _supports_temperature_top_p(hinted) and not _supports_temperature_top_p(deployment):
            temp = None
            tp = None

        for attempt in range(max_retries):
            start = time.time()
            try:
                if use_structured_output and response_model is not None:
                    # Prefer the OpenAI SDK beta parsed API (response_format).
                    client = self._get_client(model)

                    data: dict[str, Any]
                    reasoning_content = ""
                    # Returns a ParsedChatCompletion; the parsed object lives on message.parsed.
                    request_kwargs: dict[str, Any] = {
                        "model": deployment,
                        "messages": openai_messages,
                        "response_format": response_model,
                        "max_completion_tokens": available_for_output,
                    }
                    if temp is not None:
                        request_kwargs["temperature"] = temp
                    if tp is not None:
                        request_kwargs["top_p"] = tp

                    parsed = await client.beta.chat.completions.parse(**request_kwargs)
                    msg = parsed.choices[0].message
                    obj = getattr(msg, "parsed", None)
                    if obj is None:
                        raise RuntimeError("beta parse returned no parsed content")
                    # `obj` may be a Pydantic model or plain dict depending on SDK version.
                    if hasattr(obj, "model_dump"):
                        data = obj.model_dump()  # type: ignore[assignment]
                    elif isinstance(obj, dict):
                        data = obj
                    else:
                        raise TypeError(f"Unexpected parsed object type: {type(obj)}")

                    if "thinking" in data:
                        reasoning_content = str(data.get("thinking") or "")
                        data = {k: v for k, v in data.items() if k != "thinking"}

                    response_text = json.dumps(data)

                    input_tokens = self.estimate_tokens(prompt_for_logging, model)
                    output_tokens = self.estimate_tokens(response_text, model)

                    if self.conversation_logger is not None and record_index is not None:
                        generation_time = time.time() - start
                        total_tokens = input_tokens + output_tokens
                        tps = total_tokens / generation_time if generation_time > 0 else 0.0
                        self.conversation_logger.log_conversation(
                            record_index=record_index,
                            record_id=record_id,
                            prompt_strategy=prompt_strategy,
                            perturbation=perturbation,
                            model_name=deployment,
                            method=method,
                            prompt_text=prompt_for_logging,
                            response_text=response_text,
                            thinking_content=reasoning_content,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            total_tokens=total_tokens,
                            validation_success=True,
                            attempt_number=attempt + 1,
                            request_id=request_id,
                            inference_time_seconds=generation_time,
                            tokens_per_second=tps,
                        )

                    return response_text

                # Non-structured fallback
                client = self._get_client(model)

                request_kwargs = {
                    "model": deployment,
                    "messages": openai_messages,
                    "max_completion_tokens": available_for_output,
                }
                if temp is not None:
                    request_kwargs["temperature"] = temp
                if tp is not None:
                    request_kwargs["top_p"] = tp

                resp = await client.chat.completions.create(**request_kwargs)

                content = (resp.choices[0].message.content or "").strip()

                if self.conversation_logger is not None and record_index is not None:
                    input_tokens = self.estimate_tokens(prompt_for_logging, model)
                    output_tokens = self.estimate_tokens(content, model)
                    total_tokens = input_tokens + output_tokens
                    generation_time = time.time() - start
                    tps = total_tokens / generation_time if generation_time > 0 else 0.0
                    self.conversation_logger.log_conversation(
                        record_index=record_index,
                        record_id=record_id,
                        prompt_strategy=prompt_strategy,
                        perturbation=perturbation,
                        model_name=deployment,
                        method=method,
                        prompt_text=prompt_for_logging,
                        response_text=content,
                        thinking_content="",
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                        validation_success=True,
                        attempt_number=attempt + 1,
                        request_id=request_id,
                        inference_time_seconds=generation_time,
                        tokens_per_second=tps,
                    )

                return content

            except Exception as e:
                logger.warning(f"AzureOpenAIBackend attempt {attempt + 1}/{max_retries} failed: {e}")

                if self.conversation_logger is not None and record_index is not None:
                    input_tokens = self.estimate_tokens(prompt_for_logging, model)
                    self.conversation_logger.log_conversation(
                        record_index=record_index,
                        record_id=record_id,
                        prompt_strategy=prompt_strategy,
                        perturbation=perturbation,
                        model_name=deployment,
                        method=method,
                        prompt_text=prompt_for_logging,
                        response_text="",
                        thinking_content="",
                        input_tokens=input_tokens,
                        output_tokens=0,
                        total_tokens=input_tokens,
                        validation_success=False,
                        validation_error=str(e),
                        attempt_number=attempt + 1,
                        request_id=request_id,
                    )

                if attempt >= max_retries - 1:
                    raise

                if _is_retryable_error(e):
                    # Exponential backoff with jitter to avoid retry stampedes at high concurrency.
                    sleep_s = _retry_sleep_seconds(e, attempt)
                    await asyncio.sleep(sleep_s)
                else:
                    # Non-transient errors should not be retried.
                    raise

        return ""

    async def generate_responses_concurrent(self, requests: list[LLMRequest]) -> list[str]:
        return await gather_concurrent(self, requests)

    def estimate_tokens(self, text: str, model: ModelConfig) -> int:
        hint = None
        if model.azure and model.azure.model_hint:
            hint = model.azure.model_hint
        # Deployment names are not stable tokenization hints; fall back.
        return count_tokens(text, model=hint or "gpt-4o-mini")
