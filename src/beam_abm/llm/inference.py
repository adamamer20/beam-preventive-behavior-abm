"""LLM sampling utilities for micro-validation."""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from typing import Any

from beam_abm.llm.backends.factory import get_backend
from beam_abm.llm.config import get_llm_runtime_settings
from beam_abm.llm.schemas.model_config import ModelConfig


@dataclass(frozen=True)
class SampleResult:
    raw_text: str
    parsed: dict[str, Any] | None
    error: str | None
    attempts: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    raw_text_original: str | None = None
    raw_text_structurer: str | None = None


def _retry_sleep_seconds(error: Exception, attempt_index: int) -> float:
    """Compute exponential backoff with jitter for transient API failures."""

    msg = str(error).lower()
    base = 0.5
    cap = 30.0
    if "429" in msg or "rate limit" in msg or "too many requests" in msg:
        base = 2.0
        cap = 60.0
    if "timed out" in msg or "timeout" in msg:
        base = max(base, 1.0)
    delay = min(cap, base * (2.0**attempt_index))
    return delay * (0.5 + random.random())


async def sample_one(
    *,
    model: ModelConfig,
    prompt: str,
    messages: list[dict[str, str]] | None = None,
    response_model: type | None,
    use_structured_output: bool,
    temperature: float,
    top_p: float,
    top_k: int | None = None,
    min_p: float | None = None,
    max_model_len: int | None = None,
    max_retries: int,
    record_index: int | None,
    record_id: str | None = None,
    prompt_strategy: str | None = None,
    perturbation: str | None = None,
    method: str,
    semaphore: asyncio.Semaphore | None = None,
) -> SampleResult:
    backend = get_backend(model)
    reasoning_effort = None
    reasoning_summary = None
    text_verbosity = None
    model_options = getattr(model, "options", None)
    if model_options:
        if getattr(model_options, "reasoning", None):
            reasoning_effort = model_options.reasoning.effort
            reasoning_summary = model_options.reasoning.summary
        text_verbosity = model_options.text_verbosity

    attempts = 0
    last_error: str | None = None
    if messages is None:
        messages = [{"from": "human", "value": prompt}]
    for attempts in range(1, max_retries + 1):
        try:
            if semaphore is None:
                text = await backend.generate_response_async(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    max_model_len=max_model_len,
                    use_structured_output=use_structured_output,
                    response_model=response_model,
                    max_retries=1,
                    record_index=record_index,
                    record_id=record_id,
                    prompt_strategy=prompt_strategy,
                    perturbation=perturbation,
                    method=method,
                    reasoning_effort=reasoning_effort,
                    reasoning_summary=reasoning_summary,
                    text_verbosity=text_verbosity,
                )
            else:
                async with semaphore:
                    text = await backend.generate_response_async(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        min_p=min_p,
                        max_model_len=max_model_len,
                        use_structured_output=use_structured_output,
                        response_model=response_model,
                        max_retries=1,
                        record_index=record_index,
                        record_id=record_id,
                        prompt_strategy=prompt_strategy,
                        perturbation=perturbation,
                        method=method,
                        reasoning_effort=reasoning_effort,
                        reasoning_summary=reasoning_summary,
                        text_verbosity=text_verbosity,
                    )
            # Best-effort token estimate (human-visible prompt text).
            if len(messages) == 1:
                prompt_for_tokens = messages[0].get("value") or ""
            else:
                prompt_for_tokens = "\n\n".join(
                    f"[{m.get('from') or m.get('role') or ''}]\n{m.get('value') or m.get('content') or ''}"
                    for m in messages
                )
            input_tokens = backend.estimate_tokens(prompt_for_tokens, model)
            output_tokens = backend.estimate_tokens(text, model)
            total_tokens = input_tokens + output_tokens
            if use_structured_output:
                parsed = json.loads(text)
                return SampleResult(
                    raw_text=text,
                    parsed=parsed,
                    error=None,
                    attempts=attempts,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    raw_text_original=text,
                )
            parsed_local: dict[str, Any] | None = None
            if response_model is not None:
                # Best-effort local JSON parsing when we have a schema but are not
                # using provider-enforced structured outputs (e.g., belief-update
                # deltas on the Responses API, where strict schemas require
                # required-all objects).
                try:
                    maybe = json.loads(text)
                    if isinstance(maybe, dict):
                        parsed_local = maybe
                except Exception:
                    parsed_local = None
            return SampleResult(
                raw_text=text,
                parsed=parsed_local,
                error=None,
                attempts=attempts,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                raw_text_original=text,
            )
        except Exception as e:
            last_error = str(e)

            # Only sleep if we'll retry.
            if attempts < max_retries:
                await asyncio.sleep(_retry_sleep_seconds(e, attempts - 1))

    if len(messages) == 1:
        prompt_for_tokens = messages[0].get("value") or ""
    else:
        prompt_for_tokens = "\n\n".join(
            f"[{m.get('from') or m.get('role') or ''}]\n{m.get('value') or m.get('content') or ''}" for m in messages
        )
    input_tokens = backend.estimate_tokens(prompt_for_tokens, model)
    return SampleResult(
        raw_text="",
        parsed=None,
        error=last_error,
        attempts=attempts,
        input_tokens=input_tokens,
        output_tokens=0,
        total_tokens=input_tokens,
        raw_text_original="",
    )


async def sample_many_concurrent(
    *,
    model: ModelConfig,
    prompts: list[str],
    messages_list: list[list[dict[str, str]]] | None = None,
    response_model: type | list[type] | None,
    use_structured_output: bool,
    temperature: float = 0.4,
    top_p: float = 0.8,
    top_k: int | None = None,
    min_p: float | None = None,
    max_model_len: int | None = None,
    k: int = 8,
    max_retries: int | None = None,
    method: str,
    methods: list[str] | None = None,
    start_record_index: int = 0,
    max_concurrency: int | None = None,
    semaphore: asyncio.Semaphore | None = None,
    record_ids: list[str | None] | None = None,
    prompt_strategies: list[str | None] | None = None,
    perturbations: list[str | None] | None = None,
) -> list[list[SampleResult]]:
    """Sample K generations per prompt."""

    if messages_list is not None and len(messages_list) != len(prompts):
        raise ValueError("messages_list must be the same length as prompts")

    if methods is not None and len(methods) != len(prompts):
        raise ValueError("methods must be the same length as prompts")
    if record_ids is not None and len(record_ids) != len(prompts):
        raise ValueError("record_ids must be the same length as prompts")
    if prompt_strategies is not None and len(prompt_strategies) != len(prompts):
        raise ValueError("prompt_strategies must be the same length as prompts")
    if perturbations is not None and len(perturbations) != len(prompts):
        raise ValueError("perturbations must be the same length as prompts")

    if max_retries is None:
        max_retries = get_llm_runtime_settings().service.llm_max_retries

    if semaphore is None and max_concurrency is not None:
        if int(max_concurrency) < 1:
            raise ValueError("max_concurrency must be >= 1")
        semaphore = asyncio.Semaphore(int(max_concurrency))

    results: list[list[SampleResult]] = []

    per_prompt_models: list[type | None] | None = None
    if isinstance(response_model, list):
        if len(response_model) != len(prompts):
            raise ValueError("response_model list must be the same length as prompts")
        per_prompt_models = response_model

    async def run_for_prompt(i: int, p: str) -> list[SampleResult]:
        msgs = messages_list[i] if messages_list is not None else None
        rm = per_prompt_models[i] if per_prompt_models is not None else response_model
        prompt_method = methods[i] if methods is not None else method
        record_id = record_ids[i] if record_ids is not None else None
        prompt_strategy = prompt_strategies[i] if prompt_strategies is not None else None
        perturbation = perturbations[i] if perturbations is not None else None
        return await asyncio.gather(
            *[
                sample_one(
                    model=model,
                    prompt=p,
                    messages=msgs,
                    response_model=rm,
                    use_structured_output=use_structured_output,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    max_model_len=max_model_len,
                    max_retries=max_retries,
                    record_index=start_record_index + i,
                    record_id=record_id,
                    prompt_strategy=prompt_strategy,
                    perturbation=perturbation,
                    method=prompt_method,
                    semaphore=semaphore,
                )
                for _ in range(k)
            ]
        )

    all_batches = await asyncio.gather(*[run_for_prompt(i, p) for i, p in enumerate(prompts)])
    results.extend(all_batches)
    return results
