"""Backend abstraction for async LLM inference.

This module defines the minimal async interface used by processors.
Backends are responsible for:
- sending chat messages to the underlying model backend,
- (optionally) enforcing structured JSON output,
- estimating tokens (best-effort), and
- providing a batch/concurrent helper.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Sequence
from typing import Any, Protocol, TypedDict

from beam_abm.llm.schemas.model_config import ModelConfig


class LLMMessage(TypedDict):
    """A minimal message format used across backends.

    The existing codebase uses vLLM-style messages: {"from": "human", "value": "..."}.
    We keep that shape at the boundary so higher-level logic doesn't change.
    """

    from_: str
    value: str


class LLMRequest(TypedDict, total=False):
    """A request payload for concurrent generation."""

    model: ModelConfig
    messages: list[dict[str, str]]
    max_model_len: int
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    repetition_penalty: float
    presence_penalty: float
    frequency_penalty: float
    enable_thinking: bool
    use_structured_output: bool
    response_model: type | None
    max_retries: int
    record_index: int | None
    record_id: str | None
    prompt_strategy: str | None
    perturbation: str | None
    method: str
    reasoning_effort: str | None
    reasoning_summary: str | None
    text_verbosity: str | None


class AsyncLLMBackend(Protocol):
    """Protocol for async LLM backends."""

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
        """Generate a single response."""

    async def generate_responses_concurrent(self, requests: Sequence[LLMRequest]) -> list[str]:
        """Generate responses concurrently."""

    def estimate_tokens(self, text: str, model: ModelConfig) -> int:
        """Best-effort token estimate for accounting/logging."""


async def gather_concurrent(
    backend: AsyncLLMBackend,
    requests: Sequence[LLMRequest],
) -> list[str]:
    """Default concurrent helper using asyncio.gather."""

    tasks: list[Awaitable[str]] = []
    for req in requests:
        model = req["model"]
        messages = req["messages"]
        kwargs = {k: v for k, v in req.items() if k not in {"model", "messages"}}
        tasks.append(backend.generate_response_async(model=model, messages=messages, **kwargs))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    out: list[str] = []
    for result in results:
        if isinstance(result, Exception):
            raise result
        out.append(result)
    return out
