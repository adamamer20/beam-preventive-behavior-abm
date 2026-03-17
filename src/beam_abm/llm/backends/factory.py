"""Backend factory/registry.

Processors call `get_backend(model_config)` to obtain a cached backend instance.
"""

from __future__ import annotations

from dataclasses import dataclass

from beam_abm.llm.backends.base import AsyncLLMBackend
from beam_abm.llm.backends.openai_v1_backend import OpenAIV1Backend
from beam_abm.llm.backends.vllm_backend import VLLMBackend
from beam_abm.llm.schemas.model_config import ModelConfig


@dataclass(frozen=True)
class _BackendCacheKey:
    backend: str
    # Azure deployments may differ in auth/context.
    azure_endpoint: str | None
    azure_deployment: str | None


_BACKEND_CACHE: dict[_BackendCacheKey, AsyncLLMBackend] = {}


def get_backend(model: ModelConfig) -> AsyncLLMBackend:
    backend = (model.backend or "vllm").lower()

    azure_endpoint = None
    azure_deployment = None
    if model.azure is not None:
        azure_endpoint = model.azure.endpoint
        azure_deployment = model.azure.deployment

    key = _BackendCacheKey(
        backend=backend,
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
    )

    if key in _BACKEND_CACHE:
        return _BACKEND_CACHE[key]

    if backend == "vllm":
        inst: AsyncLLMBackend = VLLMBackend()
    elif backend in {"azure_openai", "azure", "openai", "openai_v1"}:
        inst = OpenAIV1Backend()
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    _BACKEND_CACHE[key] = inst
    return inst
