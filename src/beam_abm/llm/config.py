"""Typed runtime settings for LLM-facing code."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from beam_abm.settings import API_CONFIG, BASE_PROMPT


@dataclass(frozen=True, slots=True)
class LLMServiceSettings:
    llm_max_retries: int
    openai_v1_min_rate_limit_retries: int
    azure_openai_timeout_seconds: float
    azure_openai_sdk_max_retries: int
    groq_service_tier: str
    azure_openai_reasoning_effort: str
    azure_openai_reasoning_summary: str
    azure_openai_text_verbosity: str
    openai_base_url: str | None
    openai_api_key: str | None
    azure_openai_endpoint: str | None
    azure_openai_api_key: str | None
    azure_openai_api_version: str | None
    azure_openai_deployment: str | None


@dataclass(frozen=True, slots=True)
class LLMRuntimeSettings:
    base_prompt: str
    service: LLMServiceSettings


@lru_cache(maxsize=1)
def get_llm_runtime_settings() -> LLMRuntimeSettings:
    service = LLMServiceSettings(
        llm_max_retries=int(API_CONFIG.get("llm_max_retries") or 10),
        openai_v1_min_rate_limit_retries=int(API_CONFIG.get("openai_v1_min_rate_limit_retries") or 10),
        azure_openai_timeout_seconds=float(API_CONFIG.get("azure_openai_timeout_seconds") or 120.0),
        azure_openai_sdk_max_retries=int(API_CONFIG.get("azure_openai_sdk_max_retries") or 3),
        groq_service_tier=str(API_CONFIG.get("groq_service_tier") or ""),
        azure_openai_reasoning_effort=str(API_CONFIG.get("azure_openai_reasoning_effort") or ""),
        azure_openai_reasoning_summary=str(API_CONFIG.get("azure_openai_reasoning_summary") or ""),
        azure_openai_text_verbosity=str(API_CONFIG.get("azure_openai_text_verbosity") or ""),
        openai_base_url=(
            str(API_CONFIG.get("openai_base_url")) if API_CONFIG.get("openai_base_url") is not None else None
        ),
        openai_api_key=(
            str(API_CONFIG.get("openai_api_key")) if API_CONFIG.get("openai_api_key") is not None else None
        ),
        azure_openai_endpoint=(
            str(API_CONFIG.get("azure_openai_endpoint"))
            if API_CONFIG.get("azure_openai_endpoint") is not None
            else None
        ),
        azure_openai_api_key=(
            str(API_CONFIG.get("azure_openai_api_key")) if API_CONFIG.get("azure_openai_api_key") is not None else None
        ),
        azure_openai_api_version=str(
            API_CONFIG.get("azure_openai_api_version") or os.getenv("AZURE_OPENAI_API_VERSION")
        )
        if (API_CONFIG.get("azure_openai_api_version") or os.getenv("AZURE_OPENAI_API_VERSION"))
        else None,
        azure_openai_deployment=(
            str(API_CONFIG.get("azure_openai_deployment"))
            if API_CONFIG.get("azure_openai_deployment") is not None
            else None
        ),
    )
    return LLMRuntimeSettings(base_prompt=BASE_PROMPT, service=service)


__all__ = ["LLMRuntimeSettings", "LLMServiceSettings", "get_llm_runtime_settings"]
