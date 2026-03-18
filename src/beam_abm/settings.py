"""Cross-domain settings and environment defaults."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

SEED = int(os.getenv("SEED", "42"))

PREPROCESSING_OUTPUT_DIR = os.getenv("PREPROCESSING_OUTPUT_DIR", "preprocess/output")
CLEAN_PROCESSED_FILENAME = os.getenv("CLEAN_PROCESSED_FILENAME", "clean_processed_survey.csv")

CSV_FILE_CLEAN = os.path.join(PREPROCESSING_OUTPUT_DIR, CLEAN_PROCESSED_FILENAME)

PPP_DATA_FILENAME = os.getenv("PPP_DATA_FILENAME", "eurostat_ppp_2024.csv")
PPP_DATA_FILE = os.path.join(PREPROCESSING_OUTPUT_DIR, PPP_DATA_FILENAME)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/e5-large-v2")

API_CONFIG = {
    "vllm_base_url": os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
    "vllm_api_key": os.getenv("VLLM_API_KEY", "vllm"),
    "share_url": os.getenv("SHARE_URL"),
    "download_url": os.getenv("DOWNLOAD_URL"),
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "openai_base_url": os.getenv("OPENAI_BASE_URL"),
    "azure_openai_api_key": os.getenv("AZURE_OPENAI_API_KEY"),
    "azure_openai_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "azure_openai_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    "azure_openai_reasoning_effort": os.getenv("AZURE_OPENAI_REASONING_EFFORT", ""),
    "azure_openai_reasoning_summary": os.getenv("AZURE_OPENAI_REASONING_SUMMARY", "detailed"),
    "azure_openai_text_verbosity": os.getenv("AZURE_OPENAI_TEXT_VERBOSITY", "medium"),
    "azure_openai_timeout_seconds": float(os.getenv("AZURE_OPENAI_TIMEOUT_SECONDS", "120")),
    "llm_max_retries": int(os.getenv("LLM_MAX_RETRIES", "10")),
    "azure_openai_sdk_max_retries": int(os.getenv("AZURE_OPENAI_SDK_MAX_RETRIES", "3")),
    "groq_service_tier": os.getenv("GROQ_SERVICE_TIER", "flex"),
    "openai_v1_min_rate_limit_retries": int(os.getenv("OPENAI_V1_MIN_RATE_LIMIT_RETRIES", "10")),
}

__all__ = [
    "SEED",
    "PREPROCESSING_OUTPUT_DIR",
    "CSV_FILE_CLEAN",
    "PPP_DATA_FILE",
    "EMBEDDING_MODEL",
    "API_CONFIG",
]
