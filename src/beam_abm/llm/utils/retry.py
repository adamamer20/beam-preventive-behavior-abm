"""Retry utilities for LLM ABM package."""

import time

from openai import RateLimitError

from beam_abm.common.logging import get_logger
from beam_abm.llm.config import get_llm_runtime_settings

logger = get_logger(__name__)


def retry_on_quota_exceeded(func):
    """Decorator to retry API calls when rate limited."""

    def wrapper(*args, **kwargs):
        max_retries = get_llm_runtime_settings().service.llm_max_retries
        retry_count = 0

        logger.debug(f"Starting retry wrapper for function: {func.__name__}")

        while retry_count < max_retries:
            try:
                result = func(*args, **kwargs)
                if retry_count > 0:
                    logger.info(f"Successfully executed {func.__name__} after {retry_count} retries")
                return result
            except Exception as e:
                # Check if this is a quota exceeded error (429)
                if "429" in str(e) or "quota exceeded" in str(e).lower() or isinstance(e, RateLimitError):
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.warning(
                            f"Quota exceeded (429) in {func.__name__}. Waiting 10 seconds and retrying... (Attempt {retry_count}/{max_retries})"
                        )
                        print(
                            f"Quota exceeded (429). Waiting 10 seconds and retrying... (Attempt {retry_count}/{max_retries})"
                        )
                        time.sleep(10)  # Wait 10 seconds before retrying
                    else:
                        logger.error(f"Max retries ({max_retries}) reached for {func.__name__}. Giving up.")
                        print("Max retries reached. Giving up.")
                        raise
                else:
                    # If it's a different error, re-raise it
                    logger.error(f"Non-retry error in {func.__name__}: {e}")
                    raise

    return wrapper
