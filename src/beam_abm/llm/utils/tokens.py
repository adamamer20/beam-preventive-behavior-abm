"""Token counting utilities for LLM ABM package."""

from __future__ import annotations

from functools import lru_cache

import tiktoken

from beam_abm.common.logging import get_logger

logger = get_logger(__name__)

_warned_unknown_models: set[str] = set()


def _should_use_o200k_base(model: str) -> bool:
    model_l = model.lower()

    # Gemini models: closest practical approximation.
    if "gemini" in model_l:
        return True

    # OpenAI families that commonly use o200k_base. We keep this intentionally
    # conservative: unknown model names should not hard-fail token counting.
    if model_l.startswith("gpt-4o"):
        return True
    if model_l.startswith("gpt-4.1"):
        return True
    if model_l.startswith("gpt-5"):
        return True

    return False


@lru_cache(maxsize=256)
def _get_encoding(model: str):
    if _should_use_o200k_base(model):
        return tiktoken.get_encoding("o200k_base")

    try:
        return tiktoken.encoding_for_model(model)
    except Exception as e:
        if model not in _warned_unknown_models:
            _warned_unknown_models.add(model)
            logger.warning(
                "Failed to map model '{}' to a tokeniser ({}). Falling back to o200k_base.",
                model,
                e,
            )
        return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count the number of tokens in a string.

    Parameters
    ----------
    text : str
        The text to count tokens for.
    model : str
        The name of the model to use for tokenization.

    Returns
    -------
    int
        The number of tokens in the text.
    """
    logger.debug("Counting tokens for model: {}, text length: {}", model, len(text))

    encoding = _get_encoding(model)
    token_count = len(encoding.encode(text))
    logger.debug("Token count: {}", token_count)
    return token_count
