"""Shared backoff helpers for transient LLM retries."""

from __future__ import annotations

import random


def jittered_exponential_backoff(attempt_index: int, *, base: float, cap: float) -> float:
    """Return exponential backoff with uniform jitter."""

    delay = min(cap, base * (2.0**attempt_index))
    return delay * (0.5 + random.random())
