"""Regex-based fallback parsing for LLM outputs.

This module is deliberately conservative: it tries to recover a *single* JSON-like
object first (including JSON contained in markdown code fences), and only then falls
back to scalar extraction.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Final

from beam_abm.llm.schemas.predictions import extract_scalar_prediction

__all__ = [
    "regex_extract_prediction",
    "regex_extract_scalar",
]

_CODE_FENCE_RE: Final[re.Pattern[str]] = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_TRAILING_COMMA_RE: Final[re.Pattern[str]] = re.compile(r",\s*([\]}])")

_KEYED_FLOAT_RE: Final[re.Pattern[str]] = re.compile(
    r'(?i)"?(prediction|p_yes|expected)"?\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
)
_FLOAT_RE: Final[re.Pattern[str]] = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

_PAIR_AB_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?im)^(?:\s*)(A|B)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\b"
)
_PAIR_KEYED_RE: Final[re.Pattern[str]] = re.compile(
    r'(?i)"?(prediction|p_yes|expected)"?\s*[_ ]?(A|B)"?\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
)


def _iter_balanced_json_objects(text: str) -> Iterable[str]:
    """Yield candidate JSON object substrings with balanced braces."""

    start: int | None = None
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
            continue

        if ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start : i + 1]
                start = None


def _clean_json_candidate(candidate: str) -> str:
    candidate = candidate.strip()
    # Remove common markdown artifacts.
    if candidate.startswith("```") and candidate.endswith("```"):
        candidate = candidate.strip("`")
    # Fix a common LLM-json issue: trailing commas.
    candidate = _TRAILING_COMMA_RE.sub(r"\\1", candidate)
    return candidate


def _try_parse_json(candidate: str) -> object | None:
    try:
        return json.loads(_clean_json_candidate(candidate))
    except Exception:
        return None


def _extract_paired_from_text(raw_text: str) -> dict[str, object] | None:
    # Case 1: lines like `A: 0.2` and `B: 0.5`.
    found = {}
    for m in _PAIR_AB_LINE_RE.finditer(raw_text):
        found[m.group(1).upper()] = float(m.group(2))
    if set(found) >= {"A", "B"}:
        return {
            "A": {"prediction": float(found["A"])},
            "B": {"prediction": float(found["B"])},
        }

    # Case 2: keyed variants like prediction_A / prediction_B.
    keyed: dict[str, dict[str, float]] = {"A": {}, "B": {}}
    for m in _PAIR_KEYED_RE.finditer(raw_text):
        key = str(m.group(1)).lower()
        side = str(m.group(2)).upper()
        val = float(m.group(3))
        keyed[side][key] = val
    if keyed["A"] and keyed["B"]:
        # Prefer `prediction` if present; otherwise keep whatever key matched.
        def _wrap(side: str) -> dict[str, float]:
            d = keyed[side]
            if "prediction" in d:
                return {"prediction": float(d["prediction"])}
            if "p_yes" in d:
                return {"p_yes": float(d["p_yes"])}
            if "expected" in d:
                return {"expected": float(d["expected"])}
            # Fallback to any key.
            k0 = next(iter(d.keys()))
            return {k0: float(d[k0])}

        return {"A": _wrap("A"), "B": _wrap("B")}

    return None


def regex_extract_prediction(raw_text: str | None, *, allow_scalar_fallback: bool) -> object | None:
    if not raw_text:
        return None

    # 1) Prefer JSON inside markdown code fences.
    best: object | None = None
    for m in _CODE_FENCE_RE.finditer(raw_text):
        parsed = _try_parse_json(m.group(1))
        if parsed is not None:
            best = parsed

    # 2) Otherwise try balanced JSON objects anywhere in text.
    if best is None:
        for candidate in _iter_balanced_json_objects(raw_text):
            parsed = _try_parse_json(candidate)
            if parsed is not None:
                best = parsed

    # 3) If JSON parsing succeeded, return it.
    if best is not None:
        return best

    if allow_scalar_fallback:
        # Paired-profile fallback: recover A/B if both present.
        paired = _extract_paired_from_text(raw_text)
        if paired is not None:
            return paired

        key_match = _KEYED_FLOAT_RE.search(raw_text)
        if key_match:
            key = key_match.group(1)
            try:
                return {key: float(key_match.group(2))}
            except (TypeError, ValueError):
                return None
        float_match = _FLOAT_RE.search(raw_text)
        if float_match:
            try:
                return float(float_match.group(0))
            except (TypeError, ValueError):
                return None
    return None


def regex_extract_scalar(raw_text: str | None) -> float | None:
    parsed = regex_extract_prediction(raw_text, allow_scalar_fallback=True)
    if parsed is None:
        return None
    if isinstance(parsed, int | float):
        return float(parsed)
    scalar = extract_scalar_prediction(parsed)
    if scalar is None:
        return None
    return float(scalar)
