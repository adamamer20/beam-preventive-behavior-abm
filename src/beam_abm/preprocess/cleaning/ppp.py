from __future__ import annotations

import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from beam_abm.common.logging import get_logger
from beam_abm.settings import PPP_DATA_FILE

from .models import AnyType, Column

logger = get_logger(__name__)

_PPP_LOOKUP_CACHE: pd.Series | None = None
_LESS_THAN_KEYWORDS = ("less than", "up to", "under")
_GREATER_THAN_KEYWORDS = ("or more", "more than", "at least", "or above")


def _load_ppp_lookup() -> pd.Series:
    """Load PPP values from the configured CSV (cached across calls)."""
    global _PPP_LOOKUP_CACHE
    if _PPP_LOOKUP_CACHE is not None:
        return _PPP_LOOKUP_CACHE

    if not PPP_DATA_FILE or not os.path.exists(PPP_DATA_FILE):
        logger.warning(f"PPP data file missing or unspecified (expected at '{PPP_DATA_FILE}')")
        _PPP_LOOKUP_CACHE = pd.Series(dtype="float64")
        return _PPP_LOOKUP_CACHE

    try:
        ppp_df = pd.read_csv(PPP_DATA_FILE)
    except FileNotFoundError:
        logger.warning(f"PPP data file not found at '{PPP_DATA_FILE}'")
        _PPP_LOOKUP_CACHE = pd.Series(dtype="float64")
        return _PPP_LOOKUP_CACHE

    missing_columns = {"country_code", "ppp_value"} - set(ppp_df.columns)
    if missing_columns:
        logger.error(f"PPP data file missing expected columns: {missing_columns}")
        _PPP_LOOKUP_CACHE = pd.Series(dtype="float64")
        return _PPP_LOOKUP_CACHE

    lookup = (
        pd.to_numeric(ppp_df["ppp_value"], errors="coerce").rename(index=ppp_df["country_code"].str.upper()).dropna()
    )
    _PPP_LOOKUP_CACHE = lookup
    logger.info(f"Loaded PPP lookup with {len(_PPP_LOOKUP_CACHE)} country entries")
    return _PPP_LOOKUP_CACHE


def _extract_income_bounds(value: AnyType) -> tuple[float | None, float | None]:
    """Return lower/upper numeric bounds parsed from an income bracket string."""
    if not isinstance(value, str):
        return (None, None)

    text = value.strip()
    if not text:
        return (None, None)

    lowered = text.lower()
    if "prefer not" in lowered:
        return (None, None)

    tokens = re.findall(r"\d[\d\s.,]*", text)
    numbers: list[float] = []
    for token in tokens:
        normalized = token.replace(" ", "").replace(",", "")
        if normalized.count(".") > 1:
            normalized = normalized.replace(".", "")
        try:
            numbers.append(float(normalized))
        except ValueError:
            continue

    if not numbers:
        return (None, None)

    has_range_keyword = " to " in lowered or "-" in lowered

    if len(numbers) >= 2 and has_range_keyword:
        return (numbers[0], numbers[1])

    if any(keyword in lowered for keyword in _LESS_THAN_KEYWORDS):
        return (None, numbers[0])

    if any(keyword in lowered for keyword in _GREATER_THAN_KEYWORDS):
        return (numbers[-1], None)

    if len(numbers) >= 2:
        return (numbers[0], numbers[1])

    # Single point estimate
    return (numbers[0], numbers[0])


def income_ppp_normalized(
    column: pd.Series,
    *,
    df: pd.DataFrame | None = None,
    column_meta: Column | None = None,
    country_col: str = "country",
) -> pd.Series:
    """
    Convert textual income brackets into PPP-normalized numeric midpoints.

    Steps:
      1. Parse the lower/upper bounds for each bracket.
      2. Compute the midpoint, approximating open-ended bounds with typical widths.
      3. Divide the midpoint by the country's PPP value (EU27_2020 = 1 baseline).
    """
    del column_meta  # not used currently but preserved for compatibility
    result = pd.Series(np.nan, index=column.index, dtype="float64")

    if df is None:
        logger.warning("Cannot normalize income without dataframe context")
        return result
    if country_col not in df.columns:
        logger.warning(f"Country column '{country_col}' missing; skipping PPP normalization")
        return result

    ppp_lookup = _load_ppp_lookup()
    if ppp_lookup.empty:
        logger.warning("PPP lookup unavailable; skipping income normalization")
        return result

    countries = df.loc[column.index, country_col].astype("string").str.upper()
    ppp_values = countries.map(ppp_lookup)

    parsed_entries: list[tuple[object, str | None, float | None, float | None]] = []
    country_widths: defaultdict[str, list[float]] = defaultdict(list)
    all_widths: list[float] = []

    for idx in column.index:
        country = countries.get(idx)
        low, high = _extract_income_bounds(column.get(idx))
        parsed_entries.append((idx, None if pd.isna(country) else str(country), low, high))
        if low is not None and high is not None and high > low:
            width = high - low
            country_key = str(country) if not pd.isna(country) else None
            if country_key:
                country_widths[country_key].append(width)
            all_widths.append(width)

    width_by_country = {country: float(np.median(widths)) for country, widths in country_widths.items() if widths}
    fallback_width = float(np.median(all_widths)) if all_widths else None

    for idx, country, low, high in parsed_entries:
        if country is None:
            continue

        ppp_value = ppp_values.get(idx)
        if pd.isna(ppp_value) or ppp_value in {0}:
            continue

        midpoint: float | None
        if low is None and high is None:
            midpoint = None
        elif low is None:
            midpoint = high / 2 if high is not None else None
        elif high is None:
            width = width_by_country.get(country) or fallback_width
            if width is None:
                width = max(low * 0.25, 1.0)
            elevated_high = low + width
            midpoint = (low + elevated_high) / 2
        else:
            midpoint = (low + high) / 2

        if midpoint is None:
            continue

        result.loc[idx] = midpoint / ppp_value

    return result
