from __future__ import annotations

import numpy as np
import pandas as pd

from beam_abm.common.logging import get_logger

from .models import Column

logger = get_logger(__name__)

_COUNTRY_CODE_TO_NAME: dict[str, str] = {
    "DE": "Germany",
    "ES": "Spain",
    "FR": "France",
    "HU": "Hungary",
    "IT": "Italy",
    "UK": "United Kingdom",
}
_MISSING_BIRTH_VALUES = {"", "Missing", "nan", "NaN"}


def _map_country_codes(series: pd.Series) -> pd.Series:
    return series.astype("string").str.upper().map(_COUNTRY_CODE_TO_NAME)


def _clean_birth_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    return cleaned.replace(list(_MISSING_BIRTH_VALUES), pd.NA)


def born_in_residence_country(
    column: pd.Series,
    *,
    df: pd.DataFrame | None = None,
    column_meta: Column | None = None,
    birth_col: str = "birth",
    country_col: str = "country",
) -> pd.Series:
    del column_meta
    result = pd.Series(np.nan, index=column.index, dtype="float64")
    if df is None:
        logger.warning("Cannot derive born_in_residence_country without dataframe context")
        return result
    if birth_col not in df.columns or country_col not in df.columns:
        logger.warning("Missing birth/country columns for born_in_residence_country derivation")
        return result
    birth = _clean_birth_series(df.loc[column.index, birth_col])
    country = _map_country_codes(df.loc[column.index, country_col])
    valid = birth.notna() & country.notna()
    result.loc[valid] = birth.loc[valid].eq(country.loc[valid]).astype(float)
    return result


def parent_born_abroad(
    column: pd.Series,
    *,
    df: pd.DataFrame | None = None,
    column_meta: Column | None = None,
    birth_m_col: str = "birth_m",
    birth_f_col: str = "birth_f",
    country_col: str = "country",
) -> pd.Series:
    del column_meta
    result = pd.Series(np.nan, index=column.index, dtype="float64")
    if df is None:
        logger.warning("Cannot derive parent_born_abroad without dataframe context")
        return result
    if birth_m_col not in df.columns or birth_f_col not in df.columns or country_col not in df.columns:
        logger.warning("Missing birth_m/birth_f/country columns for parent_born_abroad derivation")
        return result
    birth_m = _clean_birth_series(df.loc[column.index, birth_m_col])
    birth_f = _clean_birth_series(df.loc[column.index, birth_f_col])
    country = _map_country_codes(df.loc[column.index, country_col])
    country_known = country.notna()
    abroad_mask = (birth_m.notna() & birth_m.ne(country)) | (birth_f.notna() & birth_f.ne(country))
    both_parents_known = birth_m.notna() & birth_f.notna()
    result.loc[country_known & abroad_mask] = 1.0
    result.loc[country_known & ~abroad_mask & both_parents_known] = 0.0
    return result
