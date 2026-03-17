"""Shock normalization and perturbation scaling helpers for PE_ref_P."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from .reference_model_loading import _safe_float

_QUANTILE_RE = re.compile(r"^q(?P<pct>\d{1,2})$", re.IGNORECASE)
_SHOCK_VAR_ALIASES: dict[str, tuple[str, ...]] = {
    "health_poor": ("health_good",),
    "health_good": ("health_poor",),
}


def _parse_quantile_token(raw: object) -> float | None:
    if raw is None:
        return None
    val = _safe_float(raw)
    if val is not None:
        return float(min(max(val, 0.0), 1.0))
    token = str(raw).strip().lower()
    m = _QUANTILE_RE.match(token)
    if not m:
        return None
    pct = int(m.group("pct"))
    return float(min(max(pct / 100.0, 0.0), 1.0))


def _normalize_shocks(
    signal: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_shocks = signal.get("shocks")
    if isinstance(raw_shocks, list) and raw_shocks:
        base = [s for s in raw_shocks if isinstance(s, dict)]
    else:
        raw_single = signal.get("shock")
        base = [raw_single] if isinstance(raw_single, dict) else []

    out: list[dict[str, Any]] = []
    for item in base:
        var = str(item.get("var") or item.get("variable") or "").strip()
        if not var:
            continue
        s_type = str(item.get("type") or "continuous").strip().lower()
        direction = str(item.get("direction") or "increase").strip().lower()

        if s_type == "binary":
            low = _safe_float(item.get("from"))
            high = _safe_float(item.get("to"))
            if low is None:
                low = 0.0
            if high is None:
                high = 1.0
            out.append(
                {
                    "var": var,
                    "type": "binary",
                    "low": float(low),
                    "high": float(high),
                    "direction": direction,
                }
            )
            continue

        low_q = _parse_quantile_token(item.get("from_quantile"))
        high_q = _parse_quantile_token(item.get("to_quantile"))
        if "from" in item:
            low_q = _parse_quantile_token(item.get("from"))
        if "to" in item:
            high_q = _parse_quantile_token(item.get("to"))
        if low_q is None or high_q is None:
            signal_id = str(signal.get("id") or "<unknown>")
            msg = f"Continuous shock requires explicit from/to quantiles for signal '{signal_id}', variable '{var}'."
            raise ValueError(msg)

        if direction == "decrease" and low_q < high_q:
            low_q, high_q = high_q, low_q

        out.append(
            {
                "var": var,
                "type": "continuous",
                "low_quantile": float(low_q),
                "high_quantile": float(high_q),
                "direction": direction,
            }
        )

    return out


def _country_quantile_maps(
    source_df: pd.DataFrame,
    *,
    var: str,
    country_col: str,
    q_low: float,
    q_high: float,
) -> tuple[dict[str, tuple[float, float]], tuple[float, float]]:
    if var not in source_df.columns:
        return {}, (float("nan"), float("nan"))

    vals = pd.to_numeric(source_df[var], errors="coerce")
    global_low = float(vals.quantile(q_low)) if vals.notna().any() else float("nan")
    global_high = float(vals.quantile(q_high)) if vals.notna().any() else float("nan")

    out: dict[str, tuple[float, float]] = {}
    if country_col in source_df.columns:
        groups = source_df.groupby(source_df[country_col].astype(str), dropna=False)
        for country, sub in groups:
            sub_vals = pd.to_numeric(sub[var], errors="coerce")
            if sub_vals.notna().any():
                out[str(country)] = (float(sub_vals.quantile(q_low)), float(sub_vals.quantile(q_high)))
    return out, (global_low, global_high)


def _country_std_map(source_df: pd.DataFrame, *, var: str, country_col: str) -> tuple[dict[str, float], float]:
    if var not in source_df.columns:
        return {}, float("nan")

    vals = pd.to_numeric(source_df[var], errors="coerce")
    global_std = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else float("nan")

    out: dict[str, float] = {}
    if country_col in source_df.columns:
        groups = source_df.groupby(source_df[country_col].astype(str), dropna=False)
        for country, sub in groups:
            sub_vals = pd.to_numeric(sub[var], errors="coerce")
            out[str(country)] = float(sub_vals.std(ddof=1)) if sub_vals.notna().sum() > 1 else float("nan")

    return out, global_std


def _resolve_row_ids(profiles_df: pd.DataFrame, *, id_col: str) -> np.ndarray:
    if id_col in profiles_df.columns:
        return profiles_df[id_col].astype(str).to_numpy()
    return np.asarray([str(i) for i in range(len(profiles_df))], dtype=object)


def _resolve_shock_var(*, var: str, profiles_df: pd.DataFrame, source_df: pd.DataFrame) -> str:
    if var in profiles_df.columns or var in source_df.columns:
        return var
    aliases = _SHOCK_VAR_ALIASES.get(var, ())
    for alias in aliases:
        if alias in profiles_df.columns or alias in source_df.columns:
            return str(alias)
    return var
