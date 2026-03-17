"""PE_ref_P table assembly and bundle construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .reference_model_loading import (
    RefStateModel,
    _compute_ref_term_scaler_stats,
    _load_ref_model,
    _predict_model,
    default_ref_state_models,
)
from .reference_perturbation_scaling import (
    _country_quantile_maps,
    _country_std_map,
    _normalize_shocks,
    _resolve_row_ids,
    _resolve_shock_var,
)


def _median_abs_numeric(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(np.abs(finite)))


@dataclass(frozen=True, slots=True)
class SignalReferenceBundle:
    """Reference payload attached to prompt metadata.

    Parameters
    ----------
    row_signal:
        Mapping ``(row_id, signal_id)`` -> metadata dict containing per-state
        `pe_ref_p` and `pe_ref_p_std` values.
    signal_meta:
        Signal-level derived ranking/tier metadata.
    model_status:
        Diagnostics for model resolution/loading.
    """

    row_signal: dict[tuple[str, str], dict[str, Any]]
    signal_meta: dict[str, dict[str, Any]]
    model_status: dict[str, Any]


def compute_pe_ref_p_table(
    *,
    profiles_df: pd.DataFrame,
    source_df: pd.DataFrame,
    signals: list[dict[str, Any]],
    mutable_state: list[str],
    ref_state_models: dict[str, str],
    modeling_dir: Path,
    id_col: str,
    country_col: str,
    alpha_secondary: float,
    epsilon_ref: float,
    top_n_primary: int,
    strict_missing_models: bool,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, Any]]:
    """Compute PE_ref_P rows and derived signal metadata."""

    rows_n = len(profiles_df)
    if rows_n == 0:
        return pd.DataFrame(), {}, {"loaded": {}, "missing": list(ref_state_models.values())}

    ids = _resolve_row_ids(profiles_df, id_col=id_col)
    countries = (
        profiles_df[country_col].astype(str).to_numpy(dtype=object)
        if country_col in profiles_df.columns
        else np.asarray(["GLOBAL"] * rows_n, dtype=object)
    )

    loaded_models: dict[str, RefStateModel] = {}
    missing_models: dict[str, str] = {}
    for state_var in mutable_state:
        requested = ref_state_models.get(state_var)
        if not requested:
            missing_models[state_var] = "missing mapping"
            continue
        model_name = str(requested).strip()
        model = _load_ref_model(modeling_dir=modeling_dir, model_name=model_name)
        if model is None:
            missing_models[state_var] = requested
            continue
        means, stds = _compute_ref_term_scaler_stats(model.terms, source_df)
        model = RefStateModel(
            model_name=model.model_name,
            terms=model.terms,
            coefficients=model.coefficients,
            intercept=model.intercept,
            thresholds=model.thresholds,
            term_means=means,
            term_stds=stds,
        )
        loaded_models[state_var] = model

    if strict_missing_models and len(loaded_models) < len(mutable_state):
        missing_keys = [v for v in mutable_state if v not in loaded_models]
        msg = f"Missing reduced P-models for mutable state vars: {missing_keys}"
        raise ValueError(msg)

    records: list[dict[str, Any]] = []

    for signal in signals:
        if not isinstance(signal, dict):
            continue
        signal_id = str(signal.get("id") or "").strip()
        if not signal_id:
            continue
        signal_text = str(signal.get("text") or "").strip()

        shocks = _normalize_shocks(signal)
        if not shocks:
            continue

        low_df = profiles_df.copy()
        high_df = profiles_df.copy()

        delta_x_primary = np.full(rows_n, np.nan, dtype=np.float64)
        delta_z_primary = np.full(rows_n, np.nan, dtype=np.float64)
        primary_shock_var = ""
        primary_shock_type = ""
        primary_shock_direction = ""
        primary_shock_low_q: float | None = None
        primary_shock_high_q: float | None = None
        primary_shock_base_values = np.full(rows_n, np.nan, dtype=np.float64)
        primary_shock_low_values = np.full(rows_n, np.nan, dtype=np.float64)
        primary_shock_high_values = np.full(rows_n, np.nan, dtype=np.float64)

        for s_idx, shock in enumerate(shocks):
            requested_var = str(shock["var"])
            var = _resolve_shock_var(var=requested_var, profiles_df=profiles_df, source_df=source_df)
            if var not in low_df.columns:
                low_df[var] = np.nan
                high_df[var] = np.nan

            if shock["type"] == "binary":
                low_vals = np.full(rows_n, float(shock["low"]), dtype=np.float64)
                high_vals = np.full(rows_n, float(shock["high"]), dtype=np.float64)
            else:
                q_low = float(shock["low_quantile"])
                q_high = float(shock["high_quantile"])
                by_country, glob = _country_quantile_maps(
                    source_df,
                    var=var,
                    country_col=country_col,
                    q_low=q_low,
                    q_high=q_high,
                )
                low_vals = np.empty(rows_n, dtype=np.float64)
                high_vals = np.empty(rows_n, dtype=np.float64)
                for i, c in enumerate(countries):
                    lo, hi = by_country.get(str(c), glob)
                    low_vals[i] = lo
                    high_vals[i] = hi

            low_df[var] = low_vals
            high_df[var] = high_vals

            if s_idx == 0:
                primary_shock_var = var
                primary_shock_type = str(shock["type"])
                primary_shock_direction = str(shock.get("direction") or "")
                primary_shock_low_q = (
                    float(shock["low_quantile"]) if shock["type"] == "continuous" and "low_quantile" in shock else None
                )
                primary_shock_high_q = (
                    float(shock["high_quantile"])
                    if shock["type"] == "continuous" and "high_quantile" in shock
                    else None
                )
                base_vals = pd.to_numeric(profiles_df[var], errors="coerce").to_numpy(dtype=np.float64)
                primary_shock_base_values = base_vals
                primary_shock_low_values = low_vals
                primary_shock_high_values = high_vals
                delta_x = high_vals - low_vals
                std_map, std_global = _country_std_map(source_df, var=var, country_col=country_col)
                std_vals = np.empty(rows_n, dtype=np.float64)
                for i, c in enumerate(countries):
                    std_vals[i] = std_map.get(str(c), std_global)
                with np.errstate(divide="ignore", invalid="ignore"):
                    delta_z = np.where(np.isfinite(std_vals) & (std_vals > 0.0), delta_x / std_vals, np.nan)
                delta_x_primary = delta_x
                delta_z_primary = delta_z

        for state_var, model in loaded_models.items():
            pred_low = _predict_model(model, low_df)
            pred_high = _predict_model(model, high_df)
            pe = pred_high - pred_low
            with np.errstate(divide="ignore", invalid="ignore"):
                pe_std = np.where(np.isfinite(delta_z_primary) & (delta_z_primary != 0.0), pe / delta_z_primary, np.nan)
            for i in range(rows_n):
                records.append(
                    {
                        "id": str(ids[i]),
                        "country": str(countries[i]),
                        "signal_id": signal_id,
                        "signal_text": signal_text,
                        "state_var": state_var,
                        "pe_ref_p": float(pe[i]),
                        "pe_ref_p_std": float(pe_std[i]) if np.isfinite(pe_std[i]) else float("nan"),
                        "delta_x": float(delta_x_primary[i]) if np.isfinite(delta_x_primary[i]) else float("nan"),
                        "delta_z": float(delta_z_primary[i]) if np.isfinite(delta_z_primary[i]) else float("nan"),
                        "shock_var": primary_shock_var,
                        "shock_type": primary_shock_type,
                        "shock_direction": primary_shock_direction,
                        "shock_from_quantile": (
                            float(primary_shock_low_q) if primary_shock_low_q is not None else float("nan")
                        ),
                        "shock_to_quantile": (
                            float(primary_shock_high_q) if primary_shock_high_q is not None else float("nan")
                        ),
                        "shock_base_value": (
                            float(primary_shock_base_values[i])
                            if np.isfinite(primary_shock_base_values[i])
                            else float("nan")
                        ),
                        "shock_low_value": (
                            float(primary_shock_low_values[i])
                            if np.isfinite(primary_shock_low_values[i])
                            else float("nan")
                        ),
                        "shock_high_value": (
                            float(primary_shock_high_values[i])
                            if np.isfinite(primary_shock_high_values[i])
                            else float("nan")
                        ),
                    }
                )

    pe_df = pd.DataFrame.from_records(records)
    if pe_df.empty:
        return pe_df, {}, {"loaded": sorted(loaded_models), "missing": missing_models}

    signal_meta: dict[str, dict[str, Any]] = {}
    grouped = pe_df.groupby(["signal_id", "state_var"], dropna=False)
    med_raw = grouped["pe_ref_p"].apply(_median_abs_numeric).reset_index(name="median_abs_pe_ref_p")
    med_std = grouped["pe_ref_p_std"].apply(_median_abs_numeric).reset_index(name="median_abs_pe_ref_p_std")
    med_df = med_raw.merge(med_std, on=["signal_id", "state_var"], how="inner")

    for signal_id, sub in med_df.groupby("signal_id", dropna=False):
        by_var_abs = {str(r["state_var"]): float(r["median_abs_pe_ref_p"]) for _, r in sub.iterrows()}
        by_var_abs_std = {str(r["state_var"]): float(r["median_abs_pe_ref_p_std"]) for _, r in sub.iterrows()}

        ranked = sorted(by_var_abs.items(), key=lambda kv: (-kv[1], kv[0]))
        max_val = ranked[0][1] if ranked else 0.0
        top_n = max(1, int(top_n_primary))
        primary_vars = {v for v, _ in ranked[:top_n]}

        tiers: dict[str, str] = {}
        for var, value in ranked:
            std_val = by_var_abs_std.get(var, float("nan"))
            if np.isfinite(std_val) and std_val < float(epsilon_ref):
                tiers[var] = "none"
            elif var in primary_vars:
                tiers[var] = "primary"
            elif max_val > 0.0 and value >= float(alpha_secondary) * float(max_val):
                tiers[var] = "secondary"
            else:
                tiers[var] = "weak"

        ranking = [
            {
                "var": var,
                "rank": idx + 1,
                "median_abs_pe_ref_p": float(value),
                "median_abs_pe_ref_p_std": float(by_var_abs_std.get(var, float("nan"))),
                "tier": tiers.get(var, "none"),
            }
            for idx, (var, value) in enumerate(ranked)
        ]

        signal_meta[str(signal_id)] = {
            "median_abs_pe_ref_p_by_var": by_var_abs,
            "median_abs_pe_ref_p_std_by_var": by_var_abs_std,
            "tier_by_var": tiers,
            "ranking": ranking,
            "alpha_secondary": float(alpha_secondary),
            "epsilon_ref": float(epsilon_ref),
            "top_n_primary": int(top_n),
        }

    # Attach derived tiers per row.
    tier_vals: list[str] = []
    for _, r in pe_df.iterrows():
        signal_id = str(r["signal_id"])
        state_var = str(r["state_var"])
        tier = signal_meta.get(signal_id, {}).get("tier_by_var", {}).get(state_var, "none")
        tier_vals.append(str(tier))
    pe_df["tier"] = tier_vals

    model_status = {
        "loaded": {k: v.model_name for k, v in loaded_models.items()},
        "missing": missing_models,
    }
    return pe_df, signal_meta, model_status


def build_signal_reference_bundle(
    *,
    rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]] | None,
    signals: list[dict[str, Any]],
    mutable_state: list[str],
    spec: dict[str, Any],
    id_col: str,
    country_col: str,
    modeling_dir: Path,
    alpha_secondary: float,
    epsilon_ref: float,
    top_n_primary: int,
    strict_missing_models: bool,
) -> SignalReferenceBundle:
    """Build per-(row, signal) reference payload for prompt metadata."""

    if not rows:
        return SignalReferenceBundle(row_signal={}, signal_meta={}, model_status={})

    profiles_df = pd.DataFrame(rows)
    if source_rows:
        source_df = pd.DataFrame(source_rows)
    else:
        source_df = profiles_df.copy()

    ref_map = spec.get("ref_state_models") if isinstance(spec.get("ref_state_models"), dict) else {}
    if not ref_map:
        ref_map = default_ref_state_models()

    pe_df, signal_meta, model_status = compute_pe_ref_p_table(
        profiles_df=profiles_df,
        source_df=source_df,
        signals=signals,
        mutable_state=mutable_state,
        ref_state_models={str(k): str(v) for k, v in ref_map.items()},
        modeling_dir=Path(modeling_dir),
        id_col=id_col,
        country_col=country_col,
        alpha_secondary=float(alpha_secondary),
        epsilon_ref=float(epsilon_ref),
        top_n_primary=int(top_n_primary),
        strict_missing_models=bool(strict_missing_models),
    )

    row_signal: dict[tuple[str, str], dict[str, Any]] = {}
    if not pe_df.empty:
        for (row_id, signal_id), sub in pe_df.groupby(["id", "signal_id"], dropna=False):
            by_var = {str(r["state_var"]): float(r["pe_ref_p"]) for _, r in sub.iterrows()}
            by_var_std = {
                str(r["state_var"]): float(r["pe_ref_p_std"]) if np.isfinite(float(r["pe_ref_p_std"])) else float("nan")
                for _, r in sub.iterrows()
            }
            tiers = signal_meta.get(str(signal_id), {}).get("tier_by_var", {})
            ranking = signal_meta.get(str(signal_id), {}).get("ranking", [])
            delta_z = float(np.nanmedian(pd.to_numeric(sub["delta_z"], errors="coerce")))
            delta_x = float(np.nanmedian(pd.to_numeric(sub["delta_x"], errors="coerce")))
            row_signal[(str(row_id), str(signal_id))] = {
                "pe_ref_p": by_var,
                "pe_ref_p_std": by_var_std,
                "pe_ref_p_tier_by_var": tiers,
                "pe_ref_p_ranking": ranking,
                "pe_ref_p_delta_z": delta_z if np.isfinite(delta_z) else None,
                "pe_ref_p_delta_x": delta_x if np.isfinite(delta_x) else None,
            }

    return SignalReferenceBundle(
        row_signal=row_signal,
        signal_meta=signal_meta,
        model_status=model_status,
    )
