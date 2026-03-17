"""Analysis and surrogate utilities for sensitivity outputs."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from scipy.stats import qmc, spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBRegressor

from .core import DEFAULT_PARAMETER_BOUNDS


def compute_morris_indices(
    *,
    design_df: pl.DataFrame,
    delta_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compute Morris ``mu*`` and ``sigma`` indices."""
    if delta_df.is_empty():
        raise ValueError("delta_df is empty; Morris indices cannot be computed.")

    steps = design_df.filter(pl.col("prev_point_id").is_not_null() & pl.col("changed_param").is_not_null()).select(
        pl.col("point_id").cast(pl.Int64),
        pl.col("prev_point_id").cast(pl.Int64),
        pl.col("changed_param").cast(pl.Utf8),
        pl.col("step_size").cast(pl.Float64),
    )

    current = delta_df.select(
        pl.col("point_id").cast(pl.Int64),
        pl.col("scenario").cast(pl.Utf8),
        pl.col("output").cast(pl.Utf8),
        pl.col("delta").cast(pl.Float64),
    )
    previous = current.rename({"point_id": "prev_point_id", "delta": "delta_prev"})

    joined = (
        steps.join(current, on="point_id", how="inner")
        .join(previous, on=["prev_point_id", "scenario", "output"], how="inner")
        .with_columns(((pl.col("delta") - pl.col("delta_prev")) / pl.col("step_size")).alias("ee"))
    )

    per_scenario = (
        joined.group_by("changed_param", "scenario", "output")
        .agg(
            pl.col("ee").abs().mean().alias("mu_star"),
            pl.col("ee").std(ddof=1).fill_null(0.0).alias("sigma"),
            pl.len().alias("n_effects"),
        )
        .rename({"changed_param": "parameter"})
        .sort(["scenario", "output", "mu_star"], descending=[False, False, True])
    )

    aggregate = (
        per_scenario.group_by("parameter")
        .agg(
            pl.col("mu_star").median().alias("mu_star_median"),
            pl.col("mu_star").max().alias("mu_star_max"),
            pl.col("sigma").median().alias("sigma_median"),
            pl.col("sigma").max().alias("sigma_max"),
            pl.col("n_effects").sum().alias("n_effects_total"),
        )
        .sort("mu_star_max", descending=True)
    )
    return per_scenario, aggregate


def select_shortlist(
    *,
    aggregate_indices: pl.DataFrame,
    target_size: int,
    max_size: int,
    sigma_quantile: float,
) -> pl.DataFrame:
    """Select global-SA parameters from aggregate Morris statistics."""
    if aggregate_indices.is_empty():
        raise ValueError("aggregate_indices is empty; cannot select shortlist.")

    ranked = aggregate_indices.sort("mu_star_max", descending=True).with_row_index(name="importance_rank", offset=1)
    sigma_threshold = float(ranked.select(pl.col("sigma_max").quantile(sigma_quantile)).item())

    importance_selected = ranked.head(target_size).select("parameter").to_series().to_list()
    sigma_selected = ranked.filter(pl.col("sigma_max") >= sigma_threshold).select("parameter").to_series().to_list()

    ordered_union: list[str] = []
    for parameter_name in [*importance_selected, *sigma_selected]:
        if parameter_name not in ordered_union:
            ordered_union.append(parameter_name)

    min_size = min(5, ranked.height)
    if len(ordered_union) < min_size:
        for parameter_name in ranked.select("parameter").to_series().to_list():
            if parameter_name not in ordered_union:
                ordered_union.append(parameter_name)
            if len(ordered_union) >= min_size:
                break

    ordered_union = ordered_union[:max_size]
    shortlist = ranked.filter(pl.col("parameter").is_in(ordered_union)).with_columns(
        pl.col("parameter").is_in(importance_selected).alias("selected_by_importance"),
        (pl.col("sigma_max") >= sigma_threshold).alias("selected_by_sigma"),
    )

    ordered_rank = {name: index + 1 for index, name in enumerate(ordered_union)}
    shortlist = shortlist.with_columns(
        pl.col("parameter").replace_strict(ordered_rank).cast(pl.Int64).alias("shortlist_rank")
    )
    return shortlist.sort("shortlist_rank")


def _fit_surrogate_model(
    *,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[XGBRegressor, float]:
    """Fit an XGBoost surrogate and return model with validation R²."""
    model = XGBRegressor(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=1,
    )

    if x.shape[0] >= 24:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=seed,
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        score = float(r2_score(y_test, y_pred))
    else:
        folds = max(3, min(5, x.shape[0]))
        cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
        scores = cross_val_score(model, x, y, cv=cv, scoring="r2")
        score = float(np.nanmean(scores))
        model.fit(x, y)

    return model, score


def _sanitize_name(raw: str) -> str:
    return raw.replace("/", "_").replace(" ", "_").replace(":", "_").replace("|", "_")


def _signed_label(value: float, *, eps: float = 1e-9) -> str:
    if not np.isfinite(value) or abs(value) < eps:
        return "zero"
    if value > 0.0:
        return "positive"
    return "negative"


def compute_lhc_spearman_directionality(
    *,
    lhc_delta: pl.DataFrame,
    parameter_names: tuple[str, ...],
) -> pl.DataFrame:
    """Compute Spearman sign diagnostics for LHC deltas."""
    required = {"scenario", "output", "delta"}
    missing = required.difference(set(lhc_delta.columns))
    if missing:
        msg = f"lhc_delta is missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    existing_parameters = tuple(name for name in parameter_names if name in lhc_delta.columns)
    if not existing_parameters:
        return pl.DataFrame()

    rows: list[dict[str, float | int | str]] = []
    for scenario_key, group_df in lhc_delta.group_by("scenario", "output"):
        scenario_name = str(scenario_key[0]) if isinstance(scenario_key, tuple) else str(scenario_key)
        output_name = str(scenario_key[1]) if isinstance(scenario_key, tuple) else ""
        y = group_df.get_column("delta").cast(pl.Float64).to_numpy()

        for parameter_name in existing_parameters:
            x = group_df.get_column(parameter_name).cast(pl.Float64).to_numpy()
            finite = np.isfinite(x) & np.isfinite(y)
            x_valid = x[finite]
            y_valid = y[finite]
            n_obs = int(x_valid.size)
            n_unique = int(np.unique(x_valid).size) if n_obs else 0

            rho: float = float("nan")
            p_value: float = float("nan")
            if n_obs >= 3 and n_unique >= 2:
                rho_raw, p_raw = spearmanr(x_valid, y_valid)
                rho = float(rho_raw) if np.isfinite(rho_raw) else float("nan")
                p_value = float(p_raw) if np.isfinite(p_raw) else float("nan")

            rows.append(
                {
                    "scenario": scenario_name,
                    "output": output_name,
                    "parameter": parameter_name,
                    "n_obs": n_obs,
                    "n_unique_param": n_unique,
                    "spearman_rho": rho,
                    "spearman_p_value": p_value,
                    "direction_sign": _signed_label(rho),
                    "abs_spearman_rho": abs(rho) if np.isfinite(rho) else float("nan"),
                }
            )

    if not rows:
        return pl.DataFrame()

    return pl.DataFrame(rows).sort(
        ["scenario", "output", "abs_spearman_rho"],
        descending=[False, False, True],
    )


def compute_surrogate_ice_slope_directionality(
    *,
    retained_models: dict[tuple[str, str], XGBRegressor],
    feature_df: pl.DataFrame,
    parameter_bounds: dict[str, tuple[float, float]],
) -> pl.DataFrame:
    """Compute surrogate ICE slope-sign diagnostics."""
    if not retained_models:
        return pl.DataFrame()

    param_names = list(parameter_bounds)
    if not param_names:
        return pl.DataFrame()

    required_cols = {"point_id", *param_names}
    missing = required_cols.difference(set(feature_df.columns))
    if missing:
        msg = f"feature_df is missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    x_base = feature_df.select(param_names).to_numpy()
    if x_base.shape[0] == 0:
        return pl.DataFrame()

    rows: list[dict[str, float | int | str]] = []
    for (scenario_name, output_name), model in retained_models.items():
        for idx, param_name in enumerate(param_names):
            lower, upper = parameter_bounds[param_name]
            span = float(upper - lower)
            if not np.isfinite(span) or span <= 0.0:
                continue

            x_low = x_base.copy()
            x_high = x_base.copy()
            x_low[:, idx] = lower
            x_high[:, idx] = upper

            pred_low = model.predict(x_low)
            pred_high = model.predict(x_high)
            delta = pred_high - pred_low
            slopes = delta / span

            finite = np.isfinite(slopes)
            slopes = slopes[finite]
            delta = delta[finite]
            n_effective = int(slopes.size)

            if n_effective == 0:
                mean_slope = float("nan")
                median_slope = float("nan")
                mean_delta = float("nan")
                share_positive = float("nan")
                share_negative = float("nan")
            else:
                mean_slope = float(np.mean(slopes))
                median_slope = float(np.median(slopes))
                mean_delta = float(np.mean(delta))
                share_positive = float(np.mean(slopes > 0.0))
                share_negative = float(np.mean(slopes < 0.0))

            rows.append(
                {
                    "scenario": scenario_name,
                    "output": output_name,
                    "parameter": param_name,
                    "bound_low": float(lower),
                    "bound_high": float(upper),
                    "n_points": n_effective,
                    "ice_mean_delta_low_to_high": mean_delta,
                    "ice_mean_slope": mean_slope,
                    "ice_median_slope": median_slope,
                    "ice_share_positive_slope": share_positive,
                    "ice_share_negative_slope": share_negative,
                    "direction_sign": _signed_label(mean_slope),
                    "abs_ice_mean_slope": abs(mean_slope) if np.isfinite(mean_slope) else float("nan"),
                }
            )

    if not rows:
        return pl.DataFrame()

    return pl.DataFrame(rows).sort(
        ["scenario", "output", "abs_ice_mean_slope"],
        descending=[False, False, True],
    )


def compute_sobol_indices_from_surrogate(
    *,
    model: XGBRegressor,
    parameter_bounds: dict[str, tuple[float, float]],
    base_samples: int,
    seed: int,
) -> pl.DataFrame:
    """Estimate first-order and total-order Sobol indices from a surrogate."""
    param_names = list(parameter_bounds)
    if not param_names:
        raise ValueError("parameter_bounds must not be empty.")

    exponent = int(math.ceil(math.log2(float(base_samples))))
    n_samples = int(2**exponent)

    sobol = qmc.Sobol(d=2 * len(param_names), scramble=True, seed=seed)
    unit_sample = sobol.random_base2(m=exponent)

    matrix_a = unit_sample[:, : len(param_names)]
    matrix_b = unit_sample[:, len(param_names) :]

    for col_idx, param_name in enumerate(param_names):
        lower, upper = parameter_bounds[param_name]
        matrix_a[:, col_idx] = lower + matrix_a[:, col_idx] * (upper - lower)
        matrix_b[:, col_idx] = lower + matrix_b[:, col_idx] * (upper - lower)

    pred_a = model.predict(matrix_a)
    pred_b = model.predict(matrix_b)
    variance = float(np.var(np.concatenate([pred_a, pred_b]), ddof=1))

    rows: list[dict[str, float | str | int]] = []
    if not np.isfinite(variance) or variance <= 0.0:
        for param_name in param_names:
            rows.append(
                {
                    "parameter": param_name,
                    "S_i": 0.0,
                    "S_Ti": 0.0,
                    "interaction_gap": 0.0,
                    "sobol_n": n_samples,
                }
            )
        return pl.DataFrame(rows)

    for col_idx, param_name in enumerate(param_names):
        matrix_ab_i = matrix_a.copy()
        matrix_ab_i[:, col_idx] = matrix_b[:, col_idx]
        pred_ab_i = model.predict(matrix_ab_i)

        first_order = 1.0 - float(np.mean((pred_b - pred_ab_i) ** 2) / (2.0 * variance))
        total_order = float(np.mean((pred_a - pred_ab_i) ** 2) / (2.0 * variance))

        first_order = float(np.clip(first_order, 0.0, 1.0))
        total_order = float(np.clip(total_order, 0.0, 1.0))
        rows.append(
            {
                "parameter": param_name,
                "S_i": first_order,
                "S_Ti": total_order,
                "interaction_gap": max(0.0, total_order - first_order),
                "sobol_n": n_samples,
            }
        )

    return pl.DataFrame(rows).sort("S_Ti", descending=True)


def run_posthoc_directional_diagnostics(*, output_dir: Path) -> None:
    """Backfill directional diagnostics for an existing sensitivity run."""
    output_dir = Path(output_dir)
    lhc_dir = output_dir / "lhc"
    surrogates_dir = output_dir / "surrogates"
    shortlist_dir = output_dir / "shortlist"
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    lhc_delta_path = lhc_dir / "delta.parquet"
    lhc_design_path = lhc_dir / "design.parquet"
    shortlist_path = shortlist_dir / "shortlist.csv"
    manifest_path = surrogates_dir / "manifest.csv"
    if not lhc_delta_path.exists():
        raise FileNotFoundError(f"Missing LHC delta file: {lhc_delta_path}")
    if not lhc_design_path.exists():
        raise FileNotFoundError(f"Missing LHC design file: {lhc_design_path}")
    if not shortlist_path.exists():
        raise FileNotFoundError(f"Missing shortlist file: {shortlist_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing surrogate manifest file: {manifest_path}")

    shortlist_df = pl.read_csv(shortlist_path)
    shortlisted_params = shortlist_df.select("parameter").to_series().cast(pl.Utf8).to_list()

    lhc_delta = pl.read_parquet(lhc_delta_path)
    spearman_df = compute_lhc_spearman_directionality(
        lhc_delta=lhc_delta,
        parameter_names=tuple(shortlisted_params),
    )
    spearman_path = reports_dir / "lhc_spearman_directionality.csv"
    spearman_df.write_csv(spearman_path)
    if not shortlisted_params:
        logger.warning("No shortlisted parameters found; skipping surrogate ICE diagnostics.")
        return

    lhc_design = pl.read_parquet(lhc_design_path)
    feature_df = lhc_design.select(["point_id", *shortlisted_params])
    shortlist_bounds: dict[str, tuple[float, float]] = {}
    for parameter_name in shortlisted_params:
        lower = float(feature_df.select(pl.col(parameter_name).min()).item())
        upper = float(feature_df.select(pl.col(parameter_name).max()).item())
        if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
            shortlist_bounds[parameter_name] = (lower, upper)
        else:
            shortlist_bounds[parameter_name] = DEFAULT_PARAMETER_BOUNDS[parameter_name]

    manifest_df = pl.read_csv(manifest_path)
    retained_df = manifest_df.filter(pl.col("retained") == True)  # noqa: E712
    retained_models: dict[tuple[str, str], XGBRegressor] = {}
    for row in retained_df.iter_rows(named=True):
        scenario_name = str(row["scenario"])
        output_name = str(row["output"])
        model_name = _sanitize_name(f"{scenario_name}__{output_name}")
        model_path = surrogates_dir / f"{model_name}.json"
        if not model_path.exists():
            logger.warning("Missing retained surrogate model file: {path}", path=model_path)
            continue
        model = XGBRegressor()
        model.load_model(model_path)
        retained_models[(scenario_name, output_name)] = model

    ice_df = compute_surrogate_ice_slope_directionality(
        retained_models=retained_models,
        feature_df=feature_df,
        parameter_bounds=shortlist_bounds,
    )
    ice_path = reports_dir / "surrogate_ice_slope_directionality.csv"
    ice_df.write_csv(ice_path)
