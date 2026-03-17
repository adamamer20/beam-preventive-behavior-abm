"""Fit and apply cross-fitted calibration for LLM predictions on real rows."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from beam_abm.evaluation.common.calibration import (
    BinaryPlattCalibrator,
    LinearCalibrator,
    cross_fit_predictions,
    fit_binary_platt_country,
    fit_linear_country,
    save_calibrators,
)


def _load_predictions(path: Path) -> pd.DataFrame:
    if path.is_dir():
        files = sorted(path.glob("predictions__*.csv"))
        if not files:
            raise FileNotFoundError(f"No predictions__*.csv files in {path}")
        frames = [pd.read_csv(f) for f in files]
        return pd.concat(frames, ignore_index=True)
    return pd.read_csv(path)


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _fit_and_predict_binary(
    df: pd.DataFrame,
    *,
    country_col: str,
    eps: float,
    n_folds: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    def fit_fn(frame: pd.DataFrame) -> BinaryPlattCalibrator:
        return fit_binary_platt_country(frame, p_col="y_pred", y_col="y_true", country_col=country_col, eps=eps)

    def predict_fn(cal: BinaryPlattCalibrator, frame: pd.DataFrame) -> np.ndarray:
        return cal.predict(frame["y_pred"].to_numpy(dtype=float), frame[country_col].to_numpy(dtype=str))

    preds = cross_fit_predictions(df, n_folds=int(n_folds), seed=int(seed), fit_fn=fit_fn, predict_fn=predict_fn)
    cal_full = fit_fn(df)
    return preds, cal_full.to_dict()


def _fit_and_predict_linear(
    df: pd.DataFrame,
    *,
    country_col: str,
    bounds: tuple[float, float] | None,
    n_folds: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    def fit_fn(frame: pd.DataFrame) -> LinearCalibrator:
        return fit_linear_country(
            frame,
            y_pred_col="y_pred",
            y_true_col="y_true",
            country_col=country_col,
            bounds=bounds,
        )

    def predict_fn(cal: LinearCalibrator, frame: pd.DataFrame) -> np.ndarray:
        return cal.predict(frame["y_pred"].to_numpy(dtype=float), frame[country_col].to_numpy(dtype=str))

    preds = cross_fit_predictions(df, n_folds=int(n_folds), seed=int(seed), fit_fn=fit_fn, predict_fn=predict_fn)
    cal_full = fit_fn(df)
    return preds, cal_full.to_dict()


def calibrate_predictions(
    *,
    in_path: str | Path,
    out_path: str | Path,
    calibration_out: str | Path,
    n_folds: int = 5,
    seed: int = 13,
    country_col: str = "country",
    eps: float = 1e-6,
    bounds_source: str = "true",
) -> None:
    args = SimpleNamespace(
        in_path=str(in_path),
        out_path=str(out_path),
        cal_out=str(calibration_out),
        n_folds=int(n_folds),
        seed=int(seed),
        country_col=str(country_col),
        eps=float(eps),
        bounds_source=str(bounds_source),
    )

    df = _load_predictions(Path(args.in_path))
    _ensure_columns(df, ["model", "outcome", "outcome_type", "y_true", "y_pred", args.country_col])

    df = df.copy()
    df["outcome_type"] = df["outcome_type"].astype(str).str.lower()

    calibrated_frames: list[pd.DataFrame] = []
    calibrators: list[dict] = []

    group_cols = ["model", "outcome", "outcome_type"]
    for (model, outcome, outcome_type), g in df.groupby(group_cols, dropna=False):
        work = g.dropna(subset=["y_true", "y_pred", args.country_col]).copy()
        if work.empty:
            continue

        bounds: tuple[float, float] | None = None
        if outcome_type != "binary" and args.bounds_source == "true":
            lo = float(np.nanmin(work["y_true"].to_numpy(dtype=float)))
            hi = float(np.nanmax(work["y_true"].to_numpy(dtype=float)))
            bounds = (lo, hi) if np.isfinite(lo) and np.isfinite(hi) else None

        if outcome_type == "binary":
            y_cal, cal_dict = _fit_and_predict_binary(
                work,
                country_col=args.country_col,
                eps=float(args.eps),
                n_folds=int(args.n_folds),
                seed=int(args.seed),
            )
        else:
            y_cal, cal_dict = _fit_and_predict_linear(
                work,
                country_col=args.country_col,
                bounds=bounds,
                n_folds=int(args.n_folds),
                seed=int(args.seed),
            )

        work = work.copy()
        work["y_pred_cal"] = y_cal
        calibrated_frames.append(work)

        calibrators.append(
            {
                "model": str(model),
                "outcome": str(outcome),
                "outcome_type": str(outcome_type),
                "calibrator": cal_dict,
            }
        )

    if not calibrated_frames:
        raise ValueError("No calibrated rows produced")

    out_df = pd.concat(calibrated_frames, ignore_index=True)
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_path, index=False)
    Path(args.cal_out).parent.mkdir(parents=True, exist_ok=True)
    save_calibrators(calibrators, args.cal_out)
