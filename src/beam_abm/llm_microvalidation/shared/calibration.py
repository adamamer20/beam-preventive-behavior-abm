"""Calibration utilities for LLM micro-validation outputs.

Provides simple country-aware calibration layers with cross-fitting support.
Binary outcomes use a Platt-style logit calibration with country intercepts.
Ordinal/continuous outcomes use a linear rescaling with country intercepts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Callable
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression

logger = logging.getLogger(__name__)


@beartype
def _logit(p: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


@beartype
def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


@beartype
def _one_hot(values: pd.Series) -> tuple[np.ndarray, list[str]]:
    levels = [str(x) for x in pd.Series(values).dropna().unique().tolist()]
    levels = sorted(levels)
    mapped = values.astype(str)
    mat = np.column_stack([(mapped == level).to_numpy(dtype=float) for level in levels])
    return mat, levels


@dataclass(frozen=True)
class BinaryPlattCalibrator:
    slope: float
    country_levels: list[str]
    country_intercepts: dict[str, float]
    fallback_intercept: float
    eps: float = 1e-6

    def predict(self, p_llm: np.ndarray, country: np.ndarray) -> np.ndarray:
        logits = _logit(np.asarray(p_llm, dtype=float), eps=self.eps)
        c = np.asarray(country, dtype=str)
        intercepts = np.array([self.country_intercepts.get(x, self.fallback_intercept) for x in c], dtype=float)
        return _expit(intercepts + self.slope * logits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "binary_platt_country",
            "slope": float(self.slope),
            "country_levels": list(self.country_levels),
            "country_intercepts": {str(k): float(v) for k, v in self.country_intercepts.items()},
            "fallback_intercept": float(self.fallback_intercept),
            "eps": float(self.eps),
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> BinaryPlattCalibrator:
        return BinaryPlattCalibrator(
            slope=float(payload["slope"]),
            country_levels=[str(x) for x in payload.get("country_levels", [])],
            country_intercepts={str(k): float(v) for k, v in payload.get("country_intercepts", {}).items()},
            fallback_intercept=float(payload.get("fallback_intercept", 0.0)),
            eps=float(payload.get("eps", 1e-6)),
        )


@dataclass(frozen=True)
class LinearCalibrator:
    slope: float
    country_levels: list[str]
    country_intercepts: dict[str, float]
    fallback_intercept: float
    bounds: tuple[float, float] | None = None

    def predict(self, y_llm: np.ndarray, country: np.ndarray) -> np.ndarray:
        y = np.asarray(y_llm, dtype=float)
        c = np.asarray(country, dtype=str)
        intercepts = np.array([self.country_intercepts.get(x, self.fallback_intercept) for x in c], dtype=float)
        out = intercepts + self.slope * y
        if self.bounds is not None:
            lo, hi = self.bounds
            out = np.clip(out, lo, hi)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "linear_country",
            "slope": float(self.slope),
            "country_levels": list(self.country_levels),
            "country_intercepts": {str(k): float(v) for k, v in self.country_intercepts.items()},
            "fallback_intercept": float(self.fallback_intercept),
            "bounds": list(self.bounds) if self.bounds is not None else None,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> LinearCalibrator:
        bounds = payload.get("bounds")
        return LinearCalibrator(
            slope=float(payload["slope"]),
            country_levels=[str(x) for x in payload.get("country_levels", [])],
            country_intercepts={str(k): float(v) for k, v in payload.get("country_intercepts", {}).items()},
            fallback_intercept=float(payload.get("fallback_intercept", 0.0)),
            bounds=(float(bounds[0]), float(bounds[1])) if bounds is not None else None,
        )


@dataclass(frozen=True)
class IsotonicCountryCalibrator:
    country_levels: list[str]
    country_thresholds: dict[str, dict[str, list[float]]]
    fallback_thresholds: dict[str, list[float]]
    bounds: tuple[float, float] | None = None

    def _predict_one(self, values: np.ndarray, thresholds: dict[str, list[float]]) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        if x.size == 0:
            return x
        x_thresh = np.asarray(thresholds.get("x", []), dtype=float)
        y_thresh = np.asarray(thresholds.get("y", []), dtype=float)
        if x_thresh.size == 0 or y_thresh.size == 0:
            out = np.full_like(x, np.nan, dtype=float)
        else:
            out = np.interp(x, x_thresh, y_thresh, left=y_thresh[0], right=y_thresh[-1])
        if self.bounds is not None:
            lo, hi = self.bounds
            out = np.clip(out, lo, hi)
        return out

    def predict(self, y_llm: np.ndarray, country: np.ndarray) -> np.ndarray:
        y = np.asarray(y_llm, dtype=float)
        c = np.asarray(country, dtype=str)
        out = np.full_like(y, np.nan, dtype=float)
        for level in np.unique(c):
            mask = c == level
            thresholds = self.country_thresholds.get(level, self.fallback_thresholds)
            out[mask] = self._predict_one(y[mask], thresholds)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "isotonic_country",
            "country_levels": list(self.country_levels),
            "country_thresholds": {
                str(k): {
                    "x": [float(v) for v in payload.get("x", [])],
                    "y": [float(v) for v in payload.get("y", [])],
                }
                for k, payload in self.country_thresholds.items()
            },
            "fallback_thresholds": {
                "x": [float(v) for v in self.fallback_thresholds.get("x", [])],
                "y": [float(v) for v in self.fallback_thresholds.get("y", [])],
            },
            "bounds": list(self.bounds) if self.bounds is not None else None,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> IsotonicCountryCalibrator:
        bounds = payload.get("bounds")
        country_thresholds = {
            str(k): {
                "x": [float(v) for v in threshold.get("x", [])],
                "y": [float(v) for v in threshold.get("y", [])],
            }
            for k, threshold in payload.get("country_thresholds", {}).items()
        }
        fallback = payload.get("fallback_thresholds", {})
        return IsotonicCountryCalibrator(
            country_levels=[str(x) for x in payload.get("country_levels", [])],
            country_thresholds=country_thresholds,
            fallback_thresholds={
                "x": [float(v) for v in fallback.get("x", [])],
                "y": [float(v) for v in fallback.get("y", [])],
            },
            bounds=(float(bounds[0]), float(bounds[1])) if bounds is not None else None,
        )


@beartype
def fit_binary_platt_country(
    df: pd.DataFrame,
    *,
    p_col: str = "y_pred",
    y_col: str = "y_true",
    country_col: str = "country",
    eps: float = 1e-6,
) -> BinaryPlattCalibrator:
    work = df[[p_col, y_col, country_col]].dropna()
    if work.empty:
        raise ValueError("No rows to calibrate")

    logits = _logit(work[p_col].to_numpy(dtype=float), eps=eps)
    dummies, levels = _one_hot(work[country_col])
    X = np.column_stack([logits, dummies])
    y = work[y_col].to_numpy(dtype=float)

    clf = LogisticRegression(
        fit_intercept=False,
        penalty="l2",
        C=1e6,
        solver="lbfgs",
        max_iter=1000,
    )
    clf.fit(X, y)
    coef = clf.coef_.reshape(-1)
    slope = float(coef[0])
    intercepts = {levels[i]: float(coef[i + 1]) for i in range(len(levels))}
    fallback = float(np.mean(list(intercepts.values()))) if intercepts else 0.0
    return BinaryPlattCalibrator(
        slope=slope,
        country_levels=levels,
        country_intercepts=intercepts,
        fallback_intercept=fallback,
        eps=eps,
    )


@beartype
def fit_linear_country(
    df: pd.DataFrame,
    *,
    y_pred_col: str = "y_pred",
    y_true_col: str = "y_true",
    country_col: str = "country",
    bounds: tuple[float, float] | None = None,
) -> LinearCalibrator:
    work = df[[y_pred_col, y_true_col, country_col]].dropna()
    if work.empty:
        raise ValueError("No rows to calibrate")

    y_pred = work[y_pred_col].to_numpy(dtype=float)
    dummies, levels = _one_hot(work[country_col])
    X = np.column_stack([y_pred, dummies])
    y = work[y_true_col].to_numpy(dtype=float)

    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, y)
    coef = reg.coef_.reshape(-1)
    slope = float(coef[0])
    intercepts = {levels[i]: float(coef[i + 1]) for i in range(len(levels))}
    fallback = float(np.mean(list(intercepts.values()))) if intercepts else 0.0
    return LinearCalibrator(
        slope=slope,
        country_levels=levels,
        country_intercepts=intercepts,
        fallback_intercept=fallback,
        bounds=bounds,
    )


@beartype
def fit_isotonic_country(
    df: pd.DataFrame,
    *,
    y_pred_col: str = "y_pred",
    y_true_col: str = "y_true",
    country_col: str = "country",
    bounds: tuple[float, float] | None = None,
    min_country_rows: int = 20,
) -> IsotonicCountryCalibrator:
    work = df[[y_pred_col, y_true_col, country_col]].dropna()
    if work.empty:
        raise ValueError("No rows to calibrate")

    y_min = bounds[0] if bounds is not None else None
    y_max = bounds[1] if bounds is not None else None

    country_thresholds: dict[str, dict[str, list[float]]] = {}
    country_levels: list[str] = []

    for country, g in work.groupby(country_col, dropna=False):
        if len(g) < int(min_country_rows):
            continue
        x = g[y_pred_col].to_numpy(dtype=float)
        y = g[y_true_col].to_numpy(dtype=float)
        if x.size == 0:
            continue
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=y_min, y_max=y_max)
        iso.fit(x, y)
        country_levels.append(str(country))
        country_thresholds[str(country)] = {
            "x": [float(v) for v in iso.X_thresholds_],
            "y": [float(v) for v in iso.y_thresholds_],
        }

    # Always fit a fallback calibrator.
    x_all = work[y_pred_col].to_numpy(dtype=float)
    y_all = work[y_true_col].to_numpy(dtype=float)
    iso_all = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=y_min, y_max=y_max)
    iso_all.fit(x_all, y_all)
    fallback_thresholds = {
        "x": [float(v) for v in iso_all.X_thresholds_],
        "y": [float(v) for v in iso_all.y_thresholds_],
    }

    return IsotonicCountryCalibrator(
        country_levels=country_levels,
        country_thresholds=country_thresholds,
        fallback_thresholds=fallback_thresholds,
        bounds=bounds,
    )


@beartype
def cross_fit_predictions(
    df: pd.DataFrame,
    *,
    n_folds: int,
    seed: int,
    fit_fn: Callable[[pd.DataFrame], Any],
    predict_fn: Callable[[Any, pd.DataFrame], np.ndarray],
) -> np.ndarray:
    work = df.reset_index(drop=True)
    n = len(work)
    if n == 0:
        return np.array([], dtype=float)
    if n_folds <= 1 or n < 2:
        calibrator = fit_fn(work)
        return predict_fn(calibrator, work)

    folds = min(int(n_folds), n)
    rng = np.random.default_rng(int(seed))
    indices = np.arange(n)
    rng.shuffle(indices)
    splits = np.array_split(indices, folds)
    preds = np.full(n, np.nan, dtype=float)

    for test_idx in splits:
        train_idx = np.setdiff1d(indices, test_idx, assume_unique=False)
        if train_idx.size == 0:
            continue
        calibrator = fit_fn(work.iloc[train_idx])
        preds[test_idx] = predict_fn(calibrator, work.iloc[test_idx])

    return preds


@beartype
def save_calibrators(calibrators: list[dict[str, Any]], path: str) -> None:
    payload = {"calibrators": calibrators}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


@beartype
def load_calibrators(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "calibrators" not in payload:
        raise ValueError("Calibration file missing 'calibrators' key")
    calibrators = payload.get("calibrators")
    if not isinstance(calibrators, list):
        raise TypeError("Calibration file must store a list under 'calibrators'")
    return calibrators
