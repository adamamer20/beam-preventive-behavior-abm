"""Apply saved calibration to unperturbed prediction files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from beam_abm.cli import parser_compat as cli
from beam_abm.evaluation.common.calibration import BinaryPlattCalibrator, LinearCalibrator, load_calibrators


def _select_calibrator(calibrators: list[dict], *, outcome: str, model: str | None) -> dict:
    matches = [c for c in calibrators if str(c.get("outcome")) == outcome]
    if model is not None:
        matches = [c for c in matches if str(c.get("model")) == model]
    if not matches:
        raise ValueError(f"No calibrator found for outcome={outcome} model={model}")
    if len(matches) > 1 and model is None:
        raise ValueError(f"Multiple calibrators for outcome={outcome}; provide --model")
    return matches[0]


def _apply_row(cal: dict, values: np.ndarray, countries: np.ndarray) -> np.ndarray:
    payload = cal.get("calibrator")
    if not isinstance(payload, dict):
        raise TypeError("Calibrator payload missing")
    kind = payload.get("kind")
    if kind == "binary_platt_country":
        model = BinaryPlattCalibrator.from_dict(payload)
        return model.predict(values, countries)
    if kind == "linear_country":
        model = LinearCalibrator.from_dict(payload)
        return model.predict(values, countries)
    raise ValueError(f"Unknown calibrator kind: {kind}")


def _load_prediction_files(path: Path) -> list[tuple[Path, pd.DataFrame]]:
    if path.is_dir():
        files = sorted(path.glob("predictions__*.csv"))
        if not files:
            raise FileNotFoundError(f"No predictions__*.csv files in {path}")
        return [(f, pd.read_csv(f)) for f in files]
    return [(path, pd.read_csv(path))]


def _apply_calibration_to_df(
    df: pd.DataFrame,
    *,
    calibrators: list[dict],
    pred_col: str,
    out_col: str,
    country_col: str,
    model_override: str | None,
    keep_raw: bool,
) -> pd.DataFrame:
    required = {"outcome", pred_col, country_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    if keep_raw:
        df[f"{pred_col}_raw"] = df[pred_col]

    if "model" not in df.columns and model_override is None:
        raise KeyError("Missing model column and no --model provided")

    model_col = "model" if "model" in df.columns else None

    for outcome, g in df.groupby("outcome", dropna=False):
        if model_col is None:
            cal = _select_calibrator(calibrators, outcome=str(outcome), model=model_override)
            idx = g.index.to_numpy()
            countries = df.loc[idx, country_col].to_numpy(dtype=str)
            vals = df.loc[idx, pred_col].to_numpy(dtype=float)
            df.loc[idx, out_col] = _apply_row(cal, vals, countries)
            continue

        for model_name, gm in g.groupby("model", dropna=False):
            cal = _select_calibrator(
                calibrators,
                outcome=str(outcome),
                model=(model_override if model_override is not None else str(model_name)),
            )
            idx = gm.index.to_numpy()
            countries = df.loc[idx, country_col].to_numpy(dtype=str)
            vals = df.loc[idx, pred_col].to_numpy(dtype=float)
            df.loc[idx, out_col] = _apply_row(cal, vals, countries)

    return df


def run_cli(argv: list[str] | None = None) -> None:
    parser = cli.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="Predictions CSV or directory.")
    parser.add_argument("--calibration", required=True, help="Calibration JSON from calibrate_llm_predictions.py")
    parser.add_argument("--out", dest="out_path", required=True, help="Output CSV or directory")
    parser.add_argument("--model", default=None, help="Optional model name to select calibrator")
    parser.add_argument("--pred-col", default="y_pred")
    parser.add_argument("--out-col", default="y_pred_cal")
    parser.add_argument("--country-col", default="country")
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args(argv)

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    calibrators = load_calibrators(args.calibration)
    files = _load_prediction_files(in_path)

    if in_path.is_dir():
        out_path.mkdir(parents=True, exist_ok=True)

    for src_path, df in files:
        df = _apply_calibration_to_df(
            df,
            calibrators=calibrators,
            pred_col=str(args.pred_col),
            out_col=str(args.out_col),
            country_col=str(args.country_col),
            model_override=str(args.model) if args.model is not None else None,
            keep_raw=bool(args.keep_raw),
        )

        if in_path.is_dir():
            dest = out_path / src_path.name
        else:
            dest = out_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest, index=False)



