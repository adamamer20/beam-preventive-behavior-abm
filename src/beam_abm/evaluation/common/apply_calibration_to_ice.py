"""Apply a saved calibration layer to LLM ICE tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from beam_abm.evaluation.common.calibration import (
    BinaryPlattCalibrator,
    IsotonicCountryCalibrator,
    LinearCalibrator,
    load_calibrators,
)


def _parse_country_from_id(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    parts = str(value).split("::")
    if len(parts) >= 2:
        return parts[1]
    return None


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
    if kind == "isotonic_country":
        model = IsotonicCountryCalibrator.from_dict(payload)
        return model.predict(values, countries)
    raise ValueError(f"Unknown calibrator kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-ice", required=True, help="Input LLM ICE CSV")
    parser.add_argument("--calibration", required=True, help="Calibration JSON from calibrate_llm_predictions.py")
    parser.add_argument("--out", required=True, help="Output calibrated ICE CSV")
    parser.add_argument("--model", default=None, help="Optional model name to select calibrator")
    parser.add_argument("--country-col", default="country")
    parser.add_argument("--keep-raw", action="store_true", help="Keep raw ICE columns with _raw suffix")
    args = parser.parse_args()

    llm = pd.read_csv(args.llm_ice)
    # Drop stray header rows that can appear after concatenation.
    for col in ("outcome", "target", "id"):
        if col in llm.columns:
            llm = llm[llm[col].astype(str) != col]
    calibrators = load_calibrators(args.calibration)

    if args.country_col in llm.columns:
        llm["_country"] = llm[args.country_col].astype(str)
    else:
        llm["_country"] = llm["id"].apply(_parse_country_from_id)
    if llm["_country"].isna().any():
        raise ValueError("Country could not be inferred for some rows; add a country column or fix id format.")

    ice_cols = ["ice_low", "ice_high"] + (["ice_mid"] if "ice_mid" in llm.columns else [])
    if args.keep_raw:
        for col in ice_cols:
            llm[f"{col}_raw"] = llm[col]

    for outcome, g in llm.groupby("outcome", dropna=False):
        cal = _select_calibrator(calibrators, outcome=str(outcome), model=args.model)
        idx = g.index.to_numpy()
        countries = llm.loc[idx, "_country"].to_numpy(dtype=str)
        for col in ice_cols:
            vals = llm.loc[idx, col].to_numpy(dtype=float)
            llm.loc[idx, col] = _apply_row(cal, vals, countries)

    llm.drop(columns=["_country"], inplace=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    llm.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
