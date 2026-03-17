"""Apply PE gain calibration to LLM ICE tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_calibration(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "calibrators" not in payload:
        raise ValueError("Calibration JSON missing 'calibrators'")
    if "group_cols" not in payload or not isinstance(payload["group_cols"], list):
        raise ValueError("Calibration JSON missing 'group_cols'")
    return payload


def _parse_samples(value: object) -> list[float] | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, list):
        try:
            return [float(v) for v in value]
        except Exception:
            return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return None
        if isinstance(parsed, list):
            try:
                return [float(v) for v in parsed]
            except Exception:
                return None
    return None


def _dump_samples(values: list[float] | None) -> str | None:
    if values is None:
        return None
    return json.dumps([float(v) for v in values])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-ice", required=True, help="Input LLM ICE CSV")
    parser.add_argument("--calibration", required=True, help="Calibration JSON from calibrate_pe_gains.py")
    parser.add_argument("--out", required=True, help="Output PE-calibrated ICE CSV")
    parser.add_argument(
        "--fallback-alpha",
        type=float,
        default=1.0,
        help="Fallback alpha when no calibrator is available (default: 1.0)",
    )
    parser.add_argument("--keep-raw", action="store_true", help="Keep raw ICE columns with _raw suffix")
    args = parser.parse_args()

    llm = pd.read_csv(args.llm_ice)
    payload = _load_calibration(Path(args.calibration))

    group_cols = [str(c) for c in payload.get("group_cols", [])]
    if not group_cols:
        raise ValueError("Calibration JSON missing group_cols")

    for col in group_cols:
        if col not in llm.columns:
            raise KeyError(f"LLM ICE missing required group column: {col}")

    ice_cols = ["ice_low", "ice_high"] + (["ice_mid"] if "ice_mid" in llm.columns else [])
    for col in ["ice_low", "ice_high"]:
        if col not in llm.columns:
            raise KeyError(f"LLM ICE missing required column: {col}")

    if args.keep_raw:
        for col in ice_cols:
            llm[f"{col}_raw"] = llm[col]
        for col in ["ice_low_samples", "ice_high_samples"]:
            if col in llm.columns:
                llm[f"{col}_raw"] = llm[col]

    cal_map: dict[tuple[str, ...], float] = {}
    for row in payload.get("calibrators", []):
        if not isinstance(row, dict):
            continue
        key = tuple(str(row.get(col, "")) for col in group_cols)
        alpha = row.get("alpha")
        try:
            alpha_val = float(alpha)
        except Exception:
            continue
        cal_map[key] = alpha_val

    def _row_alpha(r: pd.Series) -> float:
        key = tuple(str(r.get(col, "")) for col in group_cols)
        alpha = cal_map.get(key)
        if alpha is None or not np.isfinite(alpha):
            return float(args.fallback_alpha)
        return float(alpha)

    for idx, row in llm.iterrows():
        alpha = _row_alpha(row)
        low = row.get("ice_low")
        high = row.get("ice_high")
        if low is None or high is None:
            continue
        try:
            low_f = float(low)
            high_f = float(high)
        except Exception:
            continue
        if not (np.isfinite(low_f) and np.isfinite(high_f)):
            continue

        if "ice_mid" in llm.columns:
            mid_val = row.get("ice_mid")
            try:
                mid = float(mid_val) if mid_val is not None else float("nan")
            except Exception:
                mid = float("nan")
            if not np.isfinite(mid):
                mid = 0.5 * (low_f + high_f)
        else:
            mid = 0.5 * (low_f + high_f)

        delta = high_f - low_f
        new_delta = alpha * delta
        llm.at[idx, "ice_low"] = mid - 0.5 * new_delta
        llm.at[idx, "ice_high"] = mid + 0.5 * new_delta
        if "ice_mid" in llm.columns:
            llm.at[idx, "ice_mid"] = mid

        for sample_col in ["ice_low_samples", "ice_high_samples"]:
            if sample_col not in llm.columns:
                continue
            samples = _parse_samples(row.get(sample_col))
            if samples is None:
                continue
            adjusted = [mid + alpha * (v - mid) for v in samples]
            llm.at[idx, sample_col] = _dump_samples(adjusted)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    llm.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
