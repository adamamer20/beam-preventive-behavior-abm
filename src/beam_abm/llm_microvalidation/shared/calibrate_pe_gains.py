"""Fit PE gain calibration from reference/LLM ICE tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from beam_abm.cli import parser_compat as cli


def _ensure_columns(df: pd.DataFrame, cols: list[str], *, label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{label} missing required columns: {missing}")


def _merge_ref_llm(ref: pd.DataFrame, llm: pd.DataFrame) -> pd.DataFrame:
    join_keys: list[str] = []
    for key in ["id", "target", "outcome", "strategy"]:
        if key in ref.columns and key in llm.columns:
            join_keys.append(key)

    if not join_keys:
        raise ValueError("No overlapping join keys between ref and llm ICE tables")

    if ref.duplicated(subset=join_keys).any():
        raise ValueError(f"Reference ICE has duplicate rows for merge keys: {join_keys}")
    if llm.duplicated(subset=join_keys).any():
        raise ValueError(f"LLM ICE has duplicate rows for merge keys: {join_keys}")

    merged = ref.merge(llm, on=join_keys, how="inner", suffixes=("_ref", "_llm"))
    if merged.empty:
        raise ValueError(f"No overlapping rows after merge on keys: {join_keys}")
    return merged


def _parse_group_cols(raw: str | None) -> list[str]:
    if not raw:
        return ["outcome", "target", "strategy"]
    return [c.strip() for c in raw.split(",") if c.strip()]


def run_cli(argv: list[str] | None = None) -> None:
    parser = cli.ArgumentParser()
    parser.add_argument("--ref-ice", required=True, help="Reference ICE CSV")
    parser.add_argument("--llm-ice", required=True, help="LLM ICE CSV")
    parser.add_argument("--out", required=True, help="Output calibration JSON")
    parser.add_argument(
        "--group-cols",
        default="outcome,target,strategy",
        help="Comma-separated grouping columns (default: outcome,target,strategy)",
    )
    parser.add_argument("--min-rows", type=int, default=1, help="Minimum rows per group to fit a gain")
    args = parser.parse_args(argv)

    ref = pd.read_csv(args.ref_ice)
    llm = pd.read_csv(args.llm_ice)

    _ensure_columns(ref, ["ice_low", "ice_high"], label="Reference ICE")
    _ensure_columns(llm, ["ice_low", "ice_high"], label="LLM ICE")

    merged = _merge_ref_llm(ref, llm)

    merged["delta_ref"] = pd.to_numeric(merged["ice_high_ref"], errors="coerce") - pd.to_numeric(
        merged["ice_low_ref"], errors="coerce"
    )
    merged["delta_llm"] = pd.to_numeric(merged["ice_high_llm"], errors="coerce") - pd.to_numeric(
        merged["ice_low_llm"], errors="coerce"
    )

    group_cols = _parse_group_cols(args.group_cols)
    group_cols = [c for c in group_cols if c in merged.columns]
    if "outcome" not in group_cols or "target" not in group_cols:
        raise ValueError("group-cols must include at least outcome and target")

    calibrators: list[dict] = []
    for keys, g in merged.groupby(group_cols, dropna=False):
        group_key = keys if isinstance(keys, tuple) else (keys,)
        payload = {col: str(val) if val is not None else "" for col, val in zip(group_cols, group_key, strict=False)}

        x = pd.to_numeric(g["delta_llm"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(g["delta_ref"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        n = int(x.size)
        if n < int(args.min_rows):
            alpha = float("nan")
            sum_xx = float("nan")
            sum_xy = float("nan")
        else:
            sum_xx = float(np.sum(x * x))
            sum_xy = float(np.sum(x * y))
            if not np.isfinite(sum_xx) or sum_xx <= 0:
                alpha = float("nan")
            else:
                alpha = float(sum_xy / sum_xx)
                if not np.isfinite(alpha):
                    alpha = float("nan")
            if np.isfinite(alpha) and alpha < 0:
                alpha = 0.0

        calibrators.append(
            {
                **payload,
                "alpha": alpha,
                "n": n,
                "sum_xx": sum_xx,
                "sum_xy": sum_xy,
            }
        )

    out_payload = {
        "group_cols": group_cols,
        "ref_ice": str(args.ref_ice),
        "llm_ice": str(args.llm_ice),
        "calibrators": calibrators,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
