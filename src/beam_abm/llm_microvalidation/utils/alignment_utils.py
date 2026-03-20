"""Helpers for alignment row-level outputs and summary aggregation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def summarize_alignment_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate row-level alignment outputs into a single summary row."""
    if df.empty:
        return pd.DataFrame([{"n": 0}])

    out: dict[str, float | int] = {"n": int(len(df))}

    if "da_status" in df.columns:
        no_signal = (df["da_status"].astype(str) == "no_signal").to_numpy(dtype=bool)
        out["n_no_signal"] = int(np.sum(no_signal))
        out["n_scored"] = int(len(df) - out["n_no_signal"])
        out["da_no_signal_rate"] = float(np.mean(no_signal))
        out["da_scored_rate"] = float(1.0 - out["da_no_signal_rate"])

    # Robust directional diagnostics (do not require CIs).
    # - sign_agree: 1/0/NaN, where NaN denotes non-informative rows.
    # - low_signal_abs_pe_llm_std: |PE_z| in a low-signal band (NaN outside band).
    if "sign_agree" in df.columns:
        sign_agree = pd.to_numeric(df["sign_agree"], errors="coerce")
        sign_mask = sign_agree.notna().to_numpy(dtype=bool)
        out["n_sign_informative"] = int(np.sum(sign_mask))
        out["sign_agreement_rate"] = float(sign_agree.mean(skipna=True)) if sign_mask.any() else float("nan")

    if "false_positive_move" in df.columns:
        fp_move = pd.to_numeric(df["false_positive_move"], errors="coerce")
        fp_mask = fp_move.notna().to_numpy(dtype=bool)
        out["n_low_signal"] = int(np.sum(fp_mask))
        out["false_positive_movement_rate"] = float(fp_move.mean(skipna=True)) if fp_mask.any() else float("nan")

    if "low_signal_abs_pe_llm_std" in df.columns:
        low_abs = pd.to_numeric(df["low_signal_abs_pe_llm_std"], errors="coerce")
        out["low_signal_abs_pe_llm_std_median"] = float(low_abs.median()) if low_abs.notna().any() else float("nan")
        out["typical_drift_median"] = out["low_signal_abs_pe_llm_std_median"]

    if "epsilon_ref" in df.columns:
        eps_ref = pd.to_numeric(df["epsilon_ref"], errors="coerce")
        out["epsilon_ref"] = float(eps_ref.dropna().iloc[0]) if eps_ref.notna().any() else float("nan")
    if "epsilon_move" in df.columns:
        eps_move = pd.to_numeric(df["epsilon_move"], errors="coerce")
        out["epsilon_move"] = float(eps_move.dropna().iloc[0]) if eps_move.notna().any() else float("nan")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = {"row"}
    for col in numeric_cols:
        if col in drop_cols:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            out[f"{col}_mean"] = float(series.mean())
            out[f"{col}_std"] = float(series.std(ddof=1)) if len(series) > 1 else float("nan")

    return pd.DataFrame([out])


def merge_alignment_into_row_level(
    *,
    row_level_path: str,
    alignment_path: str,
    on_cols: Iterable[tuple[str, str]],
    prefix: str = "align_",
) -> None:
    """Merge alignment row-level metrics into row_level CSV by key columns."""
    row_df = pd.read_csv(row_level_path)
    align_df = pd.read_csv(alignment_path)

    left_cols = [left for left, _ in on_cols]
    right_cols = [right for _, right in on_cols]

    if not set(left_cols).issubset(row_df.columns):
        return
    if not set(right_cols).issubset(align_df.columns):
        return

    aligned = align_df.copy()
    rename_map = {
        col: f"{prefix}{col}" for col in aligned.columns if col not in right_cols and not col.startswith(prefix)
    }
    aligned = aligned.rename(columns=rename_map)

    overlap = [c for c in aligned.columns if c in row_df.columns and c not in left_cols]
    if overlap:
        row_df = row_df.drop(columns=overlap)

    merged = row_df.merge(
        aligned,
        how="left",
        left_on=left_cols,
        right_on=right_cols,
        suffixes=("", ""),
    )
    merged.to_csv(row_level_path, index=False)


__all__ = ["merge_alignment_into_row_level", "summarize_alignment_rows"]
