"""Post-hoc diagnostics for prompt tournament outputs (no new LLM calls)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from beam_abm.cli import parser_compat as cli
from beam_abm.evaluation.utils.jsonl import read_jsonl


def _parse_list(value: object) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, list):
        try:
            return [float(v) for v in value]
        except Exception:
            return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, list):
            try:
                return [float(v) for v in parsed]
            except Exception:
                return None
    return None


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    vx = x[mask].astype(float)
    vy = y[mask].astype(float)
    if np.std(vx) == 0 or np.std(vy) == 0:
        return None
    return float(np.corrcoef(vx, vy)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    rx = pd.Series(x[mask]).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y[mask]).rank(method="average").to_numpy(dtype=float)
    return _safe_corr(rx, ry)


def _delta(df: pd.DataFrame, *, low_col: str, high_col: str) -> pd.Series:
    return pd.to_numeric(df[high_col], errors="coerce") - pd.to_numeric(df[low_col], errors="coerce")


def _mean_prediction(samples: object) -> float | None:
    if not isinstance(samples, list):
        return None
    vals: list[float] = []
    for s in samples:
        if not isinstance(s, dict):
            continue
        scalar = s.get("prediction_scalar")
        if scalar is None:
            continue
        try:
            vals.append(float(scalar))
        except Exception:
            continue
    if not vals:
        return None
    return float(np.mean(vals))


def _load_unperturbed_predictions(samples_path: Path) -> pd.DataFrame:
    if not samples_path.exists():
        return pd.DataFrame(columns=["id", "outcome", "y_unpert", "country", "row_index"])
    rows = read_jsonl(samples_path, dicts_only=True, skip_invalid=True)
    pred_rows: list[dict[str, Any]] = []
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            continue
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        outcome = meta.get("outcome") if isinstance(meta, dict) else None
        if outcome is None:
            outcome = row.get("outcome")
        if not isinstance(outcome, str) or not outcome:
            continue
        country = meta.get("country") if isinstance(meta, dict) else None
        row_index = meta.get("row_index") if isinstance(meta, dict) else None
        y_pred = _mean_prediction(row.get("samples"))
        if y_pred is None:
            continue
        pred_rows.append(
            {
                "id": row_id,
                "outcome": outcome,
                "y_unpert": float(y_pred),
                "country": country,
                "row_index": row_index,
            }
        )
    if not pred_rows:
        return pd.DataFrame(columns=["id", "outcome", "y_unpert", "country", "row_index"])
    df = pd.DataFrame(pred_rows)
    return (
        df.groupby(["id", "outcome", "country", "row_index"], dropna=False, as_index=False)
        .agg(y_unpert=("y_unpert", "mean"))
        .reset_index(drop=True)
    )


def _load_unperturbed_from_paths(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frames.append(_load_unperturbed_predictions(path))
    if not frames:
        return pd.DataFrame(columns=["id", "outcome", "y_unpert", "country", "row_index"])
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return pd.DataFrame(columns=["id", "outcome", "y_unpert", "country", "row_index"])
    return (
        df.groupby(["id", "outcome", "country", "row_index"], dropna=False, as_index=False)
        .agg(y_unpert=("y_unpert", "mean"))
        .reset_index(drop=True)
    )


def _load_row_index_map(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["id", "outcome", "country", "row_index"])
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["id", "outcome", "country", "row_index"])
    if "dataset_id" in df.columns:
        df = df.rename(columns={"dataset_id": "id"})
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    keep = [c for c in ["id", "outcome", "country", "row_index"] if c in df.columns]
    if not keep or "id" not in keep or "outcome" not in keep:
        return pd.DataFrame(columns=["id", "outcome", "country", "row_index"])
    df = df[keep].drop_duplicates()
    df["row_index"] = pd.to_numeric(df.get("row_index"), errors="coerce")
    return df


def _bin_edges(raw: str) -> list[float]:
    edges: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        edges.append(float(part))
    if not edges:
        edges = [0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
    edges = sorted(set(edges))
    if edges[0] != 0.0:
        edges = [0.0, *edges]
    if not np.isinf(edges[-1]):
        edges.append(float("inf"))
    return edges


def _bin_labels(edges: list[float]) -> list[str]:
    labels: list[str] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        if np.isinf(hi):
            labels.append(f"{lo}+" if lo != 0.0 else "0+")
        else:
            labels.append(f"{lo}-{hi}")
    return labels


def _extract_run_context(path: Path) -> dict[str, str | None]:
    parts = path.resolve().parts
    if "prompt_tournament" not in parts:
        return {"model_slug": None, "strategy": None, "run_id": None, "outcome_batch": None, "status": None}
    idx = parts.index("prompt_tournament")
    return {
        "model_slug": parts[idx + 1] if len(parts) > idx + 1 else None,
        "strategy": parts[idx + 3] if len(parts) > idx + 3 else None,
        "run_id": parts[idx + 4] if len(parts) > idx + 4 else None,
        "outcome_batch": parts[idx + 5] if len(parts) > idx + 5 else None,
        "status": parts[idx + 6] if len(parts) > idx + 6 else None,
    }


def _merge_ref_llm(ref: pd.DataFrame, llm: pd.DataFrame) -> pd.DataFrame:
    join_keys: list[str] = []
    for key in ["id", "target", "outcome", "strategy"]:
        if key in ref.columns and key in llm.columns:
            join_keys.append(key)
    if not join_keys:
        raise ValueError("No overlapping keys between ref and llm ICE tables")
    merged = ref.merge(llm, on=join_keys, how="inner", suffixes=("_ref", "_llm"))
    if merged.empty:
        raise ValueError(f"No overlapping rows after merge on keys: {join_keys}")
    return merged


def _summarize_bins(df: pd.DataFrame, *, edges: list[float]) -> pd.DataFrame:
    labels = _bin_labels(edges)
    df = df.copy()
    df["abs_delta_ref"] = df["delta_ref"].abs()
    df["abs_delta_llm"] = df["delta_llm"].abs()
    df["bin"] = pd.cut(
        df["abs_delta_ref"],
        bins=edges,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    def _sign_agree(x: pd.Series, y: pd.Series) -> pd.Series:
        sx = np.sign(x)
        sy = np.sign(y)
        out = pd.Series(np.where((sx == 0) | (sy == 0), np.nan, sx == sy))
        return out

    df["sign_agree"] = _sign_agree(df["delta_ref"], df["delta_llm"])
    grouped = (
        df.groupby(["strategy", "outcome", "target", "calibration_status", "bin"], dropna=False)
        .agg(
            n=("delta_ref", "count"),
            sign_agreement=("sign_agree", "mean"),
            mae_delta=(
                "delta_llm",
                lambda x: float(np.nanmean(np.abs(x - df.loc[x.index, "delta_ref"]))),
            ),
            mean_abs_delta_llm=("abs_delta_llm", "mean"),
            mean_abs_delta_ref=("abs_delta_ref", "mean"),
        )
        .reset_index()
    )
    grouped["slope_ratio"] = grouped["mean_abs_delta_llm"] / grouped["mean_abs_delta_ref"].replace(0, np.nan)

    pooled = (
        df.groupby(["strategy", "outcome", "calibration_status", "bin"], dropna=False)
        .agg(
            n=("delta_ref", "count"),
            sign_agreement=("sign_agree", "mean"),
            mae_delta=(
                "delta_llm",
                lambda x: float(np.nanmean(np.abs(x - df.loc[x.index, "delta_ref"]))),
            ),
            mean_abs_delta_llm=("abs_delta_llm", "mean"),
            mean_abs_delta_ref=("abs_delta_ref", "mean"),
        )
        .reset_index()
    )
    pooled["slope_ratio"] = pooled["mean_abs_delta_llm"] / pooled["mean_abs_delta_ref"].replace(0, np.nan)
    pooled["target"] = "__all__"
    pooled = pooled[grouped.columns]

    return pd.concat([grouped, pooled], ignore_index=True)


def _summarize_beta(df: pd.DataFrame) -> pd.DataFrame:
    beta_llm = df["delta_llm"] / df["delta_z"].replace(0, np.nan)
    beta_ref = df["delta_ref"] / df["delta_z"].replace(0, np.nan)
    df = df.assign(beta_llm=beta_llm, beta_ref=beta_ref)

    def _summarize(group: pd.DataFrame) -> dict[str, Any]:
        x = group["beta_ref"].to_numpy(dtype=float)
        y = group["beta_llm"].to_numpy(dtype=float)
        bias = float(np.nanmean(y - x)) if np.isfinite(y).any() and np.isfinite(x).any() else float("nan")
        mae = float(np.nanmean(np.abs(y - x))) if np.isfinite(y).any() and np.isfinite(x).any() else float("nan")
        ratio = float(np.nanmean(np.abs(y)) / np.nanmean(np.abs(x))) if np.nanmean(np.abs(x)) > 0 else float("nan")
        return {
            "n": int((np.isfinite(x) & np.isfinite(y)).sum()),
            "pearson": _safe_corr(x, y),
            "spearman": _spearman(x, y),
            "bias": bias,
            "mae": mae,
            "slope_ratio": ratio,
        }

    grouped = (
        df.groupby(["strategy", "outcome", "target", "calibration_status"], dropna=False)
        .apply(_summarize)
        .apply(pd.Series)
        .reset_index()
    )

    pooled = (
        df.groupby(["strategy", "outcome", "calibration_status"], dropna=False)
        .apply(_summarize)
        .apply(pd.Series)
        .reset_index()
    )
    pooled["target"] = "__all__"
    pooled = pooled[grouped.columns]

    return pd.concat([grouped, pooled], ignore_index=True)


def _midpoint_invariance(
    df: pd.DataFrame,
    *,
    unperturbed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if unperturbed.empty:
        return (
            pd.DataFrame(columns=["strategy", "outcome", "target", "calibration_status", "n", "mae", "bias"]),
            pd.DataFrame(columns=["id", "outcome", "target", "strategy", "calibration_status", "m_i", "y_unpert"]),
        )
    merged = pd.DataFrame()
    if "id" in unperturbed.columns and "id" in df.columns:
        merged = df.merge(unperturbed, on=["id", "outcome"], how="inner")
    if (
        merged.empty
        and {"country", "row_index", "outcome"}.issubset(unperturbed.columns)
        and {
            "country",
            "row_index",
            "outcome",
        }.issubset(df.columns)
    ):
        left = df.copy()
        right = unperturbed.copy()
        left["row_index"] = pd.to_numeric(left["row_index"], errors="coerce")
        right["row_index"] = pd.to_numeric(right["row_index"], errors="coerce")
        merged = left.merge(right, on=["outcome", "country", "row_index"], how="inner")
        if "id" not in merged.columns:
            for candidate in ["id_x", "id_y"]:
                if candidate in merged.columns:
                    merged["id"] = merged[candidate]
                    break
            if "id" not in merged.columns:
                merged["id"] = pd.NA
    if merged.empty:
        return (
            pd.DataFrame(columns=["strategy", "outcome", "target", "calibration_status", "n", "mae", "bias"]),
            pd.DataFrame(columns=["id", "outcome", "target", "strategy", "calibration_status", "m_i", "y_unpert"]),
        )
    merged = merged.copy()
    merged["m_i"] = (merged["ice_low_llm"] + merged["ice_high_llm"]) / 2.0
    merged["err"] = merged["m_i"] - merged["y_unpert"]

    def _summarize(group: pd.DataFrame) -> dict[str, Any]:
        err = pd.to_numeric(group["err"], errors="coerce").to_numpy(dtype=float)
        m_i = pd.to_numeric(group["m_i"], errors="coerce").to_numpy(dtype=float)
        y_u = pd.to_numeric(group["y_unpert"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(err)
        return {
            "n": int(mask.sum()),
            "mae": float(np.nanmean(np.abs(err))) if mask.any() else float("nan"),
            "bias": float(np.nanmean(err)) if mask.any() else float("nan"),
            "rmse": float(np.sqrt(np.nanmean(err * err))) if mask.any() else float("nan"),
            "pearson": _safe_corr(m_i, y_u),
            "spearman": _spearman(m_i, y_u),
        }

    summary = (
        merged.groupby(["strategy", "outcome", "target", "calibration_status"], dropna=False)
        .apply(_summarize)
        .apply(pd.Series)
        .reset_index()
    )
    pooled = (
        merged.groupby(["strategy", "outcome", "calibration_status"], dropna=False)
        .apply(_summarize)
        .apply(pd.Series)
        .reset_index()
    )
    pooled["target"] = "__all__"
    pooled = pooled[summary.columns]
    summary = pd.concat([summary, pooled], ignore_index=True)

    rows = merged[["id", "outcome", "target", "strategy", "calibration_status", "m_i", "y_unpert", "err"]].copy()
    return summary, rows


def _additivity_variance(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = (
        df.groupby(["id", "outcome", "strategy", "calibration_status"], dropna=False)
        .agg(
            var_delta_llm=("delta_llm", "var"),
            var_delta_ref=("delta_ref", "var"),
            n_targets=("target", "nunique"),
        )
        .reset_index()
    )

    def _summarize(group: pd.DataFrame) -> dict[str, Any]:
        v_llm = pd.to_numeric(group["var_delta_llm"], errors="coerce").to_numpy(dtype=float)
        v_ref = pd.to_numeric(group["var_delta_ref"], errors="coerce").to_numpy(dtype=float)
        return {
            "n": int(len(group)),
            "mean_var_llm": float(np.nanmean(v_llm)) if np.isfinite(v_llm).any() else float("nan"),
            "mean_var_ref": float(np.nanmean(v_ref)) if np.isfinite(v_ref).any() else float("nan"),
            "median_var_llm": float(np.nanmedian(v_llm)) if np.isfinite(v_llm).any() else float("nan"),
            "median_var_ref": float(np.nanmedian(v_ref)) if np.isfinite(v_ref).any() else float("nan"),
            "var_ratio": float(np.nanmean(v_llm) / np.nanmean(v_ref)) if np.nanmean(v_ref) > 0 else float("nan"),
            "pearson": _safe_corr(v_ref, v_llm),
            "spearman": _spearman(v_ref, v_llm),
        }

    summary = (
        grouped.groupby(["strategy", "outcome", "calibration_status"], dropna=False)
        .apply(_summarize)
        .apply(pd.Series)
        .reset_index()
    )
    return summary, grouped


def _reliability_summary(uncal: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    join_keys: list[str] = []
    for key in ["id", "target", "outcome", "strategy"]:
        if key in uncal.columns and key in cal.columns:
            join_keys.append(key)
    if not join_keys:
        return pd.DataFrame(
            columns=["strategy", "outcome", "n", "pearson", "spearman", "bias", "mae", "calibration_status"]
        )

    merged = uncal.merge(cal, on=join_keys, suffixes=("_uncal", "_cal"), how="inner")
    if merged.empty:
        return pd.DataFrame(
            columns=["strategy", "outcome", "n", "pearson", "spearman", "bias", "mae", "calibration_status"]
        )

    merged["delta_uncal"] = _delta(merged, low_col="ice_low_uncal", high_col="ice_high_uncal")
    merged["delta_cal"] = _delta(merged, low_col="ice_low_cal", high_col="ice_high_cal")

    def _summarize(group: pd.DataFrame) -> dict[str, Any]:
        x = group["delta_uncal"].to_numpy(dtype=float)
        y = group["delta_cal"].to_numpy(dtype=float)
        return {
            "n": int((np.isfinite(x) & np.isfinite(y)).sum()),
            "pearson": _safe_corr(x, y),
            "spearman": _spearman(x, y),
            "bias": float(np.nanmean(y - x)) if np.isfinite(y).any() else float("nan"),
            "mae": float(np.nanmean(np.abs(y - x))) if np.isfinite(y).any() else float("nan"),
        }

    summary = merged.groupby(["strategy", "outcome"], dropna=False).apply(_summarize).apply(pd.Series).reset_index()
    summary["calibration_status"] = "calibrated_vs_uncalibrated"
    return summary


def _determinism_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        low_s = _parse_list(row.get("ice_low_samples"))
        high_s = _parse_list(row.get("ice_high_samples"))
        if low_s is None and high_s is None:
            continue
        low_arr = np.asarray(low_s, dtype=float) if low_s else np.asarray([], dtype=float)
        high_arr = np.asarray(high_s, dtype=float) if high_s else np.asarray([], dtype=float)
        sd_low = float(np.std(low_arr, ddof=1)) if low_arr.size >= 2 else float("nan")
        sd_high = float(np.std(high_arr, ddof=1)) if high_arr.size >= 2 else float("nan")
        sd_delta = float("nan")
        if low_arr.size >= 2 and high_arr.size >= 2 and low_arr.size == high_arr.size:
            sd_delta = float(np.std(high_arr - low_arr, ddof=1))
        rows.append(
            {
                "id": row.get("id"),
                "strategy": row.get("strategy"),
                "outcome": row.get("outcome"),
                "target": row.get("target"),
                "calibration_status": row.get("calibration_status"),
                "sd_low": sd_low,
                "sd_high": sd_high,
                "sd_delta": sd_delta,
                "n_low": int(low_arr.size),
                "n_high": int(high_arr.size),
            }
        )

    row_df = pd.DataFrame(rows)
    if row_df.empty:
        return (
            pd.DataFrame(columns=["strategy", "outcome", "target", "calibration_status", "mean_sd_delta"]),
            row_df,
        )

    summary = (
        row_df.groupby(["strategy", "outcome", "target", "calibration_status"], dropna=False)
        .agg(
            mean_sd_low=("sd_low", "mean"),
            mean_sd_high=("sd_high", "mean"),
            mean_sd_delta=("sd_delta", "mean"),
            median_sd_delta=("sd_delta", "median"),
            mean_n_low=("n_low", "mean"),
            mean_n_high=("n_high", "mean"),
        )
        .reset_index()
    )
    pooled = (
        row_df.groupby(["strategy", "outcome", "calibration_status"], dropna=False)
        .agg(
            mean_sd_low=("sd_low", "mean"),
            mean_sd_high=("sd_high", "mean"),
            mean_sd_delta=("sd_delta", "mean"),
            median_sd_delta=("sd_delta", "median"),
            mean_n_low=("n_low", "mean"),
            mean_n_high=("n_high", "mean"),
        )
        .reset_index()
    )
    pooled["target"] = "__all__"
    pooled = pooled[summary.columns]

    return pd.concat([summary, pooled], ignore_index=True), row_df


def run_cli(argv: list[str] | None = None) -> None:
    parser = cli.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="evaluation/output/prompt_tournament",
        help="Root output directory for prompt_tournament",
    )
    parser.add_argument("--model-slug", default=None, help="Optional model slug to filter")
    parser.add_argument("--run-id", default=None, help="Limit to a specific run id")
    parser.add_argument("--strategies", default=None, help="Comma-separated strategies to include")
    parser.add_argument("--bins", default="0,0.1,0.2,0.4,0.8,1.6,3.2")
    parser.add_argument("--skip-midpoint", action="store_true")
    parser.add_argument("--skip-reliability", action="store_true")
    parser.add_argument("--skip-determinism", action="store_true")

    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    strategies_filter = [s.strip() for s in args.strategies.split(",") if s.strip()] if args.strategies else None
    edges = _bin_edges(args.bins)

    ice_paths: list[Path] = []
    for ice_path in output_root.glob("*/perturbed/*/*/*/uncalibrated/ice_llm.csv"):
        ice_paths.append(ice_path)
    for ice_path in output_root.glob("*/perturbed/*/*/*/calibrated/ice_llm_calibrated.csv"):
        ice_paths.append(ice_path)

    for ice_path in sorted(ice_paths):
        ctx = _extract_run_context(ice_path)
        model_slug = ctx.get("model_slug")
        strategy = ctx.get("strategy")
        run_id = ctx.get("run_id")
        outcome_batch = ctx.get("outcome_batch")
        status = ctx.get("status")
        if args.model_slug and model_slug != args.model_slug:
            continue
        if args.run_id and run_id != args.run_id:
            continue
        if strategies_filter and strategy not in strategies_filter:
            continue
        if strategy is None or run_id is None or outcome_batch is None or status is None:
            continue

        llm = pd.read_csv(ice_path)
        if "outcome" not in llm.columns:
            llm["outcome"] = "unknown"
        llm = llm.copy()
        llm["calibration_status"] = status

        ref_frames: list[pd.DataFrame] = []
        for outcome in sorted({str(o) for o in llm["outcome"].dropna().unique().tolist()}):
            ref_path = Path(f"empirical/output/anchors/pe/mutable_engines/reduced/{outcome}/ice_ref.csv")
            if not ref_path.exists():
                continue
            ref_frames.append(pd.read_csv(ref_path))
        if not ref_frames:
            continue
        ref = pd.concat(ref_frames, ignore_index=True)

        try:
            merged = _merge_ref_llm(ref, llm)
        except ValueError:
            continue

        merged["delta_ref"] = _delta(merged, low_col="ice_low_ref", high_col="ice_high_ref")
        merged["delta_llm"] = _delta(merged, low_col="ice_low_llm", high_col="ice_high_llm")
        if "delta_z" in merged.columns:
            merged["delta_z"] = pd.to_numeric(merged["delta_z"], errors="coerce")
        elif "delta_z_ref" in merged.columns:
            merged["delta_z"] = pd.to_numeric(merged["delta_z_ref"], errors="coerce")
        elif "delta_z_llm" in merged.columns:
            merged["delta_z"] = pd.to_numeric(merged["delta_z_llm"], errors="coerce")
        else:
            merged["delta_z"] = float("nan")

        posthoc_dir = ice_path.parent / "posthoc"
        posthoc_dir.mkdir(parents=True, exist_ok=True)

        row_level_path = ice_path.parent / "row_level.csv"
        row_map = _load_row_index_map(row_level_path)
        if row_map.empty or "country" not in row_map.columns or "row_index" not in row_map.columns:
            fallback_row_level = ice_path.parent.parent / "uncalibrated" / "row_level.csv"
            if fallback_row_level.exists():
                row_map = _load_row_index_map(fallback_row_level)
        if not row_map.empty:
            merged = merged.merge(row_map, on=["id", "outcome"], how="left")

        bins_df = _summarize_bins(merged, edges=edges)
        bins_df.to_csv(posthoc_dir / "effect_size_bins.csv", index=False)

        beta_df = _summarize_beta(merged)
        beta_df.to_csv(posthoc_dir / "beta_normalized_summary.csv", index=False)

        if not args.skip_midpoint:
            family = strategy
            unperturbed_root = output_root / "unperturbed" / family
            unperturbed_paths = list(
                (unperturbed_root / run_id).glob(f"{outcome_batch}/uncalibrated/samples__{family}.jsonl")
            )
            unperturbed_paths += list((unperturbed_root / run_id).glob(f"{outcome_batch}/uncalibrated/samples.jsonl"))
            if not unperturbed_paths:
                unperturbed_paths = list(
                    unperturbed_root.glob(f"*/{outcome_batch}/uncalibrated/samples__{family}.jsonl")
                )
                unperturbed_paths += list(unperturbed_root.glob(f"*/{outcome_batch}/uncalibrated/samples.jsonl"))
            if not unperturbed_paths:
                unperturbed_paths = list(unperturbed_root.glob(f"*/*/uncalibrated/samples__{family}.jsonl"))
                unperturbed_paths += list(unperturbed_root.glob("*/*/uncalibrated/samples.jsonl"))
            unperturbed_df = _load_unperturbed_from_paths(unperturbed_paths)
            midpoint_summary, midpoint_rows = _midpoint_invariance(merged, unperturbed=unperturbed_df)
            midpoint_summary.to_csv(posthoc_dir / "midpoint_invariance_summary.csv", index=False)
            midpoint_rows.to_csv(posthoc_dir / "midpoint_invariance_rows.csv", index=False)

        additivity_summary, additivity_rows = _additivity_variance(merged)
        additivity_summary.to_csv(posthoc_dir / "additivity_variance_summary.csv", index=False)
        additivity_rows.to_csv(posthoc_dir / "additivity_variance_rows.csv", index=False)

        if not args.skip_determinism:
            determinism_summary, determinism_rows = _determinism_summary(llm)
            determinism_summary.to_csv(posthoc_dir / "determinism_summary.csv", index=False)
            determinism_rows.to_csv(posthoc_dir / "determinism_rows.csv", index=False)

        if not args.skip_reliability and status == "uncalibrated":
            cal_path = ice_path.parent.parent / "calibrated" / "ice_llm_calibrated.csv"
            if cal_path.exists():
                cal = pd.read_csv(cal_path)
                if "outcome" not in cal.columns:
                    cal["outcome"] = "unknown"
                rel_summary = _reliability_summary(llm, cal)
                rel_summary.to_csv(posthoc_dir / "reliability_summary.csv", index=False)



