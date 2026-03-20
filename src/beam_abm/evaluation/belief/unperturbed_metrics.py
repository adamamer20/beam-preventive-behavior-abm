"""Compute unperturbed belief-update accuracy metrics from sampled outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from beam_abm.cli import parser_compat as cli
from beam_abm.evaluation.utils.jsonl import read_jsonl as _read_jsonl
from beam_abm.evaluation.utils.jsonl import write_jsonl as _write_jsonl


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _safe_float(val: object) -> float | None:
    if val is None:
        return None
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _extract_post_state(parsed: object, mutable_state: list[str]) -> dict[str, float] | None:
    if not isinstance(parsed, dict):
        return None
    if "A" in parsed and "B" in parsed:
        # If paired output appears, use B by convention for post state.
        if isinstance(parsed.get("B"), dict):
            parsed = parsed.get("B")
        elif isinstance(parsed.get("A"), dict):
            parsed = parsed.get("A")
    out: dict[str, float] = {}
    for var in mutable_state:
        val = _safe_float(parsed.get(var))
        if val is None and var == "covid_perceived_danger_Y":
            val = _safe_float(parsed.get("covid_perceived_danger_T12"))
        if val is None and var == "covid_perceived_danger_T12":
            val = _safe_float(parsed.get("covid_perceived_danger_Y"))
        if val is not None:
            out[var] = val
    return out


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return None
    rx = pd.Series(x[mask]).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y[mask]).rank(method="average").to_numpy(dtype=float)
    vx = rx - float(np.mean(rx))
    vy = ry - float(np.mean(ry))
    denom = float(np.linalg.norm(vx) * np.linalg.norm(vy))
    if denom <= 0.0:
        return None
    return float(np.dot(vx, vy) / denom)


def _cosine(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) == 0:
        return None
    xv = x[mask]
    yv = y[mask]
    nx = float(np.linalg.norm(xv))
    ny = float(np.linalg.norm(yv))
    if nx <= 0.0 or ny <= 0.0:
        return None
    return float(np.dot(xv, yv) / (nx * ny))


def _group_key(row: dict[str, Any]) -> str:
    strategy = row.get("strategy")
    if isinstance(strategy, str) and strategy.strip():
        return strategy.strip()
    return "data_only"


def _row_metrics(row: dict[str, Any], mutable_state: list[str]) -> dict[str, Any]:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    baseline_state = meta.get("baseline_state") if isinstance(meta.get("baseline_state"), dict) else {}
    base_id = meta.get("base_id") or row.get("id")

    samples = row.get("samples") if isinstance(row.get("samples"), list) else []
    sample_posts: list[dict[str, float]] = []
    parsed_count = 0
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        post_state = _extract_post_state(sample.get("parsed"), mutable_state)
        if post_state is not None:
            parsed_count += 1
            sample_posts.append(post_state)

    out: dict[str, Any] = {
        "id": base_id,
        "strategy": _group_key(row),
        "country": meta.get("country"),
        "row_index": meta.get("row_index"),
        "n_samples": len(samples),
        "n_parsed": parsed_count,
    }

    for var in mutable_state:
        y_true = _safe_float(baseline_state.get(var))
        vals = [post[var] for post in sample_posts if var in post]
        y_pred = float(np.mean(vals)) if vals else None
        err = (float(y_pred) - float(y_true)) if (y_pred is not None and y_true is not None) else None
        out[f"true_{var}"] = y_true
        out[f"pred_{var}"] = y_pred
        out[f"err_{var}"] = err
        out[f"abs_err_{var}"] = abs(float(err)) if err is not None else None
        out[f"sq_err_{var}"] = (float(err) ** 2) if err is not None else None

    return out


def _strategy_summary(rows: list[dict[str, Any]], mutable_state: list[str]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "strategy": str(df["strategy"].iloc[0]) if not df.empty and "strategy" in df.columns else "data_only",
        "n_rows": float(df.shape[0]),
    }
    if df.empty:
        return summary

    n_samples = float(pd.to_numeric(df.get("n_samples"), errors="coerce").fillna(0.0).sum())
    n_parsed = float(pd.to_numeric(df.get("n_parsed"), errors="coerce").fillna(0.0).sum())
    summary["parse_n_rows"] = float(df.shape[0])
    summary["parse_n_samples"] = n_samples
    summary["parse_n_parsed"] = n_parsed
    summary["parse_rate"] = (n_parsed / n_samples) if n_samples > 0 else None
    summary["n_samples"] = n_samples
    summary["n_parsed"] = n_parsed
    summary["parsed_rate"] = summary["parse_rate"]

    var_maes: list[float] = []
    var_rmses: list[float] = []
    var_spearmans: list[float] = []
    flat_true: list[float] = []
    flat_pred: list[float] = []

    for var in mutable_state:
        true_col = f"true_{var}"
        pred_col = f"pred_{var}"
        if true_col not in df.columns or pred_col not in df.columns:
            continue
        y_true = pd.to_numeric(df[true_col], errors="coerce").to_numpy(dtype=float)
        y_pred = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        n_obs = int(mask.sum())
        summary[f"n_{var}"] = float(n_obs)
        if n_obs == 0:
            summary[f"mae_{var}"] = None
            summary[f"rmse_{var}"] = None
            summary[f"spearman_{var}"] = None
            continue
        abs_err = np.abs(y_pred[mask] - y_true[mask])
        sq_err = (y_pred[mask] - y_true[mask]) ** 2
        mae = float(np.mean(abs_err))
        rmse = float(np.sqrt(np.mean(sq_err)))
        sp = _spearman(y_true, y_pred)
        summary[f"mae_{var}"] = mae
        summary[f"rmse_{var}"] = rmse
        summary[f"spearman_{var}"] = sp
        var_maes.append(mae)
        var_rmses.append(rmse)
        if sp is not None:
            var_spearmans.append(sp)
        flat_true.extend(y_true[mask].tolist())
        flat_pred.extend(y_pred[mask].tolist())

    if flat_true and flat_pred:
        f_true = np.asarray(flat_true, dtype=float)
        f_pred = np.asarray(flat_pred, dtype=float)
        summary["mae_overall"] = float(np.mean(np.abs(f_pred - f_true)))
        summary["rmse_overall"] = float(np.sqrt(np.mean((f_pred - f_true) ** 2)))
        summary["spearman_overall"] = _spearman(f_true, f_pred)
        summary["n_obs_overall"] = float(len(flat_true))
    else:
        summary["mae_overall"] = None
        summary["rmse_overall"] = None
        summary["spearman_overall"] = None
        summary["n_obs_overall"] = 0.0

    summary["mae_mean"] = float(np.mean(var_maes)) if var_maes else None
    summary["rmse_mean"] = float(np.mean(var_rmses)) if var_rmses else None
    summary["spearman_mean"] = float(np.mean(var_spearmans)) if var_spearmans else None

    row_vector_mae: list[float] = []
    row_vector_cos: list[float] = []
    for row in rows:
        tv: list[float] = []
        pv: list[float] = []
        for var in mutable_state:
            t = _safe_float(row.get(f"true_{var}"))
            p = _safe_float(row.get(f"pred_{var}"))
            if t is None or p is None:
                continue
            tv.append(float(t))
            pv.append(float(p))
        if not tv:
            continue
        ta = np.asarray(tv, dtype=float)
        pa = np.asarray(pv, dtype=float)
        row_vector_mae.append(float(np.mean(np.abs(pa - ta))))
        c = _cosine(pa, ta)
        if c is not None:
            row_vector_cos.append(c)
    summary["vector_mae"] = float(np.mean(row_vector_mae)) if row_vector_mae else None
    summary["vector_cosine"] = float(np.mean(row_vector_cos)) if row_vector_cos else None

    return summary


def run_cli(argv: list[str] | None = None) -> None:
    parser = cli.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="Input canonical samples JSONL.")
    parser.add_argument("--spec", dest="spec_path", required=True, help="Belief target spec JSON.")
    parser.add_argument("--outdir", required=True, help="Canonical model output directory.")
    args = parser.parse_args(argv)

    spec = json.loads(Path(args.spec_path).read_text(encoding="utf-8"))
    mutable_state = [str(x) for x in spec.get("mutable_state", []) if str(x).strip()]
    rows = _read_jsonl(Path(args.in_path))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_strategy.setdefault(_group_key(row), []).append(row)

    strategy_summaries: list[dict[str, Any]] = []
    for strategy, strategy_rows in sorted(by_strategy.items()):
        row_metrics = [_row_metrics(row, mutable_state) for row in strategy_rows]
        strategy_dir = outdir / strategy
        strategy_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(strategy_dir / "rows.jsonl", row_metrics)

        if row_metrics:
            wrote_metrics = False
            try:
                import polars as pl

                pl.DataFrame(row_metrics, infer_schema_length=None).write_parquet(strategy_dir / "metrics.parquet")
                wrote_metrics = True
            except Exception:
                wrote_metrics = False
            if not wrote_metrics:
                frame = pd.DataFrame(row_metrics)
                try:
                    frame.to_parquet(strategy_dir / "metrics.parquet", index=False)
                except (OSError, ValueError):
                    frame.to_csv(strategy_dir / "metrics.csv", index=False)
        else:
            (strategy_dir / "metrics.parquet").write_bytes(b"")

        summary = _strategy_summary(row_metrics, mutable_state)
        strategy_summaries.append(summary)
        _write_json(strategy_dir / "strategy_summary.json", summary)

    _write_json(outdir / "overall_summary.json", {"strategies": strategy_summaries})
