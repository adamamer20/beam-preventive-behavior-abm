"""Compute predictions and metrics for unperturbed sampling runs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from beam_abm.evaluation.choice.support.unperturbed.unperturbed_utils import (
    SUPPORTED_PROMPT_FAMILIES,
    compute_metrics_rows,
    metrics_filename,
    parse_prompt_families,
    read_jsonl,
)
from beam_abm.llm.schemas.predictions import extract_scalar_prediction


def _metadata(row: dict) -> dict[str, Any]:
    meta = row.get("metadata")
    return meta if isinstance(meta, dict) else {}


def _outcome(row: dict) -> str:
    outcome = row.get("outcome")
    if isinstance(outcome, str) and outcome:
        return outcome
    meta = _metadata(row)
    outcome = meta.get("outcome")
    if isinstance(outcome, str) and outcome:
        return outcome
    raise ValueError("Outcome missing in samples row")


def _outcome_type(row: dict) -> str:
    outcome_type = row.get("outcome_type")
    if isinstance(outcome_type, str) and outcome_type:
        return outcome_type
    meta = _metadata(row)
    outcome_type = meta.get("outcome_type")
    if isinstance(outcome_type, str) and outcome_type:
        return outcome_type
    raise ValueError("Outcome type missing in samples row")


def _y_true(row: dict) -> float | None:
    meta = _metadata(row)
    y_true = meta.get("y_true")
    if y_true is None:
        y_true = row.get("y_true")
    if y_true is None:
        return None
    try:
        return float(y_true)
    except Exception:
        return None


def _country(row: dict) -> str:
    meta = _metadata(row)
    country = meta.get("country")
    if country is None:
        country = row.get("country", "")
    return "" if country is None else str(country)


def _pred_mean(samples: list[dict]) -> float:
    values: list[float] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        scalar = sample.get("prediction_scalar")
        if scalar is None:
            scalar = extract_scalar_prediction(sample.get("parsed"))
        if scalar is None:
            continue
        try:
            values.append(float(scalar))
        except Exception:
            continue
    if not values:
        return float("nan")
    return float(np.mean(values))


def _load_predictions(path: Path, family: str) -> pd.DataFrame:
    if path.is_dir():
        files = sorted(path.glob(f"predictions__{family}__*.csv"))
        if not files:
            raise FileNotFoundError(f"No predictions__{family}__*.csv files in {path}")
        frames = [pd.read_csv(f) for f in files]
        return pd.concat(frames, ignore_index=True)
    return pd.read_csv(path)


def _safe_token(value: object) -> str:
    text = str(value) if value is not None else ""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return cleaned or "unknown"


def _write_predictions(outdir: Path, family: str, df: pd.DataFrame) -> None:
    if df.empty:
        return

    for (model, outcome), g in df.groupby(["model", "outcome"], dropna=False):
        model_token = _safe_token(model)
        outcome_token = _safe_token(outcome)
        out_path = outdir / f"predictions__{family}__{model_token}__{outcome_token}.csv"
        g.to_csv(out_path, index=False)

    df.to_csv(outdir / f"predictions__{family}__all.csv", index=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=None, help="Samples JSONL path or directory")
    parser.add_argument("--predictions", default=None, help="Predictions CSV path or directory")
    parser.add_argument("--outdir", default="evaluation/output/debug_unperturbed")
    parser.add_argument("--prompt-family", default=None)
    parser.add_argument(
        "--prompt-families",
        default=",".join(sorted(SUPPORTED_PROMPT_FAMILIES)),
        help="Comma-separated prompt families (data_only,first_person,multi_expert)",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Skip samples and compute metrics from predictions files.",
    )
    parser.add_argument(
        "--pred-col",
        default="y_pred",
        help="Prediction column to score (e.g., y_pred or y_pred_cal).",
    )

    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    families = parse_prompt_families(args.prompt_family, args.prompt_families)

    if args.metrics_only or args.predictions:
        pred_path = Path(args.predictions) if args.predictions else outdir
        pred_col = str(args.pred_col)
        for family in families:
            df = _load_predictions(pred_path, family)
            metrics_rows = compute_metrics_rows(df, pred_col=pred_col)
            metrics_df = pd.DataFrame(metrics_rows)
            metrics_df.to_csv(outdir / metrics_filename(family=family, pred_col=pred_col), index=False)
        return

    for family in families:
        samples_path = Path(args.samples) if args.samples else outdir / f"samples__{family}.jsonl"
        if samples_path.is_dir():
            samples_path = samples_path / f"samples__{family}.jsonl"
        if not samples_path.exists():
            raise FileNotFoundError(f"Samples JSONL not found: {samples_path}")

        rows = read_jsonl(samples_path)
        prediction_rows: list[dict[str, Any]] = []

        for row in rows:
            samples = row.get("samples")
            if not isinstance(samples, list):
                continue
            y_true = _y_true(row)
            if y_true is None:
                continue

            prediction_rows.append(
                {
                    "model": row.get("model"),
                    "outcome": _outcome(row),
                    "outcome_type": _outcome_type(row),
                    "prompt_family": row.get("prompt_family", family),
                    "y_true": y_true,
                    "y_pred": _pred_mean(samples),
                    "country": _country(row),
                }
            )

        pred_df = pd.DataFrame(prediction_rows)
        _write_predictions(outdir, family, pred_df)

        metrics_rows = compute_metrics_rows(pred_df, pred_col="y_pred")
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(outdir / metrics_filename(family=family, pred_col="y_pred"), index=False)


if __name__ == "__main__":
    main()
