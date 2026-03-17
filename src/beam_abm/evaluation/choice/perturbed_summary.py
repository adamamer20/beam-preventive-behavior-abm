"""Global summary helpers for perturbed choice evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def write_global_summary(*, output_root: Path) -> None:
    """Write a coarse global summary table across canonical leaves."""

    rows: list[dict[str, Any]] = []

    for outcome_dir in sorted(p for p in output_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for model_dir in sorted(p for p in outcome_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
            for strategy_dir in sorted(p for p in model_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
                for status in ["uncalibrated", "calibrated"]:
                    metrics_path = strategy_dir / status / "summary_metrics.csv"
                    if not metrics_path.exists():
                        continue
                    try:
                        df = pd.read_csv(metrics_path)
                    except (OSError, ValueError, pd.errors.ParserError):
                        continue
                    if df.empty:
                        continue
                    first = df.iloc[0].to_dict()
                    row: dict[str, Any] = {
                        "outcome": outcome_dir.name,
                        "model": model_dir.name,
                        "strategy": strategy_dir.name,
                        "status": status,
                    }
                    # Prefix metrics columns to avoid collisions.
                    for k, v in first.items():
                        row[f"metrics_{k}"] = v
                    rows.append(row)

    if not rows:
        return

    out_dir = output_root / "_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "all_outcomes_models_strategies.csv", index=False)
