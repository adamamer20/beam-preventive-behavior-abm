"""Recompute belief-update unperturbed metrics from canonical outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from beam_abm.cli import parser_compat as cli
from beam_abm.evaluation.belief.canonicalize import (
    clear_summary_metrics,
    rebuild_canonical_from_runs,
    write_summary_metrics,
)
from beam_abm.evaluation.belief.unperturbed_metrics import run_cli as summarize_unperturbed_metrics


def _coerce_scalar(value: object) -> float | str | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return value
    return None


def _summarize_model(*, output_root: Path, model_dir: Path) -> Path | None:
    overall = model_dir / "overall_summary.json"
    if not overall.exists():
        return None

    try:
        payload = json.loads(overall.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    strategies = payload.get("strategies") if isinstance(payload, dict) else None
    if not isinstance(strategies, list):
        return None

    rows: list[dict[str, Any]] = []
    for entry in strategies:
        if not isinstance(entry, dict):
            continue
        row: dict[str, Any] = {"model": model_dir.name, "strategy": str(entry.get("strategy") or "data_only")}
        for key, value in entry.items():
            if key == "strategy":
                continue
            coerced = _coerce_scalar(value)
            if coerced is not None:
                row[key] = coerced
        rows.append(row)

    if not rows:
        return None

    out_path = output_root / "summary_metrics" / model_dir.name / "summary_metrics.csv"
    write_summary_metrics(rows=rows, out_path=out_path)
    return out_path


def run_cli(argv: list[str] | None = None) -> None:
    parser = cli.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="evaluation/output/belief_update_validation/unperturbed",
        help="Belief-update unperturbed root (contains _runs/, canonical/, summary_metrics/).",
    )
    parser.add_argument(
        "--spec",
        default="evaluation/specs/belief_update_targets.json",
        help="Belief-update spec JSON.",
    )
    parser.add_argument(
        "--skip-rebuild-canonical",
        action="store_true",
        help="Skip rebuilding canonical/<model>/samples.jsonl from _runs/.",
    )
    parser.add_argument(
        "--clear-summary-metrics",
        action="store_true",
        help="Remove summary_metrics/ before writing fresh files.",
    )
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    canonical_root = output_root / "canonical"
    if args.clear_summary_metrics:
        clear_summary_metrics(output_root=output_root)

    if not args.skip_rebuild_canonical:
        rebuild_canonical_from_runs(output_root=output_root, clear=False)

    if not canonical_root.exists():
        print(f"No canonical outputs found at {canonical_root}")
        return

    all_rows: list[dict[str, Any]] = []
    for model_dir in sorted(p for p in canonical_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        samples_path = model_dir / "samples.jsonl"
        if not samples_path.exists():
            continue

        summarize_unperturbed_metrics(
            [
                "--in",
                str(samples_path),
                "--spec",
                str(Path(args.spec)),
                "--outdir",
                str(model_dir),
            ],
        )
        out_path = _summarize_model(output_root=output_root, model_dir=model_dir)
        if out_path is None or not out_path.exists():
            continue
        try:
            df = pd.read_csv(out_path)
            all_rows.extend(df.to_dict(orient="records"))
        except (OSError, ValueError):
            continue

    if all_rows:
        write_summary_metrics(
            rows=all_rows,
            out_path=output_root / "summary_metrics" / "all_models_strategies.csv",
        )
