"""Recompute belief-update validation metrics from existing raw outputs.

This script:
- Optionally migrates legacy Phase 0 outputs into the canonical `_runs/` layout.
- Optionally rebuilds canonical `canonical/<model>/samples.jsonl` from `_runs/`.
- Re-runs belief-update alignment metrics (no LLM sampling) for each canonical model.
- Writes scalar summary artifacts under `summary_metrics/`.

It is safe to run repeatedly.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from beam_abm.cli import parser_compat as cli
from beam_abm.evaluation.belief.canonicalize import (
    clear_summary_metrics,
    migrate_legacy_prompt_tournament_outputs,
    rebuild_canonical_from_runs,
    write_summary_metrics,
)
from beam_abm.evaluation.belief.update_alignment import run_cli as update_alignment
from beam_abm.evaluation.choice._canonicalize_ids import normalize_model_slug


def _iter_jsonl_dicts(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _parse_rates_by_strategy(samples_path: Path) -> dict[str, dict[str, float]]:
    """Compute parsing/structuring diagnostics for *all* strategies in one pass."""

    counters: dict[str, dict[str, int]] = {}

    for row in _iter_jsonl_dicts(samples_path):
        strategy = str(row.get("strategy") or "data_only")
        bucket = counters.setdefault(strategy, {"n_rows": 0, "n_samples": 0, "n_parsed": 0})
        bucket["n_rows"] += 1
        samples = row.get("samples") if isinstance(row.get("samples"), list) else []
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            bucket["n_samples"] += 1
            if isinstance(sample.get("parsed"), dict):
                bucket["n_parsed"] += 1

    out: dict[str, dict[str, float]] = {}
    for strategy, c in counters.items():
        n_samples = int(c.get("n_samples") or 0)
        n_parsed = int(c.get("n_parsed") or 0)
        parsed_rate = float(n_parsed / n_samples) if n_samples else float("nan")
        out[strategy] = {
            "parse_n_rows": float(c.get("n_rows") or 0),
            "parse_n_samples": float(n_samples),
            "parse_n_parsed": float(n_parsed),
            "parse_rate": parsed_rate,
            "n_samples": float(n_samples),
            "n_parsed": float(n_parsed),
            "parsed_rate": parsed_rate,
        }
    return out


def _safe_float(x: object) -> float | None:
    try:
        val = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def _coerce_numeric_metric(value: object) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return _safe_float(value)


def _collect_moderation_metrics(canonical_model_dir: Path, *, strategy: str) -> dict[str, float]:
    """Aggregate moderation diagnostics across signals."""

    base = canonical_model_dir / strategy / "signals"
    if not base.exists():
        return {}

    moderation_passes: list[float] = []
    moderation_gaps: list[float] = []
    moderation_covered = 0
    moderation_total = 0

    for signal_dir in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("_")):
        paired_path = signal_dir / "paired_checks.json"
        if not paired_path.exists():
            continue
        try:
            payload = json.loads(paired_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        tests = payload.get("moderation_tests") if isinstance(payload, dict) else None
        if isinstance(tests, list):
            for test in tests:
                if not isinstance(test, dict):
                    continue
                moderation_total += 1
                mp = test.get("moderation_pass")
                gap = _safe_float(test.get("moderation_gap"))
                if mp is None and gap is None:
                    continue
                moderation_covered += 1
                if isinstance(mp, bool):
                    moderation_passes.append(1.0 if mp else 0.0)
                if gap is not None:
                    moderation_gaps.append(gap)

    out: dict[str, float] = {}
    if moderation_total:
        out["moderation_coverage"] = float(moderation_covered / moderation_total)
    if moderation_passes:
        out["moderation_pass_rate_mean"] = float(sum(moderation_passes) / len(moderation_passes))
    if moderation_gaps:
        out["moderation_gap_mean"] = float(sum(moderation_gaps) / len(moderation_gaps))
    return out


def _collect_signal_level_metrics(*, canonical_model_dir: Path, strategy: str) -> list[dict[str, Any]]:
    """Collect per-signal summary metrics for a strategy."""

    base = canonical_model_dir / strategy / "signals"
    if not base.exists():
        return []

    rows: list[dict[str, Any]] = []

    for signal_dir in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("_")):
        signal_id = signal_dir.name

        paired_path = signal_dir / "paired_checks.json"
        moderation_passes: list[float] = []
        moderation_gaps: list[float] = []
        moderation_covered = 0
        moderation_total = 0

        if paired_path.exists():
            try:
                payload = json.loads(paired_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None

            if isinstance(payload, dict):
                tests = payload.get("moderation_tests")
                if isinstance(tests, list):
                    for test in tests:
                        if not isinstance(test, dict):
                            continue
                        moderation_total += 1
                        mp = test.get("moderation_pass")
                        gap = _safe_float(test.get("moderation_gap"))
                        if mp is None and gap is None:
                            continue
                        moderation_covered += 1
                        if isinstance(mp, bool):
                            moderation_passes.append(1.0 if mp else 0.0)
                        if gap is not None:
                            moderation_gaps.append(gap)

        moderation_coverage = float(moderation_covered / moderation_total) if moderation_total else None
        moderation_pass_rate = float(sum(moderation_passes) / len(moderation_passes)) if moderation_passes else None
        moderation_gap_mean = float(sum(moderation_gaps) / len(moderation_gaps)) if moderation_gaps else None

        summary_path = signal_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        row: dict[str, Any] = {
            "strategy": strategy,
            "signal_id": signal_id,
            "moderation_coverage": moderation_coverage,
            "moderation_pass_rate_mean": moderation_pass_rate,
            "moderation_gap_mean": moderation_gap_mean,
        }

        for key, value in summary.items():
            if key in {"strategy", "signal_id"}:
                continue
            if isinstance(value, dict | list):
                continue
            coerced = _coerce_numeric_metric(value)
            if coerced is not None:
                row[key] = coerced

        rows.append(row)

    return rows


def _write_model_strategy_summary(*, output_root: Path, model: str, canonical_model_dir: Path) -> Path | None:
    overall = canonical_model_dir / "overall_summary.json"
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
    samples_path = canonical_model_dir / "samples.jsonl"
    parse_rates: dict[str, dict[str, float]] = {}
    if samples_path.exists():
        parse_rates = _parse_rates_by_strategy(samples_path)

    for entry in strategies:
        if not isinstance(entry, dict):
            continue
        strategy = str(entry.get("strategy") or "data_only")
        row: dict[str, Any] = {"model": model, "strategy": strategy}

        # Pull all scalar numeric strategy-level metrics.
        for key, value in entry.items():
            if key == "strategy":
                continue
            if isinstance(value, dict | list):
                continue
            coerced = _coerce_numeric_metric(value)
            if coerced is not None:
                row[key] = coerced

        # Parsing/structuring success diagnostics (from raw samples).
        if parse_rates:
            row.update(
                parse_rates.get(
                    strategy,
                    {
                        "parse_n_rows": 0.0,
                        "parse_n_samples": 0.0,
                        "parse_n_parsed": 0.0,
                        "parse_rate": float("nan"),
                        "n_samples": 0.0,
                        "n_parsed": 0.0,
                        "parsed_rate": float("nan"),
                    },
                )
            )

        # Moderation diagnostics.
        row.update(_collect_moderation_metrics(canonical_model_dir, strategy=strategy))

        rows.append(row)

    out_dir = output_root / "summary_metrics" / model
    out_path = out_dir / "summary_metrics.csv"
    write_summary_metrics(rows=rows, out_path=out_path)
    return out_path


def _write_model_signal_summary(*, output_root: Path, model: str, canonical_model_dir: Path) -> Path | None:
    strategies = [p.name for p in canonical_model_dir.iterdir() if p.is_dir() and not p.name.startswith("_")]
    if not strategies:
        return None

    rows: list[dict[str, Any]] = []
    for strategy in sorted(strategies):
        for row in _collect_signal_level_metrics(canonical_model_dir=canonical_model_dir, strategy=strategy):
            row = {"model": model, **row}
            rows.append(row)

    if not rows:
        return None

    out_dir = output_root / "summary_metrics" / model
    out_path = out_dir / "summary_metrics_by_signal.csv"
    write_summary_metrics(rows=rows, out_path=out_path)
    return out_path


def run_cli(argv: list[str] | None = None) -> None:
    parser = cli.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="evaluation/output/belief_update_validation",
        help="Belief-update output root (contains _runs/, canonical/, summary_metrics/).",
    )
    parser.add_argument(
        "--spec",
        default="evaluation/specs/belief_update_targets.json",
        help="Belief-update spec JSON (signals + expectations).",
    )

    parser.add_argument("--migrate-existing", action="store_true", help="Migrate legacy timestamp dirs into _runs/.")
    parser.add_argument("--dry-run", action="store_true", help="With --migrate-existing, only report actions.")

    parser.add_argument(
        "--rebuild-canonical",
        action="store_true",
        help="Rebuild canonical/<model>/samples.jsonl from _runs before computing metrics.",
    )
    parser.add_argument(
        "--clear-summary-metrics",
        action="store_true",
        help="Delete summary_metrics/ before writing new artifacts.",
    )
    parser.add_argument(
        "--clear-canonical",
        action="store_true",
        help="When --rebuild-canonical, delete canonical/<model> folders before rebuilding.",
    )
    parser.add_argument("--clean-spec", default="preprocess/specs/5_clean_transformed_questions.json")
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--eps-small", type=float, default=0.02)
    parser.add_argument("--eps-map", default=None)
    parser.add_argument("--moderation-quantile", type=float, default=0.25)
    parser.add_argument("--moderation-by-anchor", action="store_true")
    parser.add_argument("--epsilon-ref", type=float, default=0.03)
    parser.add_argument("--epsilon-move-mult", type=float, default=3.0)
    parser.add_argument("--alpha-secondary", type=float, default=0.33)
    parser.add_argument("--top-n-primary", type=int, default=1)
    parser.add_argument("--denom", type=float, default=1.0)

    args = parser.parse_args(argv)
    output_root = Path(args.output_root)

    if args.migrate_existing:
        migrations = migrate_legacy_prompt_tournament_outputs(output_root=output_root, dry_run=bool(args.dry_run))
        print(f"migrations={len(migrations)}")
        for m in migrations[:10]:
            print(f"- {m['run_id']} -> model={m.get('model_norm')} backend={m.get('backend')}")
        if args.dry_run:
            return

    if args.rebuild_canonical:
        if args.clear_canonical:
            canonical_root = output_root / "canonical"
            if canonical_root.exists():
                shutil.rmtree(canonical_root)
        rebuild_canonical_from_runs(output_root=output_root, clear=bool(args.clear_canonical))

    if args.clear_summary_metrics:
        clear_summary_metrics(output_root=output_root)

    canonical_root = output_root / "canonical"
    if not canonical_root.exists():
        print(f"Missing canonical folder: {canonical_root}")
        print("Run with --rebuild-canonical to create canonical/<model>/samples.jsonl first.")
        return

    summary_paths: list[Path] = []
    signal_summary_paths: list[Path] = []

    for model_dir in sorted(p for p in canonical_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        model = model_dir.name
        model_norm = normalize_model_slug(model, reasoning_effort=None)
        if model_norm != model:
            # Skip stale provider-prefixed dirs that should be merged into the normalized slug.
            continue
        samples_path = model_dir / "samples.jsonl"
        if not samples_path.exists():
            continue

        update_args = [
            "--in",
            str(samples_path),
            "--spec",
            str(Path(args.spec)),
            "--outdir",
            str(model_dir),
            "--clean-spec",
            str(Path(args.clean_spec)),
            "--eps",
            str(float(args.eps)),
            "--eps-small",
            str(float(args.eps_small)),
            "--moderation-quantile",
            str(float(args.moderation_quantile)),
            "--epsilon-ref",
            str(float(args.epsilon_ref)),
            "--epsilon-move-mult",
            str(float(args.epsilon_move_mult)),
            "--alpha-secondary",
            str(float(args.alpha_secondary)),
            "--top-n-primary",
            str(int(args.top_n_primary)),
            "--denom",
            str(float(args.denom)),
        ]
        if args.eps_map:
            update_args.extend(["--eps-map", str(args.eps_map)])
        if args.moderation_by_anchor:
            update_args.append("--moderation-by-anchor")
        update_alignment(update_args)

        # 2) Write summary_metrics.csv for this model.
        out = _write_model_strategy_summary(output_root=output_root, model=model, canonical_model_dir=model_dir)
        if out is not None:
            summary_paths.append(out)
            print(f"Wrote {out}")

        signal_out = _write_model_signal_summary(output_root=output_root, model=model, canonical_model_dir=model_dir)
        if signal_out is not None:
            signal_summary_paths.append(signal_out)
            print(f"Wrote {signal_out}")

    # 3) Write global summary.
    all_rows: list[pd.DataFrame] = []
    for path in summary_paths:
        try:
            all_rows.append(pd.read_csv(path))
        except (OSError, ValueError, pd.errors.ParserError):
            continue
    if all_rows:
        df = pd.concat(all_rows, ignore_index=True)
        out = output_root / "summary_metrics" / "all_models_strategies.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Wrote {out}")

    all_signal_rows: list[pd.DataFrame] = []
    for path in signal_summary_paths:
        try:
            all_signal_rows.append(pd.read_csv(path))
        except (OSError, ValueError, pd.errors.ParserError):
            continue
    if all_signal_rows:
        df = pd.concat(all_signal_rows, ignore_index=True)
        out = output_root / "summary_metrics" / "all_models_signals.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Wrote {out}")
