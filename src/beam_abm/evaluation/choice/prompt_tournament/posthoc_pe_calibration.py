"""Post-hoc PE calibration for existing phase0 prompt-tournament outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from beam_abm.cli import parser_compat as cli
from beam_abm.evaluation.common.apply_pe_calibration_to_ice import run_cli as apply_pe_calibration_step
from beam_abm.evaluation.common.calibrate_pe_gains import run_cli as pe_calibrate_step
from beam_abm.evaluation.common.compute_alignment import run_cli as align_step
from beam_abm.evaluation.utils import prompt_tournament_utils as pt
from beam_abm.evaluation.utils.alignment_utils import summarize_alignment_rows


def _safe_token(value: str | None) -> str:
    text = str(value or "").strip()
    out = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text).strip("_")
    return out or "unknown"


def _normalize_ref_model(name: str) -> str:
    key = str(name).strip().lower()
    if key in {"xgboost", "xgb"}:
        return "xgb"
    if key in {"linear", "lin"}:
        return "linear"
    return key


def _ref_ice_path(outcome: str, ref_model: str) -> Path:
    ref_key = _normalize_ref_model(ref_model)
    base = Path("empirical/output/anchors/pe/mutable_engines/reduced") / outcome
    if ref_key == "linear":
        return base / "ice_ref.csv"
    return base / f"ice_ref__{ref_key}.csv"


def _iter_strategy_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [p for p in root.iterdir() if p.is_dir()]


def _iter_run_dirs(strategy_dir: Path) -> list[Path]:
    return [p for p in strategy_dir.iterdir() if p.is_dir()]


def _iter_outcome_batches(run_dir: Path) -> list[Path]:
    return [p for p in run_dir.iterdir() if p.is_dir()]


def _resolve_outcomes_from_ice(ice_path: Path) -> list[str]:
    df = pd.read_csv(ice_path, usecols=["outcome"]) if ice_path.exists() else pd.DataFrame()
    if df.empty or "outcome" not in df.columns:
        return []
    outcomes = sorted({str(o) for o in df["outcome"].dropna().tolist() if str(o).strip()})
    return outcomes


def run_cli(argv: list[str] | None = None) -> None:
    parser = cli.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="evaluation/output/prompt_tournament/perturbed",
        help="Root directory that contains phase0 prompt-tournament perturbed outputs.",
    )
    parser.add_argument(
        "--ref-models",
        default="linear,xgb",
        help="Comma-separated reference model names for alignment (e.g., linear,xgb).",
    )
    parser.add_argument("--resume", action="store_true", help="Skip outputs that already exist.")
    parser.add_argument("--min-rows", type=int, default=1, help="Minimum rows per group to fit a gain.")
    parser.add_argument(
        "--fallback-alpha",
        type=float,
        default=1.0,
        help="Fallback alpha when no calibrator is available (default: 1.0)",
    )
    args = parser.parse_args(argv)

    root = Path(args.output_root)
    ref_models = [_normalize_ref_model(m) for m in args.ref_models.split(",") if m.strip()]
    if not ref_models:
        ref_models = ["linear"]

    for strategy_dir in _iter_strategy_dirs(root):
        strategy = strategy_dir.name
        for run_dir in _iter_run_dirs(strategy_dir):
            for outcome_batch_dir in _iter_outcome_batches(run_dir):
                uncal_dir = outcome_batch_dir / "uncalibrated"
                ice_path = uncal_dir / "ice_llm.csv"
                if not ice_path.exists():
                    continue

                outcomes = _resolve_outcomes_from_ice(ice_path)
                if not outcomes:
                    continue

                pe_dir = uncal_dir / "pe_calibrated"
                pe_dir.mkdir(parents=True, exist_ok=True)

                for ref_model in ref_models:
                    ref_key = _normalize_ref_model(ref_model)
                    metrics_suffix = "" if ref_key == "linear" else f"__ref_{ref_key}"

                    ref_paths = [_ref_ice_path(outcome=o, ref_model=ref_model) for o in outcomes]
                    missing = [p for p in ref_paths if not p.exists()]
                    if missing:
                        print(
                            f"Skipping PE calibration for {strategy}::{run_dir.name}::{outcome_batch_dir.name}::{ref_model}: "
                            "missing ref ICE for outcomes."
                        )
                        continue

                    ref_concat = pe_dir / f"ref_ice{metrics_suffix}.csv"
                    if not (args.resume and ref_concat.exists()):
                        ref_frames = [pd.read_csv(p) for p in ref_paths]
                        pd.concat(ref_frames, ignore_index=True).to_csv(ref_concat, index=False)

                    pe_cal_path = pe_dir / f"pe_calibration{metrics_suffix}.json"
                    pe_ice_out = pe_dir / f"ice_llm_pe_calibrated{metrics_suffix}.csv"

                    if not (args.resume and pe_cal_path.exists() and pe_ice_out.exists()):
                        pe_cal_args = pt.split_args(
                            " ".join(
                                [
                                    f"--ref-ice {ref_concat}",
                                    f"--llm-ice {ice_path}",
                                    f"--out {pe_cal_path}",
                                    "--group-cols outcome,target,strategy",
                                    f"--min-rows {int(args.min_rows)}",
                                ]
                            ),
                            strategy=strategy,
                        )
                        pt.run_step(pe_calibrate_step, pe_cal_args)

                        pe_apply_args = pt.split_args(
                            " ".join(
                                [
                                    f"--llm-ice {ice_path}",
                                    f"--calibration {pe_cal_path}",
                                    f"--out {pe_ice_out}",
                                    f"--fallback-alpha {float(args.fallback_alpha)}",
                                ]
                            ),
                            strategy=strategy,
                        )
                        pt.run_step(apply_pe_calibration_step, pe_apply_args)

                    summary_paths_by_ref: dict[str, list[Path]] = {}
                    order_paths_by_ref: dict[str, list[Path]] = {}

                    for outcome in outcomes:
                        ref_ice_outcome = _ref_ice_path(outcome, ref_model)
                        metrics_out = pe_dir / f"_summary_metrics__{_safe_token(outcome)}{metrics_suffix}.csv"
                        order_out = pe_dir / f"_order_effects__{_safe_token(outcome)}{metrics_suffix}.csv"
                        summary_paths_by_ref.setdefault(ref_key, []).append(metrics_out)
                        order_paths_by_ref.setdefault(ref_key, []).append(order_out)
                        if not (args.resume and metrics_out.exists() and order_out.exists()):
                            align_args = pt.split_args(
                                " ".join(
                                    [
                                        f"--ref {ref_ice_outcome}",
                                        f"--llm {pe_ice_out}",
                                        f"--out {metrics_out}",
                                        f"--order-out {order_out}",
                                    ]
                                ),
                                strategy=strategy,
                            )
                            pt.run_step(align_step, align_args)

                    for ref_key_out, summary_paths in summary_paths_by_ref.items():
                        summary_frames = [pd.read_csv(path) for path in summary_paths if path.exists()]
                        if summary_frames:
                            summary_rows = pd.concat(summary_frames, ignore_index=True)
                            if ref_key_out == "linear":
                                summary_out = pe_dir / "summary_metrics.csv"
                            else:
                                summary_out = pe_dir / f"summary_metrics__ref_{ref_key_out}.csv"
                            summarize_alignment_rows(summary_rows).to_csv(summary_out, index=False)

                    if strategy == "paired_profile" and order_paths_by_ref:
                        for ref_key_out, order_paths in order_paths_by_ref.items():
                            order_frames = [pd.read_csv(path) for path in order_paths if path.exists()]
                            if not order_frames:
                                continue
                            order_out = pe_dir / (
                                "order_effects.csv"
                                if ref_key_out == "linear"
                                else f"order_effects__ref_{ref_key_out}.csv"
                            )
                            pd.concat(order_frames, ignore_index=True).to_csv(order_out, index=False)

                    for summary_paths in summary_paths_by_ref.values():
                        for path in summary_paths:
                            path.unlink(missing_ok=True)
                    for order_paths in order_paths_by_ref.values():
                        for path in order_paths:
                            path.unlink(missing_ok=True)
