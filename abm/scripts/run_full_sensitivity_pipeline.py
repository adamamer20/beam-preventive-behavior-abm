#!/usr/bin/env python3
"""Run the full ABM sensitivity pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from beam_abm.abm.workflows import (
    DEFAULT_FLAGSHIP_SCENARIOS,
    DEFAULT_OUTPUT_METRICS,
    parse_csv_tuple,
    run_full_sensitivity_pipeline_workflow,
)


def _parse_csv_list(value: str) -> tuple[str, ...]:
    try:
        return parse_csv_tuple(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full ABM sensitivity pipeline (Morris → shortlist → LHC → surrogate → Sobol)."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("abm/output/full_sensitivity_pipeline"))
    parser.add_argument("--n-agents", type=int, default=2_000)
    parser.add_argument("--n-steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-offsets", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--morris-trajectories", type=int, default=8)
    parser.add_argument("--morris-levels", type=int, default=6)
    parser.add_argument("--lhc-samples", type=int, default=120)
    parser.add_argument("--sobol-base-samples", type=int, default=4096)
    parser.add_argument("--shortlist-target", type=int, default=8)
    parser.add_argument("--shortlist-max", type=int, default=10)
    parser.add_argument("--sigma-quantile", type=float, default=0.8)
    parser.add_argument("--surrogate-min-r2", type=float, default=0.30)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Sensitivity simulation workers (0=auto, 1=sequential).",
    )
    parser.add_argument(
        "--outputs",
        type=_parse_csv_list,
        default=DEFAULT_OUTPUT_METRICS,
        help="Comma-separated output metrics.",
    )
    parser.add_argument(
        "--flagship-scenarios",
        type=_parse_csv_list,
        default=DEFAULT_FLAGSHIP_SCENARIOS,
        help="Comma-separated flagship scenarios for LHC/surrogate/Sobol stages.",
    )
    parser.add_argument(
        "--effect-metric",
        choices=("final", "auc"),
        default="auc",
        help="Metric used for sensitivity deltas and downstream selection.",
    )
    parser.add_argument(
        "--persist-run-data",
        dest="persist_run_data",
        action="store_true",
        default=True,
        help="Persist full per-run artefacts (trajectories, diagnostics, populations).",
    )
    parser.add_argument(
        "--no-persist-run-data",
        dest="persist_run_data",
        action="store_false",
        help="Disable per-run artefact persistence.",
    )
    parser.add_argument(
        "--resume-from-morris-raw",
        dest="resume_from_morris_raw",
        action="store_true",
        default=False,
        help="Reuse existing morris/design.parquet and morris/runs_raw.parquet.",
    )
    return parser


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    args = _build_parser().parse_args()
    run_full_sensitivity_pipeline_workflow(
        output_dir=args.output_dir,
        n_agents=args.n_agents,
        n_steps=args.n_steps,
        seed=args.seed,
        seed_offsets=tuple(args.seed_offsets),
        morris_trajectories=args.morris_trajectories,
        morris_levels=args.morris_levels,
        lhc_samples=args.lhc_samples,
        sobol_base_samples=args.sobol_base_samples,
        shortlist_target=args.shortlist_target,
        shortlist_max=args.shortlist_max,
        sigma_quantile=args.sigma_quantile,
        surrogate_min_r2=args.surrogate_min_r2,
        max_workers=args.max_workers,
        outputs=tuple(args.outputs),
        flagship_scenarios=tuple(args.flagship_scenarios),
        effect_metric=args.effect_metric,
        persist_run_data=args.persist_run_data,
        resume_from_morris_raw=args.resume_from_morris_raw,
    )


if __name__ == "__main__":
    main()
