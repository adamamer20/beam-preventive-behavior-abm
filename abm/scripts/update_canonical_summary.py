"""Thin wrapper for rebuilding canonical ABM summary outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from beam_abm.abm.export import rebuild_canonical_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("abm/output/runs"),
        help="Directory with per-run outputs under runs/<run_id>/trajectories/outcomes.parquet.",
    )
    parser.add_argument(
        "--scenario-defs",
        type=Path,
        default=Path("src/beam_abm/abm/scenario_defs.py"),
        help="Path to scenario_defs.py for canonical scenario keys.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("abm/output/report_summary_canonical_effects.csv"),
        help="Output canonical summary CSV path.",
    )
    parser.add_argument(
        "--burn-in-tick",
        type=int,
        default=12,
        help="Post-burn start tick used for baseline-relative metrics (default: 12).",
    )
    args = parser.parse_args()
    output_path = rebuild_canonical_summary(
        runs_dir=args.runs_dir,
        scenario_defs=args.scenario_defs,
        output=args.output,
        burn_in_tick=args.burn_in_tick,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
