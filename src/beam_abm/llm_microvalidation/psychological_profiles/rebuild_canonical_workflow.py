"""Rebuild canonical belief-update validation leaves from existing `_runs/`.

Canonical contract:
evaluation/output/belief_update_validation/
  _runs/<run_id>/_batch/<model>/samples.jsonl
  canonical/<model>/samples.jsonl
  canonical/<model>/canonicalize_stats.json
"""

from __future__ import annotations

from pathlib import Path

from beam_abm.cli import parser_compat as cli
from beam_abm.llm_microvalidation.psychological_profiles.canonicalize import rebuild_canonical_from_runs


def run_cli(argv: list[str] | None = None) -> None:
    parser = cli.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="evaluation/output/belief_update_validation",
        help="Belief-update output root (contains _runs/ and canonical/).",
    )
    parser.add_argument(
        "--clear-canonical",
        action="store_true",
        help="Delete canonical/<model> folders before rebuilding.",
    )

    args = parser.parse_args(argv)
    output_root = Path(args.output_root)

    stats_by_model = rebuild_canonical_from_runs(output_root=output_root, clear=bool(args.clear_canonical))
    for model, stats in sorted(stats_by_model.items()):
        print(
            f"{model}: runs={stats.runs_seen} rows {stats.rows_in}->{stats.rows_out} "
            f"samples {stats.samples_in}->{stats.samples_out}"
        )
