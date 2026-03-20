"""CLI for running exploratory inferential statistics over survey data."""

from __future__ import annotations

import os
from pathlib import Path

import typer

from beam_abm.decision_function.taxonomy import DEFAULT_COLUMN_TYPES_PATH, DEFAULT_DERIVATIONS_PATH
from beam_abm.decision_function.workflows.inferential_statistics import run_inferential_statistics_workflow
from beam_abm.settings import CSV_FILE_CLEAN


def main(
    input_csv: Path = typer.Option(CSV_FILE_CLEAN, "--input", help="Input CSV for analysis"),
    column_types_path: Path = typer.Option(
        DEFAULT_COLUMN_TYPES_PATH,
        "--column-types",
        help="Path to TSV describing column types",
    ),
    outdir: Path = typer.Option(Path("empirical/output/inferential"), "--outdir", help="Output directory"),
    categorical_overrides: str = typer.Option(
        "",
        "--categoricals",
        help="Comma-separated list of columns to force as categorical",
    ),
    min_nominal_levels: int = typer.Option(
        3,
        "--min-nominal-levels",
        help="Minimum distinct category count to treat as nominal",
        min=2,
    ),
    workers: int | None = typer.Option(
        None,
        "--workers",
        help="Maximum worker threads for pairwise tests",
        min=1,
    ),
    max_missing_fraction: float | None = typer.Option(
        0.9,
        "--max-missing-frac",
        help="Skip variables with missing fraction above this threshold (0-1)",
        min=0.0,
        max=1.0,
    ),
    highlights: bool = typer.Option(True, "--highlights/--no-highlights", help="Write filtered highlight tables"),
    highlight_alpha: float = typer.Option(0.05, "--highlight-alpha", help="FDR q-value threshold", min=0.0, max=1.0),
    highlight_min_effect: str = typer.Option(
        "medium",
        "--highlight-min-effect",
        help="Minimum effect size label to keep (negligible, small, medium, large)",
    ),
    highlight_min_spearman_n: int = typer.Option(
        50, "--highlight-min-spearman-n", help="Minimum pairwise N for Spearman", min=0
    ),
    highlight_min_categorical_n: int = typer.Option(
        50, "--highlight-min-categorical-n", help="Minimum N for categorical pairs", min=0
    ),
    highlight_min_binary_group_n: int = typer.Option(
        20, "--highlight-min-binary-group-n", help="Minimum group size for binary vs numeric", min=0
    ),
    highlight_min_nominal_group_n: int = typer.Option(
        15, "--highlight-min-nominal-group-n", help="Minimum category size for nominal vs numeric", min=0
    ),
    derivations_tsv: Path | None = typer.Option(
        DEFAULT_DERIVATIONS_PATH if DEFAULT_DERIVATIONS_PATH.exists() else None,
        "--derivations",
        help="Path to derivations TSV to skip functionally linked pairs",
    ),
    log_level: str = typer.Option(
        os.getenv("LOG_LEVEL", "INFO"),
        "--log-level",
        help="Log level",
    ),
) -> None:
    run_inferential_statistics_workflow(
        input_csv=input_csv,
        column_types_path=column_types_path,
        outdir=outdir,
        categorical_overrides=categorical_overrides,
        min_nominal_levels=min_nominal_levels,
        workers=workers,
        max_missing_fraction=max_missing_fraction,
        highlights=highlights,
        highlight_alpha=highlight_alpha,
        highlight_min_effect=highlight_min_effect,
        highlight_min_spearman_n=highlight_min_spearman_n,
        highlight_min_categorical_n=highlight_min_categorical_n,
        highlight_min_binary_group_n=highlight_min_binary_group_n,
        highlight_min_nominal_group_n=highlight_min_nominal_group_n,
        derivations_tsv=derivations_tsv,
        log_level=log_level,
    )


if __name__ == "__main__":
    typer.run(main)
