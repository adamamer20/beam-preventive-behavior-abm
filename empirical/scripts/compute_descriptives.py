from __future__ import annotations

import os
from pathlib import Path

import typer

from beam_abm.empirical.taxonomy import DEFAULT_COLUMN_TYPES_PATH
from beam_abm.empirical.workflows.descriptives import run_descriptives_workflow
from beam_abm.settings import CSV_FILE_CLEAN


def main(
    input_path: Path = typer.Option(CSV_FILE_CLEAN, "--input", help="Input CSV path"),
    outdir: Path = typer.Option(Path("empirical/output/descriptives"), "--outdir", help="Output directory for tables"),
    column_types_path: Path = typer.Option(
        DEFAULT_COLUMN_TYPES_PATH,
        "--column-types",
        help="Path to TSV file describing column types",
    ),
    categorical_overrides: str = typer.Option(
        "",
        "--categoricals",
        help="Comma-separated list of columns to force as categorical",
    ),
    spec_path: Path = typer.Option(
        Path("preprocess/specs/5_clean_transformed_questions.json"),
        "--spec-path",
        help="Preprocessing spec JSON for derived index definitions",
    ),
    compute_alpha: bool = typer.Option(
        True,
        "--compute-alpha/--skip-alpha",
        help="Compute Cronbach's alpha for derived indexes",
    ),
    bool_groups: str = typer.Option(
        "covax_reason__,covax_no_child__",
        "--bool-groups",
        help="Comma-separated list of column prefixes for boolean group summaries",
    ),
    log_level: str = typer.Option(
        os.getenv("LOG_LEVEL", "INFO"),
        "--log-level",
        help="Log level (DEBUG, INFO, WARNING, ERROR)",
    ),
) -> None:
    run_descriptives_workflow(
        input_path=input_path,
        outdir=outdir,
        column_types_path=column_types_path,
        categorical_overrides=categorical_overrides,
        spec_path=spec_path,
        compute_alpha=compute_alpha,
        bool_groups=bool_groups,
        log_level=log_level,
    )


if __name__ == "__main__":
    typer.run(main)
