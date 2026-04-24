from __future__ import annotations

from pathlib import Path

import typer

from beam_abm.decision_function.missingness import DEFAULT_MISSINGNESS_THRESHOLD
from beam_abm.decision_function.taxonomy import DEFAULT_COLUMN_TYPES_PATH, DEFAULT_DERIVATIONS_PATH
from beam_abm.decision_function.workflows.backbone_models import (
    run_backbone_models_block_importance_workflow,
    run_backbone_models_fit_workflow,
    run_backbone_models_workflow,
)
from beam_abm.settings import CSV_FILE_CLEAN

app = typer.Typer(help="Structural backbone modeling and block importance.")


@app.command("fit")
def fit_cmd(
    input_csv: Path = typer.Option(CSV_FILE_CLEAN, "--input", help="Input CLEAN CSV"),
    column_types_path: Path = typer.Option(DEFAULT_COLUMN_TYPES_PATH, "--column-types", help="Column type TSV"),
    derivations_path: Path | None = typer.Option(
        DEFAULT_DERIVATIONS_PATH, "--derivations", help="Derivations TSV to drop redundant predictors"
    ),
    outdir: Path = typer.Option(Path("empirical/output/modeling"), "--outdir", help="Output directory"),
    models: str | None = typer.Option(None, "--models", help="Comma list of model names to run (defaults to all)"),
    weight_column: str | None = typer.Option(None, "--weight-column", help="Optional survey weight column"),
    standardize: bool = typer.Option(True, "--standardize/--no-standardize", help="Standardise numeric predictors"),
    missingness_threshold: float = typer.Option(
        DEFAULT_MISSINGNESS_THRESHOLD,
        "--missingness-threshold",
        help="Max missingness fraction for predictor imputation + missing indicator.",
    ),
    max_workers: int = typer.Option(6, "--max-workers", min=0, help="Worker threads (0 = auto)"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    run_backbone_models_fit_workflow(
        input_csv=input_csv,
        column_types_path=column_types_path,
        derivations_path=derivations_path,
        outdir=outdir,
        models=models,
        weight_column=weight_column,
        standardize=standardize,
        missingness_threshold=missingness_threshold,
        max_workers=max_workers,
        log_level=log_level,
    )


@app.command("block-importance")
def block_importance(
    input_csv: Path = typer.Option(CSV_FILE_CLEAN, "--input", help="Input CLEAN CSV"),
    column_types_path: Path = typer.Option(DEFAULT_COLUMN_TYPES_PATH, "--column-types", help="Column type TSV"),
    derivations_path: Path | None = typer.Option(
        DEFAULT_DERIVATIONS_PATH, "--derivations", help="Derivations TSV to drop redundant predictors"
    ),
    outdir: Path = typer.Option(Path("empirical/output/modeling"), "--outdir", help="Output directory"),
    models: str | None = typer.Option(None, "--models", help="Comma list of model names to run (defaults to all)"),
    baseline_blocks: str | None = typer.Option(
        "controls_demography_SES,health_history_engine", "--baseline-blocks", help="Comma list for LOBI S0"
    ),
    weight_column: str | None = typer.Option(None, "--weight-column", help="Optional survey weight column"),
    standardize: bool = typer.Option(True, "--standardize/--no-standardize", help="Standardise numeric predictors"),
    missingness_threshold: float = typer.Option(
        DEFAULT_MISSINGNESS_THRESHOLD,
        "--missingness-threshold",
        help="Max missingness fraction for predictor imputation + missing indicator.",
    ),
    max_workers: int = typer.Option(6, "--max-workers", min=0, help="Worker threads (0 = auto)"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    run_backbone_models_block_importance_workflow(
        input_csv=input_csv,
        column_types_path=column_types_path,
        derivations_path=derivations_path,
        outdir=outdir,
        models=models,
        baseline_blocks=baseline_blocks,
        weight_column=weight_column,
        standardize=standardize,
        missingness_threshold=missingness_threshold,
        max_workers=max_workers,
        log_level=log_level,
    )


@app.command("run")
def run_all(
    input_csv: Path = typer.Option(CSV_FILE_CLEAN, "--input", help="Input CLEAN CSV"),
    column_types_path: Path = typer.Option(DEFAULT_COLUMN_TYPES_PATH, "--column-types", help="Column type TSV"),
    derivations_path: Path | None = typer.Option(
        DEFAULT_DERIVATIONS_PATH, "--derivations", help="Derivations TSV to drop redundant predictors"
    ),
    outdir: Path = typer.Option(Path("empirical/output/modeling"), "--outdir", help="Output directory for fits"),
    models: str | None = typer.Option(None, "--models", help="Comma list of model names to run (defaults to all)"),
    baseline_blocks: str | None = typer.Option(
        "controls_demography_SES,health_history_engine", "--baseline-blocks", help="Comma list for LOBI S0"
    ),
    weight_column: str | None = typer.Option(None, "--weight-column", help="Optional survey weight column"),
    standardize: bool = typer.Option(True, "--standardize/--no-standardize", help="Standardise numeric predictors"),
    missingness_threshold: float = typer.Option(
        DEFAULT_MISSINGNESS_THRESHOLD,
        "--missingness-threshold",
        help="Max missingness fraction for predictor imputation + missing indicator.",
    ),
    max_workers: int = typer.Option(6, "--max-workers", min=0, help="Worker threads (0 = auto)"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    run_backbone_models_workflow(
        input_csv=input_csv,
        column_types_path=column_types_path,
        derivations_path=derivations_path,
        outdir=outdir,
        models=models,
        baseline_blocks=baseline_blocks,
        weight_column=weight_column,
        standardize=standardize,
        missingness_threshold=missingness_threshold,
        max_workers=max_workers,
        log_level=log_level,
    )


if __name__ == "__main__":
    app()
