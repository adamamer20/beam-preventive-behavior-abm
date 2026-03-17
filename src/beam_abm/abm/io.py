"""I/O utilities for reading inputs and writing simulation artefacts.

Centralises all filesystem access so the ABM core never does raw
``open()`` calls.

Examples
--------
>>> from beam_abm.abm.io import load_survey_data
>>> # df = load_survey_data(Path("preprocess/output/clean_processed_survey.csv"))
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

__all__ = [
    "load_survey_data",
    "save_run_outputs",
    "save_trajectory",
    "save_benchmark_report",
    "write_json",
    "ensure_output_dirs",
    "create_diagnostics_run_dirs",
    "DiagnosticsRunPaths",
    "RunPaths",
]


# =========================================================================
# 1.  Output directory structure
# =========================================================================


class RunPaths:
    """Organise output paths for a single simulation run.

    Parameters
    ----------
    output_dir : Path
        Base output directory.
    run_id : str
        Unique identifier for this run.
    """

    def __init__(self, output_dir: Path, run_id: str) -> None:
        self.base = output_dir / "runs" / run_id
        self.trajectories = self.base / "trajectories"
        self.diagnostics = self.base / "diagnostics"
        self.benchmarks = self.base / "benchmarks"
        self.populations = self.base / "populations"

    def ensure(self) -> None:
        """Create all directories if not present."""
        for d in (
            self.base,
            self.trajectories,
            self.diagnostics,
            self.benchmarks,
            self.populations,
        ):
            d.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True, frozen=True)
class DiagnosticsRunPaths:
    """Directory contract for one diagnostics run."""

    base: Path
    summary: Path
    suites: dict[str, Path]


def ensure_output_dirs(output_dir: Path) -> None:
    """Create the top-level output directory structure.

    Parameters
    ----------
    output_dir : Path
        Base output directory for all ABM artefacts.
    """
    for sub in ("runs", "diagnostics", "benchmarks"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)


def create_diagnostics_run_dirs(
    output_root: Path,
    run_id: str,
    suites: list[str],
) -> DiagnosticsRunPaths:
    """Create diagnostics run directory and suite subdirectories."""
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=False, exist_ok=False)

    suite_dirs: dict[str, Path] = {}
    for suite in suites:
        suite_dir = run_dir / suite
        suite_dir.mkdir(parents=True, exist_ok=True)
        suite_dirs[suite] = suite_dir

    return DiagnosticsRunPaths(
        base=run_dir,
        summary=run_dir / "summary.json",
        suites=suite_dirs,
    )


# =========================================================================
# 2.  Input loading
# =========================================================================


def _read_survey_csv(path: Path) -> pl.DataFrame:
    """Read survey CSV with a robust fallback for malformed rows/chunks."""
    csv_kwargs: dict[str, Any] = {"infer_schema_length": 5000}
    try:
        return pl.read_csv(path, **csv_kwargs)
    except pl.exceptions.ShapeError as exc:
        logger.warning(
            "CSV schema mismatch while reading {p}: {e}. Retrying with lenient parser options.",
            p=path,
            e=str(exc),
        )
        return pl.read_csv(
            path,
            **csv_kwargs,
            ignore_errors=True,
            truncate_ragged_lines=True,
        )


def _sanitize_csv_column_names(df: pl.DataFrame) -> pl.DataFrame:
    """Remove BOM/NUL artifacts from CSV headers when present."""
    rename_map: dict[str, str] = {}
    for col in df.columns:
        cleaned = col.lstrip("\ufeff").replace("\x00", "")
        if cleaned != col:
            rename_map[col] = cleaned
    if rename_map:
        df = df.rename(rename_map)
    return df


def load_survey_data(
    path: Path,
    countries: list[str] | None = None,
) -> pl.DataFrame:
    """Load the clean survey CSV as a Polars DataFrame.

    Parameters
    ----------
    path : Path
        Path to ``clean_processed_survey.csv``.
    countries : list[str] or None
        If given, filter to these countries only.

    Returns
    -------
    pl.DataFrame
        Survey data with a ``row_id`` column (created if absent).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not path.is_file():
        msg = f"Survey CSV not found: {path}"
        raise FileNotFoundError(msg)

    df = _sanitize_csv_column_names(_read_survey_csv(path))
    # Migration compatibility: keep both health aliases available while model
    # artifacts transition from `health_good` to `health_poor`.
    if "health_poor" not in df.columns and "health_good" in df.columns:
        df = df.with_columns(pl.col("health_good").alias("health_poor"))
    if "health_poor_missing" not in df.columns and "health_good_missing" in df.columns:
        df = df.with_columns(pl.col("health_good_missing").alias("health_poor_missing"))
    if "health_good" not in df.columns and "health_poor" in df.columns:
        df = df.with_columns(pl.col("health_poor").alias("health_good"))
    if "health_good_missing" not in df.columns and "health_poor_missing" in df.columns:
        df = df.with_columns(pl.col("health_poor_missing").alias("health_good_missing"))
    if "row_id" not in df.columns:
        df = df.with_row_index("row_id")

    if countries is not None:
        df = df.filter(pl.col("country").is_in(countries))

    logger.info(
        "Loaded survey data: {n} rows × {c} columns from {p}.",
        n=df.height,
        c=df.width,
        p=path.name,
    )
    return df


# =========================================================================
# 3.  Output saving
# =========================================================================


def save_trajectory(
    trajectory: pl.DataFrame,
    path: Path,
) -> None:
    """Save a trajectory DataFrame to Parquet.

    Parameters
    ----------
    trajectory : pl.DataFrame
        Trajectory data with at minimum ``tick`` and outcome columns.
    path : Path
        Output file path (Parquet).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    trajectory.write_parquet(path)
    logger.debug("Saved trajectory to {p}.", p=path)


def save_run_outputs(
    run_paths: RunPaths,
    trajectory: pl.DataFrame,
    mediator_trajectory: pl.DataFrame | None = None,
    signals_realised: pl.DataFrame | None = None,
    outcomes_by_group: pl.DataFrame | None = None,
    mediators_by_group: pl.DataFrame | None = None,
    saturation_trajectory: pl.DataFrame | None = None,
    incidence_trajectory: pl.DataFrame | None = None,
    distance_trajectory: pl.DataFrame | None = None,
    effects: pl.DataFrame | None = None,
    population_t0: pl.DataFrame | None = None,
    population_final: pl.DataFrame | None = None,
    metadata: object | None = None,
    scenario_metrics: pl.DataFrame | None = None,
    adjacency: pl.DataFrame | None = None,
    dynamics_summary: pl.DataFrame | None = None,
    dynamics_trajectory: pl.DataFrame | None = None,
) -> None:
    """Persist all artefacts for a single simulation run.

    Parameters
    ----------
    run_paths : RunPaths
        Run-specific directory structure.
    trajectory : pl.DataFrame
        Outcome trajectory over ticks.
    mediator_trajectory : pl.DataFrame or None
        Mutable-state mediator trajectories.
    population_t0 : pl.DataFrame or None
        Initial agent population snapshot.
    population_final : pl.DataFrame or None
        Final agent population snapshot.
    metadata : object or None
        Run metadata (typed object or dict).
    """
    run_paths.ensure()

    save_trajectory(
        trajectory,
        run_paths.trajectories / "outcomes.parquet",
    )

    if mediator_trajectory is not None:
        save_trajectory(
            mediator_trajectory,
            run_paths.trajectories / "mediators.parquet",
        )

    if signals_realised is not None:
        save_trajectory(
            signals_realised,
            run_paths.trajectories / "signals_realised.parquet",
        )

    if outcomes_by_group is not None:
        save_trajectory(
            outcomes_by_group,
            run_paths.trajectories / "outcomes_by_group.parquet",
        )

    if mediators_by_group is not None:
        save_trajectory(
            mediators_by_group,
            run_paths.trajectories / "mediators_by_group.parquet",
        )

    if saturation_trajectory is not None:
        save_trajectory(
            saturation_trajectory,
            run_paths.trajectories / "saturation.parquet",
        )

    if incidence_trajectory is not None:
        save_trajectory(
            incidence_trajectory,
            run_paths.trajectories / "incidence.parquet",
        )

    if distance_trajectory is not None:
        save_trajectory(
            distance_trajectory,
            run_paths.trajectories / "distance_trajectory.parquet",
        )

    if effects is not None:
        save_trajectory(
            effects,
            run_paths.trajectories / "effects.parquet",
        )

    if population_t0 is not None:
        save_trajectory(
            population_t0,
            run_paths.populations / "t0.parquet",
        )

    if population_final is not None:
        save_trajectory(
            population_final,
            run_paths.populations / "final.parquet",
        )

    if scenario_metrics is not None:
        save_trajectory(
            scenario_metrics,
            run_paths.diagnostics / "scenario_metrics.parquet",
        )

    if adjacency is not None:
        save_trajectory(
            adjacency,
            run_paths.diagnostics / "adjacency.parquet",
        )

    if dynamics_summary is not None:
        save_trajectory(
            dynamics_summary,
            run_paths.diagnostics / "dynamics_summary.parquet",
        )

    if dynamics_trajectory is not None:
        save_trajectory(
            dynamics_trajectory,
            run_paths.diagnostics / "dynamics_trajectory.parquet",
        )

    if metadata is not None:
        payload = metadata
        if hasattr(metadata, "to_dict"):
            payload = metadata.to_dict()  # type: ignore[assignment]
        elif hasattr(metadata, "model_dump"):
            payload = metadata.model_dump()  # type: ignore[assignment]
        meta_path = run_paths.base / "metadata.json"
        meta_path.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )

    logger.info("Run outputs saved to {d}.", d=run_paths.base)


def save_benchmark_report(
    report: object,
    path: Path,
) -> None:
    """Save a benchmark/validation report as JSON.

    Parameters
    ----------
    report : object
        Benchmark results (typed report or dict).
    path : Path
        Output file path (JSON).
    """
    payload = report
    if hasattr(report, "to_dict"):
        payload = report.to_dict()  # type: ignore[assignment]
    elif hasattr(report, "model_dump"):
        payload = report.model_dump()  # type: ignore[assignment]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Benchmark report saved to {p}.", p=path)


def write_json(payload: object, path: Path) -> None:
    """Write JSON payload to disk using project serialization conventions."""
    normalized = payload
    if hasattr(payload, "to_dict"):
        normalized = payload.to_dict()  # type: ignore[assignment]
    elif hasattr(payload, "model_dump"):
        normalized = payload.model_dump()  # type: ignore[assignment]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, indent=2, default=str), encoding="utf-8")
