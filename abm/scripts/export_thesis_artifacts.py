from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import polars as pl


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _mean_ci_columns(col: str) -> list[pl.Expr]:
    n_col = f"n_{col}"
    mean_col = col
    sd_col = f"{col}_sd"
    return [
        pl.col(col).count().alias(n_col),
        pl.col(col).mean().alias(mean_col),
        pl.col(col).std().alias(sd_col),
    ]


def _export_run_context_and_auc_table(repo_root: Path, abm_out: Path, artifacts_root: Path) -> None:
    runs_root = abm_out / "runs"
    if not runs_root.exists():
        raise FileNotFoundError(f"Missing required ABM runs root: {runs_root}")

    # Reuse chapter-5 run parsing to guarantee parity with thesis calculations.
    sys.path.insert(0, str((repo_root / "thesis").resolve()))
    from utils.ch05_plot_tables import load_abm_context, matched_effects

    context = load_abm_context(runs_root=runs_root)
    run_ctx_root = artifacts_root / "run_context"
    run_ctx_root.mkdir(parents=True, exist_ok=True)
    context.run_index.write_parquet(run_ctx_root / "run_index.parquet")
    context.series.write_parquet(run_ctx_root / "series.parquet")
    context.grouped_outcomes.write_parquet(run_ctx_root / "grouped_outcomes.parquet")
    context.grouped_mediators.write_parquet(run_ctx_root / "grouped_mediators.parquet")

    effects = matched_effects(context)
    effects.write_parquet(run_ctx_root / "matched_effects.parquet")

    auc_table = (
        effects.group_by("scenario")
        .agg(
            *_mean_ci_columns("delta_vax_auc"),
            *_mean_ci_columns("delta_mask_auc"),
        )
        .with_columns(
            pl.when(pl.col("n_delta_vax_auc") > 1)
            .then(1.96 * pl.col("delta_vax_auc_sd") / pl.col("n_delta_vax_auc").cast(pl.Float64).sqrt())
            .otherwise(0.0)
            .alias("delta_vax_auc_ci_half"),
            pl.when(pl.col("n_delta_mask_auc") > 1)
            .then(1.96 * pl.col("delta_mask_auc_sd") / pl.col("n_delta_mask_auc").cast(pl.Float64).sqrt())
            .otherwise(0.0)
            .alias("delta_mask_auc_ci_half"),
        )
        .with_columns(
            (pl.col("delta_vax_auc") - pl.col("delta_vax_auc_ci_half")).alias("delta_vax_auc_ci_low"),
            (pl.col("delta_vax_auc") + pl.col("delta_vax_auc_ci_half")).alias("delta_vax_auc_ci_high"),
            (pl.col("delta_mask_auc") - pl.col("delta_mask_auc_ci_half")).alias("delta_mask_auc_ci_low"),
            (pl.col("delta_mask_auc") + pl.col("delta_mask_auc_ci_half")).alias("delta_mask_auc_ci_high"),
        )
        .select(
            "scenario",
            "n_delta_vax_auc",
            "delta_vax_auc",
            "delta_vax_auc_ci_low",
            "delta_vax_auc_ci_high",
            "n_delta_mask_auc",
            "delta_mask_auc",
            "delta_mask_auc_ci_low",
            "delta_mask_auc_ci_high",
        )
        .rename(
            {
                "n_delta_vax_auc": "n_vax_auc",
                "n_delta_mask_auc": "n_mask_auc",
            }
        )
        .sort("scenario")
    )
    auc_table.write_csv(artifacts_root / "scenario_delta_auc_table.csv")


def export_thesis_artifacts(repo_root: Path) -> None:
    abm_out = repo_root / "abm" / "output"
    artifacts_root = repo_root / "thesis" / "artifacts" / "abm"

    required_map: dict[Path, Path] = {
        abm_out / "report_summary_canonical_effects.csv": artifacts_root / "report_summary_canonical_effects.csv",
    }
    missing = [str(src) for src in required_map if not src.exists()]
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing required ABM artifact inputs:\n{joined}")
    for src, dst in required_map.items():
        _copy_file(src, dst)

    optional_map: dict[Path, Path] = {
        abm_out / "report_summary.csv": artifacts_root / "report_summary.csv",
        abm_out / "leaderboard_dynamics_summary.csv": artifacts_root / "leaderboard_dynamics_summary.csv",
        abm_out / "scenario_delta_auc_table.csv": artifacts_root / "scenario_delta_auc_table_runtime.csv",
        abm_out / "full_sensitivity_pipeline" / "sobol" / "indices.csv": artifacts_root
        / "full_sensitivity_pipeline"
        / "sobol"
        / "indices.csv",
        abm_out / "full_sensitivity_pipeline" / "morris" / "indices_per_scenario.csv": artifacts_root
        / "full_sensitivity_pipeline"
        / "morris"
        / "indices_per_scenario.csv",
        abm_out / "full_sensitivity_pipeline" / "lhc" / "delta.parquet": artifacts_root
        / "full_sensitivity_pipeline"
        / "lhc"
        / "delta.parquet",
    }
    for src, dst in optional_map.items():
        if src.exists():
            _copy_file(src, dst)

    _export_run_context_and_auc_table(repo_root, abm_out, artifacts_root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export compact ABM thesis artifacts to thesis/artifacts/abm.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root containing abm/output and thesis/artifacts.",
    )
    args = parser.parse_args()
    export_thesis_artifacts(args.repo_root.resolve())
    print("Exported ABM thesis artifacts.")


if __name__ == "__main__":
    main()
