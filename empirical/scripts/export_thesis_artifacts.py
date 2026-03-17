from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import polars as pl


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_ref_ice_tree(src_root: Path, dst_root: Path) -> int:
    if not src_root.exists():
        return 0
    copied = 0
    for src in sorted(src_root.rglob("ice_ref*.csv")):
        rel = src.relative_to(src_root)
        _copy_file(src, dst_root / rel)
        copied += 1
    return copied


def export_thesis_artifacts(repo_root: Path) -> None:
    empirical_out = repo_root / "empirical" / "output"
    preprocess_out = repo_root / "preprocess" / "output"
    artifacts_root = repo_root / "thesis" / "artifacts" / "empirical"

    required_sources: dict[Path, Path] = {
        empirical_out / "descriptives" / "raw" / "binary_summary.csv": artifacts_root
        / "descriptives"
        / "binary_summary.csv",
        empirical_out / "descriptives" / "raw" / "continuous_summary.csv": artifacts_root
        / "descriptives"
        / "continuous_summary.csv",
        empirical_out / "descriptives" / "raw" / "ordinal_summary.csv": artifacts_root
        / "descriptives"
        / "ordinal_summary.csv",
        empirical_out / "modeling" / "model_coefficients_all.csv": artifacts_root
        / "modeling"
        / "model_coefficients_all.csv",
        empirical_out / "modeling" / "model_summary.csv": artifacts_root / "modeling" / "model_summary.csv",
        empirical_out / "modeling" / "model_lobo_all.csv": artifacts_root / "modeling" / "model_lobo_all.csv",
        empirical_out / "modeling" / "model_plan.json": artifacts_root / "modeling" / "model_plan.json",
        empirical_out / "modeling" / "model_plan_ref_choice.json": artifacts_root
        / "modeling"
        / "model_plan_ref_choice.json",
        empirical_out / "modeling" / "B_FLU" / "B_FLU_vaccinated_2023_2024_ENGINES" / "lobo.csv": artifacts_root
        / "modeling"
        / "B_FLU_vaccinated_2023_2024_ENGINES_lobo.csv",
        empirical_out / "inferential" / "raw" / "spearman.csv": artifacts_root / "decision_function" / "spearman.csv",
    }
    missing_required = [str(path) for path in required_sources if not path.exists()]
    if missing_required:
        joined = "\n".join(f"- {path}" for path in missing_required)
        raise FileNotFoundError(f"Missing required empirical runtime inputs:\n{joined}")

    for src, dst in required_sources.items():
        _copy_file(src, dst)

    optional_sources: dict[Path, Path] = {
        empirical_out / "anchors" / "pe" / "upstream_signals" / "reduced" / "pe_ref_p_rows.parquet": artifacts_root
        / "decision_function"
        / "pe_ref_p_rows.parquet",
        empirical_out
        / "anchors"
        / "pe"
        / "upstream_signals"
        / "reduced"
        / "pe_ref_p_signal_metadata.json": artifacts_root / "decision_function" / "pe_ref_p_signal_metadata.json",
    }
    for src, dst in optional_sources.items():
        if src.exists():
            _copy_file(src, dst)

    reduced_src = empirical_out / "anchors" / "pe" / "mutable_engines" / "reduced"
    full_src = empirical_out / "anchors" / "pe" / "mutable_engines" / "full"
    _copy_ref_ice_tree(
        reduced_src,
        artifacts_root / "decision_function" / "ref_ice" / "mutable_reduced",
    )
    _copy_ref_ice_tree(
        full_src,
        artifacts_root / "decision_function" / "ref_ice" / "mutable_full",
    )

    survey_path = preprocess_out / "clean_processed_survey.csv"
    if survey_path.exists():
        survey_df = pl.read_csv(survey_path)
        pair_specs = [
            ("vax_willingness_T12", "covid_perceived_danger_T12"),
            ("vax_willingness_T12", "fivec_confidence"),
            ("vax_willingness_T12", "vaccine_risk_avg"),
            ("vax_willingness_T12", "institutional_trust_avg"),
        ]
        rows: list[dict[str, float | str]] = []
        for var_x, var_y in pair_specs:
            if var_x not in survey_df.columns or var_y not in survey_df.columns:
                continue
            val = survey_df.select(pl.corr(var_x, var_y, method="spearman")).item()
            if val is None:
                continue
            rows.append(
                {
                    "var_x": var_x,
                    "var_y": var_y,
                    "rho_signed": float(val),
                }
            )
        signed_df = pl.DataFrame(rows) if rows else pl.DataFrame()
        signed_df.write_csv(artifacts_root / "decision_function" / "spearman_signed_selected.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export compact empirical thesis artifacts to thesis/artifacts/empirical.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root containing empirical/output and thesis/artifacts.",
    )
    args = parser.parse_args()
    export_thesis_artifacts(args.repo_root.resolve())
    print("Exported empirical thesis artifacts.")


if __name__ == "__main__":
    main()
