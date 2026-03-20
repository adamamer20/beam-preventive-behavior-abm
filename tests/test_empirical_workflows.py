from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from beam_abm.empirical.export import export_thesis_artifacts
from beam_abm.empirical.missingness import drop_nonfinite_rows_pl
from beam_abm.empirical.workflows import anchors as anchor_workflows
from beam_abm.empirical.workflows import modeling as modeling_workflows
from beam_abm.empirical.workflows.descriptives import run_descriptives_workflow
from beam_abm.empirical.workflows.inferential import run_inferential_workflow


def _write_minimal_dataset(path: Path) -> None:
    df = pl.DataFrame(
        {
            "country": ["A", "A", "B", "B", "A", "B"],
            "age": [20, 30, 40, 25, 35, 45],
            "income": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            "binary_flag": [0, 1, 0, 1, 0, 1],
            "cat_var": ["x", "x", "y", "y", "x", "y"],
            "outcome": [1, 2, 3, 2, 1, 3],
        }
    )
    df.write_csv(path)


def _write_column_types(path: Path) -> None:
    rows = [
        "column\ttype\tsection\tblock",
        "country\tcategorical\tmetadata\tstructural",
        "age\tcontinuous\tsocio-demo-econ-health\tstructural",
        "income\tcontinuous\tsocio-demo-econ-health\tstructural",
        "binary_flag\tbinary\tgroup\tsocial",
        "cat_var\tcategorical\tgroup\tsocial",
        "outcome\tordinal\topinion\toutcome",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_descriptives_workflow_smoke(tmp_path: Path) -> None:
    input_csv = tmp_path / "data.csv"
    col_types = tmp_path / "column_types.tsv"
    outdir = tmp_path / "descriptives"
    _write_minimal_dataset(input_csv)
    _write_column_types(col_types)

    run_descriptives_workflow(
        input_path=input_csv,
        outdir=outdir,
        column_types_path=col_types,
        categorical_overrides="",
        compute_alpha=False,
        bool_groups="",
        log_level="INFO",
    )

    assert (outdir / "raw" / "continuous_summary.csv").exists()
    assert (outdir / "raw" / "binary_summary.csv").exists()


def test_inferential_workflow_smoke(tmp_path: Path) -> None:
    input_csv = tmp_path / "data.csv"
    col_types = tmp_path / "column_types.tsv"
    outdir = tmp_path / "inferential"
    _write_minimal_dataset(input_csv)
    _write_column_types(col_types)

    run_inferential_workflow(
        input_csv=input_csv,
        column_types_path=col_types,
        outdir=outdir,
        categorical_overrides="",
        min_nominal_levels=2,
        workers=1,
        highlights=False,
        derivations_tsv=None,
        log_level="INFO",
    )

    assert (outdir / "raw").exists()
    assert (outdir / "views").exists()


def test_modeling_workflow_orchestration(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_load_specs_and_data(**kwargs: Any):
        calls.append(("load", kwargs))
        return [], [], object(), {}, {}, None

    def fake_run_fit(**kwargs: Any) -> None:
        calls.append(("fit", kwargs))

    def fake_run_block_importance(**kwargs: Any) -> None:
        calls.append(("block", kwargs))

    monkeypatch.setattr(modeling_workflows, "_load_specs_and_data", fake_load_specs_and_data)
    monkeypatch.setattr(modeling_workflows, "_run_fit", fake_run_fit)
    monkeypatch.setattr(modeling_workflows, "_run_block_importance", fake_run_block_importance)

    modeling_workflows.run_modeling_fit_workflow(log_level="INFO")
    modeling_workflows.run_modeling_block_importance_workflow(log_level="INFO")
    modeling_workflows.run_modeling_workflow(log_level="INFO")

    labels = [label for label, _ in calls]
    assert "load" in labels
    assert "fit" in labels
    assert "block" in labels


def test_anchor_workflow_orchestration(monkeypatch, tmp_path: Path) -> None:
    called: list[tuple[str, Any]] = []

    def fake_build(cfg: Any) -> None:
        called.append(("build", cfg))

    def fake_diag(cfg: Any) -> None:
        called.append(("diag", cfg))

    def fake_pe(cfg: Any) -> None:
        called.append(("pe", cfg))

    def fake_build_full_profiles(input_csv: Path, out_path: Path, *, country_col: str) -> None:
        called.append(("full_profiles", {"input_csv": input_csv, "out_path": out_path, "country_col": country_col}))
        pl.DataFrame({"eval_id": ["full::A::0"], "row_id": [0], "country": ["A"]}).write_parquet(out_path)

    def fake_rank_metrics(**kwargs: Any) -> None:
        called.append(("rank", kwargs))

    monkeypatch.setattr(anchor_workflows, "build_anchor_system", fake_build)
    monkeypatch.setattr(anchor_workflows, "run_anchor_diagnostics", fake_diag)
    monkeypatch.setattr(anchor_workflows, "run_anchor_pe", fake_pe)
    monkeypatch.setattr(anchor_workflows, "_build_full_eval_profiles", fake_build_full_profiles)
    monkeypatch.setattr(anchor_workflows, "_write_block_rank_metrics", fake_rank_metrics)

    anchor_workflows.run_anchor_build_workflow(features="age,income")
    anchor_workflows.run_anchor_diagnostics_workflow()
    anchor_workflows.run_anchor_pe_workflow(target_col="outcome", driver_cols="age", levers="age")
    anchor_workflows.run_anchor_full_pe_workflow(
        outdir=tmp_path / "full",
        target_col="outcome",
        driver_cols="age",
        levers="age",
        input_csv=tmp_path / "in.csv",
    )

    labels = [label for label, _ in called]
    assert "build" in labels
    assert "diag" in labels
    assert labels.count("pe") >= 2
    assert "full_profiles" in labels
    assert "rank" in labels


def test_empirical_export_library_smoke(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    required_paths = [
        "empirical/output/descriptives/raw/binary_summary.csv",
        "empirical/output/descriptives/raw/continuous_summary.csv",
        "empirical/output/descriptives/raw/ordinal_summary.csv",
        "empirical/output/modeling/model_coefficients_all.csv",
        "empirical/output/modeling/model_summary.csv",
        "empirical/output/modeling/model_lobo_all.csv",
        "empirical/output/modeling/model_plan.json",
        "empirical/output/modeling/model_plan_ref_choice.json",
        "empirical/output/modeling/B_FLU/B_FLU_vaccinated_2023_2024_ENGINES/lobo.csv",
        "empirical/output/inferential/raw/spearman.csv",
    ]

    for rel in required_paths:
        path = repo_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text("{}\n", encoding="utf-8")
        else:
            path.write_text("col\n1\n", encoding="utf-8")

    (repo_root / "thesis/artifacts").mkdir(parents=True, exist_ok=True)

    export_thesis_artifacts(repo_root)

    assert (repo_root / "thesis/artifacts/empirical/modeling/model_summary.csv").exists()
    assert (repo_root / "thesis/artifacts/empirical/descriptives/binary_summary.csv").exists()


def test_drop_nonfinite_rows_polars_handles_numeric_strings() -> None:
    df = pl.DataFrame(
        {
            "target": [1.0, 2.0, 3.0],
            "income_ppp_norm": ["100.0", None, "250.0"],
        }
    )

    cleaned = drop_nonfinite_rows_pl(df, ["target", "income_ppp_norm"], context="test")
    assert cleaned.height == 2


def test_missingness_plan_polars_handles_string_numeric_median_fill() -> None:
    df = pl.DataFrame(
        {
            "country": ["IT", "DE", "IT"],
            "income_ppp_norm": ["100.0", None, "250.0"],
        }
    )

    from beam_abm.empirical.missingness import apply_missingness_plan_pl, build_missingness_plan_pl

    plan = build_missingness_plan_pl(df, predictors=["income_ppp_norm"], threshold=0.0, country_col="country")
    out = apply_missingness_plan_pl(df, plan, country_col="country")

    assert out.schema["income_ppp_norm"] == pl.Float64
    assert out.get_column("income_ppp_norm").is_null().sum() == 0


def test_missingness_plan_polars_country_fill_with_empty_mapping_fallback() -> None:
    df = pl.DataFrame(
        {
            "country": ["IT", "DE"],
            "income_ppp_norm": [None, None],
        },
        schema={"country": pl.Utf8, "income_ppp_norm": pl.Utf8},
    )

    from beam_abm.empirical.missingness import apply_missingness_plan_pl, build_missingness_plan_pl

    plan = build_missingness_plan_pl(df, predictors=["income_ppp_norm"], threshold=0.0, country_col="country")
    out = apply_missingness_plan_pl(df, plan, country_col="country")

    assert out.schema["income_ppp_norm"] == pl.Float64
    assert out.get_column("income_ppp_norm").is_null().sum() == 0
