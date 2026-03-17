r"""Full ABM sensitivity-analysis pipeline.

Implements a staged workflow:

1. Morris screening on all uncertain parameters and all scenarios.
2. Parameter shortlist selection from aggregate Morris indices.
3. Flagship-scenario selection for computationally heavy stages.
4. Latin-hypercube sampling (LHC) on shortlisted parameters.
5. Surrogate fitting for scenario-output deltas.
6. Sobol decomposition on surrogate predictions.
7. Reporting artefacts (tables, heatmaps, summaries).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from xgboost import XGBRegressor

from beam_abm.abm.data_contracts import (
    load_all_backbone_models,
    load_anchor_system,
    load_levers_config,
)
from beam_abm.abm.io import load_survey_data
from beam_abm.abm.scenario_defs import SCENARIO_EFFECT_REGIMES, SCENARIO_LIBRARY

from .sensitivity import core as _core
from .sensitivity.analysis import (
    _fit_surrogate_model,
    _sanitize_name,
    compute_lhc_spearman_directionality,
    compute_morris_indices,
    compute_sobol_indices_from_surrogate,
    compute_surrogate_ice_slope_directionality,
    run_posthoc_directional_diagnostics,
    select_shortlist,
)
from .sensitivity.core import (
    DEFAULT_FLAGSHIP_SCENARIOS,
    DEFAULT_OUTPUT_METRICS,
    DEFAULT_PARAMETER_BOUNDS,
    FULL_PARAMETER_SWEEP_VALUES,
    PARAMETER_DEFAULTS,
    SensitivityPipelineConfig,
    _base_config,
    _ensure_output_dirs,
    _scenario_run_label,
)
from .sensitivity.design import build_lhc_design, build_morris_design
from .sensitivity.evaluation import (
    _build_delta_table,
    _output_summary_columns,
)
from .sensitivity.evaluation import (
    _evaluate_design_points as _evaluate_design_points_impl,
)
from .sensitivity.reporting import (
    _plot_morris_heatmaps,
    _write_non_flagship_morris_summary,
    _write_shortlist_rationale,
    _write_sobol_narrative,
)

__all__ = [
    "DEFAULT_FLAGSHIP_SCENARIOS",
    "DEFAULT_OUTPUT_METRICS",
    "FULL_PARAMETER_SWEEP_VALUES",
    "SensitivityPipelineConfig",
    "build_lhc_design",
    "build_morris_design",
    "compute_morris_indices",
    "compute_lhc_spearman_directionality",
    "compute_surrogate_ice_slope_directionality",
    "compute_sobol_indices_from_surrogate",
    "refresh_lhc_flagship_scenarios",
    "run_posthoc_directional_diagnostics",
    "run_full_sensitivity_pipeline",
    "select_shortlist",
]

INTEGER_PARAMETERS = _core.INTEGER_PARAMETERS
_reference_baseline = _core._reference_baseline

AUTO_MAX_SENSITIVITY_WORKERS: int = 6
AUTO_SA_WORKER_MEM_GIB: float = 3.0
AUTO_SA_WORKER_RESERVE_GIB: float = 6.0
AUTO_SA_PARENT_RSS_PER_WORKER_FACTOR: float = 0.35
AUTO_SA_PARENT_RSS_RESERVE_FACTOR: float = 0.25


def _resolve_sensitivity_workers(*, requested_workers: int, n_jobs: int) -> int:
    """Resolve worker count for sensitivity job execution."""
    if n_jobs <= 1:
        return 1
    cpu_cap = max(1, os.cpu_count() or 1)
    if requested_workers > 0:
        return max(1, min(requested_workers, cpu_cap, n_jobs))

    available_gib = _available_memory_gib()
    parent_rss_gib = _current_process_rss_gib()

    if available_gib is None:
        return max(1, min(max(1, cpu_cap - 1), AUTO_MAX_SENSITIVITY_WORKERS, n_jobs))

    reserve_gib = float(os.getenv("BEAM_ABM_SA_WORKER_RESERVE_GIB", str(AUTO_SA_WORKER_RESERVE_GIB)))
    per_worker_gib = float(os.getenv("BEAM_ABM_SA_WORKER_MEM_GIB", str(AUTO_SA_WORKER_MEM_GIB)))
    parent_rss_per_worker_factor = float(
        os.getenv(
            "BEAM_ABM_SA_PARENT_RSS_PER_WORKER_FACTOR",
            str(AUTO_SA_PARENT_RSS_PER_WORKER_FACTOR),
        )
    )
    parent_rss_reserve_factor = float(
        os.getenv(
            "BEAM_ABM_SA_PARENT_RSS_RESERVE_FACTOR",
            str(AUTO_SA_PARENT_RSS_RESERVE_FACTOR),
        )
    )
    max_auto_workers = int(os.getenv("BEAM_ABM_SA_MAX_AUTO_WORKERS", str(AUTO_MAX_SENSITIVITY_WORKERS)))
    if per_worker_gib <= 0.0:
        per_worker_gib = AUTO_SA_WORKER_MEM_GIB
    if parent_rss_per_worker_factor < 0.0:
        parent_rss_per_worker_factor = AUTO_SA_PARENT_RSS_PER_WORKER_FACTOR
    if parent_rss_reserve_factor < 0.0:
        parent_rss_reserve_factor = AUTO_SA_PARENT_RSS_RESERVE_FACTOR
    if max_auto_workers <= 0:
        max_auto_workers = AUTO_MAX_SENSITIVITY_WORKERS

    effective_per_worker_gib = per_worker_gib
    effective_reserve_gib = reserve_gib
    if parent_rss_gib is not None and np.isfinite(parent_rss_gib):
        effective_per_worker_gib += parent_rss_gib * parent_rss_per_worker_factor
        effective_reserve_gib += parent_rss_gib * parent_rss_reserve_factor

    budget_gib = max(0.0, available_gib - effective_reserve_gib)
    memory_cap = int(budget_gib // effective_per_worker_gib) if effective_per_worker_gib > 0.0 else 1
    return max(1, min(max(1, cpu_cap - 1), max_auto_workers, n_jobs, max(1, memory_cap)))


def _available_memory_gib() -> float | None:
    """Return currently available memory in GiB when supported."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) / float(1024**2)
    except (FileNotFoundError, OSError, ValueError):
        pass

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None
    return (page_size * available_pages) / float(1024**3)


def _current_process_rss_gib() -> float | None:
    """Return current process RSS in GiB when supported."""
    try:
        with open("/proc/self/status", encoding="utf-8") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) / float(1024**2)
    except (FileNotFoundError, OSError, ValueError):
        pass
    return None


def _evaluate_design_points(
    *,
    design_df: pl.DataFrame,
    scenario_targets: tuple[str, ...],
    outputs: tuple[str, ...],
    config: SensitivityPipelineConfig,
    base_cfg: Any,
    survey_df: pl.DataFrame,
    anchor_system: Any,
    models: Any,
    levers_cfg: Any,
    design_name: str,
    stage_output_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    return _evaluate_design_points_impl(
        design_df=design_df,
        scenario_targets=scenario_targets,
        outputs=outputs,
        config=config,
        base_cfg=base_cfg,
        survey_df=survey_df,
        anchor_system=anchor_system,
        models=models,
        levers_cfg=levers_cfg,
        design_name=design_name,
        stage_output_dir=stage_output_dir,
        resolve_workers=_resolve_sensitivity_workers,
    )


def refresh_lhc_flagship_scenarios(
    config: SensitivityPipelineConfig,
    *,
    base_scenarios: tuple[str, ...],
) -> None:
    """Refresh staged LHC outputs for a targeted subset of flagship scenarios."""
    paths = _ensure_output_dirs(config.output_dir)
    shortlist_path = paths["shortlist"] / "shortlist.csv"
    lhc_design_path = paths["lhc"] / "design.parquet"
    lhc_raw_path = paths["lhc"] / "runs_raw.parquet"
    lhc_delta_path = paths["lhc"] / "delta.parquet"

    if not shortlist_path.exists():
        raise FileNotFoundError(f"Missing shortlist file: {shortlist_path}")
    if not lhc_design_path.exists():
        raise FileNotFoundError(f"Missing LHC design file: {lhc_design_path}")

    valid_base = tuple(name for name in base_scenarios if name in SCENARIO_LIBRARY)
    if not valid_base:
        raise ValueError("No requested flagship scenarios are present in the scenario library.")

    scenario_targets = tuple(
        _scenario_run_label(name, effect_regime) for name in valid_base for effect_regime in SCENARIO_EFFECT_REGIMES
    )

    base_cfg = _base_config(config)
    levers_cfg = load_levers_config(base_cfg.levers_path)
    anchor_system = load_anchor_system(base_cfg.anchors_dir)
    survey_df = load_survey_data(base_cfg.survey_csv, base_cfg.countries)
    models = load_all_backbone_models(base_cfg.modeling_dir, levers_cfg, survey_df=survey_df)
    lhc_design = pl.read_parquet(lhc_design_path)

    lhc_raw_new, lhc_delta_new = _evaluate_design_points(
        design_df=lhc_design,
        scenario_targets=scenario_targets,
        outputs=config.outputs,
        config=config,
        base_cfg=base_cfg,
        survey_df=survey_df,
        anchor_system=anchor_system,
        models=models,
        levers_cfg=levers_cfg,
        design_name="LHC_REFRESH",
        stage_output_dir=paths["lhc"],
    )

    if lhc_raw_path.exists():
        lhc_raw_existing = pl.read_parquet(lhc_raw_path).filter(~pl.col("scenario").is_in(scenario_targets))
        lhc_raw_merged = (
            pl.concat([lhc_raw_existing, lhc_raw_new], how="diagonal_relaxed")
            .unique(subset=["point_id", "seed", "scenario"], keep="last")
            .sort(["scenario", "point_id", "seed"])
        )
    else:
        lhc_raw_merged = lhc_raw_new.sort(["scenario", "point_id", "seed"])
    lhc_raw_merged.write_parquet(lhc_raw_path)

    if lhc_delta_path.exists():
        lhc_delta_existing = pl.read_parquet(lhc_delta_path).filter(~pl.col("scenario").is_in(scenario_targets))
        lhc_delta_merged = (
            pl.concat([lhc_delta_existing, lhc_delta_new], how="diagonal_relaxed")
            .unique(subset=["point_id", "scenario", "output"], keep="last")
            .sort(["scenario", "output", "point_id"])
        )
    else:
        lhc_delta_merged = lhc_delta_new.sort(["scenario", "output", "point_id"])
    lhc_delta_merged.write_parquet(lhc_delta_path)

    logger.info(
        "Refreshed LHC outputs for scenarios: {scenarios}",
        scenarios=", ".join(valid_base),
    )
    logger.info("- LHC raw: {path}", path=lhc_raw_path)
    logger.info("- LHC delta: {path}", path=lhc_delta_path)


def run_full_sensitivity_pipeline(config: SensitivityPipelineConfig) -> None:
    """Execute the full ABM sensitivity pipeline."""
    paths = _ensure_output_dirs(config.output_dir)
    logger.info("Full sensitivity pipeline output directory: {path}", path=config.output_dir)

    base_cfg = _base_config(config)
    levers_cfg = load_levers_config(base_cfg.levers_path)
    anchor_system = load_anchor_system(base_cfg.anchors_dir)
    survey_df = load_survey_data(base_cfg.survey_csv, base_cfg.countries)
    models = load_all_backbone_models(base_cfg.modeling_dir, levers_cfg, survey_df=survey_df)

    all_scenarios = tuple(sorted(SCENARIO_LIBRARY))
    morris_targets_base = tuple(name for name in all_scenarios if name not in {"baseline", "baseline_importation"})

    flagship_base = tuple(name for name in config.flagship_scenarios if name in SCENARIO_LIBRARY)
    flagship_scenarios = tuple(
        _scenario_run_label(name, effect_regime) for name in flagship_base for effect_regime in SCENARIO_EFFECT_REGIMES
    )
    if not flagship_scenarios:
        raise ValueError("No flagship scenarios available in scenario library.")
    morris_targets = tuple(
        _scenario_run_label(name, effect_regime)
        for name in morris_targets_base
        for effect_regime in SCENARIO_EFFECT_REGIMES
    )
    non_flagship_scenarios = tuple(name for name in morris_targets if name not in set(flagship_scenarios))

    logger.info("Step 1/7: Morris screening")
    morris_design_path = paths["morris"] / "design.parquet"
    morris_raw_path = paths["morris"] / "runs_raw.parquet"
    if config.resume_from_morris_raw and morris_design_path.exists() and morris_raw_path.exists():
        logger.info(
            "Resuming Morris stage from existing design/raw outputs at {path}.",
            path=paths["morris"],
        )
        morris_design = pl.read_parquet(morris_design_path)
        morris_raw = pl.read_parquet(morris_raw_path)
        output_summary_columns = _output_summary_columns(config.outputs)
        aggregate_exprs = [pl.col(metric).mean().alias(metric) for metric in output_summary_columns]
        aggregate_exprs.extend(pl.col(param).first().alias(param) for param in FULL_PARAMETER_SWEEP_VALUES)
        scenario_means = morris_raw.group_by("point_id", "scenario").agg(aggregate_exprs)
        morris_delta = _build_delta_table(
            scenario_means=scenario_means,
            scenario_targets=morris_targets,
            outputs=config.outputs,
            config=config,
        )
        morris_delta.write_parquet(paths["morris"] / "delta.parquet")
    else:
        morris_design = build_morris_design(
            parameter_bounds=DEFAULT_PARAMETER_BOUNDS,
            n_trajectories=config.morris_trajectories,
            n_levels=config.morris_levels,
            seed=config.base_seed,
        )
        morris_design.write_parquet(paths["morris"] / "design.parquet")

        morris_raw, morris_delta = _evaluate_design_points(
            design_df=morris_design,
            scenario_targets=morris_targets,
            outputs=config.outputs,
            config=config,
            base_cfg=base_cfg,
            survey_df=survey_df,
            anchor_system=anchor_system,
            models=models,
            levers_cfg=levers_cfg,
            design_name="MORRIS",
            stage_output_dir=paths["morris"],
        )
        morris_raw.write_parquet(paths["morris"] / "runs_raw.parquet")
        morris_delta.write_parquet(paths["morris"] / "delta.parquet")

    morris_per_scenario, morris_aggregate = compute_morris_indices(
        design_df=morris_design,
        delta_df=morris_delta,
    )
    morris_per_scenario.write_csv(paths["morris"] / "indices_per_scenario.csv")
    morris_aggregate.write_csv(paths["morris"] / "indices_aggregate.csv")
    _plot_morris_heatmaps(morris_per_scenario=morris_per_scenario, output_dir=paths["figures"] / "morris")
    del morris_raw
    del morris_delta

    logger.info("Step 2/7: shortlist selection")
    shortlist_df = select_shortlist(
        aggregate_indices=morris_aggregate,
        target_size=config.shortlist_target,
        max_size=config.shortlist_max,
        sigma_quantile=config.shortlist_sigma_quantile,
    )
    shortlist_df.write_csv(paths["shortlist"] / "shortlist.csv")
    sigma_threshold = float(
        morris_aggregate.select(pl.col("sigma_max").quantile(config.shortlist_sigma_quantile)).item()
    )
    _write_shortlist_rationale(
        shortlist_df=shortlist_df,
        output_path=paths["shortlist"] / "rationale.md",
        sigma_threshold=sigma_threshold,
    )

    shortlisted_params = shortlist_df.select("parameter").to_series().to_list()
    shortlist_bounds = {name: DEFAULT_PARAMETER_BOUNDS[name] for name in shortlisted_params}

    logger.info("Step 3/7 and 4/7: flagship selection and LHC sampling")
    lhc_short_design = build_lhc_design(
        parameter_bounds=shortlist_bounds,
        n_samples=config.lhc_samples,
        seed=config.base_seed + 137,
    )

    full_lhc_rows: list[dict[str, float | int]] = []
    for row in lhc_short_design.iter_rows(named=True):
        expanded: dict[str, float | int] = {"point_id": int(row["point_id"])}
        for parameter_name in FULL_PARAMETER_SWEEP_VALUES:
            if parameter_name in row:
                expanded[parameter_name] = row[parameter_name]
            else:
                expanded[parameter_name] = PARAMETER_DEFAULTS[parameter_name]
        full_lhc_rows.append(expanded)
    lhc_design = pl.DataFrame(full_lhc_rows)
    lhc_design.write_parquet(paths["lhc"] / "design.parquet")

    lhc_raw, lhc_delta = _evaluate_design_points(
        design_df=lhc_design,
        scenario_targets=flagship_scenarios,
        outputs=config.outputs,
        config=config,
        base_cfg=base_cfg,
        survey_df=survey_df,
        anchor_system=anchor_system,
        models=models,
        levers_cfg=levers_cfg,
        design_name="LHC",
        stage_output_dir=paths["lhc"],
    )
    lhc_raw.write_parquet(paths["lhc"] / "runs_raw.parquet")
    lhc_delta.write_parquet(paths["lhc"] / "delta.parquet")
    del lhc_raw

    logger.info("Step 5/7: surrogate fitting")
    feature_df = lhc_design.select(["point_id", *shortlisted_params])

    surrogate_rows: list[dict[str, float | int | str | bool]] = []
    retained_models: dict[tuple[str, str], XGBRegressor] = {}
    for scenario_name in flagship_scenarios:
        for output_name in config.outputs:
            target_df = lhc_delta.filter(
                (pl.col("scenario") == scenario_name) & (pl.col("output") == output_name)
            ).select("point_id", "delta")
            merged = feature_df.join(target_df, on="point_id", how="inner")
            merged = merged.drop_nulls()
            if merged.height < 12:
                surrogate_rows.append(
                    {
                        "scenario": scenario_name,
                        "output": output_name,
                        "n_samples": merged.height,
                        "validation_r2": float("nan"),
                        "retained": False,
                        "reason": "insufficient_samples",
                    }
                )
                continue

            x = merged.select(shortlisted_params).to_numpy()
            y = merged.select("delta").to_numpy().reshape(-1)

            model, validation_r2 = _fit_surrogate_model(
                x=x,
                y=y,
                seed=config.base_seed,
            )
            retained = bool(np.isfinite(validation_r2) and validation_r2 >= config.surrogate_min_r2)
            reason = "ok" if retained else "below_threshold"

            surrogate_rows.append(
                {
                    "scenario": scenario_name,
                    "output": output_name,
                    "n_samples": merged.height,
                    "validation_r2": validation_r2,
                    "retained": retained,
                    "reason": reason,
                }
            )

            if retained:
                retained_models[(scenario_name, output_name)] = model
                model_name = _sanitize_name(f"{scenario_name}__{output_name}")
                model_path = paths["surrogates"] / f"{model_name}.json"
                model.save_model(model_path)

    surrogate_manifest = pl.DataFrame(surrogate_rows)
    surrogate_manifest.write_csv(paths["surrogates"] / "manifest.csv")

    lhc_spearman = compute_lhc_spearman_directionality(
        lhc_delta=lhc_delta,
        parameter_names=tuple(shortlisted_params),
    )
    lhc_spearman.write_csv(paths["reports"] / "lhc_spearman_directionality.csv")

    surrogate_ice_slopes = compute_surrogate_ice_slope_directionality(
        retained_models=retained_models,
        feature_df=feature_df,
        parameter_bounds=shortlist_bounds,
    )
    surrogate_ice_slopes.write_csv(paths["reports"] / "surrogate_ice_slope_directionality.csv")

    logger.info("Step 6/7: Sobol decomposition")
    sobol_rows: list[pl.DataFrame] = []
    for (scenario_name, output_name), model in retained_models.items():
        sobol_df = compute_sobol_indices_from_surrogate(
            model=model,
            parameter_bounds=shortlist_bounds,
            base_samples=config.sobol_base_samples,
            seed=config.base_seed + 503,
        ).with_columns(
            pl.lit(scenario_name).alias("scenario"),
            pl.lit(output_name).alias("output"),
        )
        sobol_rows.append(sobol_df)

    sobol_all = pl.concat(sobol_rows, how="diagonal") if sobol_rows else pl.DataFrame()
    sobol_all.write_csv(paths["sobol"] / "indices.csv")

    logger.info("Step 7/7: reporting")
    _write_sobol_narrative(
        sobol_df=sobol_all,
        output_path=paths["reports"] / "sobol_narrative.md",
    )
    _write_non_flagship_morris_summary(
        morris_per_scenario=morris_per_scenario,
        non_flagship_scenarios=non_flagship_scenarios,
        output_path=paths["reports"] / "non_flagship_morris_summary.md",
    )

    summary_payload = {
        "outputs": list(config.outputs),
        "effect_metric": config.effect_metric,
        "persist_run_data": config.persist_run_data,
        "morris_target_scenarios": list(morris_targets),
        "flagship_scenarios": list(flagship_scenarios),
        "non_flagship_scenarios": list(non_flagship_scenarios),
        "shortlisted_parameters": shortlisted_params,
        "lhc_spearman_rows": int(lhc_spearman.height),
        "surrogate_ice_slope_rows": int(surrogate_ice_slopes.height),
        "surrogate_retained_models": [
            {
                "scenario": scenario_name,
                "output": output_name,
            }
            for (scenario_name, output_name) in retained_models
        ],
    }
    (paths["reports"] / "pipeline_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    logger.info("Full sensitivity pipeline completed.")
    logger.info("- Morris indices: {path}", path=paths["morris"] / "indices_per_scenario.csv")
    logger.info("- Shortlist: {path}", path=paths["shortlist"] / "shortlist.csv")
    logger.info("- Sobol indices: {path}", path=paths["sobol"] / "indices.csv")
