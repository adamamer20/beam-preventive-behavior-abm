"""Simulation-evaluation utilities for sensitivity analysis."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from beam_abm.abm.config import SimulationConfig
from beam_abm.abm.io import RunPaths, save_run_outputs
from beam_abm.abm.network import build_contact_network
from beam_abm.abm.population import initialise_population
from beam_abm.abm.runtime import run_single_replication
from beam_abm.abm.scenario_defs import SCENARIO_EFFECT_REGIMES, SCENARIO_LIBRARY

from .analysis import _sanitize_name
from .core import (
    BACKGROUND_SCENARIOS,
    FULL_PARAMETER_SWEEP_VALUES,
    INTEGER_PARAMETERS,
    SensitivityPipelineConfig,
    _create_config_from_params,
    _parse_scenario_run_label,
    _reference_baseline,
    _scenario_run_label,
    _seed_for_point,
    _selection_metric_column,
)

_SENSITIVITY_WORKER_CONTEXT: dict[str, Any] | None = None


def _init_sensitivity_worker(
    config: SensitivityPipelineConfig,
    base_cfg: SimulationConfig,
    survey_df: pl.DataFrame,
    anchor_system: Any,
    models: Any,
    levers_cfg: Any,
    stage_output_dir: Path,
    design_name: str,
    all_scenarios_to_run: tuple[str, ...],
    output_summary_columns: tuple[str, ...],
    full_parameter_keys: tuple[str, ...],
) -> None:
    """Store static sensitivity context once per worker process."""
    global _SENSITIVITY_WORKER_CONTEXT
    _SENSITIVITY_WORKER_CONTEXT = {
        "config": config,
        "base_cfg": base_cfg,
        "survey_df": survey_df,
        "anchor_system": anchor_system,
        "models": models,
        "levers_cfg": levers_cfg,
        "stage_output_dir": stage_output_dir,
        "design_name": design_name,
        "all_scenarios_to_run": all_scenarios_to_run,
        "output_summary_columns": output_summary_columns,
        "full_parameter_keys": full_parameter_keys,
    }


def _evaluate_point_seed_job(job: tuple[int, dict[str, float | int], int]) -> list[dict[str, float | int | str]]:
    """Evaluate all scenarios for a single (point_id, seed)."""
    if _SENSITIVITY_WORKER_CONTEXT is None:
        raise RuntimeError("Sensitivity worker context not initialised.")

    point_id, params, run_seed = job
    config: SensitivityPipelineConfig = _SENSITIVITY_WORKER_CONTEXT["config"]
    base_cfg: SimulationConfig = _SENSITIVITY_WORKER_CONTEXT["base_cfg"]
    survey_df: pl.DataFrame = _SENSITIVITY_WORKER_CONTEXT["survey_df"]
    anchor_system = _SENSITIVITY_WORKER_CONTEXT["anchor_system"]
    models = _SENSITIVITY_WORKER_CONTEXT["models"]
    levers_cfg = _SENSITIVITY_WORKER_CONTEXT["levers_cfg"]
    stage_output_dir: Path = _SENSITIVITY_WORKER_CONTEXT["stage_output_dir"]
    design_name: str = _SENSITIVITY_WORKER_CONTEXT["design_name"]
    all_scenarios_to_run: tuple[str, ...] = _SENSITIVITY_WORKER_CONTEXT["all_scenarios_to_run"]
    output_summary_columns: tuple[str, ...] = _SENSITIVITY_WORKER_CONTEXT["output_summary_columns"]
    full_parameter_keys: tuple[str, ...] = _SENSITIVITY_WORKER_CONTEXT["full_parameter_keys"]

    rows: list[dict[str, float | int | str]] = []
    sim_cfg = _create_config_from_params(base_cfg, params, seed=run_seed)

    try:
        agents = initialise_population(
            n_agents=sim_cfg.n_agents,
            countries=sim_cfg.countries,
            anchor_system=anchor_system,
            survey_df=survey_df,
            seed=run_seed,
            norm_threshold_beta_alpha=sim_cfg.constants.norm_threshold_beta_alpha,
        )
        shared_network = build_contact_network(
            agents,
            bridge_prob=sim_cfg.constants.bridge_prob,
            seed=run_seed,
        )
    except (RuntimeError, ValueError, TypeError):
        logger.exception(
            "Network build failed for point_id={point_id}, seed={seed}.",
            point_id=point_id,
            seed=run_seed,
        )
        for scenario_name in all_scenarios_to_run:
            row_out: dict[str, float | int | str] = {
                "point_id": point_id,
                "seed": run_seed,
                "scenario": scenario_name,
            }
            row_out.update({name: params[name] for name in full_parameter_keys})
            row_out.update({metric: float("nan") for metric in output_summary_columns})
            rows.append(row_out)
        return rows

    for scenario_name in all_scenarios_to_run:
        base_scenario_name, effect_regime = _parse_scenario_run_label(scenario_name)
        scenario_cfg = SCENARIO_LIBRARY[base_scenario_name]
        try:
            run_cfg = sim_cfg.model_copy(deep=True)
            run_cfg.effect_regime = effect_regime
            result = run_single_replication(
                cfg=run_cfg,
                scenario=scenario_cfg,
                models=models,
                levers_cfg=levers_cfg,
                agent_df=agents,
                network=shared_network,
                rep_id=0,
                seed=run_seed,
            )
            output_values = _extract_output_summaries(result, config.outputs)
            if config.persist_run_data:
                run_id = _sanitize_name(
                    f"{design_name.lower()}__point_{point_id:05d}__seed_{run_seed}__{scenario_name}"
                )
                run_paths = RunPaths(stage_output_dir, run_id)
                save_run_outputs(
                    run_paths=run_paths,
                    trajectory=result.outcome_trajectory,
                    mediator_trajectory=result.mediator_trajectory,
                    signals_realised=result.signals_realised,
                    outcomes_by_group=result.outcomes_by_group,
                    mediators_by_group=result.mediators_by_group,
                    saturation_trajectory=result.saturation_trajectory,
                    incidence_trajectory=result.incidence_trajectory,
                    distance_trajectory=result.distance_trajectory,
                    effects=result.effects,
                    population_t0=result.population_t0,
                    population_final=result.population_final,
                    metadata=result.metadata,
                    scenario_metrics=result.scenario_metrics,
                    adjacency=result.network_adjacency,
                )
        except (RuntimeError, ValueError, TypeError):
            logger.exception(
                "Scenario run failed for point_id={point_id}, scenario={scenario}, seed={seed}.",
                point_id=point_id,
                scenario=scenario_name,
                seed=run_seed,
            )
            output_values = {metric: float("nan") for metric in output_summary_columns}
        finally:
            if "result" in locals():
                del result

        row_out: dict[str, float | int | str] = {
            "point_id": point_id,
            "seed": run_seed,
            "scenario": scenario_name,
        }
        row_out.update({name: params[name] for name in full_parameter_keys})
        row_out.update(output_values)
        rows.append(row_out)

    del shared_network
    del agents

    return rows


def _output_summary_columns(outputs: tuple[str, ...]) -> tuple[str, ...]:
    columns: list[str] = []
    for output_name in outputs:
        columns.append(output_name)
        columns.append(f"{output_name}__auc")
    return tuple(columns)


def _extract_final_outputs(result: Any, outputs: tuple[str, ...]) -> dict[str, float]:
    """Extract configured output metrics at the final simulation tick."""
    outcomes = result.outcome_trajectory
    mediators = result.mediator_trajectory
    if outcomes.is_empty():
        return {output_name: float("nan") for output_name in outputs}

    max_tick = outcomes.select("tick").max().item()
    final_outcomes = outcomes.filter(pl.col("tick") == max_tick)
    final_mediators = mediators.filter(pl.col("tick") == max_tick) if mediators is not None else None

    values: dict[str, float] = {}
    for output_name in outputs:
        if output_name in final_outcomes.columns:
            values[output_name] = float(final_outcomes.select(output_name).item())
            continue
        if final_mediators is not None and output_name in final_mediators.columns:
            values[output_name] = float(final_mediators.select(output_name).item())
            continue
        values[output_name] = float("nan")
    return values


def _extract_auc_outputs(result: Any, outputs: tuple[str, ...]) -> dict[str, float]:
    """Extract configured output AUCs over simulation ticks."""
    outcomes = result.outcome_trajectory
    mediators = result.mediator_trajectory

    if outcomes is not None and not outcomes.is_empty() and "tick" in outcomes.columns:
        outcomes = outcomes.sort("tick")
    if mediators is not None and not mediators.is_empty() and "tick" in mediators.columns:
        mediators = mediators.sort("tick")

    values: dict[str, float] = {}
    for output_name in outputs:
        source: pl.DataFrame | None = None
        if outcomes is not None and not outcomes.is_empty() and output_name in outcomes.columns:
            source = outcomes
        elif mediators is not None and not mediators.is_empty() and output_name in mediators.columns:
            source = mediators

        if source is None or "tick" not in source.columns:
            values[output_name] = float("nan")
            continue

        ticks = source.get_column("tick").cast(pl.Float64).to_numpy()
        series = source.get_column(output_name).cast(pl.Float64).to_numpy()
        finite_mask = np.isfinite(ticks) & np.isfinite(series)
        ticks = ticks[finite_mask]
        series = series[finite_mask]
        if ticks.size == 0:
            values[output_name] = float("nan")
        elif ticks.size == 1:
            values[output_name] = float(series[0])
        else:
            values[output_name] = float(np.trapezoid(series, ticks))
    return values


def _extract_output_summaries(result: Any, outputs: tuple[str, ...]) -> dict[str, float]:
    """Extract final and AUC summaries for configured outputs."""
    final_values = _extract_final_outputs(result, outputs)
    auc_values = _extract_auc_outputs(result, outputs)
    summaries: dict[str, float] = {}
    for output_name in outputs:
        summaries[output_name] = float(final_values.get(output_name, float("nan")))
        summaries[f"{output_name}__auc"] = float(auc_values.get(output_name, float("nan")))
    return summaries


def _build_delta_table(
    *,
    scenario_means: pl.DataFrame,
    scenario_targets: tuple[str, ...],
    outputs: tuple[str, ...],
    config: SensitivityPipelineConfig,
) -> pl.DataFrame:
    """Construct scenario delta table from per-point scenario means."""
    deltas: list[dict[str, float | int | str]] = []
    for point_key, point_df in scenario_means.group_by("point_id"):
        point_id = int(point_key[0]) if isinstance(point_key, tuple) else int(point_key)
        scenario_rows = point_df.to_dicts()
        by_scenario: dict[str, dict[str, Any]] = {str(row["scenario"]): row for row in scenario_rows}

        for scenario_name in scenario_targets:
            scenario_row = by_scenario.get(scenario_name)
            if scenario_row is None:
                continue
            reference_name = _reference_baseline(scenario_name)
            reference_row = by_scenario.get(reference_name)
            if reference_row is None:
                logger.warning(
                    "Missing reference row for point_id={point_id}, scenario={scenario}, reference={reference}.",
                    point_id=point_id,
                    scenario=scenario_name,
                    reference=reference_name,
                )
                continue

            for output_name in outputs:
                final_column = _selection_metric_column(output_name=output_name, effect_metric="final")
                auc_column = _selection_metric_column(output_name=output_name, effect_metric="auc")
                metric_column = _selection_metric_column(output_name=output_name, effect_metric=config.effect_metric)

                delta_final = float(scenario_row[final_column]) - float(reference_row[final_column])
                delta_auc = float(scenario_row[auc_column]) - float(reference_row[auc_column])
                delta_value = float(scenario_row[metric_column]) - float(reference_row[metric_column])
                delta_row: dict[str, float | int | str] = {
                    "point_id": point_id,
                    "scenario": scenario_name,
                    "reference_scenario": reference_name,
                    "output": output_name,
                    "effect_metric": config.effect_metric,
                    "metric_column": metric_column,
                    "delta": delta_value,
                    "delta_final": delta_final,
                    "delta_auc": delta_auc,
                }
                for parameter_name in FULL_PARAMETER_SWEEP_VALUES:
                    delta_row[parameter_name] = scenario_row[parameter_name]
                deltas.append(delta_row)

    if not deltas:
        schema: dict[str, pl.DataType] = {
            "point_id": pl.Int64,
            "scenario": pl.Utf8,
            "reference_scenario": pl.Utf8,
            "output": pl.Utf8,
            "effect_metric": pl.Utf8,
            "metric_column": pl.Utf8,
            "delta": pl.Float64,
            "delta_final": pl.Float64,
            "delta_auc": pl.Float64,
        }
        for parameter_name in FULL_PARAMETER_SWEEP_VALUES:
            schema[parameter_name] = pl.Int64 if parameter_name in INTEGER_PARAMETERS else pl.Float64
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(deltas)


def _evaluate_design_points(
    *,
    design_df: pl.DataFrame,
    scenario_targets: tuple[str, ...],
    outputs: tuple[str, ...],
    config: SensitivityPipelineConfig,
    base_cfg: SimulationConfig,
    survey_df: pl.DataFrame,
    anchor_system: Any,
    models: Any,
    levers_cfg: Any,
    design_name: str,
    stage_output_dir: Path,
    resolve_workers: Callable[..., int],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run all scenarios for each design point and compute deltas."""
    background_runs = {
        _scenario_run_label(name, effect_regime)
        for name in BACKGROUND_SCENARIOS
        for effect_regime in SCENARIO_EFFECT_REGIMES
    }
    all_scenarios_to_run = tuple(sorted(background_runs | set(scenario_targets)))
    rows: list[dict[str, float | int | str]] = []
    output_summary_columns = _output_summary_columns(outputs)

    jobs: list[tuple[int, dict[str, float | int], int]] = []
    for row in design_df.iter_rows(named=True):
        point_id = int(row["point_id"])
        params: dict[str, float | int] = {name: row[name] for name in FULL_PARAMETER_SWEEP_VALUES if name in row}
        for seed_offset in config.seed_offsets:
            run_seed = _seed_for_point(config.base_seed, point_id, seed_offset)
            jobs.append((point_id, params, run_seed))

    workers = resolve_workers(requested_workers=config.max_workers, n_jobs=len(jobs))
    logger.info(
        "{design}: evaluating {n_jobs} point-seed jobs with workers={workers}.",
        design=design_name,
        n_jobs=len(jobs),
        workers=workers,
    )

    if workers == 1:
        _init_sensitivity_worker(
            config,
            base_cfg,
            survey_df,
            anchor_system,
            models,
            levers_cfg,
            stage_output_dir,
            design_name,
            all_scenarios_to_run,
            output_summary_columns,
            tuple(FULL_PARAMETER_SWEEP_VALUES.keys()),
        )
        for index, job in enumerate(jobs, start=1):
            if index == 1 or index % max(1, len(jobs) // 10) == 0:
                logger.info(
                    "{design}: progress {current}/{total} jobs.",
                    design=design_name,
                    current=index,
                    total=len(jobs),
                )
            rows.extend(_evaluate_point_seed_job(job))
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=_init_sensitivity_worker,
            initargs=(
                config,
                base_cfg,
                survey_df,
                anchor_system,
                models,
                levers_cfg,
                stage_output_dir,
                design_name,
                all_scenarios_to_run,
                output_summary_columns,
                tuple(FULL_PARAMETER_SWEEP_VALUES.keys()),
            ),
        ) as executor:
            job_iter = iter(jobs)
            max_in_flight = max(workers * 2, workers + 1)
            pending: set[Any] = set()

            for _ in range(min(len(jobs), max_in_flight)):
                pending.add(executor.submit(_evaluate_point_seed_job, next(job_iter)))

            completed = 0
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    completed += 1
                    if completed == 1 or completed % max(1, len(jobs) // 10) == 0:
                        logger.info(
                            "{design}: progress {current}/{total} jobs.",
                            design=design_name,
                            current=completed,
                            total=len(jobs),
                        )
                    rows.extend(future.result())
                    try:
                        pending.add(executor.submit(_evaluate_point_seed_job, next(job_iter)))
                    except StopIteration:
                        pass

    raw_runs = pl.DataFrame(rows)
    aggregate_exprs = [pl.col(metric).mean().alias(metric) for metric in output_summary_columns]
    aggregate_exprs.extend(pl.col(param).first().alias(param) for param in FULL_PARAMETER_SWEEP_VALUES)
    scenario_means = raw_runs.group_by("point_id", "scenario").agg(aggregate_exprs)

    delta_df = _build_delta_table(
        scenario_means=scenario_means,
        scenario_targets=scenario_targets,
        outputs=outputs,
        config=config,
    )
    return raw_runs, delta_df
