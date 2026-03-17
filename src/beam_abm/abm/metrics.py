"""Trajectory, subgroup, and mediator summary metrics.

Computes per-tick aggregate statistics for outcome and mediator
trajectories, including overall, by-country, and by-anchor strata.

Examples
--------
>>> from beam_abm.abm.metrics import compute_outcome_summary
>>> import numpy as np
>>> summary = compute_outcome_summary(
...     {"vax_willingness_T12": np.array([3.0, 4.0, 5.0])}, tick=0
... )
>>> "vax_willingness_T12_mean" in summary
True
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from beam_abm.abm.metrics_helpers import _first_present_name, _nan_corr, _resolve_threshold, _safe_float
from beam_abm.abm.state_schema import MUTABLE_COLUMNS

if TYPE_CHECKING:
    from beam_abm.abm.data_contracts import BackboneModel

__all__ = [
    "SummaryRow",
    "ATEStats",
    "DEFAULT_OUTCOME_THRESHOLDS",
    "build_behavioral_projection_weights",
    "compute_behavioral_projection_effects",
    "compute_outcome_summary",
    "compute_mediator_summary",
    "compute_group_summary_rows",
    "compute_subgroup_summaries",
    "compute_trajectory_effects",
    "compute_effects_summary",
    "compute_paired_dynamics_diagnostics",
    "compute_local_shadow_target_diff_in_diff",
    "compute_ate",
    "compute_shock_recovery_metrics",
    "compute_diffusion_metrics",
    "compute_closed_loop_signatures",
    "assemble_parity_diagnostics_table",
]


DEFAULT_OUTCOME_THRESHOLDS: dict[str, float] = {
    "vax_willingness_T12": 4.0,
    "mask_when_symptomatic_crowded": 4.0,
    "mask_when_pressure_high": 4.0,
    "stay_home_when_symptomatic": 4.0,
    "flu_vaccinated_2023_2024": 0.5,
}
_VAX_WILLINGNESS_OUTCOME_CANDIDATES: tuple[str, ...] = ("vax_willingness_T12",)
_VAX_WILLINGNESS_MEAN_CANDIDATES: tuple[str, ...] = ("vax_willingness_T12_mean",)


@dataclass(slots=True)
class SummaryRow:
    """Typed container for per-tick summary metrics."""

    tick: int
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float | int]:
        return {"tick": self.tick, **self.metrics}


@dataclass(slots=True)
class ATEStats:
    """Average treatment effect summary."""

    ate_mean: float
    ate_std: float
    peak_effect: float
    cumulative_effect: float

    def to_dict(self) -> dict[str, float]:
        return {
            "ate_mean": self.ate_mean,
            "ate_std": self.ate_std,
            "peak_effect": self.peak_effect,
            "cumulative_effect": self.cumulative_effect,
        }


# =========================================================================
# 1.  Per-tick outcome summary
# =========================================================================


def compute_outcome_summary(
    outcomes: dict[str, np.ndarray],
    tick: int,
    reference_means: dict[str, float] | None = None,
    share_thresholds: dict[str, float] | None = None,
) -> SummaryRow:
    """Compute aggregate statistics for all outcomes at one tick.

    Parameters
    ----------
    outcomes : dict[str, np.ndarray]
        Mapping from outcome name to prediction array.
    tick : int
        Current simulation tick.

    Returns
    -------
    SummaryRow
        Container with ``tick``, and for each outcome:
        ``{name}_mean``, ``{name}_std``, ``{name}_median``,
        ``{name}_q25``, ``{name}_q75``.
    """
    metrics: dict[str, float] = {}
    thresholds = share_thresholds or DEFAULT_OUTCOME_THRESHOLDS
    for name, values in outcomes.items():
        v = values.astype(np.float64)
        mean_v = float(np.nanmean(v))
        threshold = _resolve_threshold(name, v, thresholds)
        metrics[f"{name}_mean"] = float(np.nanmean(v))
        metrics[f"{name}_std"] = float(np.nanstd(v))
        metrics[f"{name}_median"] = float(np.nanmedian(v))
        metrics[f"{name}_q25"] = float(np.nanpercentile(v, 25))
        metrics[f"{name}_q75"] = float(np.nanpercentile(v, 75))
        metrics[f"{name}_q05"] = float(np.nanpercentile(v, 5))
        metrics[f"{name}_q95"] = float(np.nanpercentile(v, 95))
        metrics[f"{name}_share_above_thresh"] = float(np.nanmean(v >= threshold))
        if reference_means is not None:
            metrics[f"{name}_delta_vs_ref_mean"] = float(mean_v - reference_means.get(name, mean_v))
    return SummaryRow(tick=tick, metrics=metrics)


# =========================================================================
# 2.  Per-tick mediator summary
# =========================================================================


def compute_mediator_summary(
    agents: pl.DataFrame,
    tick: int,
    outcomes: dict[str, np.ndarray] | None = None,
) -> SummaryRow:
    """Compute aggregate statistics for all mutable constructs at one tick.

    Parameters
    ----------
    agents : pl.DataFrame
        Current agent state.
    tick : int
        Current simulation tick.

    Returns
    -------
    SummaryRow
        Container with ``tick`` and per-construct mean/std.
    """
    metrics: dict[str, float] = {}
    for col in sorted(MUTABLE_COLUMNS):
        if col not in agents.columns:
            continue
        v = agents[col].to_numpy().astype(np.float64)
        metrics[f"{col}_mean"] = float(np.nanmean(v))
        metrics[f"{col}_std"] = float(np.nanstd(v))
        metrics[f"{col}_median"] = float(np.nanmedian(v))
        metrics[f"{col}_q25"] = float(np.nanpercentile(v, 25))
        metrics[f"{col}_q75"] = float(np.nanpercentile(v, 75))

    if outcomes is not None:
        willingness_outcome = _first_present_name(_VAX_WILLINGNESS_OUTCOME_CANDIDATES, outcomes.keys())
        key_pairs = (
            ("institutional_trust_avg", willingness_outcome, "corr_trust_willingness"),
            ("vaccine_risk_avg", willingness_outcome, "corr_risk_willingness"),
            ("social_norms_vax_avg", "mask_when_symptomatic_crowded", "corr_norms_mask"),
        )
        for mediator_col, outcome_col, metric_name in key_pairs:
            if outcome_col is None:
                continue
            if mediator_col not in agents.columns or outcome_col not in outcomes:
                continue
            mediator_values = agents[mediator_col].to_numpy().astype(np.float64)
            outcome_values = outcomes[outcome_col].astype(np.float64)
            corr = _nan_corr(mediator_values, outcome_values)
            if not np.isnan(corr):
                metrics[metric_name] = float(corr)
    return SummaryRow(tick=tick, metrics=metrics)


def compute_group_summary_rows(
    agents: pl.DataFrame,
    tick: int,
    *,
    group_type: str,
    group_column: str,
    values: dict[str, np.ndarray] | None = None,
    columns: list[str] | None = None,
) -> list[dict[str, float | int | str]]:
    """Compute long-format group summaries for one tick.

    Parameters
    ----------
    agents : pl.DataFrame
        Current agent state.
    tick : int
        Current simulation tick.
    group_type : str
        Semantic group category label (e.g., ``targeted_flag``).
    group_column : str
        Column in *agents* to group on.
    values : dict[str, np.ndarray] or None
        Optional externally computed values keyed by variable name.
    columns : list[str] or None
        Agent columns to summarise if *values* is None.

    Returns
    -------
    list[dict[str, float | int | str]]
        Long-format rows with distribution metrics.
    """
    if group_column not in agents.columns:
        return []

    working = agents.select(group_column)
    if values is not None:
        for variable, array in values.items():
            working = working.with_columns(pl.Series(name=variable, values=array.astype(np.float64)))
        variables = list(values)
    else:
        selected_cols = [col for col in (columns or sorted(MUTABLE_COLUMNS)) if col in agents.columns]
        if not selected_cols:
            return []
        working = agents.select([group_column, *selected_cols])
        variables = selected_cols

    grouped = working.group_by(group_column).agg(
        [pl.col(variable).mean().alias(f"{variable}__mean") for variable in variables]
        + [pl.col(variable).std().alias(f"{variable}__std") for variable in variables]
        + [pl.col(variable).median().alias(f"{variable}__median") for variable in variables]
        + [pl.col(variable).quantile(0.25).alias(f"{variable}__q25") for variable in variables]
        + [pl.col(variable).quantile(0.75).alias(f"{variable}__q75") for variable in variables]
        + [pl.col(variable).quantile(0.05).alias(f"{variable}__q05") for variable in variables]
        + [pl.col(variable).quantile(0.95).alias(f"{variable}__q95") for variable in variables]
    )

    rows: list[dict[str, float | int | str]] = []
    for record in grouped.iter_rows(named=True):
        group_value = record[group_column]
        for variable in variables:
            rows.append(
                {
                    "tick": tick,
                    "group_type": group_type,
                    "group_value": str(group_value),
                    "variable": variable,
                    "mean": float(record[f"{variable}__mean"]),
                    "std": _safe_float(record[f"{variable}__std"]),
                    "median": float(record[f"{variable}__median"]),
                    "q25": float(record[f"{variable}__q25"]),
                    "q75": float(record[f"{variable}__q75"]),
                    "q05": float(record[f"{variable}__q05"]),
                    "q95": float(record[f"{variable}__q95"]),
                }
            )
    return rows


# =========================================================================
# 3.  Subgroup summaries (by country, by anchor)
# =========================================================================


def compute_subgroup_summaries(
    agents: pl.DataFrame,
    outcomes: dict[str, np.ndarray],
    group_col: str = "country",
) -> pl.DataFrame:
    """Compute outcome means by subgroup.

    Parameters
    ----------
    agents : pl.DataFrame
        Agent state with *group_col*.
    outcomes : dict[str, np.ndarray]
        Current outcome predictions.
    group_col : str
        Column to group by.

    Returns
    -------
    pl.DataFrame
        One row per group with mean values for each outcome.
    """
    # Add outcomes to a temporary DataFrame
    temp = agents.select(group_col)
    for name, values in outcomes.items():
        temp = temp.with_columns(pl.Series(name=name, values=values.astype(np.float64)))

    return temp.group_by(group_col).agg([pl.col(name).mean().alias(f"{name}_mean") for name in outcomes])


# =========================================================================
# 4.  Trajectory effects (baseline vs intervention)
# =========================================================================


def compute_trajectory_effects(
    baseline_traj: pl.DataFrame,
    intervention_traj: pl.DataFrame,
    outcome_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Compute per-tick treatment effects (intervention - baseline).

    Parameters
    ----------
    baseline_traj : pl.DataFrame
        Baseline outcome trajectory.
    intervention_traj : pl.DataFrame
        Intervention outcome trajectory.
    outcome_cols : list[str] or None
        Outcome columns to compute effects for.
        Defaults to all ``*_mean`` columns.

    Returns
    -------
    pl.DataFrame
        DataFrame with ``tick`` and ``{outcome}_effect`` columns.
    """
    if outcome_cols is None:
        outcome_cols = [c for c in baseline_traj.columns if c.endswith("_mean") and c != "tick"]

    effects = baseline_traj.select("tick").clone()
    for col in outcome_cols:
        if col in baseline_traj.columns and col in intervention_traj.columns:
            effect = intervention_traj[col].to_numpy() - baseline_traj[col].to_numpy()
            effect_name = col.replace("_mean", "_effect")
            effects = effects.with_columns(pl.Series(name=effect_name, values=effect))

    return effects


def compute_ate(
    baseline_results: list,
    intervention_results: list,
    outcome_col: str,
) -> ATEStats:
    """Compute average treatment effect across replications.

    Parameters
    ----------
    baseline_results : list[SimulationResult]
        Baseline replication results.
    intervention_results : list[SimulationResult]
        Intervention replication results.
    outcome_col : str
        Outcome column name (without ``_mean`` suffix).

    Returns
    -------
    ATEStats
        ``ate_mean``, ``ate_std``, ``peak_effect``,
        ``cumulative_effect``.
    """
    col = f"{outcome_col}_mean"
    ates: list[float] = []
    peaks: list[float] = []
    cumulatives: list[float] = []

    for base, interv in zip(baseline_results, intervention_results, strict=True):
        base_vals = base.outcome_trajectory[col].to_numpy()
        interv_vals = interv.outcome_trajectory[col].to_numpy()
        effect = interv_vals - base_vals
        ates.append(float(np.mean(effect)))
        peaks.append(float(np.max(np.abs(effect))))
        cumulatives.append(float(np.sum(effect)))

    return ATEStats(
        ate_mean=float(np.mean(ates)),
        ate_std=float(np.std(ates)),
        peak_effect=float(np.mean(peaks)),
        cumulative_effect=float(np.mean(cumulatives)),
    )


def build_behavioral_projection_weights(
    models: Mapping[str, BackboneModel],
    *,
    allowed_terms: set[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Build per-outcome linear-projection weights for ``Delta eta = beta^T Delta Z``.

    Parameters
    ----------
    models : Mapping[str, BackboneModel]
        Loaded backbone models keyed by outcome.
    allowed_terms : set[str] or None
        Optional whitelist of terms to include. Defaults to mutable constructs.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping ``outcome -> {term: beta/std}`` in runtime scaling units.
    """
    allowed = allowed_terms if allowed_terms is not None else set(MUTABLE_COLUMNS)
    weights_by_outcome: dict[str, dict[str, float]] = {}

    for outcome, model in models.items():
        if model.coefficients.size == 0 or len(model.terms) == 0:
            continue

        if model.term_stds is None or model.term_stds.shape[0] != len(model.terms):
            term_stds = np.ones(len(model.terms), dtype=np.float64)
        else:
            term_stds = model.term_stds.astype(np.float64)

        outcome_weights: dict[str, float] = {}
        for term, beta, std in zip(model.terms, model.coefficients, term_stds, strict=True):
            if term not in allowed:
                continue
            if term.endswith("_missing") or term.startswith("C("):
                continue
            if not np.isfinite(beta):
                continue
            if not np.isfinite(std) or np.isclose(std, 0.0):
                continue
            outcome_weights[term] = outcome_weights.get(term, 0.0) + float(beta / std)

        if outcome_weights:
            weights_by_outcome[outcome] = outcome_weights

    return weights_by_outcome


def compute_behavioral_projection_effects(
    baseline_mediator_traj: pl.DataFrame,
    intervention_mediator_traj: pl.DataFrame,
    projection_weights: Mapping[str, Mapping[str, float]],
    *,
    epsilon: float = 0.01,
) -> pl.DataFrame:
    """Compute behavioural-space projection effects from mediator trajectories.

    Parameters
    ----------
    baseline_mediator_traj : pl.DataFrame
        Baseline mediator trajectory with ``tick`` and ``{term}_mean`` columns.
    intervention_mediator_traj : pl.DataFrame
        Intervention mediator trajectory with matching columns.
    projection_weights : Mapping[str, Mapping[str, float]]
        Per-outcome term weights from :func:`build_behavioral_projection_weights`.
    epsilon : float
        Settling tolerance around final delta.

    Returns
    -------
    pl.DataFrame
        Long-format rows following the same schema as
        :func:`compute_effects_summary`, with outcomes named
        ``eta_projection::{outcome}``.
    """
    if not projection_weights:
        return pl.DataFrame()
    if "tick" not in baseline_mediator_traj.columns or "tick" not in intervention_mediator_traj.columns:
        return pl.DataFrame()

    rows: list[dict[str, float | int | str]] = []
    needed_cols = sorted({f"{term}_mean" for terms in projection_weights.values() for term in terms})
    available_cols = [
        col
        for col in needed_cols
        if col in intervention_mediator_traj.columns and col in baseline_mediator_traj.columns
    ]
    if not available_cols:
        return pl.DataFrame()

    aligned = (
        intervention_mediator_traj.select("tick", *available_cols)
        .join(
            baseline_mediator_traj.select(
                "tick",
                *[pl.col(col).alias(f"base__{col}") for col in available_cols],
            ),
            on="tick",
            how="inner",
        )
        .sort("tick")
    )
    ticks = aligned.get_column("tick").to_numpy().astype(np.int64)
    if ticks.size == 0:
        return pl.DataFrame()

    for outcome, term_weights in projection_weights.items():
        delta_eta = np.zeros(ticks.shape[0], dtype=np.float64)
        n_terms = 0
        for term, weight in term_weights.items():
            col = f"{term}_mean"
            base_col = f"base__{col}"
            if col not in aligned.columns or base_col not in aligned.columns:
                continue
            intervention_values = aligned[col].to_numpy().astype(np.float64)
            baseline_values = aligned[base_col].to_numpy().astype(np.float64)
            delta_eta += float(weight) * (intervention_values - baseline_values)
            n_terms += 1

        if n_terms == 0:
            continue

        delta_final = float(delta_eta[-1])
        delta_auc = float(np.trapezoid(delta_eta, ticks))
        delta_abs_auc = float(np.trapezoid(np.abs(delta_eta), ticks))
        peak_idx = int(np.argmax(np.abs(delta_eta)))
        peak_delta = float(delta_eta[peak_idx])
        time_to_peak = int(ticks[peak_idx])

        diff = np.abs(delta_eta - delta_final)
        settling_candidates = np.where(diff <= epsilon)[0]
        settling_time = int(ticks[settling_candidates[0]]) if settling_candidates.size > 0 else int(ticks[-1])

        first_diff = np.diff(delta_eta)
        volatility = float(np.std(first_diff)) if first_diff.size > 0 else 0.0

        rows.append(
            {
                "outcome": f"eta_projection::{outcome}",
                "delta_final": delta_final,
                "delta_auc": delta_auc,
                "delta_abs_auc": delta_abs_auc,
                "peak_delta": peak_delta,
                "time_to_peak": time_to_peak,
                "settling_time": settling_time,
                "volatility": volatility,
            }
        )

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)


def compute_effects_summary(
    baseline_traj: pl.DataFrame,
    intervention_traj: pl.DataFrame,
    baseline_mediator_traj: pl.DataFrame | None = None,
    intervention_mediator_traj: pl.DataFrame | None = None,
    projection_weights: Mapping[str, Mapping[str, float]] | None = None,
    *,
    epsilon: float = 0.01,
) -> pl.DataFrame:
    """Compute compact trajectory effect summaries vs baseline.

    Parameters
    ----------
    baseline_traj : pl.DataFrame
        Baseline trajectory containing ``tick`` and ``*_mean`` columns.
    intervention_traj : pl.DataFrame
        Intervention trajectory containing ``tick`` and ``*_mean`` columns.
    baseline_mediator_traj : pl.DataFrame or None
        Baseline mediator trajectory with ``{term}_mean`` columns.
    intervention_mediator_traj : pl.DataFrame or None
        Intervention mediator trajectory with ``{term}_mean`` columns.
    projection_weights : Mapping[str, Mapping[str, float]] or None
        Optional per-outcome projection weights. If provided with mediator
        trajectories, appends ``eta_projection::{outcome}`` rows.
    epsilon : float
        Settling tolerance around final delta.

    Returns
    -------
    pl.DataFrame
        Long-format rows with dynamic effect descriptors.
    """
    outcome_cols = [col for col in intervention_traj.columns if col.endswith("_mean") and col != "tick"]
    if not outcome_cols:
        return pl.DataFrame()

    rows: list[dict[str, float | int | str]] = []
    ticks = intervention_traj["tick"].to_numpy().astype(np.int64)

    for outcome_col in outcome_cols:
        if outcome_col not in baseline_traj.columns:
            continue
        baseline_values = baseline_traj[outcome_col].to_numpy().astype(np.float64)
        intervention_values = intervention_traj[outcome_col].to_numpy().astype(np.float64)
        delta = intervention_values - baseline_values
        if delta.size == 0:
            continue

        delta_final = float(delta[-1])
        delta_auc = float(np.trapezoid(delta, ticks))
        delta_abs_auc = float(np.trapezoid(np.abs(delta), ticks))
        peak_idx = int(np.argmax(np.abs(delta)))
        peak_delta = float(delta[peak_idx])
        time_to_peak = int(ticks[peak_idx])

        diff = np.abs(delta - delta_final)
        settling_candidates = np.where(diff <= epsilon)[0]
        settling_time = int(ticks[settling_candidates[0]]) if settling_candidates.size > 0 else int(ticks[-1])

        first_diff = np.diff(delta)
        volatility = float(np.std(first_diff)) if first_diff.size > 0 else 0.0

        rows.append(
            {
                "outcome": outcome_col.removesuffix("_mean"),
                "delta_final": delta_final,
                "delta_auc": delta_auc,
                "delta_abs_auc": delta_abs_auc,
                "peak_delta": peak_delta,
                "time_to_peak": time_to_peak,
                "settling_time": settling_time,
                "volatility": volatility,
            }
        )

    summary = pl.DataFrame(rows)
    if (
        baseline_mediator_traj is not None
        and intervention_mediator_traj is not None
        and projection_weights is not None
        and len(projection_weights) > 0
    ):
        projection = compute_behavioral_projection_effects(
            baseline_mediator_traj=baseline_mediator_traj,
            intervention_mediator_traj=intervention_mediator_traj,
            projection_weights=projection_weights,
            epsilon=epsilon,
        )
        if projection.height > 0:
            summary = pl.concat([summary, projection], how="vertical")
    return summary


def compute_local_shadow_target_diff_in_diff(
    *,
    baseline_population_t0: pl.DataFrame,
    baseline_population_final: pl.DataFrame,
    intervention_population_t0: pl.DataFrame,
    intervention_population_final: pl.DataFrame,
    seed_rule: str,
    high_degree_target_percentile: float = 75.0,
    high_exposure_target_percentile: float = 75.0,
    low_legitimacy_target_percentile: float = 25.0,
    outcomes: tuple[str, ...] = ("vax_willingness_T12", "mask_when_pressure_high"),
) -> pl.DataFrame:
    """Compute local-targeted diffusion via shadow-target Diff-in-Diff.

    The shadow target mask is reconstructed on ``baseline_population_t0`` using
    the intervention scenario's ``seed_rule`` and target percentiles.
    """
    if (
        baseline_population_t0.height == 0
        or baseline_population_final.height == 0
        or intervention_population_t0.height == 0
        or intervention_population_final.height == 0
    ):
        return pl.DataFrame()

    baseline_t0 = _sort_population_for_effects(baseline_population_t0)
    baseline_final = _sort_population_for_effects(baseline_population_final)
    intervention_t0 = _sort_population_for_effects(intervention_population_t0)
    intervention_final = _sort_population_for_effects(intervention_population_final)

    if (
        baseline_t0.height != baseline_final.height
        or baseline_t0.height != intervention_t0.height
        or baseline_t0.height != intervention_final.height
    ):
        return pl.DataFrame()

    mask = _compute_shadow_target_mask(
        baseline_t0,
        seed_rule=seed_rule,
        high_degree_target_percentile=high_degree_target_percentile,
        high_exposure_target_percentile=high_exposure_target_percentile,
        low_legitimacy_target_percentile=low_legitimacy_target_percentile,
    )
    if mask is None:
        return pl.DataFrame()
    targeted_n = int(np.count_nonzero(mask))
    other_n = int(np.count_nonzero(~mask))
    if targeted_n == 0 or other_n == 0:
        return pl.DataFrame()

    rows: list[dict[str, float | int | str]] = []
    for outcome in outcomes:
        if not _outcome_available_in_all(
            outcome,
            baseline_t0,
            baseline_final,
            intervention_t0,
            intervention_final,
        ):
            continue

        s0 = intervention_t0[outcome].cast(pl.Float64).to_numpy()
        s1 = intervention_final[outcome].cast(pl.Float64).to_numpy()
        b0 = baseline_t0[outcome].cast(pl.Float64).to_numpy()
        b1 = baseline_final[outcome].cast(pl.Float64).to_numpy()

        delta_s = s1 - s0
        delta_b = b1 - b0

        scenario_targeted_delta = float(np.mean(delta_s[mask]))
        scenario_other_delta = float(np.mean(delta_s[~mask]))
        baseline_targeted_delta = float(np.mean(delta_b[mask]))
        baseline_other_delta = float(np.mean(delta_b[~mask]))

        scenario_gap_change = scenario_targeted_delta - scenario_other_delta
        baseline_shadow_gap_change = baseline_targeted_delta - baseline_other_delta
        did = scenario_gap_change - baseline_shadow_gap_change

        rows.append(
            {
                "outcome": f"local_shadow_did::{outcome}",
                "delta_final": did,
                "delta_auc": did,
                "delta_abs_auc": abs(did),
                "peak_delta": did,
                "time_to_peak": 0,
                "settling_time": 0,
                "volatility": 0.0,
                "source_metric": "local_shadow_did",
                "base_outcome": outcome,
                "seed_rule": seed_rule,
                "targeted_n": targeted_n,
                "other_n": other_n,
                "targeted_share": float(targeted_n / (targeted_n + other_n)),
                "scenario_targeted_delta": scenario_targeted_delta,
                "scenario_other_delta": scenario_other_delta,
                "baseline_targeted_delta": baseline_targeted_delta,
                "baseline_other_delta": baseline_other_delta,
                "scenario_gap_change": scenario_gap_change,
                "baseline_shadow_gap_change": baseline_shadow_gap_change,
                "spillover_ratio_scenario": (
                    scenario_other_delta / scenario_targeted_delta
                    if abs(scenario_targeted_delta) > 1e-12
                    else float("nan")
                ),
            }
        )

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)


def compute_paired_dynamics_diagnostics(
    *,
    baseline_outcome_traj: pl.DataFrame,
    intervention_outcome_traj: pl.DataFrame,
    baseline_mediator_traj: pl.DataFrame | None = None,
    intervention_mediator_traj: pl.DataFrame | None = None,
    signals_realised: pl.DataFrame | None = None,
    baseline_incidence_traj: pl.DataFrame | None = None,
    intervention_incidence_traj: pl.DataFrame | None = None,
    vax_outcome: str = "vax_willingness_T12_mean",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build paired dynamics diagnostics (summary + per-tick trajectory).

    The trajectory output stores aligned deltas vs baseline for key channels.
    The summary output stores run-level gains (jump/AUC per unit input), peak
    timing, and simple persistence diagnostics.
    """
    if "tick" not in baseline_outcome_traj.columns or "tick" not in intervention_outcome_traj.columns:
        return pl.DataFrame(), pl.DataFrame()
    if vax_outcome not in baseline_outcome_traj.columns or vax_outcome not in intervention_outcome_traj.columns:
        return pl.DataFrame(), pl.DataFrame()

    traj = (
        intervention_outcome_traj.select(["tick", pl.col(vax_outcome).alias("vax_intervention")])
        .join(
            baseline_outcome_traj.select(["tick", pl.col(vax_outcome).alias("vax_baseline")]),
            on="tick",
            how="inner",
        )
        .sort("tick")
        .with_columns((pl.col("vax_intervention") - pl.col("vax_baseline")).alias("delta_vax"))
    )
    if traj.is_empty():
        return pl.DataFrame(), pl.DataFrame()

    if baseline_mediator_traj is not None and intervention_mediator_traj is not None:
        med_cols = [
            "institutional_trust_avg_mean",
            "covax_legitimacy_scepticism_idx_mean",
        ]
        available = [
            c for c in med_cols if c in baseline_mediator_traj.columns and c in intervention_mediator_traj.columns
        ]
        if available:
            med_int = intervention_mediator_traj.select(["tick", *available]).rename(
                {c: f"{c}_intervention" for c in available}
            )
            med_base = baseline_mediator_traj.select(["tick", *available]).rename(
                {c: f"{c}_baseline" for c in available}
            )
            med_join = med_int.join(med_base, on="tick", how="inner").sort("tick")
            for col in available:
                med_join = med_join.with_columns(
                    (pl.col(f"{col}_intervention") - pl.col(f"{col}_baseline")).alias(f"delta_{col}")
                )
            keep_cols = ["tick"] + [f"delta_{c}" for c in available]
            traj = traj.join(med_join.select(keep_cols), on="tick", how="left")

    if baseline_incidence_traj is not None and intervention_incidence_traj is not None:
        if (
            "mean_incidence" in baseline_incidence_traj.columns
            and "mean_incidence" in intervention_incidence_traj.columns
        ):
            inc = (
                intervention_incidence_traj.select(["tick", pl.col("mean_incidence").alias("inc_intervention")])
                .join(
                    baseline_incidence_traj.select(["tick", pl.col("mean_incidence").alias("inc_baseline")]),
                    on="tick",
                    how="inner",
                )
                .sort("tick")
                .with_columns((pl.col("inc_intervention") - pl.col("inc_baseline")).alias("delta_mean_incidence"))
                .select(["tick", "delta_mean_incidence"])
            )
            traj = traj.join(inc, on="tick", how="left")

    signal_by_tick = None
    if signals_realised is not None and not signals_realised.is_empty():
        required = {"tick", "intensity_realised_mean", "intensity_realised_nominal_mean"}
        if required.issubset(set(signals_realised.columns)):
            signal_by_tick = (
                signals_realised.group_by("tick")
                .agg(
                    [
                        pl.col("intensity_realised_mean").sum().alias("signal_realised_sum"),
                        pl.col("intensity_realised_mean").abs().sum().alias("signal_realised_abs_sum"),
                        pl.col("intensity_realised_nominal_mean").abs().sum().alias("signal_nominal_abs_sum"),
                    ]
                )
                .sort("tick")
            )
            traj = traj.join(signal_by_tick, on="tick", how="left").with_columns(
                [
                    pl.col("signal_realised_sum").fill_null(0.0),
                    pl.col("signal_realised_abs_sum").fill_null(0.0),
                    pl.col("signal_nominal_abs_sum").fill_null(0.0),
                ]
            )

    ticks = traj["tick"].to_numpy().astype(np.float64)
    delta_vax = traj["delta_vax"].to_numpy().astype(np.float64)
    abs_delta_vax = np.abs(delta_vax)

    active_tick_arr = np.array([], dtype=np.float64)
    mean_abs_input_active = 0.0
    mean_input_active = 0.0
    if signal_by_tick is not None and "signal_nominal_abs_sum" in traj.columns:
        signal_nominal_abs = traj["signal_nominal_abs_sum"].to_numpy().astype(np.float64)
        active_idx = np.where(signal_nominal_abs > 1e-12)[0]
        if active_idx.size > 0:
            active_tick_arr = ticks[active_idx]
            signal_abs = traj["signal_realised_abs_sum"].to_numpy().astype(np.float64)
            signal_sum = traj["signal_realised_sum"].to_numpy().astype(np.float64)
            mean_abs_input_active = float(np.mean(signal_abs[active_idx]))
            mean_input_active = float(np.mean(signal_sum[active_idx]))
            traj = traj.with_columns(
                [
                    pl.when(pl.col("tick").is_in(active_tick_arr.tolist()))
                    .then(pl.lit(1))
                    .otherwise(pl.lit(0))
                    .alias("is_signal_active_tick"),
                ]
            )
        else:
            traj = traj.with_columns(pl.lit(0).alias("is_signal_active_tick"))
    else:
        traj = traj.with_columns(pl.lit(0).alias("is_signal_active_tick"))

    peak_idx = int(np.argmax(abs_delta_vax))
    peak_abs = float(abs_delta_vax[peak_idx])
    peak_tick = int(ticks[peak_idx])
    start_idx = (
        int(np.where(traj["is_signal_active_tick"].to_numpy().astype(bool))[0][0]) if active_tick_arr.size > 0 else 0
    )
    jump_idx = min(start_idx + 1, delta_vax.shape[0] - 1)
    jump_next_tick = float(delta_vax[jump_idx])
    vax_auc = float(np.trapezoid(delta_vax, ticks))
    vax_abs_auc = float(np.trapezoid(abs_delta_vax, ticks))
    gain_jump = float(jump_next_tick / mean_abs_input_active) if mean_abs_input_active > 0.0 else 0.0
    gain_abs_auc = float(vax_abs_auc / mean_abs_input_active) if mean_abs_input_active > 0.0 else 0.0

    summary: dict[str, float | int | str] = {
        "vax_outcome": vax_outcome,
        "n_ticks": int(traj.height),
        "signal_start_tick": int(active_tick_arr.min()) if active_tick_arr.size > 0 else -1,
        "signal_end_tick": int(active_tick_arr.max()) if active_tick_arr.size > 0 else -1,
        "signal_ticks_active": int(active_tick_arr.size),
        "input_realised_mean_active": mean_input_active,
        "input_realised_abs_mean_active": mean_abs_input_active,
        "vax_jump_next_tick": jump_next_tick,
        "vax_peak_abs": peak_abs,
        "vax_peak_tick": peak_tick,
        "vax_auc": vax_auc,
        "vax_abs_auc": vax_abs_auc,
        "gain_jump_per_abs_input": gain_jump,
        "gain_abs_auc_per_abs_input": gain_abs_auc,
        "duration_abs_gt_0p01": int(np.sum(abs_delta_vax > 0.01)),
        "duration_abs_gt_0p005": int(np.sum(abs_delta_vax > 0.005)),
        "duration_abs_gt_0p002": int(np.sum(abs_delta_vax > 0.002)),
    }

    if "delta_institutional_trust_avg_mean" in traj.columns:
        dtrust = traj["delta_institutional_trust_avg_mean"].to_numpy().astype(np.float64)
        summary["dtrust_auc"] = float(np.trapezoid(dtrust, ticks))
        summary["dtrust_peak_abs"] = float(np.max(np.abs(dtrust)))
    if "delta_covax_legitimacy_scepticism_idx_mean" in traj.columns:
        dlegit = traj["delta_covax_legitimacy_scepticism_idx_mean"].to_numpy().astype(np.float64)
        summary["dlegit_auc"] = float(np.trapezoid(dlegit, ticks))
        summary["dlegit_peak_abs"] = float(np.max(np.abs(dlegit)))
    if "delta_mean_incidence" in traj.columns:
        dinc = traj["delta_mean_incidence"].fill_null(0.0).to_numpy().astype(np.float64)
        summary["dinc_auc"] = float(np.trapezoid(dinc, ticks))
        summary["dinc_peak_abs"] = float(np.max(np.abs(dinc)))

    return pl.DataFrame([summary]), traj


def compute_shock_recovery_metrics(trajectory: pl.DataFrame) -> pl.DataFrame:
    """Compute peak, half-life, and residual scar from mean trajectories."""
    mean_cols = [c for c in trajectory.columns if c.endswith("_mean") and c != "tick"]
    if not mean_cols or "tick" not in trajectory.columns:
        return pl.DataFrame()

    ticks = trajectory["tick"].to_numpy().astype(np.int64)
    rows: list[dict[str, float | int | str]] = []
    for col in mean_cols:
        values = trajectory[col].to_numpy().astype(np.float64)
        if values.size == 0:
            continue
        baseline = float(values[0])
        delta = values - baseline
        peak_idx = int(np.argmax(np.abs(delta)))
        peak = float(delta[peak_idx])
        half_target = 0.5 * abs(peak)
        post = np.abs(delta[peak_idx:])
        half_idx = (
            int(np.where(post <= half_target)[0][0]) + peak_idx if np.any(post <= half_target) else len(delta) - 1
        )
        rows.append(
            {
                "metric": "shock_recovery",
                "variable": col,
                "peak_effect": peak,
                "peak_tick": int(ticks[peak_idx]),
                "half_life_ticks": int(ticks[half_idx] - ticks[peak_idx]),
                "residual_scar": float(delta[-1]),
            }
        )
    return pl.DataFrame(rows)


def compute_diffusion_metrics(distance_trajectory: pl.DataFrame) -> pl.DataFrame:
    """Compute diffusion timing metrics from distance-binned trajectories."""
    required = {"tick", "distance", "mean_vax_willingness"}
    if not required.issubset(distance_trajectory.columns):
        return pl.DataFrame()

    rows: list[dict[str, float | int | str]] = []
    for distance in sorted(distance_trajectory["distance"].unique().to_list()):
        subset = distance_trajectory.filter(pl.col("distance") == distance).sort("tick")
        values = subset["mean_vax_willingness"].to_numpy().astype(np.float64)
        ticks = subset["tick"].to_numpy().astype(np.int64)
        if values.size == 0:
            continue
        peak_idx = int(np.argmax(values))
        rows.append(
            {
                "metric": "diffusion_distance_peak",
                "distance": int(distance),
                "time_to_peak": int(ticks[peak_idx]),
                "peak_value": float(values[peak_idx]),
            }
        )
    return pl.DataFrame(rows)


def compute_closed_loop_signatures(
    incidence_trajectory: pl.DataFrame,
    outcome_trajectory: pl.DataFrame,
) -> pl.DataFrame:
    """Compute incidence-protection coupling signatures."""
    willingness_col = _first_present_name(
        _VAX_WILLINGNESS_MEAN_CANDIDATES,
        outcome_trajectory.columns,
    )
    if "mean_incidence" not in incidence_trajectory.columns or willingness_col is None:
        return pl.DataFrame()

    merged = incidence_trajectory.join(
        outcome_trajectory.select("tick", willingness_col),
        on="tick",
        how="inner",
    )
    if merged.height == 0:
        return pl.DataFrame()

    incidence = merged["mean_incidence"].to_numpy().astype(np.float64)
    protection = merged[willingness_col].to_numpy().astype(np.float64)

    if incidence.size <= 1:
        delta_corr = 0.0
        delta_slope = 0.0
        rolling_delta_corr = 0.0
    else:
        protection_t = protection[:-1]
        delta_inc_next = incidence[1:] - incidence[:-1]
        delta_corr = _nan_corr(protection_t, delta_inc_next)
        if np.isnan(delta_corr):
            delta_corr = 0.0

        finite = np.isfinite(protection_t) & np.isfinite(delta_inc_next)
        if int(np.sum(finite)) <= 1:
            delta_slope = 0.0
        else:
            x = protection_t[finite]
            y = delta_inc_next[finite]
            x_centered = x - float(np.mean(x))
            denom = float(np.sum(x_centered * x_centered))
            if denom <= 0.0:
                delta_slope = 0.0
            else:
                delta_slope = float(np.sum(x_centered * y) / denom)

        n_window = int(min(6, delta_inc_next.size))
        if n_window < 3:
            rolling_delta_corr = float(delta_corr)
        else:
            window_corrs: list[float] = []
            for start in range(0, delta_inc_next.size - n_window + 1):
                end = start + n_window
                corr = _nan_corr(protection_t[start:end], delta_inc_next[start:end])
                if np.isfinite(corr):
                    window_corrs.append(float(corr))
            rolling_delta_corr = float(np.mean(window_corrs)) if window_corrs else float(delta_corr)

    peak_idx = int(np.argmax(incidence))
    rise_corr = _nan_corr(incidence[: peak_idx + 1], protection[: peak_idx + 1]) if peak_idx >= 1 else 0.0
    fall_corr = _nan_corr(incidence[peak_idx:], protection[peak_idx:]) if (incidence.size - peak_idx) >= 2 else 0.0
    rise_corr = 0.0 if np.isnan(rise_corr) else float(rise_corr)
    fall_corr = 0.0 if np.isnan(fall_corr) else float(fall_corr)

    return pl.DataFrame(
        [
            {
                "metric": "closed_loop_signature",
                "protection_to_next_delta_incidence_corr": float(delta_corr),
                "protection_to_next_delta_incidence_slope": float(delta_slope),
                "rolling_delta_incidence_corr_mean": float(rolling_delta_corr),
                "hysteresis_delta": float(rise_corr - fall_corr),
            }
        ]
    )


def assemble_parity_diagnostics_table(
    outcome_trajectory: pl.DataFrame,
    incidence_trajectory: pl.DataFrame,
    distance_trajectory: pl.DataFrame,
) -> pl.DataFrame:
    """Assemble long-format diagnostics table for runtime artifacts."""
    parts: list[pl.DataFrame] = []
    shock = compute_shock_recovery_metrics(outcome_trajectory)
    if shock.height > 0:
        parts.append(shock)
    diffusion = compute_diffusion_metrics(distance_trajectory)
    if diffusion.height > 0:
        parts.append(diffusion)
    closed_loop = compute_closed_loop_signatures(incidence_trajectory, outcome_trajectory)
    if closed_loop.height > 0:
        parts.append(closed_loop)
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="diagonal_relaxed")


def _sort_population_for_effects(population: pl.DataFrame) -> pl.DataFrame:
    if "unique_id" in population.columns:
        return population.sort("unique_id")
    return population.clone()


def _outcome_available_in_all(outcome: str, *frames: pl.DataFrame) -> bool:
    return all(outcome in frame.columns for frame in frames)


def _compute_low_legitimacy_shadow_mask(
    agents_t0: pl.DataFrame,
    *,
    low_legitimacy_target_percentile: float,
) -> np.ndarray | None:
    n = agents_t0.height
    if n == 0:
        return None

    columns = set(agents_t0.columns)
    legitimacy_score: np.ndarray | None = None
    if "legitimacy_engine" in columns:
        legitimacy_score = agents_t0["legitimacy_engine"].cast(pl.Float64).to_numpy()
    elif "covax_legitimacy_scepticism_idx" in columns:
        scepticism = agents_t0["covax_legitimacy_scepticism_idx"].cast(pl.Float64).to_numpy()
        legitimacy_score = -scepticism
    if legitimacy_score is None:
        return None

    cutoff = float(np.nanpercentile(legitimacy_score, low_legitimacy_target_percentile))
    if not np.isfinite(cutoff):
        return None
    return legitimacy_score <= cutoff


def _compute_shadow_target_mask(
    agents_t0: pl.DataFrame,
    *,
    seed_rule: str,
    high_degree_target_percentile: float,
    high_exposure_target_percentile: float,
    low_legitimacy_target_percentile: float,
) -> np.ndarray | None:
    n = agents_t0.height
    if n == 0:
        return None

    if seed_rule == "high_degree":
        if "core_degree" not in agents_t0.columns:
            return None
        degree = agents_t0["core_degree"].cast(pl.Float64).to_numpy()
        cutoff = float(np.percentile(degree, high_degree_target_percentile))
        return degree >= cutoff

    if seed_rule in {"high_exposure", "high_degree_high_exposure", "community_targeted"}:
        score = (
            agents_t0["activity_score"].cast(pl.Float64).to_numpy()
            if "activity_score" in agents_t0.columns
            else np.zeros(n, dtype=np.float64)
        )
        local = np.ones(n, dtype=bool)
        if seed_rule == "community_targeted" and "locality_id" in agents_t0.columns:
            communities = agents_t0["locality_id"].cast(pl.Utf8).to_numpy().astype(str)
            unique = np.unique(communities)
            means_by_comm: dict[str, float] = {}
            for comm in unique:
                idx = communities == comm
                if np.any(idx):
                    means_by_comm[comm] = float(np.nanmean(score[idx]))
            if means_by_comm:
                target_community = max(means_by_comm.items(), key=lambda item: item[1])[0]
                local = communities == target_community
        local_scores = score[local]
        if local_scores.size == 0:
            return None
        threshold = float(np.nanpercentile(local_scores, high_exposure_target_percentile))
        mask = np.zeros(n, dtype=bool)
        mask[local] = score[local] >= threshold
        return mask

    if seed_rule == "low_legitimacy":
        return _compute_low_legitimacy_shadow_mask(
            agents_t0,
            low_legitimacy_target_percentile=low_legitimacy_target_percentile,
        )

    if seed_rule == "low_legitimacy_high_degree":
        low_legitimacy_mask = _compute_low_legitimacy_shadow_mask(
            agents_t0,
            low_legitimacy_target_percentile=low_legitimacy_target_percentile,
        )
        if low_legitimacy_mask is None:
            return None
        if "core_degree" not in agents_t0.columns:
            return low_legitimacy_mask
        degree = agents_t0["core_degree"].cast(pl.Float64).to_numpy()
        cutoff = float(np.percentile(degree, high_degree_target_percentile))
        return low_legitimacy_mask & (degree >= cutoff)

    return None
