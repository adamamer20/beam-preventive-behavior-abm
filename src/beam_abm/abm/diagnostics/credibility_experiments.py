"""Controlled diagnostic worlds for credibility diagnostics."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from beam_abm.abm.scenario_defs import SCENARIO_LIBRARY, ScenarioConfig

from .credibility_instrumentation import run_instrumented_detailed
from .credibility_shared import (
    LONG_HORIZON_STEPS,
    POLICY_RANK_TIE_TOLERANCE,
    POLICY_RANKING_SCENARIOS,
    PULSE_MILD_MAX_THRESHOLD,
    PULSE_MILD_RANGE_THRESHOLD,
    PULSE_MILD_STD_THRESHOLD,
    SharedData,
    build_model,
)


def run_2x2_social_ablation(shared: SharedData) -> pl.DataFrame:
    """D6: Run 2x2 social ablation grid: peer_pull x observed_norms."""
    configs = [
        ("both_off", False, False),
        ("peer_only", True, False),
        ("obs_only", False, True),
        ("both_on", True, True),
    ]
    all_records: list[dict[str, Any]] = []

    for label, peer_on, obs_on in configs:
        logger.info(f"2x2 ablation: {label} (peer={peer_on}, obs={obs_on})")
        scenario = ScenarioConfig(
            name=f"ablation_{label}",
            family="diagnostic",
            communication_enabled=(peer_on or obs_on),
            peer_pull_enabled=peer_on,
            observed_norms_enabled=obs_on,
            enable_endogenous_loop=False,
            incidence_mode="none",
        )
        for attractor_on in [False, True]:
            att_label = "att_on" if attractor_on else "att_off"
            full_label = f"{label}_{att_label}"
            substep_recs, tick_recs = run_instrumented_detailed(
                shared, scenario, attractor_on=attractor_on, label=full_label
            )
            for rec in tick_recs:
                rec["cell"] = label
                rec["attractor"] = att_label
                all_records.append(rec)

    return pl.DataFrame(all_records)


def run_pulse_response(shared: SharedData) -> pl.DataFrame:
    """D5: Run baseline_importation with/without pulse to measure response."""

    # Baseline importation (canonical)
    scenario_base = ScenarioConfig(
        name="pulse_baseline",
        family="diagnostic",
        communication_enabled=True,
        peer_pull_enabled=True,
        observed_norms_enabled=True,
        enable_endogenous_loop=True,
        incidence_mode="importation",
    )

    # Baseline importation with pulse
    scenario_pulse = ScenarioConfig(
        name="pulse_test",
        family="diagnostic",
        communication_enabled=True,
        peer_pull_enabled=True,
        observed_norms_enabled=True,
        enable_endogenous_loop=True,
        incidence_mode="importation",
    )

    all_records: list[dict[str, Any]] = []

    for scenario, label in [(scenario_base, "baseline_import"), (scenario_pulse, "pulse_import")]:
        model = build_model(shared, scenario, attractor_on=True)
        # For the pulse scenario, manually inject a pulse at tick 3
        if label == "pulse_import":
            model.cfg.constants.importation_pulse_amplitude = 0.05  # big pulse
            model.cfg.constants.importation_pulse_period_ticks = 6
            model.cfg.constants.importation_pulse_start_tick = 3

        for tick in range(int(shared.cfg.n_steps)):
            pre_incidence = dict(model.community_incidence)
            pre_mediators = {}
            for col in ["covid_perceived_danger_T12", "disease_fear_avg", "infection_likelihood_avg"]:
                if col in model.agents.df.columns:
                    pre_mediators[col] = float(model.agents.df[col].cast(pl.Float64).to_numpy().mean())

            model.step()

            post_incidence = dict(model.community_incidence)
            post_mediators = {}
            for col in ["covid_perceived_danger_T12", "disease_fear_avg", "infection_likelihood_avg"]:
                if col in model.agents.df.columns:
                    post_mediators[col] = float(model.agents.df[col].cast(pl.Float64).to_numpy().mean())

            post_outcomes = {}
            if model.latest_outcomes:
                for oname, ovals in model.latest_outcomes.items():
                    post_outcomes[f"outcome_{oname}_mean"] = float(np.mean(ovals))

            mean_inc_pre = float(np.mean(list(pre_incidence.values())))
            mean_inc_post = float(np.mean(list(post_incidence.values())))

            rec = {
                "tick": tick,
                "label": label,
                "mean_incidence": mean_inc_post,
                "delta_incidence": mean_inc_post - mean_inc_pre,
            }
            for col, val in post_mediators.items():
                rec[f"post_{col}"] = val
                if col in pre_mediators:
                    rec[f"delta_{col}"] = val - pre_mediators[col]
            rec.update(post_outcomes)
            all_records.append(rec)

    return pl.DataFrame(all_records)


def run_long_horizon_stationarity(
    shared: SharedData,
    *,
    steps: int = LONG_HORIZON_STEPS,
) -> pl.DataFrame:
    """Run a long-horizon stability check for descriptive norms."""
    scenario = ScenarioConfig(
        name="social_only_long_horizon",
        family="diagnostic",
        communication_enabled=True,
        peer_pull_enabled=True,
        observed_norms_enabled=True,
        enable_endogenous_loop=False,
        incidence_mode="none",
    )

    records: list[dict[str, Any]] = []
    for attractor_on in (False, True):
        model = build_model(shared, scenario, attractor_on=attractor_on)
        if "social_norms_descriptive_vax" not in model.agents.df.columns:
            continue

        start_mean = float(model.agents.df["social_norms_descriptive_vax"].cast(pl.Float64).to_numpy().mean())
        for _ in range(steps):
            model.step()
        end_mean = float(model.agents.df["social_norms_descriptive_vax"].cast(pl.Float64).to_numpy().mean())

        drift_total = end_mean - start_mean
        drift_per_month = drift_total / float(max(steps, 1))
        records.append(
            {
                "horizon_steps": int(steps),
                "attractor_on": bool(attractor_on),
                "start_mean": start_mean,
                "end_mean": end_mean,
                "drift_total": float(drift_total),
                "drift_per_month": float(drift_per_month),
            }
        )

    return pl.DataFrame(records)


def summarize_pulse_band(pulse_df: pl.DataFrame) -> dict[str, Any]:
    """Summarize pulse incidence variability and mild-band checks."""
    out: dict[str, Any] = {}
    if pulse_df.height == 0:
        return out

    baseline_rows = pulse_df.filter(pl.col("label") == "baseline_import")
    pulse_rows = pulse_df.filter(pl.col("label") == "pulse_import")
    if baseline_rows.height == 0 or pulse_rows.height == 0:
        return out

    baseline = baseline_rows["mean_incidence"].to_numpy().astype(np.float64)
    pulse = pulse_rows["mean_incidence"].to_numpy().astype(np.float64)
    pulse_delta = pulse - baseline

    out["baseline_incidence_std"] = float(np.std(baseline))
    out["baseline_incidence_min"] = float(np.min(baseline))
    out["baseline_incidence_max"] = float(np.max(baseline))
    out["baseline_incidence_range"] = float(np.max(baseline) - np.min(baseline))

    out["pulse_incidence_std"] = float(np.std(pulse))
    out["pulse_incidence_min"] = float(np.min(pulse))
    out["pulse_incidence_max"] = float(np.max(pulse))
    out["pulse_incidence_range"] = float(np.max(pulse) - np.min(pulse))

    out["pulse_peak_delta_vs_baseline"] = float(np.max(pulse - baseline))
    out["pulse_delta_mean"] = float(np.mean(pulse_delta))
    out["pulse_delta_std"] = float(np.std(pulse_delta))
    out["pulse_delta_range"] = float(np.max(pulse_delta) - np.min(pulse_delta))

    # Use incremental effect variability (pulse-baseline) instead of absolute
    # pulse trajectory variance, which is confounded by shared trend.
    out["pulse_mild_band_std"] = out["pulse_delta_std"] <= PULSE_MILD_STD_THRESHOLD
    out["pulse_mild_band_range"] = out["pulse_incidence_range"] <= PULSE_MILD_RANGE_THRESHOLD
    out["pulse_mild_band_max"] = out["pulse_incidence_max"] <= PULSE_MILD_MAX_THRESHOLD
    out["pulse_mild_band_all"] = (
        out["pulse_mild_band_std"] and out["pulse_mild_band_range"] and out["pulse_mild_band_max"]
    )

    return out


def _pairwise_rank_agreement(
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    *,
    tie_tol: float,
) -> tuple[float, int, int]:
    """Return tie-aware pairwise ordering agreement.

    Pairs that are ties (within ``tie_tol``) in either horizon are excluded
    from decisive-pair scoring to avoid unstable near-tie penalties.
    """
    common = sorted(set(scores_a).intersection(scores_b))
    if len(common) < 2:
        return float("nan"), 0, 0

    total_pairs = 0
    decisive_pairs = 0
    agree = 0
    for idx, left in enumerate(common):
        for right in common[idx + 1 :]:
            total_pairs += 1
            diff_a = float(scores_a[left] - scores_a[right])
            diff_b = float(scores_b[left] - scores_b[right])
            if abs(diff_a) <= tie_tol or abs(diff_b) <= tie_tol:
                continue
            decisive_pairs += 1
            if (diff_a > 0.0 and diff_b > 0.0) or (diff_a < 0.0 and diff_b < 0.0):
                agree += 1

    if decisive_pairs == 0:
        return 1.0, decisive_pairs, total_pairs
    return float(agree / decisive_pairs), decisive_pairs, total_pairs


def _top_tied_scenarios(scores: dict[str, float], *, tie_tol: float) -> list[str]:
    """Return scenarios tied for top score within tolerance."""
    if not scores:
        return []
    ranking = sorted(scores.keys(), key=lambda name: scores[name], reverse=True)
    top_score = float(scores[ranking[0]])
    return [name for name in ranking if (top_score - float(scores[name])) <= tie_tol]


def run_policy_ranking_stability(
    shared: SharedData,
    *,
    horizons: tuple[int, int] | None = None,
    scenario_names: tuple[str, ...] = POLICY_RANKING_SCENARIOS,
    tie_tolerance: float = POLICY_RANK_TIE_TOLERANCE,
) -> dict[str, Any]:
    """Check whether policy ranking changes between short and long horizons."""
    from beam_abm.abm.engine import _resolve_vax_outcome_key

    if horizons is None:
        horizons = (int(shared.cfg.n_steps), LONG_HORIZON_STEPS)

    ranking_by_horizon: dict[str, dict[str, Any]] = {}
    for horizon in horizons:
        scores: dict[str, float] = {}
        for scenario_name in scenario_names:
            if scenario_name not in SCENARIO_LIBRARY:
                continue
            scenario = copy.deepcopy(SCENARIO_LIBRARY[scenario_name])
            model = build_model(shared, scenario, attractor_on=True)
            for _ in range(int(horizon)):
                model.step()

            outcomes = model.latest_outcomes or {}
            vax_key = _resolve_vax_outcome_key(outcomes)
            if vax_key is None:
                continue
            scores[scenario_name] = float(np.mean(outcomes[vax_key]))

        ranking = sorted(scores.keys(), key=lambda name: scores[name], reverse=True)
        ranking_by_horizon[str(horizon)] = {
            "scores": scores,
            "ranking": ranking,
        }

    short_key = str(horizons[0])
    long_key = str(horizons[1])
    short_info = ranking_by_horizon.get(short_key, {})
    long_info = ranking_by_horizon.get(long_key, {})
    short_rank = short_info.get("ranking", [])
    long_rank = long_info.get("ranking", [])
    short_scores = short_info.get("scores", {})
    long_scores = long_info.get("scores", {})
    top_tied_short = _top_tied_scenarios(short_scores, tie_tol=float(tie_tolerance))
    top_tied_long = _top_tied_scenarios(long_scores, tie_tol=float(tie_tolerance))
    agreement, decisive_pairs, total_pairs = _pairwise_rank_agreement(
        short_scores,
        long_scores,
        tie_tol=float(tie_tolerance),
    )

    return {
        "horizons": [int(horizons[0]), int(horizons[1])],
        "scenario_names": list(scenario_names),
        "ranking_by_horizon": ranking_by_horizon,
        "tie_tolerance": float(tie_tolerance),
        "top_scenario_short": short_rank[0] if short_rank else None,
        "top_scenario_long": long_rank[0] if long_rank else None,
        "top_tied_short": top_tied_short,
        "top_tied_long": top_tied_long,
        "top_scenario_stable": bool(set(top_tied_short).intersection(top_tied_long)),
        "full_ranking_stable": bool(np.isfinite(agreement) and agreement == 1.0),
        "pairwise_rank_agreement": agreement,
        "pairwise_rank_decisive_pairs": int(decisive_pairs),
        "pairwise_rank_total_pairs": int(total_pairs),
    }
