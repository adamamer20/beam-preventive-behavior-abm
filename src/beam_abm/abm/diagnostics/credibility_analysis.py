"""Decision-tree analysis and reporting for credibility diagnostics."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .credibility_experiments import summarize_pulse_band
from .credibility_shared import (
    DRIFT_ABS_THRESHOLD,
    DRIFT_PER_MONTH_THRESHOLD,
    LONG_HORIZON_STEPS,
    PULSE_MILD_MAX_THRESHOLD,
    PULSE_MILD_RANGE_THRESHOLD,
    PULSE_MILD_STD_THRESHOLD,
    TRACKED_CONSTRUCTS,
)


def analyze_decision_tree(
    substep_df: pl.DataFrame,
    tick_df: pl.DataFrame,
    ablation_df: pl.DataFrame,
    *,
    short_horizon_steps: int,
    pulse_df: pl.DataFrame | None = None,
    long_horizon_df: pl.DataFrame | None = None,
    policy_ranking: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze diagnostic outputs and classify the drift bucket."""
    findings: dict[str, Any] = {
        "buckets": [],
        "fix_recommendations": [],
        "invariant_checks": {},
    }

    # --- Check 1: Drift in descriptive norms under attractor OFF ---
    desc_off = tick_df.filter(
        (pl.col("construct") == "social_norms_descriptive_vax") & (pl.col("attractor_on") == False)  # noqa
    )
    if desc_off.height > 0:
        cumulative_drift = desc_off["drift"].sum()
        drift_per_month = float(cumulative_drift) / float(max(short_horizon_steps, 1))
        findings["desc_norm_drift_attractor_off"] = float(cumulative_drift)
        findings["desc_norm_drift_per_month_attractor_off"] = drift_per_month
        findings["invariant_checks"]["stationarity_desc_off"] = abs(float(cumulative_drift)) < DRIFT_ABS_THRESHOLD
        findings["invariant_checks"]["stationarity_desc_off_per_month"] = (
            abs(drift_per_month) < DRIFT_PER_MONTH_THRESHOLD
        )

    # --- Check 2: Drift in descriptive norms under attractor ON ---
    desc_on = tick_df.filter(
        (pl.col("construct") == "social_norms_descriptive_vax") & (pl.col("attractor_on") == True)  # noqa
    )
    if desc_on.height > 0:
        cumulative_drift = desc_on["drift"].sum()
        drift_per_month = float(cumulative_drift) / float(max(short_horizon_steps, 1))
        findings["desc_norm_drift_attractor_on"] = float(cumulative_drift)
        findings["desc_norm_drift_per_month_attractor_on"] = drift_per_month
        findings["invariant_checks"]["stationarity_desc_on"] = abs(float(cumulative_drift)) < DRIFT_ABS_THRESHOLD
        findings["invariant_checks"]["stationarity_desc_on_per_month"] = (
            abs(drift_per_month) < DRIFT_PER_MONTH_THRESHOLD
        )

    findings["short_horizon_steps"] = int(short_horizon_steps)
    findings["drift_abs_threshold"] = float(DRIFT_ABS_THRESHOLD)
    findings["drift_per_month_threshold"] = float(DRIFT_PER_MONTH_THRESHOLD)

    if long_horizon_df is not None and long_horizon_df.height > 0:
        for row in long_horizon_df.iter_rows(named=True):
            steps = int(row["horizon_steps"])
            attractor_on = bool(row["attractor_on"])
            mode = "on" if attractor_on else "off"
            drift_total = float(row["drift_total"])
            drift_per_month = float(row["drift_per_month"])

            findings[f"desc_norm_drift_attractor_{mode}_{steps}"] = drift_total
            findings[f"desc_norm_drift_per_month_attractor_{mode}_{steps}"] = drift_per_month
            findings["invariant_checks"][f"stationarity_desc_{mode}_{steps}"] = abs(drift_total) < DRIFT_ABS_THRESHOLD
            findings["invariant_checks"][f"stationarity_desc_{mode}_{steps}_per_month"] = (
                abs(drift_per_month) < DRIFT_PER_MONTH_THRESHOLD
            )

    # --- 2x2 ablation analysis ---
    if ablation_df.height > 0:
        for cell in ["both_off", "peer_only", "obs_only", "both_on"]:
            for att in ["att_off", "att_on"]:
                subset = ablation_df.filter(
                    (pl.col("cell") == cell)
                    & (pl.col("attractor") == att)
                    & (pl.col("construct") == "social_norms_descriptive_vax")
                )
                if subset.height > 0:
                    cum_drift = float(subset["drift"].sum())
                    var_change = float(subset.tail(1)["std_post"][0]) - float(subset.head(1)["std_post"][0])
                    findings[f"ablation_{cell}_{att}_drift"] = cum_drift
                    findings[f"ablation_{cell}_{att}_var_change"] = var_change

    # --- Decision tree classification ---
    # Check if drift dominated by observed norms
    desc_substeps = substep_df.filter(pl.col("construct") == "social_norms_descriptive_vax")
    if desc_substeps.height > 0:
        mean_d_obs = float(desc_substeps["delta_observed_norm"].mean())
        mean_d_att = float(desc_substeps["delta_attractor"].mean())
        mean_d_peer = float(desc_substeps["delta_peer_pull"].mean())
        mean_d_clip = float(desc_substeps["delta_clip"].mean())

        findings["mean_delta_obs"] = mean_d_obs
        findings["mean_delta_att"] = mean_d_att
        findings["mean_delta_peer"] = mean_d_peer
        findings["mean_delta_clip"] = mean_d_clip

        if abs(mean_d_att) > abs(mean_d_obs) and abs(mean_d_att) > 0.01:
            findings["buckets"].append("relax_attractor_dominant")
            if abs(mean_d_clip) > 0.5 * abs(mean_d_att):
                findings["buckets"].append("relax_clip_cancellation")
                findings["fix_recommendations"].append("relaxation_attractor_correction")
            else:
                findings["fix_recommendations"].append("relaxation_attractor_correction")

        if abs(mean_d_obs) > 0.001 and "mean_observed_share" in desc_substeps.columns:
            obs_hat_cols = desc_substeps.select(["mean_observed_share", "mean_share_hat0"]).drop_nulls()
            if obs_hat_cols.height > 0:
                mean_obs = float(obs_hat_cols["mean_observed_share"].mean())
                mean_hat0 = float(obs_hat_cols["mean_share_hat0"].mean())
                obs_hat_gap = mean_obs - mean_hat0
                findings["perceived_share_gap"] = obs_hat_gap
                if abs(obs_hat_gap) > 0.05:
                    findings["buckets"].append("perceived_share_input_bias")
                    findings["fix_recommendations"].append("perceived_share_anchoring")

            if "pop_mean_behavior" in desc_substeps.columns and "mean_obs_share" in desc_substeps.columns:
                sampling_cols = desc_substeps.select(["pop_mean_behavior", "mean_obs_share"]).drop_nulls()
                if sampling_cols.height > 0:
                    pop_mean = float(sampling_cols["pop_mean_behavior"].mean())
                    obs_mean = float(sampling_cols["mean_obs_share"].mean())
                    sampling_bias = obs_mean - pop_mean
                    findings["sampling_bias"] = sampling_bias
                    if abs(sampling_bias) > 0.05:
                        findings["buckets"].append("sampling_bias")
                        findings["fix_recommendations"].append("sampling_correction")

        if abs(mean_d_peer) > 0.01:
            # Check 2x2: does peer-only drift?
            peer_only_drift = findings.get("ablation_peer_only_att_off_drift", 0.0)
            if abs(peer_only_drift) > 0.5:
                findings["buckets"].append("peer_pull_not_mean_preserving")
                findings["fix_recommendations"].append("peer_pull_correction")

    # --- Invariant: no large relax+clip cancellation ---
    if desc_substeps.height > 0 and abs(mean_d_att) > 0.0 and abs(mean_d_clip) > 0.0:
        cancellation_ratio = abs(mean_d_clip) / max(abs(mean_d_att), 1e-10)
        findings["relax_clip_cancellation_ratio"] = cancellation_ratio
        findings["invariant_checks"]["no_large_relax_clip"] = cancellation_ratio < 0.5

    if pulse_df is not None and pulse_df.height > 0:
        pulse_stats = summarize_pulse_band(pulse_df)
        findings["pulse_stats"] = pulse_stats
        if pulse_stats:
            findings["invariant_checks"]["pulse_mild_band_std"] = bool(pulse_stats.get("pulse_mild_band_std", False))
            findings["invariant_checks"]["pulse_mild_band_range"] = bool(
                pulse_stats.get("pulse_mild_band_range", False)
            )
            findings["invariant_checks"]["pulse_mild_band_max"] = bool(pulse_stats.get("pulse_mild_band_max", False))
            findings["invariant_checks"]["pulse_mild_band_all"] = bool(pulse_stats.get("pulse_mild_band_all", False))

    if policy_ranking is not None:
        findings["policy_ranking"] = policy_ranking
        findings["invariant_checks"]["policy_top_rank_stable"] = bool(policy_ranking.get("top_scenario_stable", False))
        findings["invariant_checks"]["policy_full_rank_stable"] = bool(policy_ranking.get("full_ranking_stable", False))

    return findings


def generate_report(
    substep_df: pl.DataFrame,
    tick_df: pl.DataFrame,
    ablation_df: pl.DataFrame,
    pulse_df: pl.DataFrame,
    findings: dict[str, Any],
    *,
    n_agents: int,
    n_steps: int,
    seed: int,
    output_dir: Path,
) -> str:
    """Generate the final diagnostic report."""
    lines: list[str] = []
    lines.append("# ABM Credibility Diagnostics Report\n")
    lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**N_agents**: {n_agents}, **N_steps**: {n_steps}, **seed**: {seed}\n")

    # --- Section 1: Drift buckets ---
    lines.append("## 1. Identified Drift Buckets\n")
    buckets = findings.get("buckets", [])
    if not buckets:
        lines.append("No significant drift buckets identified.\n")
    else:
        for b in buckets:
            lines.append(f"- **{b}**")
    lines.append("")

    # --- Section 2: Fix recommendations ---
    lines.append("## 2. Fix Recommendations\n")
    fixes = findings.get("fix_recommendations", [])
    if not fixes:
        lines.append("No fixes needed.\n")
    else:
        for f in fixes:
            lines.append(f"- {f}")
    lines.append("")

    # --- Section 3: D1 contribution decomposition ---
    lines.append("## 3. D1: Additive Contribution Decomposition (per-substep means)\n")
    if substep_df.height > 0:
        for col in TRACKED_CONSTRUCTS:
            sub = substep_df.filter(pl.col("construct") == col)
            if sub.height == 0:
                continue
            lines.append(f"### {col}\n")
            lines.append("| Channel | Mean Δ/substep |")
            lines.append("|---------|---------------|")
            for channel in [
                "delta_peer_pull",
                "delta_observed_norm",
                "delta_attractor",
                "delta_noise",
                "delta_clip",
                "delta_total",
            ]:
                val = float(sub[channel].mean())
                lines.append(f"| {channel.replace('delta_', '')} | {val:+.6f} |")
            lines.append("")

    # --- Section 4: D2 perceived-share pipeline ---
    lines.append("## 4. D2: Perceived-Share Pipeline Diagnostics\n")
    desc_sub = substep_df.filter(pl.col("construct") == "social_norms_descriptive_vax")
    if desc_sub.height > 0 and "mean_observed_share" in desc_sub.columns:
        obs_vals = desc_sub.select(
            [
                "mean_observed_share",
                "mean_share_hat",
                "mean_share_hat0",
                "mean_logit_share_hat",
                "mean_logit_share_hat0",
                "mean_delta_e",
                "frac_obs_below_hat0",
                "frac_share_hat_saturating",
            ]
        ).drop_nulls()
        if obs_vals.height > 0:
            lines.append("| Metric | Mean |")
            lines.append("|--------|------|")
            for col_name in obs_vals.columns:
                val = float(obs_vals[col_name].mean())
                lines.append(f"| {col_name} | {val:.6f} |")
            lines.append("")
            gap = findings.get("perceived_share_gap", None)
            if gap is not None:
                lines.append(f"**observed_share - share_hat0 gap**: {gap:+.4f}\n")

    # --- Section 5: D3 sampling bias ---
    lines.append("## 5. D3: Sampling Bias Diagnostics\n")
    sb = findings.get("sampling_bias", None)
    if sb is not None:
        lines.append(f"**Sampling bias (obs_share - pop_mean_behavior)**: {sb:+.4f}\n")
    else:
        lines.append("No sampling bias data available.\n")

    # --- Section 6: D5 pulse response ---
    lines.append("## 6. D5: Incidence Pulse Response\n")
    if pulse_df.height > 0:
        baseline_rows = pulse_df.filter(pl.col("label") == "baseline_import")
        pulse_rows = pulse_df.filter(pl.col("label") == "pulse_import")
        if baseline_rows.height > 0 and pulse_rows.height > 0:
            base_inc = baseline_rows["mean_incidence"].to_numpy()
            pulse_inc = pulse_rows["mean_incidence"].to_numpy()
            lines.append(f"- Baseline mean incidence (last tick): {base_inc[-1]:.6f}")
            lines.append(f"- Pulse mean incidence (last tick): {pulse_inc[-1]:.6f}")
            lines.append(f"- Peak Δincidence: {float(np.max(pulse_inc - base_inc)):.6f}")

            # Check if risk mediators responded
            for col in ["delta_covid_perceived_danger_T12", "delta_disease_fear_avg"]:
                if col in pulse_rows.columns:
                    base_deltas = (
                        baseline_rows[col].to_numpy()
                        if col in baseline_rows.columns
                        else np.zeros(baseline_rows.height)
                    )
                    pulse_deltas = pulse_rows[col].to_numpy()
                    diff = float(np.mean(pulse_deltas) - np.mean(base_deltas))
                    lines.append(f"- Mean {col} difference (pulse - baseline): {diff:+.6f}")

            pulse_stats = findings.get("pulse_stats", {})
            if pulse_stats:
                lines.append(
                    f"- Baseline incidence std/range: {pulse_stats.get('baseline_incidence_std', float('nan')):.6f} / "
                    f"{pulse_stats.get('baseline_incidence_range', float('nan')):.6f}"
                )
                lines.append(
                    f"- Pulse incidence std/range: {pulse_stats.get('pulse_incidence_std', float('nan')):.6f} / "
                    f"{pulse_stats.get('pulse_incidence_range', float('nan')):.6f}"
                )
                lines.append(
                    f"- Pulse effect std/range (pulse-baseline): "
                    f"{pulse_stats.get('pulse_delta_std', float('nan')):.6f} / "
                    f"{pulse_stats.get('pulse_delta_range', float('nan')):.6f}"
                )
                lines.append(
                    f"- Pulse incidence min/max: {pulse_stats.get('pulse_incidence_min', float('nan')):.6f} / "
                    f"{pulse_stats.get('pulse_incidence_max', float('nan')):.6f}"
                )
                lines.append(
                    "- Mild-band thresholds used "
                    f"(std<={PULSE_MILD_STD_THRESHOLD:.3f}, range<={PULSE_MILD_RANGE_THRESHOLD:.3f}, "
                    f"max<={PULSE_MILD_MAX_THRESHOLD:.3f}; std on pulse-baseline effect): "
                    f"{'PASS' if pulse_stats.get('pulse_mild_band_all', False) else 'FAIL'}"
                )
            lines.append("")

    # --- Section 7: D6 2x2 ablation ---
    lines.append("## 7. D6: 2×2 Social Ablation Grid\n")
    if ablation_df.height > 0:
        lines.append("| Cell | Attractor | Cumulative Drift | Var Change |")
        lines.append("|------|-----------|-----------------|------------|")
        for cell in ["both_off", "peer_only", "obs_only", "both_on"]:
            for att in ["att_off", "att_on"]:
                drift_key = f"ablation_{cell}_{att}_drift"
                var_key = f"ablation_{cell}_{att}_var_change"
                drift = findings.get(drift_key, float("nan"))
                var_c = findings.get(var_key, float("nan"))
                lines.append(f"| {cell} | {att} | {drift:+.4f} | {var_c:+.4f} |")
        lines.append("")

    # --- Section 8: Horizon-aware stationarity summary ---
    lines.append("## 8. Horizon-Aware Stationarity\n")
    short_steps = int(findings.get("short_horizon_steps", n_steps))
    lines.append(
        f"- Short horizon: {short_steps} months; "
        f"thresholds |drift|<={DRIFT_ABS_THRESHOLD:.3f} and |drift/month|<={DRIFT_PER_MONTH_THRESHOLD:.3f}."
    )
    off_total = float(findings.get("desc_norm_drift_attractor_off", float("nan")))
    off_rate = float(findings.get("desc_norm_drift_per_month_attractor_off", float("nan")))
    on_total = float(findings.get("desc_norm_drift_attractor_on", float("nan")))
    on_rate = float(findings.get("desc_norm_drift_per_month_attractor_on", float("nan")))
    lines.append(f"- Descriptive drift (att_off): {off_total:+.4f} total, {off_rate:+.4f}/month")
    lines.append(f"- Descriptive drift (att_on): {on_total:+.4f} total, {on_rate:+.4f}/month")

    long_off_key = f"desc_norm_drift_attractor_off_{LONG_HORIZON_STEPS}"
    long_on_key = f"desc_norm_drift_attractor_on_{LONG_HORIZON_STEPS}"
    if long_off_key in findings and long_on_key in findings:
        long_off_total = float(findings[long_off_key])
        long_on_total = float(findings[long_on_key])
        long_off_rate = float(
            findings.get(f"desc_norm_drift_per_month_attractor_off_{LONG_HORIZON_STEPS}", float("nan"))
        )
        long_on_rate = float(findings.get(f"desc_norm_drift_per_month_attractor_on_{LONG_HORIZON_STEPS}", float("nan")))
        lines.append(
            f"- Long horizon: {LONG_HORIZON_STEPS} months (att_off {long_off_total:+.4f}, {long_off_rate:+.4f}/month; "
            f"att_on {long_on_total:+.4f}, {long_on_rate:+.4f}/month)"
        )
    lines.append("")

    # --- Section 9: Policy ranking stability ---
    lines.append("## 9. Policy Ranking Stability\n")
    policy_info = findings.get("policy_ranking", {})
    if policy_info:
        horizons = policy_info.get("horizons", [n_steps, LONG_HORIZON_STEPS])
        ranking_by_horizon = policy_info.get("ranking_by_horizon", {})
        lines.append(f"- Tie tolerance for rank stability: {float(policy_info.get('tie_tolerance', 0.0)):.6f}")
        for horizon in horizons:
            horizon_key = str(horizon)
            horizon_info = ranking_by_horizon.get(horizon_key, {})
            ranking = horizon_info.get("ranking", [])
            scores = horizon_info.get("scores", {})
            if ranking:
                lines.append(f"- Ranking @ {horizon} months: {', '.join(ranking)}")
                top = ranking[0]
                lines.append(f"  - Top scenario score ({top}): {float(scores.get(top, float('nan'))):.6f}")
        agreement = policy_info.get("pairwise_rank_agreement", float("nan"))
        decisive = int(policy_info.get("pairwise_rank_decisive_pairs", 0))
        total_pairs = int(policy_info.get("pairwise_rank_total_pairs", 0))
        lines.append(f"- Pairwise ranking agreement (short vs long): {float(agreement):.3f}")
        lines.append(f"- Pairwise decisive pairs used: {decisive}/{total_pairs}")
        top_tied_short = policy_info.get("top_tied_short", [])
        top_tied_long = policy_info.get("top_tied_long", [])
        if top_tied_short:
            lines.append(f"- Top tied set @ {horizons[0]} months: {', '.join(top_tied_short)}")
        if top_tied_long:
            lines.append(f"- Top tied set @ {horizons[1]} months: {', '.join(top_tied_long)}")
        lines.append(
            f"- Top scenario stable: {'YES' if policy_info.get('top_scenario_stable', False) else 'NO'}; "
            f"full ranking stable: {'YES' if policy_info.get('full_ranking_stable', False) else 'NO'}"
        )
    else:
        lines.append("No policy ranking stability data available.")
    lines.append("")

    # --- Section 10: Invariant checks ---
    lines.append("## 10. Invariant Checks\n")
    checks = findings.get("invariant_checks", {})
    lines.append("| Invariant | Pass |")
    lines.append("|-----------|------|")
    for name, passed in checks.items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines.append("")

    # --- Section 11: Acceptance criteria ---
    lines.append("## 11. Acceptance Criteria Status\n")
    criteria: dict[str, bool] = {}

    desc_off_drift = abs(findings.get("desc_norm_drift_attractor_off", float("inf")))
    desc_on_drift = abs(findings.get("desc_norm_drift_attractor_on", float("inf")))
    desc_off_rate = abs(findings.get("desc_norm_drift_per_month_attractor_off", float("inf")))
    desc_on_rate = abs(findings.get("desc_norm_drift_per_month_attractor_on", float("inf")))
    criteria["C1_stationarity_att_off"] = (
        desc_off_drift < DRIFT_ABS_THRESHOLD and desc_off_rate < DRIFT_PER_MONTH_THRESHOLD
    )
    criteria["C1_stationarity_att_on"] = (
        desc_on_drift < DRIFT_ABS_THRESHOLD and desc_on_rate < DRIFT_PER_MONTH_THRESHOLD
    )

    long_off_rate = abs(findings.get(f"desc_norm_drift_per_month_attractor_off_{LONG_HORIZON_STEPS}", float("inf")))
    long_on_rate = abs(findings.get(f"desc_norm_drift_per_month_attractor_on_{LONG_HORIZON_STEPS}", float("inf")))
    criteria[f"C1_long_horizon_{LONG_HORIZON_STEPS}m_att_off"] = long_off_rate < DRIFT_PER_MONTH_THRESHOLD
    criteria[f"C1_long_horizon_{LONG_HORIZON_STEPS}m_att_on"] = long_on_rate < DRIFT_PER_MONTH_THRESHOLD

    peer_only_drift = abs(findings.get("ablation_peer_only_att_off_drift", float("inf")))
    obs_only_drift = abs(findings.get("ablation_obs_only_att_off_drift", float("inf")))
    criteria["C2_peer_only_no_drift"] = peer_only_drift < DRIFT_ABS_THRESHOLD
    criteria["C2_obs_only_no_drift"] = obs_only_drift < DRIFT_ABS_THRESHOLD

    criteria["C3_policy_top_rank_stable"] = bool(findings.get("policy_ranking", {}).get("top_scenario_stable", False))

    criteria["C4_pulse_mild_band"] = bool(findings.get("pulse_stats", {}).get("pulse_mild_band_all", False))

    criteria["C5_no_relax_clip_cancel"] = findings.get("invariant_checks", {}).get("no_large_relax_clip", True)

    for name, passed in criteria.items():
        lines.append(f"- **{name}**: {'PASS' if passed else 'FAIL'}")
    lines.append("")

    # --- Section 12: Generated artifact paths ---
    lines.append("## 12. Generated Artifacts\n")
    lines.append(f"- Substep timeseries: `{output_dir / 'substep_timeseries.csv'}`")
    lines.append(f"- Tick summaries: `{output_dir / 'tick_summaries.csv'}`")
    lines.append(f"- 2x2 ablation: `{output_dir / 'ablation_2x2.csv'}`")
    lines.append(f"- Pulse response: `{output_dir / 'pulse_response.csv'}`")
    lines.append(f"- Long-horizon stationarity: `{output_dir / 'long_horizon_stationarity.csv'}`")
    lines.append(f"- Policy ranking stability: `{output_dir / 'policy_ranking.json'}`")
    lines.append(f"- Decision tree findings: `{output_dir / 'findings.json'}`")
    lines.append(f"- This report: `{output_dir / 'report.md'}`")

    return "\n".join(lines)
