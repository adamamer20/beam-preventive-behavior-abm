"""Calibrate homophily gamma against survey friend-self similarity.

Fits homophily strength so the ABM's peer-mean correlation aligns with
survey-reported friend/self similarity measures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from beam_abm.abm.config import SimulationConfig
from beam_abm.abm.data_contracts import load_anchor_system
from beam_abm.abm.io import load_survey_data
from beam_abm.abm.network import build_contact_network, compute_homophily_weighted_peer_means
from beam_abm.abm.population import initialise_population


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return 0.0
    x_clean = x[mask]
    y_clean = y[mask]
    if np.std(x_clean) <= 1e-12 or np.std(y_clean) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x_clean, y_clean)[0, 1])


def _available_target_pairs(survey_df: pl.DataFrame) -> list[tuple[str, str]]:
    candidates = [
        ("progressive_attitudes_nonvax_idx", "friends_progressive_norms_nonvax_idx"),
        ("progressive_attitudes_nonvax_idx", "family_progressive_norms_nonvax_idx"),
        ("progressive_attitudes_nonvax_idx", "colleagues_progressive_norms_nonvax_idx"),
        ("att_vaccines_importance", "friends_vaccines_importance"),
        ("att_vaccines_importance", "family_vaccines_importance"),
        ("att_vaccines_importance", "colleagues_vaccines_importance"),
    ]
    available = []
    cols = set(survey_df.columns)
    for self_col, peer_col in candidates:
        if self_col in cols and peer_col in cols:
            available.append((self_col, peer_col))
    return available


def _compute_target_corr(survey_df: pl.DataFrame) -> float:
    pairs = _available_target_pairs(survey_df)
    if not pairs:
        msg = "No friend/self proxy columns found for homophily calibration."
        raise ValueError(msg)

    corrs: list[float] = []
    for self_col, peer_col in pairs:
        x = survey_df[self_col].cast(pl.Float64).to_numpy()
        y = survey_df[peer_col].cast(pl.Float64).to_numpy()
        corr = _pearson(x, y)
        if np.isfinite(corr):
            corrs.append(corr)
    if not corrs:
        msg = "Friend/self proxy columns are present but correlations are undefined."
        raise ValueError(msg)
    return float(np.mean(corrs))


def _compute_peer_corr(
    agents: pl.DataFrame,
    network_seed: int,
    gamma: float,
    bridge_prob: float,
) -> float:
    network = build_contact_network(agents, bridge_prob=bridge_prob, seed=network_seed)
    means, _ = compute_homophily_weighted_peer_means(
        agents,
        network,
        columns=("institutional_trust_avg", "moral_progressive_orientation"),
        homophily_gamma=gamma,
        homophily_w_trust=0.5,
        homophily_w_moral=0.5,
    )

    corrs: list[float] = []
    for col, peer_vals in means.items():
        if col not in agents.columns:
            continue
        self_vals = agents[col].cast(pl.Float64).to_numpy()
        corrs.append(_pearson(self_vals, peer_vals))

    if not corrs:
        return 0.0
    return float(np.mean(corrs))


def _calibrate_gamma(
    agents: pl.DataFrame,
    target_corr: float,
    bridge_prob: float,
    seed: int,
    gamma_min: float,
    gamma_max: float,
    gamma_steps: int,
) -> dict[str, object]:
    grid = np.linspace(gamma_min, gamma_max, gamma_steps)
    best_gamma = None
    best_error = float("inf")
    rows: list[dict[str, float]] = []

    for gamma in grid:
        peer_corr = _compute_peer_corr(agents, seed, float(gamma), bridge_prob)
        error = abs(peer_corr - target_corr)
        rows.append({"gamma": float(gamma), "peer_corr": peer_corr, "abs_error": float(error)})
        if error < best_error:
            best_error = error
            best_gamma = float(gamma)

    if best_gamma is None:
        msg = "Gamma calibration failed to produce a candidate."
        raise RuntimeError(msg)

    return {
        "target_corr": float(target_corr),
        "best_gamma": float(best_gamma),
        "grid": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate homophily gamma from survey data.")
    parser.add_argument("--n-agents", type=int, default=4000, help="Number of agents to simulate for calibration.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for population/network building.")
    parser.add_argument("--gamma-min", type=float, default=0.2, help="Minimum gamma value to scan.")
    parser.add_argument("--gamma-max", type=float, default=4.0, help="Maximum gamma value to scan.")
    parser.add_argument("--gamma-steps", type=int, default=25, help="Number of gamma grid points.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write calibration JSON output.",
    )
    args = parser.parse_args()

    cfg = SimulationConfig(n_agents=args.n_agents, seed=args.seed)
    survey_df = load_survey_data(cfg.survey_csv, cfg.countries)
    target_corr = _compute_target_corr(survey_df)

    anchor_system = load_anchor_system(cfg.anchors_dir)
    agents = initialise_population(
        n_agents=cfg.n_agents,
        countries=cfg.countries,
        anchor_system=anchor_system,
        survey_df=survey_df,
        seed=cfg.seed,
    )

    report = _calibrate_gamma(
        agents=agents,
        target_corr=target_corr,
        bridge_prob=cfg.constants.bridge_prob,
        seed=cfg.seed,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        gamma_steps=args.gamma_steps,
    )

    logger.info("Target correlation: {t:.3f}", t=report["target_corr"])
    logger.info("Recommended homophily_gamma: {g:.3f}", g=report["best_gamma"])

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Saved calibration report to {p}", p=args.output)


if __name__ == "__main__":
    main()
