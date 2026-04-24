"""Calibrate homophily gamma against survey friend-self similarity."""

from __future__ import annotations

import argparse
from pathlib import Path

from beam_abm.abm.workflows import calibrate_homophily_gamma_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    calibrate_homophily_gamma_workflow(
        n_agents=args.n_agents,
        seed=args.seed,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        gamma_steps=args.gamma_steps,
        output=args.output,
    )


if __name__ == "__main__":
    main()
