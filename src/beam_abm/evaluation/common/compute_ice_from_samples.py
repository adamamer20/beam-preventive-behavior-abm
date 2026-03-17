"""CLI wrapper for ICE table rebuild from samples."""

from __future__ import annotations

import argparse

from beam_abm.evaluation.common.ice_from_samples import compute_ice_from_samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    parser.add_argument(
        "--summary-out",
        dest="summary_out",
        default=None,
        help="Optional CSV path for noncompliance summary by (strategy, target) and overall.",
    )
    parser.add_argument(
        "--paired-profile-swap-on-b-then-a",
        action="store_true",
        help=(
            "If set, treat paired_profile A/B as display-ordered under paired_order=B_then_A and swap them "
            "so that ice_low always corresponds to the low profile and ice_high to the high profile."
        ),
    )
    args = parser.parse_args()
    compute_ice_from_samples(
        in_path=args.in_path,
        out_path=args.out_path,
        summary_out=args.summary_out,
        paired_profile_swap_on_b_then_a=args.paired_profile_swap_on_b_then_a,
    )
