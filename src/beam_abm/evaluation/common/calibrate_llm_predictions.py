"""CLI wrapper for prediction calibration."""

from __future__ import annotations

import argparse

from beam_abm.evaluation.common.calibrate_predictions import calibrate_predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="Predictions CSV or directory.")
    parser.add_argument("--out", dest="out_path", required=True, help="Output calibrated CSV.")
    parser.add_argument("--calibration-out", dest="cal_out", required=True, help="Output calibration JSON.")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of cross-fitting folds.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--country-col", default="country")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--bounds-source",
        choices=["true", "none"],
        default="true",
        help="Use y_true min/max as bounds for linear calibration.",
    )
    args = parser.parse_args()
    calibrate_predictions(
        in_path=args.in_path,
        out_path=args.out_path,
        calibration_out=args.cal_out,
        n_folds=args.n_folds,
        seed=args.seed,
        country_col=args.country_col,
        eps=args.eps,
        bounds_source=args.bounds_source,
    )
