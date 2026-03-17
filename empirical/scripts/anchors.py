"""Anchor-based heterogeneity pipeline CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from beam_abm.anchoring.build import AnchorConfig, build_anchor_system, default_core_features
from beam_abm.anchoring.diagnostics import DiagnosticsConfig, run_anchor_diagnostics
from beam_abm.anchoring.sampling import PerturbationConfig, run_anchor_pe
from beam_abm.empirical.io import parse_list_arg, slugify
from beam_abm.empirical.missingness import DEFAULT_MISSINGNESS_THRESHOLD
from beam_abm.empirical.plan_io import extract_predictors_from_model_plan
from beam_abm.empirical.taxonomy import DEFAULT_COLUMN_TYPES_PATH
from beam_abm.settings import CSV_FILE_CLEAN, SEED


def main() -> None:
    parser = argparse.ArgumentParser(description="Anchor-based heterogeneity pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build anchors, memberships, and neighborhoods.")
    build.add_argument("--input", default=CSV_FILE_CLEAN)
    build.add_argument("--outdir", default="empirical/output/anchors/system")
    build.add_argument("--column-types", default=str(DEFAULT_COLUMN_TYPES_PATH))
    build.add_argument("--features", default=None, help="Comma-separated feature list.")
    build.add_argument("--country-col", default="country")
    build.add_argument("--normalization", default="demean_country_then_global_scale")
    build.add_argument("--missingness-threshold", type=float, default=0.35)
    build.add_argument("--anchor-k", type=int, default=10)
    build.add_argument("--neighbor-k", type=int, default=200)
    build.add_argument("--globality-min-countries", type=int, default=3)
    build.add_argument("--globality-pi-threshold", type=float, default=0.05)
    build.add_argument("--globality-pi-factor", type=float, default=None)
    build.add_argument("--pca-components", type=int, default=0)
    build.add_argument("--pca-variance-target", type=float, default=0.97)
    build.add_argument("--tau", type=float, default=None)
    build.add_argument("--tau-method", default="knn_quantile")
    build.add_argument("--tau-keff-target", type=float, default=None)
    build.add_argument("--tau-knn-k", type=int, default=10)
    build.add_argument("--tau-quantile", type=float, default=0.5)
    build.add_argument("--tau-sample-n", type=int, default=5000)
    build.add_argument("--seed", type=int, default=SEED)

    diag = sub.add_parser("diagnostics", help="Run anchor diagnostics.")
    diag.add_argument("--input", default=CSV_FILE_CLEAN)
    diag.add_argument("--anchors-dir", default="empirical/output/anchors/system")
    diag.add_argument("--outdir", default="empirical/output/anchors/diagnostics")
    diag.add_argument("--country-col", default="country")
    diag.add_argument("--missingness-threshold", type=float, default=0.35)
    diag.add_argument("--normalization", default="demean_country_then_global_scale")
    diag.add_argument("--coverage-k-grid", default="5,10,15,20,25")
    diag.add_argument("--bootstrap-runs", type=int, default=20)
    diag.add_argument("--bootstrap-subsample-frac", type=float, default=0.8)
    diag.add_argument("--globality-pi-threshold", type=float, default=0.05)
    diag.add_argument("--seed", type=int, default=SEED)

    pe = sub.add_parser("pe", help="Compute perturbation effects with anchors.")
    pe.add_argument("--input", default=CSV_FILE_CLEAN)
    pe.add_argument("--anchors-dir", default="empirical/output/anchors/system")
    pe.add_argument("--outdir", default=None)
    pe.add_argument("--country-col", default="country")
    pe.add_argument("--target-col", default="protective_behaviour_avg")
    pe.add_argument("--driver-cols", default=None)
    pe.add_argument("--levers", default=None)
    pe.add_argument("--lever-config", default=None)
    pe.add_argument("--lever-key", default=None)
    pe.add_argument(
        "--lever-scope",
        choices=("policy", "non_policy", "all_predictors"),
        default="policy",
        help="Which predictors to perturb: policy levers (default), non-policy predictors, or all predictors.",
    )
    pe.add_argument("--neighborhood-features", default=None)
    pe.add_argument("--neighbor-k", type=int, default=200)
    pe.add_argument("--min-n", type=int, default=200)
    pe.add_argument("--oom-percentile", type=float, default=0.95)
    pe.add_argument("--propensity-slices", default=None)
    pe.add_argument("--propensity-outcome", default=None)
    pe.add_argument("--propensity-slice-col", default="slice")
    pe.add_argument(
        "--ref-models",
        default="linear",
        help="Comma-separated reference model names to build ICE refs (e.g., linear,xgb).",
    )
    pe.add_argument("--xgb-n-estimators", type=int, default=300)
    pe.add_argument("--xgb-max-depth", type=int, default=4)
    pe.add_argument("--xgb-learning-rate", type=float, default=0.05)
    pe.add_argument("--xgb-subsample", type=float, default=0.8)
    pe.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    pe.add_argument(
        "--fallback-min-n",
        type=int,
        default=300,
        help="Minimum stratum size before widening quantiles for shock sizing.",
    )
    pe.add_argument(
        "--fallback-min-delta-z",
        type=float,
        default=0.25,
        help="Minimum standardized shock size before widening quantiles for small strata.",
    )
    pe.add_argument(
        "--fallback-quantiles",
        default="0.10,0.90",
        help="Quantile pair used for fallback when IQR collapses (comma-separated).",
    )
    pe.add_argument(
        "--fallback-quantiles-wide",
        default="0.05,0.95",
        help="Wider quantile pair used when fallback quantiles still collapse.",
    )
    pe.add_argument(
        "--fallback-nonzero-quantile",
        type=float,
        default=0.50,
        help="Quantile among non-zero values used for zero-inflated levers.",
    )
    pe.add_argument(
        "--predictor-missingness-threshold",
        type=float,
        default=None,
        help="Max missingness fraction for predictor imputation + missing indicator.",
    )
    pe.add_argument("--seed", type=int, default=SEED)
    pe.add_argument(
        "--model-plan",
        default="empirical/output/modeling/model_plan.json",
        help="Model plan JSON used to resolve driver cols when lever-config entries use model_ref.",
    )

    args = parser.parse_args()

    if args.command == "build":
        features = parse_list_arg(args.features) if args.features else default_core_features()
        cfg = AnchorConfig(
            input_csv=Path(args.input),
            outdir=Path(args.outdir),
            column_types_path=Path(args.column_types),
            features=features,
            country_col=args.country_col,
            normalization=args.normalization,
            missingness_threshold=float(args.missingness_threshold),
            anchor_k=int(args.anchor_k),
            neighbor_k=int(args.neighbor_k),
            globality_min_countries=int(args.globality_min_countries),
            globality_pi_threshold=float(args.globality_pi_threshold),
            globality_pi_factor=float(args.globality_pi_factor) if args.globality_pi_factor is not None else None,
            pca_components=int(args.pca_components),
            pca_variance_target=float(args.pca_variance_target) if args.pca_variance_target else None,
            tau=args.tau,
            tau_method=str(args.tau_method),
            tau_keff_target=float(args.tau_keff_target) if args.tau_keff_target is not None else None,
            tau_knn_k=int(args.tau_knn_k),
            tau_quantile=float(args.tau_quantile),
            tau_sample_n=int(args.tau_sample_n),
            seed=int(args.seed),
        )
        build_anchor_system(cfg)
    elif args.command == "diagnostics":
        k_grid = tuple(int(x) for x in parse_list_arg(args.coverage_k_grid))
        cfg = DiagnosticsConfig(
            input_csv=Path(args.input),
            anchors_dir=Path(args.anchors_dir),
            outdir=Path(args.outdir) if args.outdir else None,
            country_col=args.country_col,
            missingness_threshold=float(args.missingness_threshold),
            normalization=args.normalization,
            coverage_k_grid=k_grid,
            bootstrap_runs=int(args.bootstrap_runs),
            bootstrap_subsample_frac=float(args.bootstrap_subsample_frac),
            globality_pi_threshold=float(args.globality_pi_threshold),
            seed=int(args.seed),
        )
        run_anchor_diagnostics(cfg)
    elif args.command == "pe":
        levers = tuple(parse_list_arg(args.levers)) if args.levers else ()
        driver_cols = tuple(parse_list_arg(args.driver_cols)) if args.driver_cols else ()
        policy_levers: tuple[str, ...] = levers
        if args.lever_config:
            config_path = Path(args.lever_config)
            lever_data = json.loads(config_path.read_text(encoding="utf-8"))
            lever_key = args.lever_key or args.target_col
            if lever_key not in lever_data:
                raise ValueError(f"lever_key '{lever_key}' not found in {config_path}")
            entry = lever_data[lever_key]
            if isinstance(entry, list):
                levers = tuple(str(c) for c in entry)
                policy_levers = levers
                if not driver_cols:
                    driver_cols = levers
            elif isinstance(entry, dict):
                levers_obj = entry.get("levers") or entry.get("targets") or entry.get("cols")
                if not isinstance(levers_obj, list) or not levers_obj:
                    raise ValueError(f"lever_key '{lever_key}' missing non-empty 'levers' in {config_path}")
                levers = tuple(str(c) for c in levers_obj)
                policy_levers = levers
                if not driver_cols:
                    model_ref = entry.get("model_ref")
                    if isinstance(model_ref, str) and model_ref.strip():
                        driver_cols = tuple(
                            extract_predictors_from_model_plan(
                                model_plan_path=Path(args.model_plan),
                                model_name=model_ref.strip(),
                                drop_suffixes=("_missing",),
                                drop_cols=(str(args.country_col),),
                            )
                        )
                    else:
                        predictors_obj = entry.get("predictors") or entry.get("driver_cols") or entry.get("attributes")
                        if not isinstance(predictors_obj, list) or not predictors_obj:
                            raise ValueError(
                                f"lever_key '{lever_key}' missing non-empty 'predictors' or 'model_ref' in {config_path}"
                            )
                        # Driver cols must be numeric; exclude country if present in the config.
                        country_col = str(args.country_col)
                        driver_cols = tuple(str(c) for c in predictors_obj if str(c) != country_col)
            else:
                raise ValueError(f"Unsupported lever_config entry type for key '{lever_key}': {type(entry)}")

        if levers:
            merged = list(dict.fromkeys([*driver_cols, *levers]))
            driver_cols = tuple(merged)

        lever_scope = str(args.lever_scope).strip()
        if lever_scope == "policy":
            selected_levers = tuple(dict.fromkeys(policy_levers or levers))
        elif lever_scope == "all_predictors":
            selected_levers = tuple(dict.fromkeys(driver_cols))
        elif lever_scope == "non_policy":
            policy_set = set(policy_levers or levers)
            selected_levers = tuple(col for col in driver_cols if col not in policy_set)
        else:
            raise ValueError(f"Unsupported lever scope: {lever_scope}")

        if not selected_levers:
            raise ValueError(
                f"No levers selected for lever scope '{lever_scope}' and target '{args.target_col}'. "
                "Check lever config / model plan coverage."
            )

        anchors_dir = Path(args.anchors_dir)
        root = anchors_dir.parent
        outdir = Path(args.outdir) if args.outdir else (root / "pe" / slugify(str(args.target_col)))

        cfg = PerturbationConfig(
            input_csv=Path(args.input),
            anchors_dir=anchors_dir,
            outdir=outdir,
            country_col=args.country_col,
            target_col=args.target_col,
            driver_cols=driver_cols,
            levers=selected_levers,
            neighborhood_features=tuple(parse_list_arg(args.neighborhood_features))
            if args.neighborhood_features
            else (),
            neighbor_k=int(args.neighbor_k),
            min_n=int(args.min_n),
            oom_percentile=float(args.oom_percentile),
            propensity_slices_path=Path(args.propensity_slices) if args.propensity_slices else None,
            propensity_outcome=args.propensity_outcome,
            propensity_slice_col=args.propensity_slice_col,
            predictor_missingness_threshold=(
                float(args.predictor_missingness_threshold)
                if args.predictor_missingness_threshold is not None
                else DEFAULT_MISSINGNESS_THRESHOLD
            ),
            seed=int(args.seed),
            fallback_min_n=int(args.fallback_min_n),
            fallback_min_delta_z=float(args.fallback_min_delta_z),
            fallback_quantiles=tuple(float(x) for x in str(args.fallback_quantiles).split(",")),
            fallback_quantiles_wide=tuple(float(x) for x in str(args.fallback_quantiles_wide).split(",")),
            fallback_nonzero_quantile=float(args.fallback_nonzero_quantile),
            ref_models=tuple(parse_list_arg(args.ref_models)),
            xgb_n_estimators=int(args.xgb_n_estimators),
            xgb_max_depth=int(args.xgb_max_depth),
            xgb_learning_rate=float(args.xgb_learning_rate),
            xgb_subsample=float(args.xgb_subsample),
            xgb_colsample_bytree=float(args.xgb_colsample_bytree),
        )
        run_anchor_pe(cfg)


if __name__ == "__main__":
    main()
