"""Compute reference PE on the full respondent set.

This script mirrors `anchors.py pe` but evaluates ICE/PE on all respondents
from the input CSV (not the sampled anchor eval subset). It also emits
eval-level off-manifold diagnostics:

- `pe_off_manifold_eval.csv` (per eval_id x lever x grid_point flag)
- `pe_off_manifold_eval_summary.csv` (aggregated rates/counts)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import polars as pl

from beam_abm.anchoring.sampling import PerturbationConfig, run_anchor_pe
from beam_abm.empirical.io import parse_list_arg
from beam_abm.empirical.missingness import DEFAULT_MISSINGNESS_THRESHOLD
from beam_abm.empirical.plan_io import extract_predictors_from_model_plan
from beam_abm.settings import CSV_FILE_CLEAN, SEED


def _iter_plan_entries(model_plan: object) -> list[dict]:
    if isinstance(model_plan, list):
        return [x for x in model_plan if isinstance(x, dict)]
    if isinstance(model_plan, dict):
        models = model_plan.get("models") or model_plan.get("plan")
        if isinstance(models, list):
            return [x for x in models if isinstance(x, dict)]
        if isinstance(models, dict):
            return [x for x in models.values() if isinstance(x, dict)]
        return [model_plan]
    return []


def _norm_block_name(block: str) -> str:
    return str(block).strip()


def _extract_target_block_map(*, model_plan_path: Path, model_ref: str | None, outcome: str) -> dict[str, str]:
    if not model_plan_path.exists():
        return {}
    raw = json.loads(model_plan_path.read_text(encoding="utf-8"))
    entries = _iter_plan_entries(raw)
    spec: dict | None = None
    if model_ref:
        spec = next(
            (e for e in entries if str(e.get("model") or e.get("name") or "").strip() == str(model_ref).strip()),
            None,
        )
    if spec is None:
        candidates = [e for e in entries if str(e.get("outcome") or "").strip() == str(outcome).strip()]
        if len(candidates) == 1:
            spec = candidates[0]
    if spec is None:
        return {}

    predictors = spec.get("predictors")
    if not isinstance(predictors, dict):
        return {}

    mapping: dict[str, str] = {}
    for block, cols in predictors.items():
        if not isinstance(cols, list):
            continue
        block_norm = _norm_block_name(str(block))
        for col in cols:
            target = str(col).strip()
            if target and target not in mapping:
                mapping[target] = block_norm
    return mapping


def _average_ranks_desc(values_by_block: dict[str, float]) -> dict[str, float]:
    items = sorted(values_by_block.items(), key=lambda kv: (-kv[1], kv[0]))
    ranks: dict[str, float] = {}
    i = 0
    while i < len(items):
        j = i
        while j + 1 < len(items) and items[j + 1][1] == items[i][1]:
            j += 1
        avg_rank = ((i + 1) + (j + 1)) / 2.0
        for k in range(i, j + 1):
            ranks[items[k][0]] = avg_rank
        i = j + 1
    return ranks


def _kendall_tau_b(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y):
        raise ValueError("x and y must have the same length for Kendall tau.")
    n = len(x)
    if n < 2:
        return None

    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0

    def _sgn(v: float) -> int:
        if v > 0:
            return 1
        if v < 0:
            return -1
        return 0

    for i in range(n - 1):
        for j in range(i + 1, n):
            sx = _sgn(float(x[i]) - float(x[j]))
            sy = _sgn(float(y[i]) - float(y[j]))
            if sx == 0 and sy == 0:
                continue
            if sx == 0:
                ties_x += 1
                continue
            if sy == 0:
                ties_y += 1
                continue
            if sx == sy:
                concordant += 1
            else:
                discordant += 1

    denom = float((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom <= 0:
        return None
    return float((concordant - discordant) / math.sqrt(denom))


def _write_block_rank_metrics(
    *,
    outdir: Path,
    model_plan_path: Path,
    model_ref: str | None,
    outcome: str,
) -> None:
    ice_path = outdir / "ice_ref.csv"
    if not ice_path.exists():
        return

    ice_df = pl.read_csv(ice_path)
    required_ice = {"id", "target", "ice_low", "ice_high", "delta_z"}
    if not required_ice.issubset(set(ice_df.columns)):
        return

    pe_df = (
        ice_df.select(
            pl.col("id").cast(pl.Utf8),
            pl.col("target").cast(pl.Utf8),
            pl.col("ice_low").cast(pl.Float64),
            pl.col("ice_high").cast(pl.Float64),
            pl.col("delta_z").cast(pl.Float64),
        )
        .with_columns(
            pl.when(pl.col("delta_z").is_finite() & (pl.col("delta_z") != 0))
            .then(((pl.col("ice_high") - pl.col("ice_low")) / pl.col("delta_z")).abs())
            .otherwise(float("nan"))
            .alias("pe_ref_std")
        )
        .filter(pl.col("pe_ref_std").is_finite())
        .select(["id", "target", "pe_ref_std"])
    )
    if pe_df.is_empty():
        return

    off_path = outdir / "pe_off_manifold_eval.csv"
    if off_path.exists():
        off_df = pl.read_csv(off_path)
        required_off = {"eval_id", "lever", "is_off_manifold"}
        if required_off.issubset(set(off_df.columns)):
            off_any = (
                off_df.select(
                    pl.col("eval_id").cast(pl.Utf8).alias("id"),
                    pl.col("lever").cast(pl.Utf8).alias("target"),
                    pl.col("is_off_manifold").cast(pl.Boolean),
                )
                .group_by(["id", "target"])
                .agg(pl.col("is_off_manifold").max().alias("is_off_manifold_any"))
            )
        else:
            off_any = pl.DataFrame(
                {
                    "id": pl.Series(name="id", values=[], dtype=pl.Utf8),
                    "target": pl.Series(name="target", values=[], dtype=pl.Utf8),
                    "is_off_manifold_any": pl.Series(name="is_off_manifold_any", values=[], dtype=pl.Boolean),
                }
            )
    else:
        off_any = pl.DataFrame(
            {
                "id": pl.Series(name="id", values=[], dtype=pl.Utf8),
                "target": pl.Series(name="target", values=[], dtype=pl.Utf8),
                "is_off_manifold_any": pl.Series(name="is_off_manifold_any", values=[], dtype=pl.Boolean),
            }
        )

    pe_df = pe_df.join(off_any, on=["id", "target"], how="left").with_columns(
        pl.col("is_off_manifold_any").fill_null(False)
    )
    target_to_block = _extract_target_block_map(
        model_plan_path=model_plan_path,
        model_ref=model_ref,
        outcome=outcome,
    )
    pe_df = pe_df.with_columns(
        pl.col("target")
        .map_elements(lambda x: target_to_block.get(str(x)), return_dtype=pl.Utf8)
        .fill_null("unmapped")
        .alias("block")
    )

    full_stats = pe_df.group_by("block").agg(
        pl.len().alias("n_full"),
        pl.col("pe_ref_std").median().alias("pe_abs_p50_full"),
    )
    typical_stats = (
        pe_df.filter(~pl.col("is_off_manifold_any"))
        .group_by("block")
        .agg(
            pl.len().alias("n_typical"),
            pl.col("pe_ref_std").median().alias("pe_abs_p50_typical"),
        )
    )
    rank_df = full_stats.join(typical_stats, on="block", how="full").with_columns(
        pl.coalesce([pl.col("block"), pl.col("block_right")]).alias("block")
    )
    if "block_right" in rank_df.columns:
        rank_df = rank_df.drop("block_right")

    full_vals: dict[str, float] = {}
    typical_vals: dict[str, float] = {}
    for row in rank_df.to_dicts():
        block = str(row.get("block", ""))
        v_full = row.get("pe_abs_p50_full")
        v_typ = row.get("pe_abs_p50_typical")
        if isinstance(v_full, int | float) and math.isfinite(float(v_full)):
            full_vals[block] = float(v_full)
        if isinstance(v_typ, int | float) and math.isfinite(float(v_typ)):
            typical_vals[block] = float(v_typ)

    ranks_full = _average_ranks_desc(full_vals) if full_vals else {}
    ranks_typical = _average_ranks_desc(typical_vals) if typical_vals else {}

    rows: list[dict[str, object]] = []
    for row in rank_df.to_dicts():
        block = str(row.get("block", ""))
        rank_full = ranks_full.get(block)
        rank_typical = ranks_typical.get(block)
        rows.append(
            {
                "outcome": outcome,
                "model_ref": model_ref,
                "block": block,
                "n_full": row.get("n_full"),
                "n_typical": row.get("n_typical"),
                "pe_abs_p50_full": row.get("pe_abs_p50_full"),
                "pe_abs_p50_typical": row.get("pe_abs_p50_typical"),
                "rank_full": rank_full,
                "rank_typical": rank_typical,
                "rank_shift_typical_minus_full": (
                    float(rank_typical) - float(rank_full)
                    if rank_full is not None and rank_typical is not None
                    else None
                ),
            }
        )

    rank_out = pl.DataFrame(rows).sort(
        by=["rank_full", "rank_typical", "block"],
        descending=[False, False, False],
        nulls_last=True,
    )
    rank_out.write_csv(outdir / "pe_block_rank_metrics.csv")

    common_blocks = sorted(set(full_vals) & set(typical_vals))
    tau = None
    if len(common_blocks) >= 2:
        tau = _kendall_tau_b(
            [full_vals[b] for b in common_blocks],
            [typical_vals[b] for b in common_blocks],
        )

    n_total = int(pe_df.height)
    n_typical = int(pe_df.filter(~pl.col("is_off_manifold_any")).height)
    n_atypical = n_total - n_typical
    summary = pl.DataFrame(
        {
            "outcome": [outcome],
            "model_ref": [model_ref],
            "n_blocks_full": [len(full_vals)],
            "n_blocks_typical": [len(typical_vals)],
            "n_blocks_common": [len(common_blocks)],
            "kendall_tau_b_block_rank_full_vs_typical": [tau],
            "n_profiles_total": [n_total],
            "n_profiles_typical": [n_typical],
            "n_profiles_atypical": [n_atypical],
            "typical_share": [float(n_typical / n_total) if n_total else None],
        }
    )
    summary.write_csv(outdir / "pe_block_rank_kendall_tau.csv")


def _resolve_levers_and_drivers(args: argparse.Namespace) -> tuple[tuple[str, ...], tuple[str, ...], str | None]:
    levers = tuple(parse_list_arg(args.levers)) if args.levers else ()
    driver_cols = tuple(parse_list_arg(args.driver_cols)) if args.driver_cols else ()
    model_ref: str | None = None
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
            model_ref_obj = entry.get("model_ref")
            if isinstance(model_ref_obj, str) and model_ref_obj.strip():
                model_ref = model_ref_obj.strip()
            if not driver_cols:
                if model_ref:
                    driver_cols = tuple(
                        extract_predictors_from_model_plan(
                            model_plan_path=Path(args.model_plan),
                            model_name=model_ref,
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

    return selected_levers, driver_cols, model_ref


def _build_full_eval_profiles(input_csv: Path, out_path: Path, *, country_col: str) -> None:
    df = pl.read_csv(input_csv, infer_schema_length=10_000).with_row_index(name="row_id")
    if country_col not in df.columns:
        raise ValueError(f"country column not found in input CSV: {country_col}")
    df = df.with_columns(
        pl.concat_str(
            [
                pl.lit("full::"),
                pl.col(country_col).cast(pl.Utf8, strict=False).fill_null("unknown"),
                pl.lit("::"),
                pl.col("row_id").cast(pl.Utf8),
            ]
        ).alias("eval_id")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute full-population anchor PE references.")
    parser.add_argument("--input", default=CSV_FILE_CLEAN)
    parser.add_argument("--anchors-dir", default="empirical/output/anchors/system")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--country-col", default="country")
    parser.add_argument("--target-col", required=True)
    parser.add_argument("--driver-cols", default=None)
    parser.add_argument("--levers", default=None)
    parser.add_argument("--lever-config", default=None)
    parser.add_argument("--lever-key", default=None)
    parser.add_argument(
        "--lever-scope",
        choices=("policy", "non_policy", "all_predictors"),
        default="policy",
        help="Which predictors to perturb: policy levers (default), non-policy predictors, or all predictors.",
    )
    parser.add_argument("--neighborhood-features", default=None)
    parser.add_argument("--neighbor-k", type=int, default=200)
    parser.add_argument("--min-n", type=int, default=200)
    parser.add_argument("--oom-percentile", type=float, default=0.95)
    parser.add_argument(
        "--off-manifold-method",
        choices=("distance_nn", "conditional_tail"),
        default="conditional_tail",
        help="Off-manifold rule for perturbed points (default: conditional_tail).",
    )
    parser.add_argument(
        "--conditional-tail-alpha",
        type=float,
        default=0.05,
        help="Tail-probability cutoff for conditional-tail off-manifold flags.",
    )
    parser.add_argument(
        "--conditional-tail-knn-k",
        type=int,
        default=200,
        help="k for local conditional CDF estimation in conditional-tail mode.",
    )
    parser.add_argument("--propensity-slices", default=None)
    parser.add_argument("--propensity-outcome", default=None)
    parser.add_argument("--propensity-slice-col", default="slice")
    parser.add_argument("--ref-models", default="linear")
    parser.add_argument("--xgb-n-estimators", type=int, default=300)
    parser.add_argument("--xgb-max-depth", type=int, default=4)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--fallback-min-n", type=int, default=300)
    parser.add_argument("--fallback-min-delta-z", type=float, default=0.25)
    parser.add_argument("--fallback-quantiles", default="0.10,0.90")
    parser.add_argument("--fallback-quantiles-wide", default="0.05,0.95")
    parser.add_argument("--fallback-nonzero-quantile", type=float, default=0.50)
    parser.add_argument("--predictor-missingness-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--model-plan", default="empirical/output/modeling/model_plan.json")
    parser.add_argument(
        "--keep-full-eval-profiles",
        action="store_true",
        help="Keep intermediate full eval parquet in output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    levers, driver_cols, model_ref = _resolve_levers_and_drivers(args)

    full_eval_path = outdir / "_eval_profiles_full.parquet"
    _build_full_eval_profiles(Path(args.input), full_eval_path, country_col=str(args.country_col))

    cfg = PerturbationConfig(
        input_csv=Path(args.input),
        anchors_dir=Path(args.anchors_dir),
        outdir=outdir,
        country_col=str(args.country_col),
        target_col=str(args.target_col),
        driver_cols=driver_cols,
        levers=levers,
        neighborhood_features=tuple(parse_list_arg(args.neighborhood_features)) if args.neighborhood_features else (),
        neighbor_k=int(args.neighbor_k),
        min_n=int(args.min_n),
        oom_percentile=float(args.oom_percentile),
        off_manifold_method=str(args.off_manifold_method),
        conditional_tail_alpha=float(args.conditional_tail_alpha),
        conditional_tail_knn_k=int(args.conditional_tail_knn_k),
        propensity_slices_path=Path(args.propensity_slices) if args.propensity_slices else None,
        propensity_outcome=args.propensity_outcome,
        propensity_slice_col=str(args.propensity_slice_col),
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
        eval_profiles_path=full_eval_path,
        compute_eval_off_manifold=True,
        xgb_n_estimators=int(args.xgb_n_estimators),
        xgb_max_depth=int(args.xgb_max_depth),
        xgb_learning_rate=float(args.xgb_learning_rate),
        xgb_subsample=float(args.xgb_subsample),
        xgb_colsample_bytree=float(args.xgb_colsample_bytree),
    )
    run_anchor_pe(cfg)
    _write_block_rank_metrics(
        outdir=outdir,
        model_plan_path=Path(args.model_plan),
        model_ref=model_ref,
        outcome=str(args.target_col),
    )

    if not args.keep_full_eval_profiles and full_eval_path.exists():
        full_eval_path.unlink()


if __name__ == "__main__":
    main()
