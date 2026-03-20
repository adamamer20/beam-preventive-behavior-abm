from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _gini_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0 or y_pred.size == 0 or y_true.size != y_pred.size:
        return float("nan")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = int(y_true.size)
    if n < 2:
        return float("nan")
    unique_preds = np.unique(y_pred)
    pred_ranks = np.searchsorted(unique_preds, y_pred) + 1
    m = int(unique_preds.size)
    order = np.argsort(y_true, kind="mergesort")
    y_true = y_true[order]
    pred_ranks = pred_ranks[order]
    tree = np.zeros(m + 2, dtype=np.int64)

    def _add(idx: int, value: int) -> None:
        while idx < tree.size:
            tree[idx] += value
            idx += idx & -idx

    def _sum(idx: int) -> int:
        out = 0
        while idx > 0:
            out += tree[idx]
            idx -= idx & -idx
        return out

    total_pairs = 0
    concordant = 0
    ties = 0
    i = 0
    while i < n:
        j = i + 1
        while j < n and y_true[j] == y_true[i]:
            j += 1
        total_prev = _sum(m)
        for k in range(i, j):
            rank = int(pred_ranks[k])
            less = _sum(rank - 1)
            equal = _sum(rank) - less
            concordant += less
            ties += equal
            total_pairs += total_prev
        for k in range(i, j):
            _add(int(pred_ranks[k]), 1)
        i = j
    if total_pairs == 0:
        return float("nan")
    cindex = float((concordant + 0.5 * ties) / total_pairs)
    return float(2.0 * cindex - 1.0)


def _mae_skill_cal_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    yt = y_true[mask]
    yp = y_pred[mask]
    mae = float(np.mean(np.abs(yp - yt)))
    baseline = float(np.median(yt))
    baseline_mae = float(np.mean(np.abs(yt - baseline)))
    if baseline_mae <= 0:
        return float("nan")
    return float(1.0 - mae / baseline_mae)


def _canonical_outcome_name(outcome: str) -> str:
    raw = str(outcome).strip()
    if raw == "vax_willingness_Y":
        return "vax_willingness_T12"
    if raw == "flu_vaccinated_2023_2024_no_habit":
        return "flu_vaccinated_2023_2024"
    return raw


def _outcome_candidates_for_lookup(outcome: str) -> list[str]:
    canonical = _canonical_outcome_name(outcome)
    if canonical == "vax_willingness_T12":
        return ["vax_willingness_T12", "vax_willingness_Y"]
    if canonical == "flu_vaccinated_2023_2024":
        return ["flu_vaccinated_2023_2024", "flu_vaccinated_2023_2024_no_habit"]
    return [canonical]


def _export_choice_summary_files(evaluation_out: Path, artifact_root: Path) -> None:
    choice_root = evaluation_out / "choice_validation"
    out_root = artifact_root / "choice_validation"
    required_map: dict[Path, Path] = {
        choice_root / "unperturbed" / "_summary" / "posthoc_metrics_all.csv": out_root
        / "unperturbed"
        / "posthoc_metrics_all.csv",
        choice_root / "unperturbed" / "_summary" / "posthoc_metrics_bootstrap.csv": out_root
        / "unperturbed"
        / "posthoc_metrics_bootstrap.csv",
        choice_root / "unperturbed" / "_summary" / "strategy_bootstrap_cis.csv": out_root
        / "unperturbed"
        / "strategy_bootstrap_cis.csv",
        choice_root / "unperturbed" / "_summary" / "cell_bootstrap_cis.csv": out_root
        / "unperturbed"
        / "cell_bootstrap_cis.csv",
        choice_root / "unperturbed" / "_summary" / "outcome_bootstrap_cis.csv": out_root
        / "unperturbed"
        / "outcome_bootstrap_cis.csv",
        choice_root / "unperturbed" / "_summary" / "ref_benchmark.csv": out_root / "unperturbed" / "ref_benchmark.csv",
        choice_root / "perturbed" / "_summary" / "all_outcomes_models_strategies.csv": out_root
        / "perturbed"
        / "all_outcomes_models_strategies.csv",
        choice_root / "unperturbed_block_ablation_canonical" / "_summary" / "unperturbed_block_ablation.csv": out_root
        / "unperturbed_block_ablation.csv",
    }
    missing = [str(src) for src in required_map if not src.exists()]
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing required evaluation choice-summary inputs:\n{joined}")
    for src, dst in required_map.items():
        _copy_file(src, dst)


def _load_ref_ice_for_outcome(ref_root: Path, outcome: str) -> pl.DataFrame:
    src: Path | None = None
    for outcome_key in _outcome_candidates_for_lookup(outcome):
        out_dir = ref_root / outcome_key
        if not out_dir.exists():
            continue
        candidates = [out_dir / "ice_ref.csv", *sorted(out_dir.glob("ice_ref__*.csv"))]
        src = next((path for path in candidates if path.exists()), None)
        if src is not None:
            break
    if src is None:
        return pl.DataFrame()
    df = pl.read_csv(src)
    required = {"id", "target", "ice_low", "ice_high", "delta_z"}
    if not required.issubset(set(df.columns)):
        return pl.DataFrame()
    return (
        df.select("id", "target", "ice_low", "ice_high", "delta_z")
        .with_columns(
            (pl.col("ice_high") - pl.col("ice_low")).alias("pe_ref"),
            pl.col("delta_z").cast(pl.Float64),
        )
        .with_columns(
            pl.when(pl.col("delta_z").is_finite() & (pl.col("delta_z") != 0))
            .then(pl.col("pe_ref") / pl.col("delta_z"))
            .otherwise(float("nan"))
            .alias("pe_ref_std")
        )
        .select(
            pl.col("id").cast(pl.Utf8).alias("id"),
            pl.col("target").cast(pl.Utf8).alias("lever"),
            pl.col("ice_low").cast(pl.Float64).alias("ice_low_ref"),
            pl.col("ice_high").cast(pl.Float64).alias("ice_high_ref"),
            pl.col("pe_ref_std").cast(pl.Float64),
        )
        .drop_nulls("pe_ref_std")
    )


def _export_perturbed_compact_tables(repo_root: Path, evaluation_out: Path, artifact_root: Path) -> None:
    perturbed_root = evaluation_out / "choice_validation" / "perturbed"
    out_root = artifact_root / "choice_validation" / "perturbed"
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[pl.DataFrame] = []
    for summary_path in sorted(perturbed_root.rglob("summary_metrics.csv")):
        try:
            rel = summary_path.relative_to(perturbed_root)
        except ValueError:
            continue
        if len(rel.parts) < 5:
            continue
        outcome, model, strategy, calibration = rel.parts[0], rel.parts[1], rel.parts[2], rel.parts[3]
        if calibration not in {"calibrated", "uncalibrated"}:
            continue
        frame = pl.read_csv(summary_path)
        if frame.is_empty():
            continue
        summary_rows.append(
            frame.head(1).with_columns(
                pl.all().cast(pl.Float64, strict=False),
                pl.lit(outcome).alias("outcome"),
                pl.lit(model).alias("model"),
                pl.lit(strategy).alias("strategy"),
                pl.lit(calibration).alias("calibration"),
            )
        )

    summary_cells = pl.concat(summary_rows, how="diagonal") if summary_rows else pl.DataFrame()
    summary_cells.write_csv(out_root / "summary_metrics_cells.csv")

    levers_cfg_path = repo_root / "config" / "levers.json"
    lever_config_raw = json.loads(levers_cfg_path.read_text(encoding="utf-8"))
    lever_config = lever_config_raw if isinstance(lever_config_raw, dict) else {}
    model_plan_path = repo_root / "empirical" / "output" / "modeling" / "model_plan.json"
    model_plan_obj = json.loads(model_plan_path.read_text(encoding="utf-8")) if model_plan_path.exists() else []
    model_plan_entries = model_plan_obj if isinstance(model_plan_obj, list) else []
    model_lobo_path = repo_root / "empirical" / "output" / "modeling" / "model_lobo_all.csv"
    model_lobo = pl.read_csv(model_lobo_path) if model_lobo_path.exists() else pl.DataFrame()
    ref_root = repo_root / "empirical" / "output" / "anchors" / "pe" / "mutable_engines" / "reduced"

    lever_rows: list[dict[str, object]] = []
    block_alignment_rows: list[dict[str, object]] = []
    repeatability_rows: list[dict[str, object]] = []
    reference_model_by_outcome: dict[str, str] = {}
    block_order_map: dict[str, list[str]] = {}
    lever_block_rows: list[dict[str, str]] = []
    raw_outcomes = (
        sorted({str(v) for v in summary_cells.get_column("outcome").to_list()})
        if not summary_cells.is_empty() and "outcome" in summary_cells.columns
        else []
    )
    outcome_groups: dict[str, list[str]] = {}
    for raw_outcome in raw_outcomes:
        canonical = _canonical_outcome_name(raw_outcome)
        outcome_groups.setdefault(canonical, []).append(raw_outcome)
    outcomes = sorted(outcome_groups)

    def _config_for_outcome(outcome: str) -> dict[str, object]:
        for key in _outcome_candidates_for_lookup(outcome):
            payload = lever_config.get(key)
            if isinstance(payload, dict):
                return payload
        return {}

    def _select_model_plan_entry(outcome: str, model_ref: str | None) -> dict[str, object] | None:
        candidates: list[dict[str, object]] = []
        for row in model_plan_entries:
            if not isinstance(row, dict):
                continue
            row_outcome = _canonical_outcome_name(str(row.get("outcome", "")).strip())
            if row_outcome != _canonical_outcome_name(outcome):
                continue
            predictors = row.get("predictors")
            if not isinstance(predictors, dict):
                continue
            candidates.append(row)
        if not candidates:
            return None
        if model_ref:
            for row in candidates:
                if str(row.get("model", "")).strip() == str(model_ref).strip():
                    return row
        return candidates[0]

    def _repeatability_summary(rows: list[dict[str, float]]) -> dict[str, float]:
        if not rows:
            return {
                "n": 0.0,
                "mean_within_sd": float("nan"),
                "median_within_sd": float("nan"),
                "mean_abs_bias": float("nan"),
                "median_abs_bias": float("nan"),
                "var_share": float("nan"),
                "bias_share": float("nan"),
            }
        within_sd = np.asarray([float(r["within_sd"]) for r in rows], dtype=np.float64)
        abs_bias = np.asarray([float(r["abs_bias"]) for r in rows], dtype=np.float64)
        within_var = np.asarray([float(r["within_var"]) for r in rows], dtype=np.float64)
        bias_sq = np.asarray([float(r["bias_sq"]) for r in rows], dtype=np.float64)
        mse = np.asarray([float(r["mse"]) for r in rows], dtype=np.float64)
        total = float(np.sum(mse))
        var_share = float(np.sum(within_var) / total) if total > 0 else float("nan")
        bias_share = float(np.sum(bias_sq) / total) if total > 0 else float("nan")
        return {
            "n": float(len(rows)),
            "mean_within_sd": float(np.mean(within_sd)),
            "median_within_sd": float(np.median(within_sd)),
            "mean_abs_bias": float(np.mean(abs_bias)),
            "median_abs_bias": float(np.median(abs_bias)),
            "var_share": var_share,
            "bias_share": bias_share,
        }

    for outcome in outcomes:
        config_payload = _config_for_outcome(outcome)
        keep_levers = config_payload.get("levers", []) if isinstance(config_payload, dict) else []
        keep_levers = [str(v) for v in keep_levers if str(v).strip()]
        model_ref = str(config_payload.get("model_ref", "")).strip() if isinstance(config_payload, dict) else ""
        if model_ref:
            reference_model_by_outcome[outcome] = model_ref

        entry = _select_model_plan_entry(outcome, model_ref or None)
        predictors = entry.get("predictors", {}) if isinstance(entry, dict) else {}
        predictors = predictors if isinstance(predictors, dict) else {}
        blocks = entry.get("blocks", []) if isinstance(entry, dict) else []
        block_order_map[outcome] = [str(v) for v in blocks if isinstance(v, str) and str(v).strip()]
        lever_to_block: dict[str, str] = {}
        for block, cols in predictors.items():
            if not isinstance(block, str) or not isinstance(cols, list):
                continue
            for col in cols:
                lever = str(col).strip()
                if lever:
                    lever_to_block[lever] = str(block).strip()
        for lever, block in lever_to_block.items():
            lever_block_rows.append({"outcome": outcome, "lever": lever, "block": block})

        ref_df = _load_ref_ice_for_outcome(ref_root, outcome)
        if ref_df.is_empty() or not keep_levers:
            continue

        for raw_outcome in outcome_groups.get(outcome, [outcome]):
            for ice_path in sorted((perturbed_root / raw_outcome).rglob("calibrated/ice_llm.csv")):
                try:
                    rel = ice_path.relative_to(perturbed_root)
                except ValueError:
                    continue
                if len(rel.parts) < 5:
                    continue
                model = rel.parts[1]
                strategy = rel.parts[2]
                llm_raw = pl.read_csv(ice_path)
                if llm_raw.is_empty() or "target" not in llm_raw.columns:
                    continue
                select_cols = [
                    pl.col("id").cast(pl.Utf8).alias("id"),
                    pl.col("target").cast(pl.Utf8).alias("lever"),
                    pl.col("pe_llm_std").cast(pl.Float64),
                ]
                if "ice_low_samples" in llm_raw.columns:
                    select_cols.append(pl.col("ice_low_samples").cast(pl.Utf8, strict=False).alias("ice_low_samples"))
                if "ice_high_samples" in llm_raw.columns:
                    select_cols.append(pl.col("ice_high_samples").cast(pl.Utf8, strict=False).alias("ice_high_samples"))
                llm = (
                    llm_raw.filter(pl.col("target").is_in(sorted(keep_levers)))
                    .with_columns(
                        (pl.col("ice_high") - pl.col("ice_low")).alias("pe_llm"),
                        pl.col("delta_z").cast(pl.Float64),
                    )
                    .with_columns(
                        pl.when(pl.col("delta_z").is_finite() & (pl.col("delta_z") != 0))
                        .then(pl.col("pe_llm") / pl.col("delta_z"))
                        .otherwise(float("nan"))
                        .alias("pe_llm_std")
                    )
                    .select(select_cols)
                )
                joined = llm.join(ref_df, on=["id", "lever"], how="inner")
                if joined.is_empty():
                    continue
                for lever in keep_levers:
                    sub = joined.filter(pl.col("lever") == lever)
                    if sub.is_empty():
                        continue
                    pe_ref = sub.get_column("pe_ref_std").to_numpy()
                    pe_llm = sub.get_column("pe_llm_std").to_numpy()
                    mask = np.isfinite(pe_ref) & np.isfinite(pe_llm)
                    active = mask & (np.abs(pe_ref) > 0.03)
                    use = active if int(active.sum()) >= 3 else mask
                    if int(use.sum()) < 3:
                        continue
                    gini = _gini_np(np.abs(pe_ref[use]), np.abs(pe_llm[use]))
                    abs_err = np.abs(pe_llm[use] - pe_ref[use])
                    denom = float(np.mean(np.abs(pe_ref[use])))
                    mae = float(np.mean(abs_err))
                    skill = (
                        float(1.0 - mae / denom)
                        if np.isfinite(mae) and np.isfinite(denom) and denom > 0
                        else float("nan")
                    )
                    lever_rows.append(
                        {
                            "outcome": outcome,
                            "model": model,
                            "strategy": strategy,
                            "lever": lever,
                            "gini_pe": _safe_float(gini),
                            "skill_pe0": _safe_float(skill),
                        }
                    )

                if lever_to_block:
                    block_df = pl.DataFrame([{"lever": k, "block": v} for k, v in lever_to_block.items()])
                    block_joined = joined.join(block_df, on="lever", how="inner")
                    for block in sorted({str(v) for v in block_joined.get_column("block").drop_nulls().to_list()}):
                        sub = block_joined.filter(pl.col("block") == block)
                        if sub.is_empty():
                            continue
                        sub_active = sub.filter(pl.col("pe_ref_std").abs() > 0.03)
                        sub_use = sub_active if sub_active.height >= 3 else sub
                        if sub_use.height < 3:
                            continue
                        m_ref = _safe_float(sub_use.select(pl.col("pe_ref_std").abs().median()).item())
                        m_llm = _safe_float(sub_use.select(pl.col("pe_llm_std").abs().median()).item())
                        if np.isfinite(m_ref) and np.isfinite(m_llm):
                            block_alignment_rows.append(
                                {
                                    "outcome": outcome,
                                    "model": model,
                                    "strategy": strategy,
                                    "block": block,
                                    "m_ref": m_ref,
                                    "m_llm": m_llm,
                                }
                            )

                if "ice_low_samples" in joined.columns and "ice_high_samples" in joined.columns:
                    for row in joined.select(
                        "ice_low_samples",
                        "ice_high_samples",
                        "ice_low_ref",
                        "ice_high_ref",
                    ).iter_rows(named=True):
                        try:
                            low_vals = ast.literal_eval(str(row.get("ice_low_samples")))
                            high_vals = ast.literal_eval(str(row.get("ice_high_samples")))
                        except (ValueError, SyntaxError):
                            continue
                        low = np.asarray(low_vals, dtype=np.float64)
                        high = np.asarray(high_vals, dtype=np.float64)
                        low = low[np.isfinite(low)]
                        high = high[np.isfinite(high)]
                        n = min(low.size, high.size)
                        if n == 0:
                            continue
                        pe_samples = high[:n] - low[:n]
                        pe_ref = _safe_float(_safe_float(row.get("ice_high_ref")) - _safe_float(row.get("ice_low_ref")))
                        if not np.isfinite(pe_ref):
                            continue
                        pe_mean = float(np.mean(pe_samples))
                        within_var = float(np.mean((pe_samples - pe_mean) ** 2))
                        bias = float(pe_mean - pe_ref)
                        repeatability_rows.append(
                            {
                                "strategy": strategy,
                                "within_var": within_var,
                                "within_sd": float(np.sqrt(within_var)),
                                "abs_bias": abs(bias),
                                "bias_sq": float(bias**2),
                                "mse": float(np.mean((pe_samples - pe_ref) ** 2)),
                            }
                        )

    lever_df = pl.DataFrame(lever_rows) if lever_rows else pl.DataFrame()
    lever_df.write_csv(out_root / "lever_metrics.csv")

    lever_block_df = pl.DataFrame(lever_block_rows) if lever_block_rows else pl.DataFrame()
    if not lever_block_df.is_empty():
        lever_block_df.write_csv(out_root / "lever_block_map.csv")

    block_metrics = (
        lever_df.join(lever_block_df, on=["outcome", "lever"], how="inner")
        if (not lever_df.is_empty() and not lever_block_df.is_empty())
        else pl.DataFrame()
    )
    if not block_metrics.is_empty():
        block_metrics = block_metrics.group_by("outcome", "model", "strategy", "block").agg(
            pl.median("gini_pe").alias("gini_block_median"),
            pl.median("skill_pe0").alias("skill_block_median"),
        )
    block_metrics.write_csv(out_root / "block_metrics.csv")

    block_sensitivity = pl.DataFrame(block_alignment_rows) if block_alignment_rows else pl.DataFrame()
    if not block_sensitivity.is_empty():
        block_sensitivity = (
            block_sensitivity.with_columns(
                pl.col("m_ref").cast(pl.Float64, strict=False),
                pl.col("m_llm").cast(pl.Float64, strict=False),
            )
            .group_by("outcome", "block")
            .agg(
                pl.median("m_ref").alias("m_ref"),
                pl.median("m_llm").alias("m_llm"),
            )
        )

    importance_rows: list[dict[str, object]] = []
    if not model_lobo.is_empty():
        for outcome in outcomes:
            model_ref = reference_model_by_outcome.get(outcome)
            outcome_df = model_lobo.filter(
                pl.col("kind").cast(pl.Utf8) == "lobo",
            )
            outcome_df = outcome_df.filter(
                pl.col("outcome").cast(pl.Utf8).is_in(_outcome_candidates_for_lookup(outcome))
            )
            if model_ref:
                outcome_df = outcome_df.filter(pl.col("model").cast(pl.Utf8) == model_ref)
            if outcome_df.is_empty():
                continue
            weighted = (
                outcome_df.select(
                    pl.col("block").cast(pl.Utf8).alias("block"),
                    pl.col("lobo_bic_weight").cast(pl.Float64, strict=False).alias("importance_share"),
                )
                .filter(pl.col("importance_share").is_finite() & (pl.col("importance_share") >= 0.0))
                .group_by("block")
                .agg(pl.median("importance_share").alias("importance_share"))
            )
            if weighted.is_empty():
                continue
            total = _safe_float(weighted.select(pl.sum("importance_share")).item())
            if np.isfinite(total) and total > 0:
                weighted = weighted.with_columns((pl.col("importance_share") / pl.lit(total)).alias("importance_share"))
            for row in weighted.iter_rows(named=True):
                importance_rows.append(
                    {
                        "outcome": outcome,
                        "block": str(row.get("block")),
                        "importance_share": _safe_float(row.get("importance_share")),
                    }
                )

    importance_df = pl.DataFrame(importance_rows) if importance_rows else pl.DataFrame()
    if not block_sensitivity.is_empty() and not importance_df.is_empty():
        block_sensitivity = block_sensitivity.join(importance_df, on=["outcome", "block"], how="left")
    block_sensitivity.write_csv(out_root / "block_sensitivity.csv")

    repeatability_overall = _repeatability_summary(
        [{k: float(v) for k, v in row.items() if k != "strategy"} for row in repeatability_rows],
    )
    by_strategy: dict[str, dict[str, float]] = {}
    for strategy in sorted({str(row.get("strategy")) for row in repeatability_rows}):
        subset = [
            {k: float(v) for k, v in row.items() if k != "strategy"}
            for row in repeatability_rows
            if str(row.get("strategy")) == strategy
        ]
        by_strategy[strategy] = _repeatability_summary(subset)
    (out_root / "repeatability_summary.json").write_text(
        json.dumps({"overall": repeatability_overall, "by_strategy": by_strategy}, indent=2),
        encoding="utf-8",
    )

    (out_root / "reference_model_by_outcome.json").write_text(
        json.dumps(reference_model_by_outcome, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_root / "block_order_map.json").write_text(
        json.dumps(block_order_map, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _export_belief_summary_files(evaluation_out: Path, artifact_root: Path) -> None:
    belief_root = evaluation_out / "belief_update_validation"
    out_root = artifact_root / "belief_update_validation"
    required_map: dict[Path, Path] = {
        belief_root / "summary_metrics" / "all_models_strategies.csv": out_root / "all_models_strategies.csv",
        belief_root / "summary_metrics" / "all_models_signals.csv": out_root / "all_models_signals.csv",
    }
    missing = [str(src) for src in required_map if not src.exists()]
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing required evaluation belief-summary inputs:\n{joined}")
    for src, dst in required_map.items():
        _copy_file(src, dst)

    unpert_src = belief_root / "unperturbed" / "summary_metrics" / "all_models_strategies.csv"
    unpert_dst = out_root / "unperturbed_all_models_strategies.csv"
    if unpert_src.exists():
        _copy_file(unpert_src, unpert_dst)
    else:
        unpert_dst.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame().write_csv(unpert_dst)

    _copy_file(
        belief_root / "summary_metrics" / "all_models_strategies.csv",
        out_root / "frontier_metrics.csv",
    )


def _export_belief_unperturbed_engine_metrics(evaluation_out: Path, artifact_root: Path) -> None:
    root = evaluation_out / "belief_update_validation" / "unperturbed" / "canonical"
    rows: list[dict[str, object]] = []
    vector_rows: list[dict[str, object]] = []
    if root.exists():
        for model_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
            for strategy_dir in sorted(p for p in model_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
                metrics_path = strategy_dir / "metrics.parquet"
                if not metrics_path.exists():
                    continue
                frame = pl.read_parquet(metrics_path)
                if frame.is_empty():
                    continue
                columns = set(frame.columns)
                for column in sorted(col for col in columns if col.startswith("true_")):
                    engine = column[5:]
                    pred_col = f"pred_{engine}"
                    if pred_col not in columns:
                        continue
                    true_vals = frame.select(pl.col(column).cast(pl.Float64, strict=False)).to_series().to_numpy()
                    pred_vals = frame.select(pl.col(pred_col).cast(pl.Float64, strict=False)).to_series().to_numpy()
                    rows.append(
                        {
                            "variant": "canonical",
                            "model": model_dir.name,
                            "strategy": strategy_dir.name,
                            "engine": engine,
                            "gini": _safe_float(_gini_np(true_vals, pred_vals)),
                            "skill": _safe_float(_mae_skill_cal_np(true_vals, pred_vals)),
                        }
                    )

                engines = sorted(
                    {
                        col[5:]
                        for col in columns
                        if col.startswith("true_") and f"pred_{col[5:]}" in columns and col[5:] != "vaccine_risk_avg"
                    }
                )
                if len(engines) >= 2:
                    true_mat = np.column_stack(
                        [
                            frame.select(pl.col(f"true_{engine}").cast(pl.Float64, strict=False)).to_series().to_numpy()
                            for engine in engines
                        ]
                    )
                    pred_mat = np.column_stack(
                        [
                            frame.select(pl.col(f"pred_{engine}").cast(pl.Float64, strict=False)).to_series().to_numpy()
                            for engine in engines
                        ]
                    )
                    mu_true = np.nanmean(true_mat, axis=0)
                    sd_true = np.nanstd(true_mat, axis=0, ddof=1)
                    valid = np.isfinite(sd_true) & (sd_true > 0)
                    if int(valid.sum()) >= 2:
                        true_z = (true_mat[:, valid] - mu_true[valid]) / sd_true[valid]
                        pred_z = (pred_mat[:, valid] - mu_true[valid]) / sd_true[valid]
                        row_scores = []
                        for idx in range(true_z.shape[0]):
                            xv = pred_z[idx, :]
                            yv = true_z[idx, :]
                            mask = np.isfinite(xv) & np.isfinite(yv)
                            if int(mask.sum()) < 2:
                                continue
                            nx = float(np.linalg.norm(xv[mask]))
                            ny = float(np.linalg.norm(yv[mask]))
                            if nx <= 0 or ny <= 0:
                                continue
                            row_scores.append(float(np.dot(xv[mask], yv[mask]) / (nx * ny)))
                        if row_scores:
                            vector_rows.append(
                                {
                                    "model": model_dir.name,
                                    "strategy": strategy_dir.name,
                                    "vector_cosine_std": _safe_float(float(np.mean(row_scores))),
                                }
                            )
    out = pl.DataFrame(rows) if rows else pl.DataFrame()
    dst = artifact_root / "belief_update_validation" / "unperturbed_engine_metrics.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(dst)

    vector_df = pl.DataFrame(vector_rows) if vector_rows else pl.DataFrame()
    vector_df.write_csv(artifact_root / "belief_update_validation" / "unperturbed_vector_cosine.csv")

    core_engines = [
        "institutional_trust_avg",
        "group_trust_vax",
        "social_norms_vax_avg",
        "covid_perceived_danger_T12",
        "fivec_low_constraints",
    ]
    compression_rows: list[dict[str, object]] = []
    model_strategy_rows = (
        vector_df.select("model", "strategy").unique().iter_rows(named=True) if not vector_df.is_empty() else []
    )
    for row in model_strategy_rows:
        model = str(row.get("model"))
        strategy = str(row.get("strategy"))
        for engine in core_engines:
            metrics_path = root / model / strategy / "metrics.parquet"
            if not metrics_path.exists():
                continue
            frame = pl.read_parquet(metrics_path)
            if frame.is_empty():
                continue
            true_col = f"true_{engine}"
            pred_col = f"pred_{engine}"
            if true_col not in frame.columns or pred_col not in frame.columns:
                continue
            pair = frame.select(
                pl.col(true_col).cast(pl.Float64, strict=False).alias("y_true"),
                pl.col(pred_col).cast(pl.Float64, strict=False).alias("y_pred"),
            ).drop_nulls()
            if pair.height < 5:
                continue
            y_true = pair.get_column("y_true").to_numpy()
            y_pred = pair.get_column("y_pred").to_numpy()
            iqr_true = float(np.quantile(y_true, 0.75) - np.quantile(y_true, 0.25))
            iqr_pred = float(np.quantile(y_pred, 0.75) - np.quantile(y_pred, 0.25))
            if not np.isfinite(iqr_true) or iqr_true <= 0 or not np.isfinite(iqr_pred):
                continue
            compression_rows.append(
                {
                    "model": model,
                    "strategy": strategy,
                    "engine": engine,
                    "source_variant": "canonical",
                    "ratio": _safe_float(iqr_pred / iqr_true),
                    "n_rows": float(pair.height),
                }
            )
    compression_df = pl.DataFrame(compression_rows) if compression_rows else pl.DataFrame()
    compression_df.write_csv(artifact_root / "belief_update_validation" / "unperturbed_compression.csv")


def _export_belief_signal_block_alignment(evaluation_out: Path, artifact_root: Path) -> None:
    canonical_root = evaluation_out / "belief_update_validation" / "canonical"
    rows: list[dict[str, object]] = []
    if canonical_root.exists():
        for model_dir in sorted(p for p in canonical_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
            for strategy_dir in sorted(p for p in model_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
                signal_root = strategy_dir / "signals"
                if not signal_root.exists():
                    continue
                for signal_dir in sorted(p for p in signal_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
                    if signal_dir.name == "__unperturbed__":
                        continue
                    path = signal_dir / "pe_alignment_rows.parquet"
                    if not path.exists():
                        continue
                    frame = pl.read_parquet(path)
                    if frame.is_empty():
                        continue
                    required = {"pe_ref_p_std", "pe_llm_p_std"}
                    if not required.issubset(set(frame.columns)):
                        continue
                    rows.append(
                        {
                            "model": model_dir.name,
                            "strategy": strategy_dir.name,
                            "signal_id": signal_dir.name,
                            "ref_abs_median": _safe_float(
                                frame.select(pl.col("pe_ref_p_std").abs().median()).item(),
                            ),
                            "llm_abs_median": _safe_float(
                                frame.select(pl.col("pe_llm_p_std").abs().median()).item(),
                            ),
                        }
                    )
    out = pl.DataFrame(rows) if rows else pl.DataFrame()
    dst = artifact_root / "belief_update_validation" / "signal_block_alignment.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(dst)


def export_thesis_artifacts(repo_root: Path) -> None:
    evaluation_out = repo_root / "evaluation" / "output"
    artifacts_root = repo_root / "thesis" / "artifacts" / "evaluation"
    _export_choice_summary_files(evaluation_out, artifacts_root)
    _export_perturbed_compact_tables(repo_root, evaluation_out, artifacts_root)
    _export_belief_summary_files(evaluation_out, artifacts_root)
    _export_belief_unperturbed_engine_metrics(evaluation_out, artifacts_root)
    _export_belief_signal_block_alignment(evaluation_out, artifacts_root)


__all__ = ["export_thesis_artifacts"]
