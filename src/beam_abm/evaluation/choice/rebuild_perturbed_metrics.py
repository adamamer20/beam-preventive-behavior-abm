"""Recompute perturbed choice-validation metrics from existing raw outputs.

Primary use cases:
- After migrating/copying legacy runs into
  evaluation/output/choice_validation/perturbed/_runs/...
- After editing metric code or reference anchors.

This script does NOT rerun LLM sampling.

It can optionally ingest existing prompt-tournament outputs (copy-only) into the
canonical `_runs/` layout, with a migration log.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from beam_abm.evaluation.choice.canonicalize import normalize_model_slug
from beam_abm.evaluation.choice.perturbed import (
    ingest_prompt_tournament_root_to_runs,
    normalize_model_dirname,
    rebuild_canonical_from_runs,
    rename_prompt_tournament_model_dirs,
    write_global_summary,
)
from beam_abm.evaluation.common.metrics import gini_alignment, spearman_alignment
from beam_abm.evaluation.utils.alignment_utils import summarize_alignment_rows


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start


@lru_cache(maxsize=4)
def _load_model_plan(model_plan_path: str) -> list[dict]:
    path = Path(model_plan_path)
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"model_plan.json must be a list, got {type(obj).__name__}")
    return [x for x in obj if isinstance(x, dict)]


def _select_model_plan_entry(plan: list[dict], *, outcome: str, model_name: str) -> dict | None:
    for entry in plan:
        if entry.get("outcome") != outcome:
            continue
        if str(entry.get("model") or "") != model_name:
            continue
        if not isinstance(entry.get("predictors"), dict):
            continue
        return entry
    return None


@lru_cache(maxsize=256)
def _target_to_block_map(model_plan_path: str, outcome: str, model_name: str) -> dict[str, str]:
    plan = _load_model_plan(model_plan_path)
    entry = _select_model_plan_entry(plan, outcome=outcome, model_name=model_name)
    if entry is None:
        return {}

    predictors = entry.get("predictors")
    if not isinstance(predictors, dict):
        return {}

    out: dict[str, str] = {}
    for block, preds in predictors.items():
        if not isinstance(block, str) or not block.strip():
            continue
        if not isinstance(preds, list):
            continue
        for p in preds:
            if not isinstance(p, str) or not p.strip():
                continue
            # If there is disagreement across blocks, keep the first-seen mapping.
            # (These model_plan blocks should be mutually exclusive in practice.)
            out.setdefault(p, block)
    return out


def _compute_block_alignment(
    *,
    status_dir: Path,
    outcome: str,
    ref: str,
    ref_ice: Path,
    llm_ice: Path,
    model_plan_path: Path,
    ref_model_by_outcome: dict[str, str],
    topk: int,
) -> dict[str, float]:
    """Compute block-level alignment metrics for this leaf.

    Returns scalar metrics suitable for attaching to summary_metrics.
    """

    from beam_abm.evaluation.common.compute_block_alignment import main as block_main

    try:
        llm_df = pd.read_csv(llm_ice, usecols=["target"])  # type: ignore[arg-type]
    except (OSError, ValueError, pd.errors.ParserError):
        return {}
    if llm_df.empty or "target" not in llm_df.columns:
        return {}

    targets = sorted({t for t in llm_df["target"].astype(str).tolist() if t and t != "nan"})
    if not targets:
        return {}

    model_ref = str(ref_model_by_outcome.get(outcome, "")).strip()
    if not model_ref:
        return {}

    target_to_block = _target_to_block_map(str(model_plan_path), outcome, model_ref)
    if not target_to_block:
        return {}

    n_targets = len(targets)
    n_targets_mapped = sum(1 for t in targets if t in target_to_block)

    block_map_path = status_dir / "_block_map.csv"
    block_map_df = pd.DataFrame(
        [{"target": t, "block": target_to_block.get(t, "")} for t in targets],
    )
    block_map_df.to_csv(block_map_path, index=False)

    block_out = status_dir / ("_block_alignment.csv" if ref == "linear" else f"_block_alignment__ref_{ref}.csv")

    saved = list(sys.argv)
    try:
        sys.argv = [
            sys.argv[0],
            "--ref-ice",
            str(ref_ice),
            "--llm-ice",
            str(llm_ice),
            "--block-map",
            str(block_map_path),
            "--topk",
            str(int(topk)),
            "--out",
            str(block_out),
        ]
        block_main()
    except (SystemExit, ValueError):
        return {}
    finally:
        sys.argv = saved

    try:
        block_df = pd.read_csv(block_out) if block_out.exists() else pd.DataFrame([])
    except (OSError, ValueError, pd.errors.ParserError):
        block_df = pd.DataFrame([])

    if block_df.empty:
        return {
            "bsa_target_map_n_targets": float(n_targets),
            "bsa_target_map_n_targets_mapped": float(n_targets_mapped),
            "bsa_target_map_coverage": float(n_targets_mapped / n_targets) if n_targets else float("nan"),
        }

    first = block_df.iloc[0].to_dict()
    out = {
        "bsa": float(first.get("bsa")) if first.get("bsa") is not None else float("nan"),
        "bsa_topk_overlap": float(first.get("bsa_topk_overlap"))
        if first.get("bsa_topk_overlap") is not None
        else float("nan"),
        "bsa_highest_quartile_concordance": float(first.get("bsa_highest_quartile_concordance"))
        if first.get("bsa_highest_quartile_concordance") is not None
        else float("nan"),
        "bsa_n_blocks": float(block_df["block"].nunique()) if "block" in block_df.columns else float("nan"),
        "bsa_target_map_n_targets": float(n_targets),
        "bsa_target_map_n_targets_mapped": float(n_targets_mapped),
        "bsa_target_map_coverage": float(n_targets_mapped / n_targets) if n_targets else float("nan"),
    }
    return out


def _write_migration_log_line(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _attach_alignment_dispersion_metrics(summary: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    if metrics_df.empty:
        return
    if not {"pe_ref_std", "pe_llm_std"}.issubset(metrics_df.columns):
        return
    if summary.empty:
        summary.loc[0, "n"] = 0

    ref = pd.to_numeric(metrics_df["pe_ref_std"], errors="coerce").to_numpy(dtype=float)
    llm = pd.to_numeric(metrics_df["pe_llm_std"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(ref) & np.isfinite(llm)
    if mask.sum() < 2:
        summary.loc[0, "pe_std_var_ratio"] = float("nan")
        summary.loc[0, "spearman_pe_std_abs"] = float("nan")
        summary.loc[0, "spearman_pe_std_pos"] = float("nan")
        summary.loc[0, "spearman_pe_std_neg"] = float("nan")
        summary.loc[0, "gini_pe_std_abs"] = float("nan")
        summary.loc[0, "gini_pe_std_pos"] = float("nan")
        summary.loc[0, "gini_pe_std_neg"] = float("nan")
        summary.loc[0, "gini_pe_std_mean"] = float("nan")
        summary.loc[0, "gini_pe_mean"] = float("nan")
        return

    ref_use = ref[mask]
    llm_use = llm[mask]
    summary.loc[0, "gini_pe_std_mean"] = gini_alignment(ref_use, llm_use)
    var_ref = float(np.var(ref_use, ddof=1)) if ref_use.size > 1 else float("nan")
    var_llm = float(np.var(llm_use, ddof=1)) if llm_use.size > 1 else float("nan")
    summary.loc[0, "pe_std_var_ratio"] = float(var_llm / var_ref) if var_ref > 0 else float("nan")
    summary.loc[0, "spearman_pe_std_abs"] = spearman_alignment(np.abs(ref_use), np.abs(llm_use))
    summary.loc[0, "gini_pe_std_abs"] = gini_alignment(np.abs(ref_use), np.abs(llm_use))

    pos_mask = mask & (ref > 0)
    if pos_mask.sum() >= 2:
        summary.loc[0, "spearman_pe_std_pos"] = spearman_alignment(ref[pos_mask], llm[pos_mask])
        summary.loc[0, "gini_pe_std_pos"] = gini_alignment(ref[pos_mask], llm[pos_mask])
    else:
        summary.loc[0, "spearman_pe_std_pos"] = float("nan")
        summary.loc[0, "gini_pe_std_pos"] = float("nan")

    neg_mask = mask & (ref < 0)
    if neg_mask.sum() >= 2:
        summary.loc[0, "spearman_pe_std_neg"] = spearman_alignment(ref[neg_mask], llm[neg_mask])
        summary.loc[0, "gini_pe_std_neg"] = gini_alignment(ref[neg_mask], llm[neg_mask])
    else:
        summary.loc[0, "spearman_pe_std_neg"] = float("nan")
        summary.loc[0, "gini_pe_std_neg"] = float("nan")


def _compute_block_metrics_for_leaf(
    *,
    llm_ice: Path,
    ref_ice: Path,
    block_map: dict[str, str],
    outcome: str,
    model: str,
    strategy: str,
) -> pd.DataFrame:
    try:
        llm_df = pd.read_csv(llm_ice)
        ref_df = pd.read_csv(ref_ice)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame([])

    keep_cols = {"id", "target", "ice_low", "ice_high", "delta_z"}
    if not keep_cols.issubset(llm_df.columns) or not keep_cols.issubset(ref_df.columns):
        return pd.DataFrame([])

    llm_df = llm_df[llm_df["target"].astype(str).isin(block_map.keys())].copy()
    ref_df = ref_df[ref_df["target"].astype(str).isin(block_map.keys())].copy()
    if llm_df.empty or ref_df.empty:
        return pd.DataFrame([])

    llm_df["pe_llm"] = pd.to_numeric(llm_df["ice_high"], errors="coerce") - pd.to_numeric(
        llm_df["ice_low"], errors="coerce"
    )
    llm_df["delta_z"] = pd.to_numeric(llm_df["delta_z"], errors="coerce")
    llm_df["pe_llm_std"] = llm_df["pe_llm"] / llm_df["delta_z"].replace(0, np.nan)

    ref_df["pe_ref"] = pd.to_numeric(ref_df["ice_high"], errors="coerce") - pd.to_numeric(
        ref_df["ice_low"], errors="coerce"
    )
    ref_df["delta_z"] = pd.to_numeric(ref_df["delta_z"], errors="coerce")
    ref_df["pe_ref_std"] = ref_df["pe_ref"] / ref_df["delta_z"].replace(0, np.nan)

    join = llm_df.merge(ref_df, on=["id", "target"], suffixes=("_llm", "_ref"))
    if join.empty:
        return pd.DataFrame([])

    rows: list[dict[str, object]] = []
    for lever in sorted(set(join["target"].astype(str))):
        sub = join[join["target"].astype(str) == lever]
        pe_ref = pd.to_numeric(sub["pe_ref_std"], errors="coerce").to_numpy(dtype=float)
        pe_llm = pd.to_numeric(sub["pe_llm_std"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(pe_ref) & np.isfinite(pe_llm)
        gini = gini_alignment(pe_ref[mask], pe_llm[mask]) if mask.sum() >= 2 else float("nan")

        ice_low_ref = pd.to_numeric(sub["ice_low_ref"], errors="coerce").to_numpy(dtype=float)
        ice_high_ref = pd.to_numeric(sub["ice_high_ref"], errors="coerce").to_numpy(dtype=float)
        ice_low_llm = pd.to_numeric(sub["ice_low_llm"], errors="coerce").to_numpy(dtype=float)
        ice_high_llm = pd.to_numeric(sub["ice_high_llm"], errors="coerce").to_numpy(dtype=float)
        ice_err = np.concatenate([np.abs(ice_low_llm - ice_low_ref), np.abs(ice_high_llm - ice_high_ref)])
        mae = float(np.nanmean(ice_err)) if ice_err.size else float("nan")
        ref_vals = np.concatenate([ice_low_ref, ice_high_ref])
        baseline = float(np.nanmedian(ref_vals)) if ref_vals.size else float("nan")
        baseline_mae = float(np.nanmean(np.abs(ref_vals - baseline))) if ref_vals.size else float("nan")
        skill = float("nan")
        if np.isfinite(mae) and np.isfinite(baseline_mae) and baseline_mae > 0:
            skill = float(1.0 - mae / baseline_mae)

        rows.append(
            {
                "outcome": outcome,
                "model": model,
                "strategy": strategy,
                "lever": lever,
                "block": block_map.get(lever, ""),
                "gini_pe": float(gini) if np.isfinite(gini) else float("nan"),
                "skill_ice": float(skill) if np.isfinite(skill) else float("nan"),
            }
        )

    if not rows:
        return pd.DataFrame([])

    df = pd.DataFrame(rows)
    df = df[df["block"].astype(str).str.strip().astype(bool)]
    if df.empty:
        return pd.DataFrame([])

    return (
        df.groupby(["outcome", "model", "strategy", "block"], as_index=False)
        .agg(
            gini_block_median=("gini_pe", "median"),
            skill_block_median=("skill_ice", "median"),
        )
        .reset_index(drop=True)
    )


def _oracle_uplift_summary(df: pd.DataFrame, value_col: str, *, epsilon: float) -> dict[str, float]:
    if df.empty or value_col not in df.columns:
        return {}

    use = df[pd.to_numeric(df[value_col], errors="coerce").notna()].copy()
    if use.empty:
        return {}

    deltas: list[float] = []
    winners: dict[str, float] = {}
    cell_count = 0
    win_count = 0
    meaningful = 0

    for (_outcome, _block), cell in use.groupby(["outcome", "block"], dropna=False):
        data_only = cell[cell["strategy"] == "data_only"]
        if data_only.empty:
            continue
        s0 = float(data_only[value_col].iloc[0])
        max_val = float(cell[value_col].max())
        if not np.isfinite(max_val) or not np.isfinite(s0):
            continue
        delta = max_val - s0
        deltas.append(delta)
        cell_count += 1

        best = cell[cell[value_col] == max_val]
        best_strats = [str(s) for s in best["strategy"].tolist()]
        if not (len(best_strats) == 1 and best_strats[0] == "data_only"):
            if "data_only" not in best_strats:
                win_count += 1
        if delta > epsilon:
            meaningful += 1
        share = 1.0 / len(best_strats) if best_strats else 0.0
        for s in best_strats:
            winners[s] = winners.get(s, 0.0) + share

    if cell_count == 0:
        return {}

    deltas_arr = np.array(deltas, dtype=float)
    med = float(np.nanmedian(deltas_arr))
    q25 = float(np.nanquantile(deltas_arr, 0.25))
    q75 = float(np.nanquantile(deltas_arr, 0.75))
    win_rate = win_count / cell_count if cell_count else float("nan")
    meaningful_rate = meaningful / cell_count if cell_count else float("nan")

    total = sum(winners.values())
    if total <= 0:
        h_norm = float("nan")
    else:
        weights = np.array([w / total for w in winners.values()])
        h = -np.sum(weights * np.log(weights + 1e-12))
        k = len(weights)
        h_norm = float(h / np.log(k)) if k > 1 else 0.0

    return {
        "n_cells": float(cell_count),
        "median_uplift": med,
        "iqr_low": q25,
        "iqr_high": q75,
        "win_rate": float(win_rate),
        "meaningful_rate": float(meaningful_rate),
        "winner_entropy": float(h_norm),
    }


def _write_oracle_uplift_summary(
    *,
    output_root: Path,
    anchors_root: Path,
    model_plan_path: Path,
    ref_model_by_outcome: dict[str, str],
    epsilon: float = 0.02,
) -> None:
    out_summary = output_root / "_summary"
    out_summary.mkdir(parents=True, exist_ok=True)

    rows: list[pd.DataFrame] = []
    for outcome_dir in sorted(p for p in output_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        outcome = outcome_dir.name
        ref_model = ref_model_by_outcome.get(outcome, "")
        if not ref_model:
            continue
        ref_ice = _resolve_ref_ice_path(
            anchors_root=anchors_root,
            outcome=outcome,
            ref_model="linear",
            use_model_ref_for_linear=True,
            ref_model_by_outcome=ref_model_by_outcome,
        )
        if not ref_ice.exists():
            continue

        block_map = _target_to_block_map(str(model_plan_path), outcome, ref_model)
        if not block_map:
            continue

        for model_dir in sorted(p for p in outcome_dir.iterdir() if p.is_dir()):
            model = model_dir.name
            for strategy_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                status_dir = strategy_dir / "calibrated"
                llm_ice = status_dir / "ice_llm.csv"
                if not llm_ice.exists():
                    continue
                rows.append(
                    _compute_block_metrics_for_leaf(
                        llm_ice=llm_ice,
                        ref_ice=ref_ice,
                        block_map=block_map,
                        outcome=outcome,
                        model=model,
                        strategy=strategy_dir.name,
                    )
                )

    if not rows:
        return

    block_df = pd.concat([r for r in rows if not r.empty], ignore_index=True) if rows else pd.DataFrame([])
    if block_df.empty:
        return

    block_df.to_csv(out_summary / "oracle_uplift_block_cells.csv", index=False)

    summaries: dict[str, object] = {"epsilon": float(epsilon), "by_model": {}}
    for model, sub in block_df.groupby("model"):
        model_summary: dict[str, object] = {
            "gini_pe": _oracle_uplift_summary(sub, "gini_block_median", epsilon=epsilon),
            "skill": _oracle_uplift_summary(sub, "skill_block_median", epsilon=epsilon),
            "per_outcome": {},
            "per_block": {},
        }

        per_outcome: dict[str, dict[str, object]] = {}
        for outcome, sub_o in sub.groupby("outcome"):
            per_outcome[str(outcome)] = {
                "gini_pe": _oracle_uplift_summary(sub_o, "gini_block_median", epsilon=epsilon),
                "skill": _oracle_uplift_summary(sub_o, "skill_block_median", epsilon=epsilon),
            }
        model_summary["per_outcome"] = per_outcome

        per_block: dict[str, dict[str, object]] = {}
        for block, sub_b in sub.groupby("block"):
            per_block[str(block)] = {
                "gini_pe": _oracle_uplift_summary(sub_b, "gini_block_median", epsilon=epsilon),
                "skill": _oracle_uplift_summary(sub_b, "skill_block_median", epsilon=epsilon),
            }
        model_summary["per_block"] = per_block

        summaries["by_model"][str(model)] = model_summary

    (out_summary / "oracle_uplift_summary.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )


def _iter_jsonl_dicts(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _row_outcome(row: dict) -> str | None:
    raw = row.get("outcome")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    meta = row.get("metadata")
    if isinstance(meta, dict):
        m = meta.get("outcome")
        if isinstance(m, str) and m.strip():
            return m.strip()
    rid = row.get("dataset_id") or row.get("id")
    if isinstance(rid, str) and rid.strip() and "__" in rid:
        return rid.split("__", 1)[0].strip() or None
    return None


def _safe_token(value: object) -> str:
    text = str(value) if value is not None else ""
    out = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text).strip("_")
    return out or "unknown"


def _stream_split_jsonl_by_outcome(*, src_path: Path, dst_root_by_outcome: dict[str, Path], filename: str) -> None:
    handles: dict[str, object] = {}
    try:
        with src_path.open("r", encoding="utf-8") as f:
            for line in f:
                line_s = line.strip()
                if not line_s:
                    continue
                try:
                    obj = json.loads(line_s)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                outcome = _row_outcome(obj)
                if not outcome:
                    continue
                dst_root = dst_root_by_outcome.get(outcome)
                if dst_root is None:
                    continue
                out_path = dst_root / filename
                if outcome not in handles:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    handles[outcome] = out_path.open("a", encoding="utf-8")
                handles[outcome].write(line_s + "\n")
    finally:
        for h in handles.values():
            try:
                h.close()  # type: ignore[attr-defined]
            except OSError:
                pass


def _copy_prompt_tournament_calibrated_leaf(*, src_status_dir: Path, dest_status_dir: Path) -> None:
    """Copy a phase0 calibrated leaf into the canonical layout.

    Phase0 calibrated leaves typically do not include samples/prompts, but do
    include an ICE table (ice_llm_calibrated.csv) and row_level.csv.
    """
    dest_status_dir.mkdir(parents=True, exist_ok=True)

    ice_src = src_status_dir / "ice_llm_calibrated.csv"
    if ice_src.exists():
        shutil.copy2(ice_src, dest_status_dir / "ice_llm.csv")

    for name in ["row_level.csv", "run_metadata.json", "summary_metrics.csv"]:
        src = src_status_dir / name
        if src.exists():
            shutil.copy2(src, dest_status_dir / name)

    src_conv = src_status_dir / "conversations"
    if src_conv.exists() and src_conv.is_dir():
        dst_conv = dest_status_dir / "conversations"
        if not dst_conv.exists():
            shutil.copytree(src_conv, dst_conv)


def _ref_ice_path(*, anchors_root: Path, outcome: str, ref_model: str) -> Path:
    # Matches the phase0 prompt tournament convention.
    if ref_model == "linear":
        return anchors_root / outcome / "ice_ref__linear.csv"
    return anchors_root / outcome / f"ice_ref__{ref_model}.csv"


def _load_ref_model_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    out: dict[str, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            outcome = str(k).strip()
            if not outcome:
                continue
            if isinstance(v, str) and v.strip():
                out[outcome] = v.strip()
                continue
            if isinstance(v, dict):
                model_ref = v.get("model_ref")
                if isinstance(model_ref, str) and model_ref.strip():
                    out[outcome] = model_ref.strip()
    return out


def _resolve_ref_ice_path(
    *,
    anchors_root: Path,
    outcome: str,
    ref_model: str,
    use_model_ref_for_linear: bool,
    ref_model_by_outcome: dict[str, str],
) -> Path:
    if ref_model == "linear" and use_model_ref_for_linear:
        mapped = ref_model_by_outcome.get(outcome)
        if mapped:
            candidate = anchors_root / outcome / f"ice_ref__{mapped}.csv"
            if candidate.exists():
                return candidate
    return _ref_ice_path(anchors_root=anchors_root, outcome=outcome, ref_model=ref_model)


def _iter_leaf_status_dirs(root: Path) -> list[Path]:
    out: list[Path] = []

    runs_root = root / "_runs"
    if not runs_root.exists():
        return out

    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for outcome_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
            for model_dir in sorted(p for p in outcome_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
                for strategy_dir in sorted(p for p in model_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
                    for status_dir in sorted(
                        p for p in strategy_dir.iterdir() if p.is_dir() and p.name in {"uncalibrated", "calibrated"}
                    ):
                        out.append(status_dir)

    return out


def _iter_uncalibrated_leaf_status_dirs(root: Path) -> list[Path]:
    out: list[Path] = []

    runs_root = root / "_runs"
    if not runs_root.exists():
        return out

    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for outcome_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
            for model_dir in sorted(p for p in outcome_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
                for strategy_dir in sorted(p for p in model_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
                    status_dir = strategy_dir / "uncalibrated"
                    if status_dir.exists() and status_dir.is_dir():
                        out.append(status_dir)

    return out


def _rebuild_ice(*, status_dir: Path) -> None:
    from beam_abm.evaluation.common.compute_ice_from_samples import main as ice_main

    samples_path = status_dir / "samples.jsonl"
    if not samples_path.exists():
        return

    out_path = status_dir / "ice_llm.csv"
    summary_out = status_dir / "_ice_noncompliance.csv"

    saved = list(sys.argv)
    try:
        sys.argv = [
            sys.argv[0],
            "--in",
            str(samples_path),
            "--out",
            str(out_path),
            "--summary-out",
            str(summary_out),
        ]
        # Ensure paired-profile order handling is stable for this codebase.
        # (This matches the runner's behavior when enabled.)
        sys.argv.append("--paired-profile-swap-on-b-then-a")
        ice_main()
    finally:
        sys.argv = saved


def _rebuild_alignment(
    *,
    status_dir: Path,
    anchors_root: Path,
    ref_models: list[str],
    model_plan_path: Path,
    block_topk: int,
    skip_block_alignment: bool,
    use_model_ref_for_linear: bool,
    ref_model_by_outcome: dict[str, str],
) -> None:
    from beam_abm.evaluation.common.compute_alignment import main as align_main

    ice_path = status_dir / "ice_llm.csv"
    if not ice_path.exists():
        return

    outcome = status_dir.parent.parent.parent.name  # _runs/<run>/<outcome>/<model>/<strategy>/<status>

    for ref in ref_models:
        ref_ice = _resolve_ref_ice_path(
            anchors_root=anchors_root,
            outcome=outcome,
            ref_model=ref,
            use_model_ref_for_linear=use_model_ref_for_linear,
            ref_model_by_outcome=ref_model_by_outcome,
        )

        # Compute row-level alignment table.
        metrics_out = status_dir / (
            "_alignment_metrics.csv" if ref == "linear" else f"_alignment_metrics__ref_{ref}.csv"
        )
        order_out = status_dir / ("_order_effects.csv" if ref == "linear" else f"_order_effects__ref_{ref}.csv")

        if not ref_ice.exists():
            if metrics_out.exists():
                try:
                    metrics_df = pd.read_csv(metrics_out)
                except (OSError, ValueError, pd.errors.ParserError):
                    metrics_df = pd.DataFrame([])

                summary = summarize_alignment_rows(metrics_df)
                _attach_alignment_dispersion_metrics(summary, metrics_df)

                summary_out = status_dir / (
                    "summary_metrics.csv" if ref == "linear" else f"summary_metrics__ref_{ref}.csv"
                )
                summary.to_csv(summary_out, index=False)
            continue

        saved = list(sys.argv)
        try:
            sys.argv = [
                sys.argv[0],
                "--ref",
                str(ref_ice),
                "--llm",
                str(ice_path),
                "--out",
                str(metrics_out),
                "--order-out",
                str(order_out),
            ]
            align_main()
        finally:
            sys.argv = saved

        # Summarize alignment into the canonical summary_metrics.csv.
        try:
            metrics_df = pd.read_csv(metrics_out) if metrics_out.exists() else pd.DataFrame([])
        except (OSError, ValueError, pd.errors.ParserError):
            metrics_df = pd.DataFrame([])

        summary = summarize_alignment_rows(metrics_df)
        _attach_alignment_dispersion_metrics(summary, metrics_df)

        # Attach block-level alignment summary (BSA / overlap / concordance).
        if not skip_block_alignment and model_plan_path.exists():
            block_metrics = _compute_block_alignment(
                status_dir=status_dir,
                outcome=outcome,
                ref=ref,
                ref_ice=ref_ice,
                llm_ice=ice_path,
                model_plan_path=model_plan_path,
                ref_model_by_outcome=ref_model_by_outcome,
                topk=block_topk,
            )
            for k, v in block_metrics.items():
                summary[k] = v

        # Attach ICE noncompliance (JSON/schema compliance proxy) to the summary.
        noncomp_path = status_dir / "_ice_noncompliance.csv"
        if noncomp_path.exists():
            try:
                noncomp = pd.read_csv(noncomp_path)
            except (OSError, ValueError, pd.errors.ParserError):
                noncomp = pd.DataFrame([])
            if not noncomp.empty:
                overall = noncomp[(noncomp["strategy"] == "__all__") & (noncomp["target"] == "__all__")]
                if not overall.empty:
                    row0 = overall.iloc[0].to_dict()
                    summary["ice_n_total"] = row0.get("n_total")
                    summary["ice_n_ok"] = row0.get("n_ok")
                    summary["ice_n_error"] = row0.get("n_error")
                    summary["ice_n_off_manifold"] = row0.get("n_off_manifold")
                    summary["ice_noncompliance_rate"] = row0.get("noncompliance_rate")
                    summary["ice_off_manifold_rate"] = row0.get("off_manifold_rate")

        # Attach paired-profile order effects summary (A_then_B vs B_then_A).
        if order_out.exists():
            try:
                order_df = pd.read_csv(order_out)
            except (OSError, ValueError, pd.errors.ParserError):
                order_df = pd.DataFrame([])
            if not order_df.empty and "paired_order" in order_df.columns:
                by_order = order_df.set_index("paired_order")
                for order_key, prefix in [("A_then_B", "order_A_then_B"), ("B_then_A", "order_B_then_A")]:
                    if order_key in by_order.index:
                        r0 = by_order.loc[order_key]
                        for col in ["n", "dar", "tie_rate", "mean_pe_llm", "mean_pe_llm_std", "mean_delta_x"]:
                            if col in by_order.columns:
                                summary[f"{prefix}__{col}"] = float(r0[col]) if pd.notna(r0[col]) else float("nan")
                if "dar_delta_A_minus_B" in order_df.columns:
                    v = order_df["dar_delta_A_minus_B"].dropna().tolist()
                    summary["order__dar_delta_A_minus_B"] = float(v[0]) if v else float("nan")

        summary_out = status_dir / ("summary_metrics.csv" if ref == "linear" else f"summary_metrics__ref_{ref}.csv")
        summary.to_csv(summary_out, index=False)

        # Keep the row-level alignment tables for debugging unless caller wants them deleted later.


def _clear_runs_and_canonical(*, output_root: Path) -> None:
    """Remove `_runs/`, `_summary/`, and canonical outcome folders.

    Keeps `_legacy/`, `_migration/`, `_locks/` for traceability.
    """

    for name in ["_runs", "_summary"]:
        path = output_root / name
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

    for child in sorted(output_root.iterdir() if output_root.exists() else []):
        if not child.is_dir():
            continue
        if child.name.startswith("_"):
            continue
        shutil.rmtree(child, ignore_errors=True)


def _clear_canonical_only(*, output_root: Path) -> None:
    """Remove `_summary/` and canonical outcome folders.

    Keeps `_runs/` intact so we can rebuild canonical leaves deterministically.
    """

    summary = output_root / "_summary"
    if summary.exists():
        shutil.rmtree(summary, ignore_errors=True)

    for child in sorted(output_root.iterdir() if output_root.exists() else []):
        if not child.is_dir():
            continue
        if child.name.startswith("_"):
            continue
        shutil.rmtree(child, ignore_errors=True)


def _copy_row_level_with_status(*, src: Path, dst: Path, calibration_status: str) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    wrote_any = False
    for chunk in pd.read_csv(src, chunksize=200_000):
        chunk["calibration_status"] = calibration_status
        chunk.to_csv(dst, index=False, mode="a", header=not wrote_any)
        wrote_any = True


def _find_unperturbed_calibration(
    *,
    unperturbed_root: Path,
    outcome: str,
    model: str,
    strategy: str,
) -> Path | None:
    candidates: list[Path] = []

    def _add(m: str) -> None:
        candidates.append(unperturbed_root / outcome / m / strategy / "calibrated" / "calibration.json")

    _add(model)

    # Common historical convention: gpt-oss-20b medium is stored as gpt-oss-20b_medium.
    if model == "gpt-oss-20b":
        _add("gpt-oss-20b_medium")

    for path in candidates:
        if path.exists():
            return path
    return None


def _rerun_calibration_from_unperturbed(
    *,
    output_root: Path,
    unperturbed_root: Path,
    anchors_root: Path,
    ref_models: list[str],
    model_plan_path: Path,
    block_topk: int,
    skip_block_alignment: bool,
    force: bool,
    use_model_ref_for_linear: bool,
    ref_model_by_outcome: dict[str, str],
) -> None:
    from beam_abm.evaluation.common.apply_calibration_to_ice import main as apply_calibration_step

    uncal_leaves = _iter_uncalibrated_leaf_status_dirs(output_root)
    print(f"Found {len(uncal_leaves)} uncalibrated _runs leaves")

    for uncal_dir in uncal_leaves:
        # _runs/<run>/<outcome>/<model>/<strategy>/<status>
        strategy_dir = uncal_dir.parent
        strategy = strategy_dir.name
        model_dir = strategy_dir.parent
        model = model_dir.name
        outcome_dir = model_dir.parent
        outcome = outcome_dir.name

        cal_file = _find_unperturbed_calibration(
            unperturbed_root=unperturbed_root,
            outcome=outcome,
            model=model,
            strategy=strategy,
        )
        if cal_file is None and strategy == "paired_profile":
            cal_file = _find_unperturbed_calibration(
                unperturbed_root=unperturbed_root,
                outcome=outcome,
                model=model,
                strategy="data_only",
            )
        if cal_file is None:
            continue

        ice_in = uncal_dir / "ice_llm.csv"
        if not ice_in.exists():
            continue

        cal_dir = strategy_dir / "calibrated"
        ice_out = cal_dir / "ice_llm.csv"
        if ice_out.exists() and not force and ice_out.stat().st_mtime >= ice_in.stat().st_mtime:
            continue

        cal_dir.mkdir(parents=True, exist_ok=True)

        # Copy-through artifacts for parity.
        for name in [
            "prompts.jsonl",
            "samples.jsonl",
            "raw_conversations.jsonl",
            "run_metadata.json",
        ]:
            src = uncal_dir / name
            if src.exists() and (force or not (cal_dir / name).exists()):
                shutil.copy2(src, cal_dir / name)

        src_conv = uncal_dir / "conversations"
        if src_conv.exists() and src_conv.is_dir() and not (cal_dir / "conversations").exists():
            shutil.copytree(src_conv, cal_dir / "conversations")

        src_row = uncal_dir / "row_level.csv"
        if src_row.exists() and (force or not (cal_dir / "row_level.csv").exists()):
            _copy_row_level_with_status(src=src_row, dst=cal_dir / "row_level.csv", calibration_status="calibrated")

        saved = list(sys.argv)
        try:
            sys.argv = [
                sys.argv[0],
                "--llm-ice",
                str(ice_in),
                "--calibration",
                str(cal_file),
                "--out",
                str(ice_out),
            ]
            apply_calibration_step()
        except SystemExit:
            # Treat CLI failures as a skip; do not stop the rebuild.
            continue
        finally:
            sys.argv = saved

        # Recompute alignment summaries for the calibrated leaf.
        _rebuild_alignment(
            status_dir=cal_dir,
            anchors_root=anchors_root,
            ref_models=ref_models,
            model_plan_path=model_plan_path,
            block_topk=block_topk,
            skip_block_alignment=skip_block_alignment,
            use_model_ref_for_linear=use_model_ref_for_linear,
            ref_model_by_outcome=ref_model_by_outcome,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="evaluation/output/choice_validation/perturbed",
        help="Choice-validation perturbed root (contains _runs and canonical).",
    )

    parser.add_argument(
        "--clear-runs-and-canonical",
        action="store_true",
        help="Delete _runs/, _summary/, and canonical outcome folders before rebuilding.",
    )

    parser.add_argument(
        "--clear-canonical",
        action="store_true",
        help="Delete _summary/ and canonical outcome folders (keeps _runs/) before rebuilding.",
    )

    parser.add_argument(
        "--ingest-only-status",
        choices=["uncalibrated", "calibrated"],
        default=None,
        help=(
            "If set, ingest only this leaf status from legacy sources. "
            "Useful to backfill calibrated without re-ingesting uncalibrated."
        ),
    )

    parser.add_argument(
        "--ingest-prompt-tournament-root",
        default=None,
        help="Optional prompt-tournament root to ingest (copy-only) into _runs.",
    )
    parser.add_argument(
        "--rename-prompt-tournament-model-dirs",
        action="store_true",
        help="If set, rename output/prompt_tournament/<model_slug> dirs to normalized names before ingestion.",
    )
    parser.add_argument(
        "--ingest-run-type",
        default="perturbed",
        help="prompt-tournament run type to ingest (default: perturbed).",
    )

    parser.add_argument(
        "--ingest-phase0-root",
        default=None,
        help="Optional legacy phase0 root (evaluation/output/prompt_tournament) to ingest.",
    )
    parser.add_argument(
        "--ingest-temp-sweep-root",
        default=None,
        help="Optional legacy temp sweep root (temp_sweep_fixed) to ingest.",
    )
    parser.add_argument(
        "--ingest-llm-responses-root",
        default=None,
        help="Optional legacy llm_responses root (evaluation/output/llm_responses) to ingest.",
    )
    parser.add_argument(
        "--migration-log",
        default=None,
        help="Optional JSONL log of ingest copy operations.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Dry-run ingestion (no writes).")

    parser.add_argument(
        "--anchors-root",
        default="empirical/output/anchors/pe/mutable_engines/reduced",
        help="Root for reference ICE tables.",
    )

    parser.add_argument(
        "--model-plan",
        default="empirical/output/modeling/model_plan.json",
        help="Path to model_plan.json (used to map ICE targets to modelling predictor blocks).",
    )
    parser.add_argument(
        "--block-topk",
        type=int,
        default=5,
        help="k for block-level top-k overlap metric (default: 5).",
    )
    parser.add_argument(
        "--skip-block-alignment",
        action="store_true",
        help="Skip computing block-level alignment metrics (BSA / top-k overlap / quartile concordance).",
    )
    parser.add_argument("--ref-models", default="linear", help="Comma-separated reference models (default: linear).")
    parser.add_argument(
        "--use-model-ref-for-linear",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When aligning against ref_model=linear, prefer per-outcome model_ref ICE "
            "(ice_ref__<model_ref>.csv) resolved from --ref-model-map or --lever-config. "
            "Default: enabled."
        ),
    )
    parser.add_argument(
        "--ref-model-map",
        default=None,
        help="Optional JSON mapping outcome -> reference model name (or objects with model_ref).",
    )
    parser.add_argument(
        "--lever-config",
        default="config/levers.json",
        help="Fallback mapping source for outcome -> model_ref when --use-model-ref-for-linear is set.",
    )

    parser.add_argument("--skip-ice", action="store_true", help="Skip rebuilding ice_llm.csv")
    parser.add_argument("--skip-alignment", action="store_true", help="Skip rebuilding summary_metrics.csv")
    parser.add_argument(
        "--skip-canonical",
        action="store_true",
        help="Skip rebuilding canonical leaves from _runs.",
    )

    parser.add_argument(
        "--rerun-calibration",
        action="store_true",
        help="Apply unperturbed calibration.json to perturbed ICE and rebuild calibrated leaves.",
    )
    parser.add_argument(
        "--unperturbed-root",
        default="evaluation/output/choice_validation/unperturbed",
        help="Root for canonical unperturbed leaves (used to source calibration.json files).",
    )
    parser.add_argument(
        "--force-calibration",
        action="store_true",
        help="Overwrite existing calibrated leaves during --rerun-calibration.",
    )

    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    anchors_root = Path(args.anchors_root)
    unperturbed_root = Path(args.unperturbed_root)

    repo_root = _find_repo_root(Path(__file__).resolve())
    model_plan_path = Path(args.model_plan)
    if not model_plan_path.is_absolute():
        model_plan_path = repo_root / model_plan_path
    ref_models = [m.strip() for m in str(args.ref_models).split(",") if m.strip()]
    if not ref_models:
        ref_models = ["linear"]
    use_model_ref_for_linear = bool(args.use_model_ref_for_linear)
    ref_model_by_outcome: dict[str, str] = {}
    if use_model_ref_for_linear:
        if args.ref_model_map:
            ref_model_by_outcome = _load_ref_model_map(Path(args.ref_model_map))
        if not ref_model_by_outcome:
            ref_model_by_outcome = _load_ref_model_map(Path(args.lever_config))

    migration_log = Path(args.migration_log) if args.migration_log else None

    if args.clear_runs_and_canonical and not args.dry_run:
        _clear_runs_and_canonical(output_root=output_root)
    elif args.clear_canonical and not args.dry_run:
        _clear_canonical_only(output_root=output_root)

    if args.ingest_prompt_tournament_root:
        ingest_root = Path(args.ingest_prompt_tournament_root)
        if args.rename_prompt_tournament_model_dirs:
            rename_prompt_tournament_model_dirs(prompt_tournament_root=ingest_root, migration_log=migration_log)
        counts = ingest_prompt_tournament_root_to_runs(
            prompt_tournament_root=ingest_root,
            output_root=output_root,
            run_type=str(args.ingest_run_type),
            dry_run=bool(args.dry_run),
            migration_log=migration_log,
        )
        print(f"Ingest: leaves_seen={counts['leaves_seen']} outcomes_written={counts['outcomes_written']}")

    # Best-effort legacy ingests. We keep these implementations local to avoid expanding the public API.
    def _infer_effort_from_run_meta(run_meta: dict) -> str | None:
        snap = run_meta.get("config_snapshot")
        if not isinstance(snap, dict):
            return None
        if isinstance(snap.get("reasoning_effort"), str):
            return str(snap.get("reasoning_effort")).strip().lower() or None
        reasoning = snap.get("reasoning")
        if isinstance(reasoning, dict) and isinstance(reasoning.get("effort"), str):
            return str(reasoning.get("effort")).strip().lower() or None
        model_options = snap.get("model_options")
        if isinstance(model_options, dict) and isinstance(model_options.get("reasoning_effort"), str):
            return str(model_options.get("reasoning_effort")).strip().lower() or None
        if isinstance(model_options, str):
            try:
                obj = json.loads(model_options)
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict):
                if isinstance(obj.get("reasoning_effort"), str):
                    return str(obj.get("reasoning_effort")).strip().lower() or None
                reasoning2 = obj.get("reasoning")
                if isinstance(reasoning2, dict) and isinstance(reasoning2.get("effort"), str):
                    return str(reasoning2.get("effort")).strip().lower() or None
        return None

    def _model_slug_from_run_meta(run_meta: dict, *, fallback: str) -> str:
        snap = run_meta.get("config_snapshot")
        if isinstance(snap, dict) and isinstance(snap.get("model_slug"), str) and str(snap.get("model_slug")).strip():
            return str(snap.get("model_slug")).strip()
        model = None
        if isinstance(snap, dict) and isinstance(snap.get("model"), str):
            model = str(snap.get("model")).strip()
        effort = _infer_effort_from_run_meta(run_meta)
        return normalize_model_slug(model or fallback, reasoning_effort=effort)

    def _copytree_if_missing(src: Path, dst: Path) -> None:
        if dst.exists():
            return
        shutil.copytree(src, dst)

    def _copy_leaf_dir(src_status_dir: Path, dst_status_dir: Path) -> None:
        dst_status_dir.mkdir(parents=True, exist_ok=True)
        for name in [
            "prompts.jsonl",
            "samples.jsonl",
            "raw_conversations.jsonl",
            "row_level.csv",
            "ice_llm.csv",
            "summary_metrics.csv",
            "run_metadata.json",
        ]:
            src = src_status_dir / name
            if src.exists():
                shutil.copy2(src, dst_status_dir / name)
        src_conv = src_status_dir / "conversations"
        if src_conv.exists() and src_conv.is_dir():
            _copytree_if_missing(src_conv, dst_status_dir / "conversations")

    def _infer_model_slug_from_samples(samples_path: Path, *, fallback: str) -> str:
        for r in _iter_jsonl_dicts(samples_path):
            model = r.get("model")
            if isinstance(model, str) and model.strip():
                return normalize_model_dirname(normalize_model_slug(model.strip(), reasoning_effort=None))
            break
        return normalize_model_dirname(normalize_model_slug(fallback, reasoning_effort=None))

    def _ingest_leaf_split_by_outcome(
        *,
        src_status_dir: Path,
        run_id: str,
        model_slug: str,
        strategy: str,
        status: str,
        output_root: Path,
        migration_log: Path | None,
        action: str,
    ) -> None:
        samples_path = src_status_dir / "samples.jsonl"
        if not samples_path.exists():
            return

        outcomes_seen: set[str] = set()
        for r in _iter_jsonl_dicts(samples_path):
            o = _row_outcome(r)
            if o:
                outcomes_seen.add(o)
        if not outcomes_seen:
            return

        dest_by_outcome: dict[str, Path] = {}
        for outcome in sorted(outcomes_seen):
            dest_by_outcome[outcome] = (
                output_root / "_runs" / run_id / _safe_token(outcome) / model_slug / strategy / status
            )

        # Skip already-ingested destinations to avoid duplicating JSONL rows.
        dest_by_outcome = {
            outcome: dest for outcome, dest in dest_by_outcome.items() if not (dest / "samples.jsonl").exists()
        }
        if not dest_by_outcome:
            return

        if migration_log is not None:
            for outcome, dest_leaf in dest_by_outcome.items():
                _write_migration_log_line(
                    migration_log,
                    {
                        "ts_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "action": action,
                        "source": str(src_status_dir),
                        "dest": str(dest_leaf),
                        "run_id": run_id,
                        "outcome": outcome,
                        "model_dest": model_slug,
                        "strategy": strategy,
                        "status": status,
                    },
                )

        for dest in dest_by_outcome.values():
            dest.mkdir(parents=True, exist_ok=True)

        _stream_split_jsonl_by_outcome(
            src_path=samples_path, dst_root_by_outcome=dest_by_outcome, filename="samples.jsonl"
        )

        for jsonl_name in ["prompts.jsonl", "raw_conversations.jsonl"]:
            src_jsonl = src_status_dir / jsonl_name
            if src_jsonl.exists():
                _stream_split_jsonl_by_outcome(
                    src_path=src_jsonl,
                    dst_root_by_outcome=dest_by_outcome,
                    filename=jsonl_name,
                )

        row_level_path = src_status_dir / "row_level.csv"
        if row_level_path.exists():
            for chunk in pd.read_csv(row_level_path, chunksize=200_000):
                if "outcome" not in chunk.columns:
                    break
                for outcome_val, g in chunk.groupby(chunk["outcome"].astype(str), dropna=False):
                    key = str(outcome_val).strip()
                    if key not in dest_by_outcome:
                        continue
                    out_path = dest_by_outcome[key] / "row_level.csv"
                    g.to_csv(out_path, mode="a", header=not out_path.exists(), index=False)

        ice_path = src_status_dir / "ice_llm.csv"
        if ice_path.exists():
            df = pd.read_csv(ice_path)
            if "outcome" in df.columns:
                for outcome_val, g in df.groupby(df["outcome"].astype(str), dropna=False):
                    key = str(outcome_val).strip()
                    if key in dest_by_outcome:
                        g.to_csv(dest_by_outcome[key] / "ice_llm.csv", index=False)
            else:
                # No outcome column: copy as-is to all outcomes.
                for dest in dest_by_outcome.values():
                    shutil.copy2(ice_path, dest / "ice_llm.csv")

        for name in ["run_metadata.json", "summary_metrics.csv"]:
            src = src_status_dir / name
            if not src.exists():
                continue
            for dest in dest_by_outcome.values():
                shutil.copy2(src, dest / name)

        src_conv = src_status_dir / "conversations"
        if src_conv.exists() and src_conv.is_dir():
            for dest in dest_by_outcome.values():
                _copytree_if_missing(src_conv, dest / "conversations")

    if args.ingest_prompt_tournament_root and not args.dry_run:
        prompt_tournament_root = Path(args.ingest_prompt_tournament_root)
        perturbed_root = prompt_tournament_root / "perturbed"
        if perturbed_root.exists():
            for strategy_dir in sorted(
                p for p in perturbed_root.iterdir() if p.is_dir() and not p.name.startswith("_")
            ):
                strategy = strategy_dir.name
                for run_dir in sorted(p for p in strategy_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
                    run_id = run_dir.name
                    # Inside run_dir we may have either outcome dirs or batch dirs.
                    for maybe_outcome_dir in sorted(
                        p for p in run_dir.iterdir() if p.is_dir() and not p.name.startswith("_")
                    ):
                        for status_dir in sorted(
                            p
                            for p in maybe_outcome_dir.iterdir()
                            if p.is_dir() and p.name in {"uncalibrated", "calibrated"}
                        ):
                            if args.ingest_only_status and status_dir.name != args.ingest_only_status:
                                continue
                            run_meta_path = status_dir / "run_metadata.json"
                            run_meta = {}
                            if run_meta_path.exists():
                                try:
                                    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
                                except json.JSONDecodeError:
                                    run_meta = {}

                            model_slug = normalize_model_dirname(
                                _model_slug_from_run_meta(run_meta, fallback="gpt-oss-20b")
                            )

                            if status_dir.name == "calibrated" and not (status_dir / "samples.jsonl").exists():
                                outcome = maybe_outcome_dir.name
                                dest_status_dir = (
                                    output_root / "_runs" / run_id / outcome / model_slug / strategy / "calibrated"
                                )
                                _copy_prompt_tournament_calibrated_leaf(
                                    src_status_dir=status_dir, dest_status_dir=dest_status_dir
                                )
                                if migration_log is not None:
                                    _write_migration_log_line(
                                        migration_log,
                                        {
                                            "ts_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                            "action": "ingest_prompt_tournament_calibrated_leaf",
                                            "source": str(status_dir),
                                            "dest": str(dest_status_dir),
                                            "run_id": run_id,
                                            "outcome": outcome,
                                            "model_dest": model_slug,
                                            "strategy": strategy,
                                            "status": "calibrated",
                                        },
                                    )
                            else:
                                _ingest_leaf_split_by_outcome(
                                    src_status_dir=status_dir,
                                    run_id=run_id,
                                    model_slug=model_slug,
                                    strategy=strategy,
                                    status=status_dir.name,
                                    output_root=output_root,
                                    migration_log=migration_log,
                                    action="ingest_prompt_tournament_leaf_split",
                                )

    if args.ingest_temp_sweep_root and not args.dry_run:
        sweep_root = Path(args.ingest_temp_sweep_root)
        for run_dir in sorted(p for p in sweep_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
            run_id = run_dir.name
            for batch_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
                for status_dir in sorted(
                    p for p in batch_dir.iterdir() if p.is_dir() and p.name in {"uncalibrated", "calibrated"}
                ):
                    if args.ingest_only_status and status_dir.name != args.ingest_only_status:
                        continue
                    run_meta_path = status_dir / "run_metadata.json"
                    run_meta = {}
                    if run_meta_path.exists():
                        try:
                            run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
                        except json.JSONDecodeError:
                            run_meta = {}
                    model_slug = normalize_model_dirname(
                        _model_slug_from_run_meta(run_meta, fallback="openai/gpt-oss-20b")
                    )
                    strategy = "data_only"

                    _ingest_leaf_split_by_outcome(
                        src_status_dir=status_dir,
                        run_id=run_id,
                        model_slug=model_slug,
                        strategy=strategy,
                        status=status_dir.name,
                        output_root=output_root,
                        migration_log=migration_log,
                        action="ingest_temp_sweep_leaf_split",
                    )

    if args.ingest_llm_responses_root and not args.dry_run:
        llm_root = Path(args.ingest_llm_responses_root)
        # Ingest only known strategy dirs.
        for strategy in ["data_only_OK", "paired_profile_OK"]:
            src = llm_root / strategy
            if not src.exists():
                continue
            run_id = f"legacy-llm_responses-{strategy}"
            # Infer samples/prompts filenames.
            samples = None
            prompts = None
            for cand in ["samples.jsonl", f"samples__direct_repaired_v3__{strategy.replace('_OK', '')}.jsonl"]:
                if (src / cand).exists():
                    samples = src / cand
                    break
            for cand in [f"prompts__{strategy.replace('_OK', '')}.jsonl", "prompts.jsonl"]:
                if (src / cand).exists():
                    prompts = src / cand
                    break
            if samples is None:
                continue
            # Model is not encoded; assume gpt-oss-20b medium (normalized -> gpt-oss-20b).
            model_slug = normalize_model_dirname(normalize_model_slug("gpt-oss-20b", reasoning_effort="medium"))
            strat = strategy.replace("_OK", "")
            status = "uncalibrated"

            # Determine outcomes.
            outcomes_seen: set[str] = set()
            for r in _iter_jsonl_dicts(samples):
                o = r.get("outcome") or (r.get("metadata") or {}).get("outcome")
                if isinstance(o, str) and o.strip():
                    outcomes_seen.add(o.strip())
            if not outcomes_seen:
                continue

            dest_by_outcome: dict[str, Path] = {
                o: output_root / "_runs" / run_id / o / model_slug / strat / status for o in outcomes_seen
            }
            for dest in dest_by_outcome.values():
                dest.mkdir(parents=True, exist_ok=True)
            _stream_split_jsonl_by_outcome(
                src_path=samples, dst_root_by_outcome=dest_by_outcome, filename="samples.jsonl"
            )
            if prompts is not None:
                _stream_split_jsonl_by_outcome(
                    src_path=prompts, dst_root_by_outcome=dest_by_outcome, filename="prompts.jsonl"
                )

            # Archive the original legacy directory for traceability.
            legacy_archive = output_root / "_legacy" / "llm_responses" / strategy
            if not legacy_archive.exists():
                shutil.copytree(src, legacy_archive)

        # Also ingest legacy llm_responses/prompt_tournament persona dirs (first_person/third_person/multi_expert/...)
        legacy_pt = llm_root / "prompt_tournament"
        if legacy_pt.exists():
            run_id = "legacy-llm_responses-prompt_tournament"
            for persona_dir in sorted(p for p in legacy_pt.iterdir() if p.is_dir() and not p.name.startswith("_")):
                strategy = persona_dir.name
                samples_path = persona_dir / "samples.jsonl"
                if not samples_path.exists():
                    continue
                model_slug = _infer_model_slug_from_samples(samples_path, fallback="gpt-oss-20b")
                status = "uncalibrated"

                # If legacy provides row_level/ice, copy through; otherwise rebuild later.
                _ingest_leaf_split_by_outcome(
                    src_status_dir=persona_dir,
                    run_id=run_id,
                    model_slug=model_slug,
                    strategy=strategy,
                    status=status,
                    output_root=output_root,
                    migration_log=migration_log,
                    action="ingest_llm_responses_prompt_tournament_split",
                )

            legacy_archive = output_root / "_legacy" / "llm_responses" / "prompt_tournament"
            if not legacy_archive.exists():
                shutil.copytree(legacy_pt, legacy_archive)

    if args.dry_run:
        return

    leaf_status_dirs = _iter_leaf_status_dirs(output_root)
    print(f"Found {len(leaf_status_dirs)} _runs leaf status dirs")

    for status_dir in leaf_status_dirs:
        if not args.skip_ice:
            _rebuild_ice(status_dir=status_dir)
        if not args.skip_alignment:
            _rebuild_alignment(
                status_dir=status_dir,
                anchors_root=anchors_root,
                ref_models=ref_models,
                model_plan_path=model_plan_path,
                block_topk=int(args.block_topk),
                skip_block_alignment=bool(args.skip_block_alignment),
                use_model_ref_for_linear=use_model_ref_for_linear,
                ref_model_by_outcome=ref_model_by_outcome,
            )

    if args.rerun_calibration and not args.skip_alignment:
        _rerun_calibration_from_unperturbed(
            output_root=output_root,
            unperturbed_root=unperturbed_root,
            anchors_root=anchors_root,
            ref_models=ref_models,
            model_plan_path=model_plan_path,
            block_topk=int(args.block_topk),
            skip_block_alignment=bool(args.skip_block_alignment),
            force=bool(args.force_calibration),
            use_model_ref_for_linear=use_model_ref_for_linear,
            ref_model_by_outcome=ref_model_by_outcome,
        )

    if not args.skip_canonical:
        rebuild_canonical_from_runs(
            output_root=output_root,
            merge_runs=True,
            anchors_root=anchors_root,
            ref_models=ref_models,
            model_plan_path=model_plan_path,
            block_topk=int(args.block_topk),
            skip_block_alignment=bool(args.skip_block_alignment),
            use_model_ref_for_linear=use_model_ref_for_linear,
            ref_model_by_outcome=ref_model_by_outcome,
        )
        write_global_summary(output_root=output_root)
        if not args.skip_alignment:
            _write_oracle_uplift_summary(
                output_root=output_root,
                anchors_root=anchors_root,
                model_plan_path=model_plan_path,
                ref_model_by_outcome=ref_model_by_outcome,
            )


if __name__ == "__main__":
    # Respect the project root when invoked from elsewhere.
    os.environ.setdefault("PROJECT_ROOT", str(Path(__file__).resolve().parents[4]))
    main()
