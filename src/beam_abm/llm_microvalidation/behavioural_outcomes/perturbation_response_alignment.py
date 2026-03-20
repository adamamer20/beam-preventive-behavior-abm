"""Alignment orchestration helpers for perturbed choice evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from beam_abm.llm_microvalidation.shared.metrics import gini_alignment, spearman_alignment
from beam_abm.llm_microvalidation.utils.alignment_utils import summarize_alignment_rows


def _ref_ice_path(*, anchors_root: Path, outcome: str, ref_model: str) -> Path:
    if ref_model == "linear":
        return anchors_root / outcome / "ice_ref__linear.csv"
    return anchors_root / outcome / f"ice_ref__{ref_model}.csv"


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

    # Active/containment split: when epsilon_ref is available (written by
    # compute_alignment), compute magnitude/ordering diagnostics on the active
    # subset |PE_ref_std| > epsilon_ref.
    eps_ref = float("nan")
    if "epsilon_ref" in metrics_df.columns:
        eps_series = pd.to_numeric(metrics_df["epsilon_ref"], errors="coerce")
        if eps_series.notna().any():
            eps_ref = float(eps_series.dropna().iloc[0])

    active_mask = np.ones(ref_use.shape[0], dtype=bool)
    if np.isfinite(eps_ref):
        active_mask = np.abs(ref_use) > eps_ref

    ref_active = ref_use[active_mask]
    llm_active = llm_use[active_mask]

    summary.loc[0, "gini_pe_std_mean"] = (
        gini_alignment(ref_active, llm_active) if ref_active.size >= 2 else float("nan")
    )
    var_ref = float(np.var(ref_active, ddof=1)) if ref_active.size > 1 else float("nan")
    var_llm = float(np.var(llm_active, ddof=1)) if llm_active.size > 1 else float("nan")
    summary.loc[0, "pe_std_var_ratio"] = float(var_llm / var_ref) if var_ref > 0 else float("nan")
    summary.loc[0, "spearman_pe_std_abs"] = (
        spearman_alignment(np.abs(ref_active), np.abs(llm_active)) if ref_active.size >= 2 else float("nan")
    )
    summary.loc[0, "gini_pe_std_abs"] = (
        gini_alignment(np.abs(ref_active), np.abs(llm_active)) if ref_active.size >= 2 else float("nan")
    )

    pos_active = ref_active > 0
    if np.sum(pos_active) >= 2:
        summary.loc[0, "spearman_pe_std_pos"] = spearman_alignment(ref_active[pos_active], llm_active[pos_active])
        summary.loc[0, "gini_pe_std_pos"] = gini_alignment(ref_active[pos_active], llm_active[pos_active])
    else:
        summary.loc[0, "spearman_pe_std_pos"] = float("nan")
        summary.loc[0, "gini_pe_std_pos"] = float("nan")

    neg_active = ref_active < 0
    if np.sum(neg_active) >= 2:
        summary.loc[0, "spearman_pe_std_neg"] = spearman_alignment(ref_active[neg_active], llm_active[neg_active])
        summary.loc[0, "gini_pe_std_neg"] = gini_alignment(ref_active[neg_active], llm_active[neg_active])
    else:
        summary.loc[0, "spearman_pe_std_neg"] = float("nan")
        summary.loc[0, "gini_pe_std_neg"] = float("nan")

    if {"pe_ref", "pe_llm"}.issubset(metrics_df.columns):
        ref_raw = pd.to_numeric(metrics_df["pe_ref"], errors="coerce").to_numpy(dtype=float)
        llm_raw = pd.to_numeric(metrics_df["pe_llm"], errors="coerce").to_numpy(dtype=float)
        raw_mask = np.isfinite(ref_raw) & np.isfinite(llm_raw)
        summary.loc[0, "gini_pe_mean"] = (
            gini_alignment(ref_raw[raw_mask], llm_raw[raw_mask]) if raw_mask.sum() >= 2 else float("nan")
        )


def _load_model_plan(model_plan_path: Path) -> list[dict]:
    with model_plan_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"model_plan.json must be a list, got {type(obj).__name__}")
    return [x for x in obj if isinstance(x, dict)]


def _select_model_plan_entry(
    plan: list[dict],
    *,
    outcome: str,
    model_name: str | None = None,
) -> dict | None:
    candidates: list[dict] = []
    for entry in plan:
        if entry.get("outcome") != outcome:
            continue
        if not isinstance(entry.get("predictors"), dict):
            continue
        candidates.append(entry)

    if not candidates:
        return None

    model_name_norm = str(model_name or "").strip()
    if model_name_norm:
        for entry in candidates:
            if str(entry.get("model") or "").strip() == model_name_norm:
                return entry

    def _score(entry: dict) -> tuple[int, str]:
        model = str(entry.get("model") or "")
        desc = str(entry.get("description") or "").lower()
        if model.endswith("_ENGINES_REF"):
            return (0, model)
        if "ENGINES_REF" in model:
            return (1, model)
        if "ref engines" in desc:
            return (2, model)
        return (10, model)

    return sorted(candidates, key=_score)[0]


def _target_to_block_map(
    model_plan_path: Path,
    outcome: str,
    *,
    model_name: str | None = None,
) -> dict[str, str]:
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
    topk: int,
    ref_model_by_outcome: dict[str, str] | None = None,
) -> dict[str, float]:
    from beam_abm.llm_microvalidation.shared.block_alignment import compute_block_alignment

    try:
        llm_df = pd.read_csv(llm_ice, usecols=["target"])  # type: ignore[arg-type]
    except (OSError, ValueError, pd.errors.ParserError):
        return {}
    if llm_df.empty or "target" not in llm_df.columns:
        return {}

    targets = sorted({t for t in llm_df["target"].astype(str).tolist() if t and t != "nan"})
    if not targets:
        return {}

    model_ref = str((ref_model_by_outcome or {}).get(outcome, "")).strip()
    target_to_block = _target_to_block_map(
        model_plan_path,
        outcome,
        model_name=model_ref if model_ref else None,
    )
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

    try:
        compute_block_alignment(
            ref_ice=ref_ice,
            llm_ice=llm_ice,
            block_map=block_map_path,
            topk=int(topk),
            out=block_out,
        )
    except ValueError:
        return {}

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


def rebuild_alignment_summary(
    *,
    status_dir: Path,
    anchors_root: Path,
    ref_models: list[str],
    model_plan_path: Path | None,
    block_topk: int,
    skip_block_alignment: bool,
    use_model_ref_for_linear: bool = False,
    ref_model_by_outcome: dict[str, str] | None = None,
) -> None:
    from beam_abm.llm_microvalidation.shared.alignment import compute_alignment_metrics

    ice_path = status_dir / "ice_llm.csv"
    if not ice_path.exists():
        return

    outcome = status_dir.parent.parent.parent.name

    ref_model_by_outcome = ref_model_by_outcome or {}
    for ref in ref_models:
        ref_ice = _resolve_ref_ice_path(
            anchors_root=anchors_root,
            outcome=outcome,
            ref_model=ref,
            use_model_ref_for_linear=use_model_ref_for_linear,
            ref_model_by_outcome=ref_model_by_outcome,
        )

        metrics_out = status_dir / (
            "_alignment_metrics.csv" if ref == "linear" else f"_alignment_metrics__ref_{ref}.csv"
        )
        order_out = status_dir / ("_order_effects.csv" if ref == "linear" else f"_order_effects__ref_{ref}.csv")

        if not ref_ice.exists():
            if metrics_out.exists():
                try:
                    metrics_df = pd.read_csv(metrics_out) if metrics_out.exists() else pd.DataFrame([])
                except (OSError, ValueError, pd.errors.ParserError):
                    metrics_df = pd.DataFrame([])

                summary = summarize_alignment_rows(metrics_df)
                _attach_alignment_dispersion_metrics(summary, metrics_df)

                summary_out = status_dir / (
                    "summary_metrics.csv" if ref == "linear" else f"summary_metrics__ref_{ref}.csv"
                )
                summary.to_csv(summary_out, index=False)
            continue

        compute_alignment_metrics(
            ref=ref_ice,
            llm=ice_path,
            out=metrics_out,
            order_out=order_out,
        )

        try:
            metrics_df = pd.read_csv(metrics_out) if metrics_out.exists() else pd.DataFrame([])
        except (OSError, ValueError, pd.errors.ParserError):
            metrics_df = pd.DataFrame([])

        summary = summarize_alignment_rows(metrics_df)
        _attach_alignment_dispersion_metrics(summary, metrics_df)

        if model_plan_path is not None and model_plan_path.exists() and not skip_block_alignment:
            block_metrics = _compute_block_alignment(
                status_dir=status_dir,
                outcome=outcome,
                ref=ref,
                ref_ice=ref_ice,
                llm_ice=ice_path,
                model_plan_path=model_plan_path,
                topk=block_topk,
                ref_model_by_outcome=ref_model_by_outcome,
            )
            for k, v in block_metrics.items():
                summary[k] = v

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
