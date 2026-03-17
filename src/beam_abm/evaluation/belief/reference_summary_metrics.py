"""Choice-like summary metrics for belief perturbation evaluation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from beam_abm.evaluation.common.metrics import gini_alignment, spearman_alignment


def _kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(x.size)
    if n < 2:
        return float("nan")

    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = float(x[i] - x[j])
            dy = float(y[i] - y[j])
            if dx == 0.0 and dy == 0.0:
                continue
            if dx == 0.0:
                ties_x += 1
                continue
            if dy == 0.0:
                ties_y += 1
                continue
            if dx * dy > 0.0:
                concordant += 1
            elif dx * dy < 0.0:
                discordant += 1

    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom <= 0.0:
        return float("nan")
    return float((concordant - discordant) / denom)


def _cosine_alignment(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return float("nan")
    x_use = x[mask]
    y_use = y[mask]
    nx = float(np.linalg.norm(x_use))
    ny = float(np.linalg.norm(y_use))
    if nx <= 0.0 or ny <= 0.0:
        return float("nan")
    return float(np.dot(x_use, y_use) / (nx * ny))


def _mae_skill_zero_baseline(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    yt = y_true[mask]
    yp = y_pred[mask]
    mae = float(np.mean(np.abs(yp - yt)))
    baseline_mae = float(np.mean(np.abs(yt)))
    if baseline_mae <= 0.0:
        return float("nan")
    return float(1.0 - mae / baseline_mae)


def compute_choice_like_summary(
    *,
    long_df: pd.DataFrame,
    epsilon_ref: float,
    epsilon_move_mult: float,
    denom: float,
) -> dict[str, float]:
    """Compute choice-style perturbed metrics on PE rows."""

    out: dict[str, float] = {}
    if long_df.empty:
        return out

    row_ids = long_df.get("id")
    if isinstance(row_ids, pd.Series):
        row_ids_np = row_ids.astype(str).to_numpy(dtype=object)
    else:
        row_ids_np = np.asarray(["__all__"] * len(long_df), dtype=object)

    ref_std = pd.to_numeric(long_df.get("pe_ref_p_std"), errors="coerce").to_numpy(dtype=float)
    llm_std = pd.to_numeric(long_df.get("pe_llm_p_std"), errors="coerce").to_numpy(dtype=float)
    ref = pd.to_numeric(long_df.get("pe_ref_p"), errors="coerce").to_numpy(dtype=float)
    llm = pd.to_numeric(long_df.get("pe_llm_p"), errors="coerce").to_numpy(dtype=float)

    abs_ref = np.abs(ref_std)
    informative = np.isfinite(abs_ref) & np.isfinite(llm_std) & (abs_ref > float(epsilon_ref))
    low_signal = np.isfinite(abs_ref) & np.isfinite(llm_std) & (abs_ref <= float(epsilon_ref))

    sign_ref = np.sign(ref_std)
    sign_llm = np.sign(llm_std)
    sign_agree = np.where(informative, (sign_ref == sign_llm).astype(float), np.nan)

    eps_move = float(epsilon_move_mult) * float(epsilon_ref)
    false_pos = np.where(low_signal, (np.abs(llm_std) > eps_move).astype(float), np.nan)

    out["n_pe_rows"] = float(len(long_df))
    out["n_sign_informative"] = float(np.sum(np.isfinite(sign_agree)))
    out["sign_agreement_rate"] = float(np.nanmean(sign_agree)) if np.isfinite(sign_agree).any() else float("nan")
    out["n_low_signal"] = float(np.sum(low_signal))
    out["false_positive_movement_rate"] = float(np.nanmean(false_pos)) if np.isfinite(false_pos).any() else float("nan")
    out["low_signal_abs_pe_llm_std_median"] = (
        float(np.nanmedian(np.where(low_signal, np.abs(llm_std), np.nan))) if np.any(low_signal) else float("nan")
    )
    out["epsilon_ref"] = float(epsilon_ref)
    out["epsilon_move"] = float(eps_move)
    out["tie_rate"] = float(np.mean(~informative)) if informative.size else float("nan")
    out["dar"] = float(np.nanmean(sign_agree)) if np.isfinite(sign_agree).any() else float("nan")
    out["da_no_signal_rate"] = out["tie_rate"]
    out["da_scored_rate"] = float(1.0 - out["tie_rate"]) if np.isfinite(out["tie_rate"]) else float("nan")
    out["n_scored"] = float(np.sum(informative))
    out["n_no_signal"] = float(len(long_df) - int(np.sum(informative)))
    out["typical_drift_median"] = out["low_signal_abs_pe_llm_std_median"]

    mask_pair_std = np.isfinite(ref_std) & np.isfinite(llm_std)
    if np.any(mask_pair_std):
        abs_err_std = np.abs(llm_std[mask_pair_std] - ref_std[mask_pair_std])
        out["mae_pe_std"] = float(np.mean(abs_err_std))
        out["smae"] = float(out["mae_pe_std"] / float(max(denom, 1e-12)))
        out["spearman_pe_std"] = float(spearman_alignment(ref_std[mask_pair_std], llm_std[mask_pair_std]))
        out["gini_pe_std"] = float(gini_alignment(ref_std[mask_pair_std], llm_std[mask_pair_std]))
        out["gini_pe_std_mean"] = out["gini_pe_std"]
        out["skill_pe_std"] = float(_mae_skill_zero_baseline(ref_std[informative], llm_std[informative]))
        out["pe_std_var_ratio"] = (
            float(np.nanvar(llm_std[mask_pair_std]) / np.nanvar(ref_std[mask_pair_std]))
            if np.nanvar(ref_std[mask_pair_std]) > 0
            else float("nan")
        )
        out["cosine_pe_std"] = _cosine_alignment(ref_std[mask_pair_std], llm_std[mask_pair_std])
    else:
        out["mae_pe_std"] = float("nan")
        out["smae"] = float("nan")
        out["spearman_pe_std"] = float("nan")
        out["gini_pe_std"] = float("nan")
        out["gini_pe_std_mean"] = float("nan")
        out["skill_pe_std"] = float("nan")
        out["pe_std_var_ratio"] = float("nan")
        out["cosine_pe_std"] = float("nan")

    mask_pair = np.isfinite(ref) & np.isfinite(llm)
    if np.any(mask_pair):
        abs_err = np.abs(llm[mask_pair] - ref[mask_pair])
        out["mae_pe"] = float(np.mean(abs_err))
        out["spearman_pe"] = float(spearman_alignment(ref[mask_pair], llm[mask_pair]))
        out["gini_pe"] = float(gini_alignment(ref[mask_pair], llm[mask_pair]))
        out["gini_pe_mean"] = out["gini_pe"]
        out["skill_pe"] = float(_mae_skill_zero_baseline(ref[informative], llm[informative]))
        out["cosine_pe"] = _cosine_alignment(ref[mask_pair], llm[mask_pair])
    else:
        out["mae_pe"] = float("nan")
        out["spearman_pe"] = float("nan")
        out["gini_pe"] = float("nan")
        out["gini_pe_mean"] = float("nan")
        out["skill_pe"] = float("nan")
        out["cosine_pe"] = float("nan")

    pos_mask = informative & (ref_std > 0)
    neg_mask = informative & (ref_std < 0)
    out["spearman_pe_std_abs"] = (
        float(abs(out["spearman_pe_std"])) if np.isfinite(out["spearman_pe_std"]) else float("nan")
    )
    out["gini_pe_std_abs"] = float(abs(out["gini_pe_std"])) if np.isfinite(out["gini_pe_std"]) else float("nan")
    out["spearman_pe_std_pos"] = (
        float(spearman_alignment(ref_std[pos_mask], llm_std[pos_mask])) if np.any(pos_mask) else float("nan")
    )
    out["gini_pe_std_pos"] = (
        float(gini_alignment(ref_std[pos_mask], llm_std[pos_mask])) if np.any(pos_mask) else float("nan")
    )
    out["spearman_pe_std_neg"] = (
        float(spearman_alignment(ref_std[neg_mask], llm_std[neg_mask])) if np.any(neg_mask) else float("nan")
    )
    out["gini_pe_std_neg"] = (
        float(gini_alignment(ref_std[neg_mask], llm_std[neg_mask])) if np.any(neg_mask) else float("nan")
    )

    # Per-profile ranking metrics, analogous to choice perturbed ranking checks.
    tmp_df = pd.DataFrame(
        {
            "id": row_ids_np,
            "ref_std": ref_std,
            "llm_std": llm_std,
        }
    )
    kendall_rows: list[float] = []
    top1_rows: list[float] = []
    for _, sub in tmp_df.groupby("id", dropna=False):
        ref_abs = np.abs(pd.to_numeric(sub["ref_std"], errors="coerce").to_numpy(dtype=float))
        llm_abs = np.abs(pd.to_numeric(sub["llm_std"], errors="coerce").to_numpy(dtype=float))
        mask = np.isfinite(ref_abs) & np.isfinite(llm_abs)
        if np.sum(mask) < 2:
            continue
        ref_use = ref_abs[mask]
        llm_use = llm_abs[mask]
        kendall_rows.append(_kendall_tau_b(ref_use, llm_use))
        top1_rows.append(float(int(np.argmax(ref_use) == np.argmax(llm_use))))
    out["rank_kendall_abs_mean"] = (
        float(np.nanmean(np.asarray(kendall_rows, dtype=float))) if kendall_rows else float("nan")
    )
    out["top1_match_rate"] = float(np.mean(top1_rows)) if top1_rows else float("nan")

    # Placeholders for compatibility with choice summary schema.
    out.setdefault("bsa", float("nan"))
    out.setdefault("bsa_topk_overlap", float("nan"))
    out.setdefault("bsa_highest_quartile_concordance", float("nan"))
    out.setdefault("bsa_n_blocks", float("nan"))
    out.setdefault("bsa_target_map_n_targets", float("nan"))
    out.setdefault("bsa_target_map_n_targets_mapped", float("nan"))
    out.setdefault("bsa_target_map_coverage", float("nan"))
    out.setdefault("ice_n_total", float("nan"))
    out.setdefault("ice_n_ok", float("nan"))
    out.setdefault("ice_n_error", float("nan"))
    out.setdefault("ice_n_off_manifold", float("nan"))
    out.setdefault("ice_noncompliance_rate", float("nan"))
    out.setdefault("ice_off_manifold_rate", float("nan"))

    return out
