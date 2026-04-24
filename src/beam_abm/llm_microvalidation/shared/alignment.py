"""Compute alignment metrics from ICE tables."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from beam_abm.llm_microvalidation.shared.metrics import (
    bootstrap_ci,
    curvature,
    dar,
    directional_agreement,
    gini_alignment,
    perturbation_effect,
    segment_contrasts,
    smae,
    spearman_alignment,
)


def compute_alignment_metrics(
    *,
    ref: str | Path,
    llm: str | Path,
    out: str | Path,
    order_out: str | Path,
    truth: str | Path | None = None,
    truth_y_col: str = "y_true",
    denom: float = 1.0,
    boot: int = 1000,
    epsilon_ref: float = 0.03,
    epsilon_move_mult: float = 3.0,
) -> None:
    args = SimpleNamespace(
        ref=str(ref),
        llm=str(llm),
        truth=str(truth) if truth is not None else None,
        truth_y_col=str(truth_y_col),
        out=str(out),
        order_out=str(order_out),
        denom=float(denom),
        boot=int(boot),
        epsilon_ref=float(epsilon_ref),
        epsilon_move_mult=float(epsilon_move_mult),
    )

    ref = pd.read_csv(args.ref)
    llm = pd.read_csv(args.llm)

    # Prefer key-based alignment over positional alignment.
    join_keys: list[str] = []
    for key in ["id", "target", "strategy", "outcome"]:
        if key in ref.columns and key in llm.columns:
            join_keys.append(key)

    if join_keys:
        # Guard against duplicate keys producing a many-to-many merge.
        if ref.duplicated(subset=join_keys).any():
            raise ValueError(f"Reference ICE has duplicate rows for merge keys: {join_keys}")
        if llm.duplicated(subset=join_keys).any():
            raise ValueError(f"LLM ICE has duplicate rows for merge keys: {join_keys}")

        merged = ref.merge(llm, on=join_keys, how="inner", suffixes=("_ref", "_llm"))
        if merged.empty:
            raise ValueError(f"No overlapping rows after merge on keys: {join_keys}")

        # Carry-through columns that only exist on the LLM side (i.e., not in ref) but
        # are needed for downstream summaries.
        carry_llm_cols: list[str] = []
        for c in ["paired_order", "strategy"]:
            if c in merged.columns and c not in join_keys:
                carry_llm_cols.append(c)

        ref = merged[[c for c in merged.columns if c.endswith("_ref") or c in join_keys]].copy()
        llm = merged[[c for c in merged.columns if c.endswith("_llm") or c in join_keys or c in carry_llm_cols]].copy()
        ref.columns = [c[:-4] if c.endswith("_ref") else c for c in ref.columns]
        llm.columns = [c[:-4] if c.endswith("_llm") else c for c in llm.columns]
    else:
        if len(ref) != len(llm):
            raise ValueError(
                "ref and llm must have same number of rows (or include id/target[/strategy] columns for merging)"
            )

    has_mid = "ice_mid" in ref.columns and "ice_mid" in llm.columns

    truth: pd.DataFrame | None = None
    if args.truth:
        truth = pd.read_csv(args.truth)
        if args.truth_y_col not in truth.columns:
            raise ValueError(f"Truth CSV is missing required column: {args.truth_y_col}")

        truth_keys: list[str] = []
        for key in ["id", "target", "strategy", "outcome"]:
            if key in truth.columns and key in llm.columns:
                truth_keys.append(key)

        if not truth_keys:
            raise ValueError(
                "Truth merge keys could not be inferred. Provide at least one common column between truth and llm "
                "(e.g., id, outcome, target, strategy)."
            )
        if truth.duplicated(subset=truth_keys).any():
            raise ValueError(f"Truth CSV has duplicate rows for merge keys: {truth_keys}")

        # Attach y_true onto both ref and llm so we can compute both ref-vs-true and llm-vs-true.
        truth_small = truth[truth_keys + [args.truth_y_col]].copy()
        truth_small = truth_small.rename(columns={args.truth_y_col: "y_true"})
        ref = ref.merge(truth_small, on=truth_keys, how="left")
        llm = llm.merge(truth_small, on=truth_keys, how="left")

    def _maybe_parse_list(v: object) -> list[float] | None:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        if isinstance(v, list):
            try:
                return [float(x) for x in v]
            except (TypeError, ValueError):
                return None
        if isinstance(v, str):
            try:
                arr = json.loads(v)
                if isinstance(arr, list):
                    return [float(x) for x in arr]
            except (json.JSONDecodeError, TypeError, ValueError):
                return None
        return None

    agreements: list[bool | None] = []
    tie_flags: list[bool] = []
    rows: list[dict] = []

    def _mae_skill_median_baseline(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """MAE skill against a constant baseline equal to the median of y_true."""

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

    def _mae_skill_zero_baseline(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """MAE skill against the zero-effect baseline (predict 0 everywhere).

        For perturbation effects, the natural null predictor is PE=0, yielding
        baseline MAE = E|PE_ref|.
        """

        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(mask):
            return float("nan")
        yt = y_true[mask]
        yp = y_pred[mask]
        mae = float(np.mean(np.abs(yp - yt)))
        baseline_mae = float(np.mean(np.abs(yt)))
        if baseline_mae <= 0:
            return float("nan")
        return float(1.0 - mae / baseline_mae)

    def _bootstrap_spearman_ci(
        x: np.ndarray, y: np.ndarray, *, n_boot: int, alpha: float = 0.05
    ) -> tuple[float, float]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]
        if x.size < 2:
            return float("nan"), float("nan")

        rng = np.random.default_rng(0)
        rhos: list[float] = []
        for _ in range(int(n_boot)):
            idx = rng.integers(0, x.size, size=x.size)
            rhos.append(spearman_alignment(x[idx], y[idx]))
        lo, hi = np.quantile(np.asarray(rhos, dtype=float), [alpha / 2, 1 - alpha / 2])
        return float(lo), float(hi)

    for i in range(len(ref)):
        r = ref.iloc[i]
        ll = llm.iloc[i]

        llm_low_s = _maybe_parse_list(ll.get("ice_low_samples")) if "ice_low_samples" in llm.columns else None
        llm_high_s = _maybe_parse_list(ll.get("ice_high_samples")) if "ice_high_samples" in llm.columns else None

        pe_ref = perturbation_effect(r["ice_low"], r["ice_high"])
        pe_llm = perturbation_effect(ll["ice_low"], ll["ice_high"])

        dz = float(ll["delta_z"]) if "delta_z" in llm.columns else float("nan")
        dx = float(ll["delta_x"]) if "delta_x" in llm.columns else float("nan")
        pe_ref_std = float(pe_ref / dz) if np.isfinite(dz) and dz != 0 else float("nan")
        pe_llm_std = float(pe_llm / dz) if np.isfinite(dz) and dz != 0 else float("nan")

        # Bootstrap over per-sample ICE arrays if available.
        ref_low_s = _maybe_parse_list(r.get("ice_low_samples")) if "ice_low_samples" in ref.columns else None
        ref_high_s = _maybe_parse_list(r.get("ice_high_samples")) if "ice_high_samples" in ref.columns else None

        if ref_low_s and ref_high_s and len(ref_low_s) == len(ref_high_s):
            pe_ref_samples = np.asarray(ref_high_s, dtype=float) - np.asarray(ref_low_s, dtype=float)
            ci_ref = bootstrap_ci(pe_ref_samples, n_boot=args.boot)
        else:
            # Without reference predictive uncertainty, do not manufacture a false-zero CI.
            ci_ref = bootstrap_ci(np.array([], dtype=float), n_boot=args.boot)

        if llm_low_s and llm_high_s and len(llm_low_s) == len(llm_high_s):
            pe_llm_samples = np.asarray(llm_high_s, dtype=float) - np.asarray(llm_low_s, dtype=float)
            ci_llm = bootstrap_ci(pe_llm_samples, n_boot=args.boot)
        else:
            # LLM ICE should be sample-based; fall back to no-signal if not.
            ci_llm = bootstrap_ci(np.array([], dtype=float), n_boot=args.boot)

        agree, status = directional_agreement(pe_ref=pe_ref, pe_llm=pe_llm, ci_ref=ci_ref, ci_llm=ci_llm)
        agreements.append(agree)
        tie_flags.append(status == "no_signal")

        out = {
            "row": i,
            "id": ll.get("id", r.get("id")),
            "target": ll.get("target", r.get("target")),
            "strategy": ll.get("strategy", r.get("strategy")),
            "outcome": ll.get("outcome", r.get("outcome")),
            "pe_ref": pe_ref,
            "pe_llm": pe_llm,
            "pe_ref_std": pe_ref_std,
            "pe_llm_std": pe_llm_std,
            "delta_z": dz,
            "delta_x": dx,
            "da": agree,
            "da_status": status,
            "ci_ref_low": ci_ref.low,
            "ci_ref_high": ci_ref.high,
            "ci_llm_low": ci_llm.low,
            "ci_llm_high": ci_llm.high,
        }

        if "y_true" in llm.columns:
            out["y_true"] = ll.get("y_true")

        if "paired_order" in llm.columns:
            out["paired_order"] = ll.get("paired_order")

        if has_mid:
            d1_ref, d2_ref = segment_contrasts(r["ice_low"], r["ice_mid"], r["ice_high"])
            d1_llm, d2_llm = segment_contrasts(ll["ice_low"], ll["ice_mid"], ll["ice_high"])
            out.update(
                {
                    "d_low_mid_ref": d1_ref,
                    "d_mid_high_ref": d2_ref,
                    "d_low_mid_llm": d1_llm,
                    "d_mid_high_llm": d2_llm,
                    "d_low_mid_delta": float(d1_llm - d1_ref),
                    "d_mid_high_delta": float(d2_llm - d2_ref),
                    "monotone_ref": np.sign(d1_ref) == np.sign(d2_ref),
                    "monotone_llm": np.sign(d1_llm) == np.sign(d2_llm),
                    "monotone_mismatch": bool(
                        (np.sign(d1_ref) == np.sign(d2_ref)) != (np.sign(d1_llm) == np.sign(d2_llm))
                    ),
                    "curvature_ref": curvature(r["ice_low"], r["ice_mid"], r["ice_high"]),
                    "curvature_llm": curvature(ll["ice_low"], ll["ice_mid"], ll["ice_high"]),
                    "curvature_delta": float(
                        curvature(ll["ice_low"], ll["ice_mid"], ll["ice_high"])
                        - curvature(r["ice_low"], r["ice_mid"], r["ice_high"])
                    ),
                }
            )

        rows.append(out)

    metrics = pd.DataFrame(rows)

    # ---------------------------------------------------------------------
    # Robust directional diagnostics (no CI required)
    #
    # CI-based DA (directional_agreement) is intentionally conservative and will
    # mark rows as no-signal when predictive uncertainty is unavailable.
    # For perturbed evaluation against point-estimated references, we also
    # provide sign-agreement on an informative subset defined by |PE_z|.
    # ---------------------------------------------------------------------
    pe_ref_std = pd.to_numeric(metrics.get("pe_ref_std"), errors="coerce").to_numpy(dtype=float)
    pe_llm_std = pd.to_numeric(metrics.get("pe_llm_std"), errors="coerce").to_numpy(dtype=float)
    abs_ref = np.abs(pe_ref_std)

    sign_ref = np.sign(pe_ref_std)
    sign_llm = np.sign(pe_llm_std)

    epsilon_ref = float(args.epsilon_ref)
    epsilon_move = float(args.epsilon_move_mult) * epsilon_ref

    informative_mask = np.isfinite(abs_ref) & np.isfinite(pe_llm_std) & (abs_ref > epsilon_ref)
    low_signal_mask = np.isfinite(abs_ref) & np.isfinite(pe_llm_std) & (abs_ref <= epsilon_ref)

    abs_llm = np.abs(pe_llm_std)

    metrics["sign_ref"] = sign_ref
    metrics["sign_llm"] = sign_llm
    metrics["sign_informative"] = informative_mask
    metrics["sign_agree"] = np.where(informative_mask, (sign_ref == sign_llm).astype(float), np.nan)
    metrics["low_signal"] = low_signal_mask
    metrics["low_signal_abs_pe_llm_std"] = np.where(low_signal_mask, abs_llm, np.nan)
    metrics["false_positive_move"] = np.where(low_signal_mask, (abs_llm > epsilon_move).astype(float), np.nan)
    metrics["epsilon_ref"] = epsilon_ref
    metrics["epsilon_move"] = epsilon_move

    metrics["dar"] = dar(agreements)
    metrics["tie_rate"] = float(np.mean(np.asarray(tie_flags, dtype=bool))) if tie_flags else float("nan")

    ice_cols = ["ice_low", "ice_high"] + (["ice_mid"] if has_mid else [])
    metrics["smae"] = smae(llm[ice_cols].to_numpy(), ref[ice_cols].to_numpy(), denom=args.denom)
    metrics["spearman_pe"] = spearman_alignment(metrics["pe_ref"].to_numpy(), metrics["pe_llm"].to_numpy())
    metrics["spearman_pe_std"] = spearman_alignment(metrics["pe_ref_std"].to_numpy(), metrics["pe_llm_std"].to_numpy())
    metrics["gini_pe"] = gini_alignment(metrics["pe_ref"].to_numpy(), metrics["pe_llm"].to_numpy())
    metrics["gini_pe_std"] = gini_alignment(metrics["pe_ref_std"].to_numpy(), metrics["pe_llm_std"].to_numpy())
    # Perturbation-effect magnitude accuracy: report skill against the no-effect
    # baseline PE=0, and restrict to the informative/active set.
    pe_ref = metrics["pe_ref"].to_numpy(dtype=float)
    pe_llm = metrics["pe_llm"].to_numpy(dtype=float)
    pe_ref_std = metrics["pe_ref_std"].to_numpy(dtype=float)
    pe_llm_std = metrics["pe_llm_std"].to_numpy(dtype=float)

    metrics["skill_pe"] = _mae_skill_zero_baseline(pe_ref[informative_mask], pe_llm[informative_mask])
    metrics["skill_pe_std"] = _mae_skill_zero_baseline(pe_ref_std[informative_mask], pe_llm_std[informative_mask])

    # Bootstrap CIs for top-line metrics.
    da_status = metrics["da_status"].astype(str) if "da_status" in metrics.columns else pd.Series([], dtype=str)
    scored_mask = (da_status == "scored").to_numpy(dtype=bool) if len(da_status) else np.zeros(len(metrics), dtype=bool)
    if "da" in metrics.columns:
        da_scored = metrics.loc[scored_mask, "da"]
        da_scored_num = da_scored.astype(float).to_numpy(dtype=float) if len(da_scored) else np.array([], dtype=float)
    else:
        da_scored_num = np.array([], dtype=float)

    dar_ci = bootstrap_ci(da_scored_num, n_boot=args.boot)
    tie_ci = (
        bootstrap_ci(np.asarray(tie_flags, dtype=float), n_boot=args.boot)
        if tie_flags
        else bootstrap_ci(np.array([], dtype=float), n_boot=args.boot)
    )

    llm_ice = llm[ice_cols].to_numpy(dtype=float)
    ref_ice = ref[ice_cols].to_numpy(dtype=float)
    abs_err = np.abs(llm_ice - ref_ice)
    per_row_smae = np.nanmean(abs_err, axis=1) / float(args.denom) if args.denom > 0 else np.full(len(metrics), np.nan)
    smae_ci = bootstrap_ci(per_row_smae, n_boot=args.boot)

    spearman_lo, spearman_hi = _bootstrap_spearman_ci(
        metrics["pe_ref"].to_numpy(), metrics["pe_llm"].to_numpy(), n_boot=args.boot
    )
    spearman_std_lo, spearman_std_hi = _bootstrap_spearman_ci(
        metrics["pe_ref_std"].to_numpy(), metrics["pe_llm_std"].to_numpy(), n_boot=args.boot
    )

    metrics["dar_ci_low"] = dar_ci.low
    metrics["dar_ci_high"] = dar_ci.high
    metrics["tie_rate_ci_low"] = tie_ci.low
    metrics["tie_rate_ci_high"] = tie_ci.high
    metrics["smae_ci_low"] = smae_ci.low
    metrics["smae_ci_high"] = smae_ci.high
    metrics["spearman_pe_ci_low"] = spearman_lo
    metrics["spearman_pe_ci_high"] = spearman_hi
    metrics["spearman_pe_std_ci_low"] = spearman_std_lo
    metrics["spearman_pe_std_ci_high"] = spearman_std_hi

    # Optional: unperturbed alignment vs y_true (requires ice_mid + truth merge).
    # We treat ice_mid as the unperturbed point.
    if "y_true" in llm.columns and "ice_mid" in llm.columns:
        y_true = pd.to_numeric(llm["y_true"], errors="coerce").to_numpy(dtype=float)
        y_llm = pd.to_numeric(llm["ice_mid"], errors="coerce").to_numpy(dtype=float)
        y_ref = (
            pd.to_numeric(ref["ice_mid"], errors="coerce").to_numpy(dtype=float) if "ice_mid" in ref.columns else None
        )

        mask_llm = ~(np.isnan(y_true) | np.isnan(y_llm))
        mae_llm = float(np.mean(np.abs(y_llm[mask_llm] - y_true[mask_llm]))) if np.any(mask_llm) else float("nan")
        smae_llm = float(mae_llm / args.denom) if np.isfinite(mae_llm) and args.denom > 0 else float("nan")
        rho_llm = spearman_alignment(y_true, y_llm)

        metrics["unperturbed_mae_true_llm"] = mae_llm
        metrics["unperturbed_smae_true_llm"] = smae_llm
        metrics["unperturbed_spearman_true_llm"] = rho_llm
        metrics["unperturbed_gini_true_llm"] = gini_alignment(y_true, y_llm)
        metrics["unperturbed_mae_skill_true_llm"] = _mae_skill_median_baseline(y_true, y_llm)
        metrics["unperturbed_err_true_llm"] = y_llm - y_true
        metrics["unperturbed_abs_err_true_llm"] = np.abs(metrics["unperturbed_err_true_llm"].to_numpy(dtype=float))

        if y_ref is not None:
            mask_ref = ~(np.isnan(y_true) | np.isnan(y_ref))
            mae_ref = float(np.mean(np.abs(y_ref[mask_ref] - y_true[mask_ref]))) if np.any(mask_ref) else float("nan")
            smae_ref = float(mae_ref / args.denom) if np.isfinite(mae_ref) and args.denom > 0 else float("nan")
            rho_ref = spearman_alignment(y_true, y_ref)

            metrics["unperturbed_mae_true_ref"] = mae_ref
            metrics["unperturbed_smae_true_ref"] = smae_ref
            metrics["unperturbed_spearman_true_ref"] = rho_ref
            metrics["unperturbed_gini_true_ref"] = gini_alignment(y_true, y_ref)
            metrics["unperturbed_mae_skill_true_ref"] = _mae_skill_median_baseline(y_true, y_ref)
            metrics["unperturbed_err_true_ref"] = y_ref - y_true
            metrics["unperturbed_abs_err_true_ref"] = np.abs(metrics["unperturbed_err_true_ref"].to_numpy(dtype=float))

    metrics.to_csv(args.out, index=False)

    out_rows: list[dict] = []
    if "paired_order" in metrics.columns:
        # Focus on paired_profile strategy if present; otherwise summarize any rows that have paired_order.
        if "strategy" in metrics.columns:
            m = metrics
        elif "strategy" in llm.columns:
            m = metrics.join(llm[["strategy"]], how="left")
        else:
            m = metrics
        if "strategy" in m.columns:
            m = m[m["strategy"] == "paired_profile"] if (m["strategy"] == "paired_profile").any() else m

        grp = m.groupby("paired_order", dropna=True)
        for order, g in grp:
            agreements = g["da"].tolist()
            tie_rate = float(np.mean((g["da_status"] == "no_signal").to_numpy(dtype=bool))) if len(g) else float("nan")
            out_rows.append(
                {
                    "paired_order": order,
                    "n": int(len(g)),
                    "dar": dar(agreements),
                    "tie_rate": tie_rate,
                    "mean_pe_llm": float(np.nanmean(g["pe_llm"].to_numpy(dtype=float))),
                    "mean_pe_llm_std": float(np.nanmean(g["pe_llm_std"].to_numpy(dtype=float))),
                    "mean_delta_x": float(np.nanmean(g["delta_x"].to_numpy(dtype=float)))
                    if "delta_x" in g.columns
                    else float("nan"),
                }
            )

    out_cols = [
        "paired_order",
        "n",
        "dar",
        "tie_rate",
        "mean_pe_llm",
        "mean_pe_llm_std",
        "mean_delta_x",
        "dar_delta_A_minus_B",
    ]
    out_df = pd.DataFrame(out_rows, columns=out_cols)
    if not out_df.empty and set(out_df["paired_order"]) >= {"A_then_B", "B_then_A"}:
        a = out_df.set_index("paired_order").loc["A_then_B"]
        b = out_df.set_index("paired_order").loc["B_then_A"]
        out_df["dar_delta_A_minus_B"] = (
            float(a["dar"] - b["dar"]) if np.isfinite(a["dar"]) and np.isfinite(b["dar"]) else float("nan")
        )

    out_df.to_csv(args.order_out, index=False)
