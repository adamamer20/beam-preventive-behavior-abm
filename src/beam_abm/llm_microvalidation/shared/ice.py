"""Aggregate sampled LLM outputs into ICE tables."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from beam_abm.llm_microvalidation.utils.id_utils import normalize_perturbed_dataset_id
from beam_abm.llm_microvalidation.utils.jsonl import read_jsonl as _read_jsonl


def _extract_samples(row: dict) -> list[float]:
    # Most strategies: row["samples"] is a list of SampleResult dicts.
    if isinstance(row.get("samples"), list):
        out: list[float] = []
        for s in row["samples"]:
            if not isinstance(s, dict):
                continue
            v = s.get("prediction_scalar")
            if v is None:
                continue
            try:
                out.append(float(v))
            except Exception:
                continue
        return out

    # Legacy fallback: multi_expert used to store a single scalar at top-level.
    v = row.get("prediction_scalar")
    if v is None:
        return []
    try:
        return [float(v)]
    except Exception:
        return []


def _extract_paired_samples(
    row: dict,
    *,
    swap_on_b_then_a: bool,
) -> tuple[list[float], list[float], dict[str, int]]:
    """Extract paired-profile samples.

    Convention: A == low profile, B == high profile (independent of display order).

    Some model runs appear to return A/B in *display order* (first/second) rather than
    by label. If `swap_on_b_then_a` is enabled, we swap A/B for paired_order=B_then_A
    to recover the intended low/high mapping.
    """

    total = 0
    ok_low = 0
    ok_high = 0
    missing_low = 0
    missing_high = 0
    errors = 0

    low: list[float] = []
    high: list[float] = []

    samples = row.get("samples")
    if not isinstance(samples, list):
        return (
            low,
            high,
            {
                "n_total": 0,
                "n_ok_low": 0,
                "n_ok_high": 0,
                "n_missing_low": 0,
                "n_missing_high": 0,
                "n_error": 0,
            },
        )

    paired_order: str | None = None
    meta = row.get("metadata") or {}
    if isinstance(meta, dict):
        po = meta.get("paired_order")
        if isinstance(po, str) and po.strip():
            paired_order = po.strip()

    do_swap = bool(swap_on_b_then_a and paired_order == "B_then_A")

    for s in samples:
        if not isinstance(s, dict):
            continue
        total += 1
        if s.get("error"):
            errors += 1
        a = s.get("prediction_scalar_A")
        b = s.get("prediction_scalar_B")
        if do_swap:
            a, b = b, a

        if a is None:
            missing_low += 1
        else:
            try:
                low.append(float(a))
                ok_low += 1
            except Exception:
                missing_low += 1

        if b is None:
            missing_high += 1
        else:
            try:
                high.append(float(b))
                ok_high += 1
            except Exception:
                missing_high += 1

    return (
        low,
        high,
        {
            "n_total": total,
            "n_ok_low": ok_low,
            "n_ok_high": ok_high,
            "n_missing_low": missing_low,
            "n_missing_high": missing_high,
            "n_error": errors,
        },
    )


def compute_ice_from_samples(
    *,
    in_path: str | Path,
    out_path: str | Path,
    summary_out: str | Path | None = None,
    paired_profile_swap_on_b_then_a: bool = False,
) -> None:
    args = SimpleNamespace(
        in_path=str(in_path),
        out_path=str(out_path),
        summary_out=str(summary_out) if summary_out is not None else None,
        paired_profile_swap_on_b_then_a=bool(paired_profile_swap_on_b_then_a),
    )

    rows = _read_jsonl(Path(args.in_path))

    packed: list[dict] = []
    for r in rows:
        meta = r.get("metadata") or {}

        strat = r.get("strategy")
        tgt = meta.get("target", r.get("target"))
        outcome = meta.get("outcome", r.get("outcome"))
        rid = normalize_perturbed_dataset_id(r.get("id"), outcome=outcome) or r.get("id")

        def _get_delta_z_from_meta(m: object) -> float | None:
            if not isinstance(m, dict):
                return None
            v = m.get("delta_z")
            try:
                return float(v)
            except Exception:
                return None

        dz = _get_delta_z_from_meta(meta.get("metadata")) if isinstance(meta.get("metadata"), dict) else None
        if dz is None:
            dz = _get_delta_z_from_meta(meta)

        if strat == "paired_profile":
            # For paired-profile prompts, the grid-point metadata is stored as metadata_low/high.
            # Use delta_z from low/high and verify they match when both are present.
            dz_low = _get_delta_z_from_meta(meta.get("metadata_low"))
            dz_high = _get_delta_z_from_meta(meta.get("metadata_high"))
            if dz_low is not None and dz_high is not None and np.isfinite(dz_low) and np.isfinite(dz_high):
                dz = dz_low if abs(dz_low - dz_high) < 1e-9 else float("nan")
            elif dz_low is not None:
                dz = dz_low
            elif dz_high is not None:
                dz = dz_high

            low, high, counts = _extract_paired_samples(
                r,
                swap_on_b_then_a=bool(args.paired_profile_swap_on_b_then_a),
            )
            # Paired-profile encodes both low/high in a single prompt response.
            packed.append(
                {
                    "id": rid,
                    "strategy": strat,
                    "target": tgt,
                    "outcome": outcome,
                    "grid_point": "low",
                    "delta_z": dz,
                    "paired_order": meta.get("paired_order"),
                    "samples": low,
                    "n_total": counts["n_total"],
                    "n_ok": counts["n_ok_low"],
                    "n_missing": counts["n_missing_low"],
                    "n_error": counts["n_error"],
                }
            )
            packed.append(
                {
                    "id": rid,
                    "strategy": strat,
                    "target": tgt,
                    "outcome": outcome,
                    "grid_point": "high",
                    "delta_z": dz,
                    "paired_order": meta.get("paired_order"),
                    "samples": high,
                    "n_total": counts["n_total"],
                    "n_ok": counts["n_ok_high"],
                    "n_missing": counts["n_missing_high"],
                    "n_error": counts["n_error"],
                }
            )
            continue

        samples = _extract_samples(r)
        n_total = (
            len(r.get("samples") or [])
            if isinstance(r.get("samples"), list)
            else (1 if r.get("prediction_scalar") is not None else 0)
        )
        n_ok = len(samples)
        packed.append(
            {
                "id": rid,
                "strategy": strat,
                "target": tgt,
                "outcome": outcome,
                "grid_point": meta.get("grid_point", r.get("grid_point")),
                "delta_z": dz,
                "paired_order": meta.get("paired_order"),
                "samples": samples,
                "n_total": int(n_total),
                "n_ok": int(n_ok),
                "n_missing": int(max(0, n_total - n_ok)),
                "n_error": int(
                    sum(1 for s in (r.get("samples") or []) if isinstance(s, dict) and s.get("error"))
                    if isinstance(r.get("samples"), list)
                    else 0
                ),
            }
        )

    df = pd.DataFrame(packed)
    if df.empty:
        raise ValueError("No rows loaded")

    # Normalize grid_point labels
    df["grid_point"] = df["grid_point"].astype(str)

    out_rows: list[dict] = []
    group_cols = ["id", "strategy", "target"]
    if "outcome" in df.columns:
        group_cols.append("outcome")
    if "outcome" in df.columns:
        group_cols.append("outcome")
    for (rid, strat, tgt, *rest), g in df.groupby(group_cols, dropna=False):
        grouped = pd.DataFrame.groupby(g, "grid_point", dropna=False)
        by_point = grouped["samples"].apply(list).to_dict()
        by_counts_total = grouped["n_total"].sum().to_dict()
        by_counts_ok = grouped["n_ok"].sum().to_dict()
        by_counts_missing = grouped["n_missing"].sum().to_dict()
        by_counts_error = grouped["n_error"].sum().to_dict()

        def _flatten(vals: pd.Series) -> list[float]:
            out: list[float] = []
            for v in vals:
                if isinstance(v, list):
                    out.extend(v)
            return out

        row: dict = {
            "id": rid,
            "strategy": strat,
            "target": tgt,
        }
        if rest:
            row["outcome"] = rest[0]

        for grid_point, series in by_point.items():
            samples = _flatten(series)
            col = f"ice_{grid_point}"
            row[col] = float(np.mean(samples)) if samples else float("nan")
            row[f"{col}_samples"] = json.dumps(samples)
            row[f"{col}_n_total"] = int(by_counts_total.get(grid_point, 0))
            row[f"{col}_n_ok"] = int(by_counts_ok.get(grid_point, 0))
            row[f"{col}_n_missing"] = int(by_counts_missing.get(grid_point, 0))
            row[f"{col}_n_error"] = int(by_counts_error.get(grid_point, 0))

        # carry through delta_z for downstream standardization
        if "delta_z" in g.columns:
            delta_z_vals = g["delta_z"].dropna().unique().tolist()
            row["delta_z"] = float(delta_z_vals[0]) if len(delta_z_vals) == 1 else float("nan")

        if "paired_order" in g.columns:
            paired_order_vals = g["paired_order"].dropna().unique().tolist()
            if len(paired_order_vals) == 1:
                row["paired_order"] = paired_order_vals[0]

        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_path, index=False)

    if args.summary_out:
        summary = out_df.copy()
        summary["n_total"] = summary[[c for c in out_df.columns if c.endswith("_n_total")]].sum(axis=1)
        summary["n_ok"] = summary[[c for c in out_df.columns if c.endswith("_n_ok")]].sum(axis=1)
        summary["n_missing"] = summary[[c for c in out_df.columns if c.endswith("_n_missing")]].sum(axis=1)
        summary["n_error"] = summary[[c for c in out_df.columns if c.endswith("_n_error")]].sum(axis=1)
        summary["noncompliance_rate"] = 1.0 - summary["n_ok"].replace(0, float("nan")) / summary["n_total"].replace(
            0, float("nan")
        )

        cols = ["strategy", "target"]
        if "outcome" in summary.columns:
            cols.append("outcome")

        per_strategy = (
            summary.groupby(cols, dropna=False)
            .agg(n_total=("n_total", "sum"), n_ok=("n_ok", "sum"), n_error=("n_error", "sum"))
            .reset_index()
        )
        per_strategy["noncompliance_rate"] = 1.0 - per_strategy["n_ok"].replace(0, float("nan")) / per_strategy[
            "n_total"
        ].replace(0, float("nan"))

        overall = pd.DataFrame(
            [
                {
                    "strategy": "__all__",
                    "target": "__all__",
                    "n_total": int(per_strategy["n_total"].sum()),
                    "n_ok": int(per_strategy["n_ok"].sum()),
                    "n_error": int(per_strategy["n_error"].sum()),
                    "noncompliance_rate": 1.0 - per_strategy["n_ok"].sum() / max(1, per_strategy["n_total"].sum()),
                }
            ]
        )
        summary_df = pd.concat([per_strategy, overall], ignore_index=True)
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(args.summary_out, index=False)
