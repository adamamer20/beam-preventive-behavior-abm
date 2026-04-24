"""Compute block-level alignment."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from beam_abm.llm_microvalidation.shared.metrics import spearman_alignment


def _topk_overlap(a: pd.Series, b: pd.Series, k: int) -> float:
    if k <= 0:
        return float("nan")
    a_top = set(a.sort_values(ascending=False).head(k).index)
    b_top = set(b.sort_values(ascending=False).head(k).index)
    if not a_top and not b_top:
        return 1.0
    if not a_top or not b_top:
        return 0.0
    return len(a_top & b_top) / float(k)


def _highest_quartile_concordance(a: pd.Series, b: pd.Series) -> float:
    """Share of reference top-quartile blocks that are also top-quartile in LLM."""

    if a.empty or b.empty:
        return float("nan")

    n = int(min(len(a), len(b)))
    if n == 0:
        return float("nan")

    q = max(1, int(np.ceil(0.25 * n)))
    a_top = set(a.sort_values(ascending=False).head(q).index)
    b_top = set(b.sort_values(ascending=False).head(q).index)
    if not a_top:
        return float("nan")
    return float(len(a_top & b_top) / len(a_top))


def compute_block_alignment(
    *,
    out: str | Path,
    ref: str | Path | None = None,
    llm: str | Path | None = None,
    ref_ice: str | Path | None = None,
    llm_ice: str | Path | None = None,
    block_map: str | Path | None = None,
    ref_cko: str | Path | None = None,
    llm_cko: str | Path | None = None,
    topk: int = 5,
) -> None:
    args = SimpleNamespace(
        ref=str(ref) if ref is not None else None,
        llm=str(llm) if llm is not None else None,
        ref_ice=str(ref_ice) if ref_ice is not None else None,
        llm_ice=str(llm_ice) if llm_ice is not None else None,
        block_map=str(block_map) if block_map is not None else None,
        ref_cko=str(ref_cko) if ref_cko is not None else None,
        llm_cko=str(llm_cko) if llm_cko is not None else None,
        topk=int(topk),
        out=str(out),
    )

    if args.ref and args.llm:
        ref = pd.read_csv(args.ref)
        llm = pd.read_csv(args.llm)
    else:
        if not (args.ref_ice and args.llm_ice and args.block_map):
            raise ValueError("Provide either --ref/--llm or --ref-ice/--llm-ice/--block-map")

        ref_ice = pd.read_csv(args.ref_ice)
        llm_ice = pd.read_csv(args.llm_ice)
        bmap = pd.read_csv(args.block_map)
        if "target" not in bmap.columns or "block" not in bmap.columns:
            raise ValueError("--block-map must have columns: target, block")

        def _abs_pe_std(df: pd.DataFrame) -> pd.DataFrame:
            if "delta_z" not in df.columns:
                raise ValueError("ICE table must include delta_z")
            pe = df["ice_high"].astype(float) - df["ice_low"].astype(float)
            dz = df["delta_z"].astype(float)
            pe_std = pe / dz.replace({0.0: pd.NA})
            out = pd.DataFrame({"target": df["target"], "abs_pe_std": pe_std.abs().astype(float)})
            return out.merge(bmap[["target", "block"]], on="target", how="left")

        ref = _abs_pe_std(ref_ice)
        llm = _abs_pe_std(llm_ice)

    ref_med = ref.groupby("block", as_index=False)["abs_pe_std"].median().rename(columns={"abs_pe_std": "m_ref"})
    llm_med = llm.groupby("block", as_index=False)["abs_pe_std"].median().rename(columns={"abs_pe_std": "m_llm"})
    merged = ref_med.merge(llm_med, on="block", how="inner")

    bsa = spearman_alignment(merged["m_ref"].to_numpy(), merged["m_llm"].to_numpy())
    merged["bsa"] = bsa

    by_block = merged.set_index("block")[["m_ref", "m_llm"]]
    merged["bsa_topk_overlap"] = _topk_overlap(by_block["m_ref"], by_block["m_llm"], args.topk)
    merged["bsa_highest_quartile_concordance"] = _highest_quartile_concordance(by_block["m_ref"], by_block["m_llm"])

    # Optional: CKO importance alignment.
    if args.ref_cko and args.llm_cko:
        ref_cko = pd.read_csv(args.ref_cko)
        llm_cko = pd.read_csv(args.llm_cko)
        if "block" not in ref_cko.columns or "importance" not in ref_cko.columns:
            raise ValueError("--ref-cko must have columns: block, importance")
        if "block" not in llm_cko.columns or "importance" not in llm_cko.columns:
            raise ValueError("--llm-cko must have columns: block, importance")

        both = ref_cko.set_index("block")[["importance"]].join(
            llm_cko.set_index("block")[["importance"]],
            how="inner",
            lsuffix="_ref",
            rsuffix="_llm",
        )
        rho = spearman_alignment(both["importance_ref"].to_numpy(), both["importance_llm"].to_numpy())
        topk = _topk_overlap(both["importance_ref"], both["importance_llm"], args.topk)
        merged["cko_spearman"] = rho
        merged["cko_topk_overlap"] = topk

    merged.to_csv(args.out, index=False)
