"""CLI wrapper for block alignment computation."""

from __future__ import annotations

import argparse

from beam_abm.evaluation.common.block_alignment import compute_block_alignment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default=None, help="CSV with columns block, abs_pe_std")
    parser.add_argument("--llm", default=None, help="CSV with columns block, abs_pe_std")
    parser.add_argument("--ref-ice", default=None, help="Reference ICE CSV")
    parser.add_argument("--llm-ice", default=None, help="LLM ICE CSV")
    parser.add_argument("--block-map", default=None, help="CSV mapping with columns target, block")
    parser.add_argument("--ref-cko", default=None, help="CSV with columns block, importance")
    parser.add_argument("--llm-cko", default=None, help="CSV with columns block, importance")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    compute_block_alignment(
        out=args.out,
        ref=args.ref,
        llm=args.llm,
        ref_ice=args.ref_ice,
        llm_ice=args.llm_ice,
        block_map=args.block_map,
        ref_cko=args.ref_cko,
        llm_cko=args.llm_cko,
        topk=args.topk,
    )
