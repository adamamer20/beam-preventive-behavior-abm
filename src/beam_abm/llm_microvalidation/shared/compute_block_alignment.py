"""CLI wrapper for block alignment computation."""

from __future__ import annotations

from pathlib import Path

from beam_abm.cli import parser_compat as cli
from beam_abm.llm_microvalidation.shared.block_alignment import compute_block_alignment


def compute_block_alignment_step(
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
    compute_block_alignment(
        out=out,
        ref=ref,
        llm=llm,
        ref_ice=ref_ice,
        llm_ice=llm_ice,
        block_map=block_map,
        ref_cko=ref_cko,
        llm_cko=llm_cko,
        topk=topk,
    )


def run_cli(argv: list[str] | None = None) -> None:
    parser = cli.ArgumentParser()
    parser.add_argument("--ref", default=None, help="CSV with columns block, abs_pe_std")
    parser.add_argument("--llm", default=None, help="CSV with columns block, abs_pe_std")
    parser.add_argument("--ref-ice", default=None, help="Reference ICE CSV")
    parser.add_argument("--llm-ice", default=None, help="LLM ICE CSV")
    parser.add_argument("--block-map", default=None, help="CSV mapping with columns target, block")
    parser.add_argument("--ref-cko", default=None, help="CSV with columns block, importance")
    parser.add_argument("--llm-cko", default=None, help="CSV with columns block, importance")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    compute_block_alignment_step(
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
