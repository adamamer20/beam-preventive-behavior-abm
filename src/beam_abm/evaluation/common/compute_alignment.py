"""CLI wrapper for alignment metrics computation."""

from __future__ import annotations

from pathlib import Path

from beam_abm.cli import parser_compat as cli
from beam_abm.evaluation.common.alignment_metrics import compute_alignment_metrics


def compute_alignment_step(
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
    compute_alignment_metrics(
        ref=ref,
        llm=llm,
        out=out,
        order_out=order_out,
        truth=truth,
        truth_y_col=truth_y_col,
        denom=denom,
        boot=boot,
        epsilon_ref=epsilon_ref,
        epsilon_move_mult=epsilon_move_mult,
    )


def run_cli(argv: list[str] | None = None) -> None:
    parser = cli.ArgumentParser()
    parser.add_argument("--ref", required=True, help="Reference ICE CSV")
    parser.add_argument("--llm", required=True, help="LLM ICE CSV")
    parser.add_argument(
        "--truth",
        default=None,
        help=(
            "Optional CSV with ground-truth outcomes to compute unperturbed (ice_mid) error against y_true. "
            "Must include a y_true column plus merge keys (typically id + outcome)."
        ),
    )
    parser.add_argument("--truth-y-col", default="y_true", help="Column name for ground truth values (default: y_true)")
    parser.add_argument("--out", required=True, help="Output metrics CSV")
    parser.add_argument(
        "--order-out",
        required=True,
        help="CSV path to summarize paired-profile order effects (A_then_B vs B_then_A).",
    )
    parser.add_argument("--denom", type=float, default=1.0, help="Denominator for NMAE (e.g. range(Y) or sd(Y))")
    parser.add_argument("--boot", type=int, default=1000)
    parser.add_argument(
        "--epsilon-ref",
        type=float,
        default=0.03,
        help=(
            "Reference-defined low-signal threshold in PE_z units (default: 0.03). "
            "Defines S0: |pe_ref_std| <= epsilon_ref and S1: |pe_ref_std| > epsilon_ref."
        ),
    )
    parser.add_argument(
        "--epsilon-move-mult",
        type=float,
        default=3.0,
        help=("Multiplier for the false-movement threshold epsilon_move = epsilon_ref * multiplier (default: 3)."),
    )
    args = parser.parse_args(argv)
    compute_alignment_step(
        ref=args.ref,
        llm=args.llm,
        out=args.out,
        order_out=args.order_out,
        truth=args.truth,
        truth_y_col=args.truth_y_col,
        denom=args.denom,
        boot=args.boot,
        epsilon_ref=args.epsilon_ref,
        epsilon_move_mult=args.epsilon_move_mult,
    )
