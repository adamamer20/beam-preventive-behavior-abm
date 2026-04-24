"""Rebuild heterogeneity CDF knots from the survey population."""

from __future__ import annotations

import argparse
from pathlib import Path

from beam_abm.abm.workflows import rebuild_heterogeneity_cdf_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path (defaults to heterogeneity_loadings.json).",
    )
    args = parser.parse_args()
    rebuild_heterogeneity_cdf_workflow(output=args.output)


if __name__ == "__main__":
    main()
