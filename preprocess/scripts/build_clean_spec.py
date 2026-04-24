from __future__ import annotations

import argparse
from pathlib import Path

from beam_abm.preprocess.specs.clean_spec import (
    DEFAULT_OUTPUT_POST_GENERAL,
    DEFAULT_OUTPUT_RAW,
    build_clean_analysis_spec,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean analysis spec")
    parser.add_argument(
        "--stage",
        choices=["raw", "post-general"],
        default="raw",
        help="Apply spec to raw data or to general-processed data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output path for the generated spec",
    )
    args = parser.parse_args()

    output = args.output or (DEFAULT_OUTPUT_RAW if args.stage == "raw" else DEFAULT_OUTPUT_POST_GENERAL)
    build_clean_analysis_spec(output_path=output, stage=args.stage)


if __name__ == "__main__":
    main()
