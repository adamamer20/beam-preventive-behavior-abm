from __future__ import annotations

import argparse
from pathlib import Path

from beam_abm.preprocess.pipeline import DEFAULT_CLEAN_SPEC, run_clean_pipeline
from beam_abm.settings import CSV_FILE_CLEAN


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the clean-only preprocessing pipeline")
    parser.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_CLEAN_SPEC,
        help="Path to the clean transform spec JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(CSV_FILE_CLEAN),
        help="Destination CSV for the clean processed dataset",
    )
    args = parser.parse_args()
    run_clean_pipeline(model_path=args.spec, output_csv=args.output)


if __name__ == "__main__":
    main()
