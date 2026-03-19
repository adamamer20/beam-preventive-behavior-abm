from __future__ import annotations

from pathlib import Path

import pandas as pd

from beam_abm.common.logging import get_logger
from beam_abm.settings import CSV_FILE_CLEAN

from .cleaning import load_model, process_data
from .ingest import load_raw_survey_df

logger = get_logger(__name__)

DEFAULT_CLEAN_SPEC = Path("preprocess/specs/5_clean_transformed_questions.json")


def run_clean_pipeline(
    *,
    model_path: Path = DEFAULT_CLEAN_SPEC,
    output_csv: Path = Path(CSV_FILE_CLEAN),
) -> pd.DataFrame:
    """Run the clean-only preprocessing pipeline and persist the output CSV."""
    logger.info("Starting clean-only preprocessing pipeline")
    logger.info("Loading raw survey data")
    raw_df = load_raw_survey_df()
    logger.info(f"Raw survey shape: {raw_df.shape}")

    logger.info(f"Loading clean model from: {model_path}")
    model = load_model(str(model_path))
    processed_df = process_data(model, raw_df)

    logger.info(f"Writing clean dataset to: {output_csv}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_csv, index=False)
    logger.info("Clean-only preprocessing pipeline completed")
    return processed_df
