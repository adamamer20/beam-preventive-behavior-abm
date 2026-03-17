from __future__ import annotations

import os

import pandas as pd

from beam_abm.common.logging import get_logger

from .models import DataCleaningModel

logger = get_logger(__name__)


def load_model(file_path: str) -> DataCleaningModel:
    """Load a DataCleaningModel from a JSON file."""
    logger.info(f"Loading data cleaning model from: {file_path}")

    try:
        if not os.path.exists(file_path):
            logger.error(f"Model file not found: {file_path}")
            raise FileNotFoundError(f"Model file not found: {file_path}")

        with open(file_path) as f:
            model_json = f.read()

        logger.debug(f"Model file size: {len(model_json)} characters")

        model = DataCleaningModel.model_validate_json(model_json)

        # Log model statistics
        old_cols = len(model.old_columns) if model.old_columns else 0
        new_cols = len(model.new_columns) if model.new_columns else 0
        encodings = len(model.encodings) if model.encodings else 0
        bins = len(model.bins) if model.bins else 0

        logger.info(
            f"Successfully loaded model: {old_cols} old columns, {new_cols} new columns, {encodings} encodings, {bins} bin configs"
        )

        return model

    except Exception as e:
        logger.error(f"Error loading model from {file_path}: {e}")
        raise


def process_data(model: DataCleaningModel, df: pd.DataFrame) -> pd.DataFrame:
    """Process a dataframe using the given model."""
    logger.info(f"Processing dataframe with shape {df.shape} using data cleaning model")

    # Log input dataframe statistics
    logger.debug(f"Input dataframe columns: {len(df.columns)}")
    logger.debug(f"Input dataframe dtypes: {df.dtypes.value_counts().to_dict()}")

    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        logger.info(f"Input dataframe has {total_missing} missing values")

    try:
        from .processor import DataCleaningProcessor

        processor = DataCleaningProcessor(model)
        result_df = processor.process_transformations(df)

        logger.info(f"Data processing completed successfully - output shape: {result_df.shape}")

        return result_df

    except Exception as e:
        logger.error(f"Error during data processing: {e}")
        raise


def generate_report(df: pd.DataFrame, model: DataCleaningModel, name: str, output_file: str) -> None:
    """Generate and save a report for the processed dataframe."""
    logger.info(
        "Report generation is currently disabled in this environment. "
        "To re-enable profiling, uncomment the implementation in this function and "
        "restore the `ProfileReport` import at the top of the module."
    )

    # The original implementation using ydata_profiling.ProfileReport is preserved
    # below as commented lines so it can be re-enabled if desired.
    # try:
    #     # Create descriptions dictionary
    #     descriptions: dict[str, str] = {}
    #     columns = model.old_columns
    #     if model.new_columns:
    #         columns += model.new_columns

    #     logger.debug(f"Processing {len(columns)} columns for report descriptions")

    #     descriptions_added = 0
    #     for col in columns:
    #         try:
    #             if col.new_question:
    #                 descriptions[col.id] = col.new_question
    #                 descriptions_added += 1
    #             elif col.question:
    #                 descriptions[col.id] = col.question
    #                 descriptions_added += 1
    #         except AttributeError:
    #             logger.debug(f"Column '{col.id}' has no question attribute")

    #     logger.debug(f"Added {descriptions_added} column descriptions")

    #     # Generate report
    #     logger.info(f"Creating ProfileReport for dataframe with shape {df.shape}")

    #     report = ProfileReport(
    #         df,
    #         title=f"{name} Processed Survey Data",
    #         variables={"descriptions": descriptions},
    #         minimal=True,
    #     )

    #     logger.info(f"Saving report to: {output_file}")
    #     report.to_file(output_file)

    #     logger.info(f"Report generated successfully: {output_file}")

    # except Exception as e:
    #     logger.error(f"Error generating report for {name}: {e}")
    #     raise
