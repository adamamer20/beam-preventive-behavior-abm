from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import typer

# Note: generate_report import intentionally commented out above to disable
# ydata_profiling-based report generation when running this script. The call
# sites were preserved as commented lines so the feature can be re-enabled.
from beam_abm.common.logging import get_logger
from beam_abm.settings import CSV_FILE_CLEAN, CSV_FILE_NN, PREPROCESSING_OUTPUT_DIR
from beam_abm.survey.cleaning.io import load_model, process_data
from beam_abm.survey.load import load_df
from beam_abm.survey.splits import get_balanced_dataset_splits

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Preprocessing-only paths and specs
# -----------------------------------------------------------------------------
# Define here since they are only used by this script.
CSV_FILE_GENERAL = os.path.join(PREPROCESSING_OUTPUT_DIR, "general_processed_survey.csv")
CSV_FILE_LLM = os.path.join(PREPROCESSING_OUTPUT_DIR, "llm_processed_survey.csv")

PREPROCESSING_SPECS_DIR = "preprocess/specs/"
GENERAL_TRANSFORM_FILE = os.path.join(PREPROCESSING_SPECS_DIR, "5_general_transformed_questions.json")
CLEAN_TRANSFORM_FILE = os.path.join(PREPROCESSING_SPECS_DIR, "5_clean_transformed_questions.json")
NN_TRANSFORM_FILE = os.path.join(PREPROCESSING_SPECS_DIR, "6_nn_transformed_questions.json")
LLM_TRANSFORM_FILE = os.path.join(PREPROCESSING_SPECS_DIR, "6_llm_transformed_questions.json")


def process_dataset(model_path: str, input_df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Process a dataset and generate reports."""
    logger.info(f"Processing {name} dataset using model: {model_path}")
    logger.debug(f"Input dataframe shape: {input_df.shape}")

    try:
        model = load_model(model_path)
        logger.info(f"Processing {name} dataset")
        processed_df = process_data(model, input_df)

        base_dir = Path(PREPROCESSING_OUTPUT_DIR)
        output_csv = base_dir / f"{name.lower()}_processed_survey.csv"
        # output_report = base_dir / f"{name.lower()}_dataset_report.html"

        logger.info(f"Saving processed dataset to: {output_csv}")
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        processed_df.to_csv(str(output_csv), index=False)

        logger.info("Processed dataset saved successfully")

        # Report generation (ydata_profiling) commented out to avoid heavy profiling
        # and extra dependencies during pipeline runs. If you want to re-enable
        # profile reports, uncomment the lines below.
        # logger.info("Generating dataset report")
        # generate_report(processed_df, model, name, str(output_report))

        logger.info(f"{name} dataset processing completed successfully")
        logger.info(f"  - Processed CSV: {output_csv}")
        # Report generation disabled; the report file will not be created by default
        # logger.info(f"  - Report: {output_report}")

        return processed_df

    except Exception as exc:
        logger.error(f"Error processing {name} dataset: {exc}")
        raise


def main(selected: set[str] | None = None) -> None:
    """Run the dataset preprocessing pipeline.

    Args:
        selected: optional set of dataset names to process (any of 'clean', 'nn', 'llm').
                  If None or empty, all datasets are processed.
    """
    logger.info("Starting dataset preprocessing pipeline")

    # Normalize selection
    if selected is None or len(selected) == 0:
        selected_set = {"clean", "nn", "llm"}
    else:
        selected_set = {s.lower() for s in selected}

    logger.info(f"Datasets selected for processing: {sorted(selected_set)}")

    try:
        # Process GENERAL only when NN or LLM are requested (GENERAL is the base
        # dataset used to derive NN/LLM variations). If only CLEAN is requested,
        # skip GENERAL entirely.
        general_df = None
        if "nn" in selected_set or "llm" in selected_set:
            logger.info("Processing general dataset")

            if os.path.exists(CSV_FILE_GENERAL):
                logger.warning(
                    f"General dataset already exists at {CSV_FILE_GENERAL}; it will be regenerated and overwritten."
                )

            logger.info("Loading source survey data")
            input_survey_df = load_df()
            logger.info(f"Source survey data loaded with shape: {input_survey_df.shape}")

            general_df = process_dataset(
                GENERAL_TRANSFORM_FILE,
                input_survey_df,
                "GENERAL",
            )

            logger.info("Adding balanced dataset splits")
            general_df = get_balanced_dataset_splits(general_df)

            output_path = CSV_FILE_GENERAL
            logger.info(f"Saving general dataset with splits to: {output_path}")
            Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
            general_df.to_csv(output_path, index=False)

            general_df = pd.read_csv(CSV_FILE_GENERAL)
            logger.info(f"Reloaded general dataset with shape: {general_df.shape}")

        # NN
        if "nn" in selected_set:
            logger.info("Processing NN dataset")

            if os.path.exists(CSV_FILE_NN):
                logger.warning(f"NN dataset already exists at {CSV_FILE_NN}; it will be regenerated and overwritten.")

            if general_df is None:
                # If general_df wasn't created in this run, regenerate it from source data
                # to avoid loading a cached CSV.
                logger.info("General dataset not present in memory; regenerating from source (avoiding cached CSV).")
                input_survey_df = load_df()
                logger.info(f"Source survey data loaded with shape: {input_survey_df.shape}")

                general_df = process_dataset(
                    GENERAL_TRANSFORM_FILE,
                    input_survey_df,
                    "GENERAL",
                )

                logger.info("Adding balanced dataset splits")
                general_df = get_balanced_dataset_splits(general_df)

                output_path = CSV_FILE_GENERAL
                logger.info(f"Saving regenerated general dataset with splits to: {output_path}")
                Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
                general_df.to_csv(output_path, index=False)

                general_df = pd.read_csv(CSV_FILE_GENERAL)
                logger.info(f"Regenerated general dataset with shape: {general_df.shape}")

            logger.info("Creating NN dataset from general dataset")
            process_dataset(NN_TRANSFORM_FILE, general_df, "NN")
            logger.info("NN dataset processing completed")

        # LLM
        if "llm" in selected_set:
            logger.info("Processing LLM dataset")

            if os.path.exists(CSV_FILE_LLM):
                logger.warning(f"LLM dataset already exists at {CSV_FILE_LLM}; it will be regenerated and overwritten.")

            if general_df is None:
                # If general_df wasn't created in this run, regenerate it from source data
                # to avoid loading a cached CSV.
                logger.info("General dataset not present in memory; regenerating from source (avoiding cached CSV).")
                input_survey_df = load_df()
                logger.info(f"Source survey data loaded with shape: {input_survey_df.shape}")

                general_df = process_dataset(
                    GENERAL_TRANSFORM_FILE,
                    input_survey_df,
                    "GENERAL",
                )

                logger.info("Adding balanced dataset splits")
                general_df = get_balanced_dataset_splits(general_df)

                output_path = CSV_FILE_GENERAL
                logger.info(f"Saving regenerated general dataset with splits to: {output_path}")
                Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
                general_df.to_csv(output_path, index=False)

                general_df = pd.read_csv(CSV_FILE_GENERAL)
                logger.info(f"Regenerated general dataset with shape: {general_df.shape}")

            logger.info("Creating LLM dataset from general dataset")
            process_dataset(LLM_TRANSFORM_FILE, general_df, "LLM")
            logger.info("LLM dataset processing completed")

        # CLEAN dataset can be created independently from the raw survey data.
        if "clean" in selected_set:
            logger.info("Processing CLEAN dataset")

            if os.path.exists(CSV_FILE_CLEAN):
                logger.warning(
                    f"CLEAN dataset already exists at {CSV_FILE_CLEAN}; it will be regenerated and overwritten."
                )

            logger.info("Loading source survey data for CLEAN dataset")
            clean_input_df = load_df()
            logger.info(f"Raw survey data loaded with shape: {clean_input_df.shape}")

            logger.info(f"Using clean transform file: {CLEAN_TRANSFORM_FILE}")
            process_dataset(CLEAN_TRANSFORM_FILE, clean_input_df, "CLEAN")
            logger.info("CLEAN dataset processing completed")

        logger.info("Dataset preprocessing pipeline completed successfully")

    except Exception as exc:
        logger.error(f"Error in dataset preprocessing pipeline: {exc}")
        raise


def _parse_selection_arg(raw: str | None) -> set[str] | None:
    """Parse the comma-separated selection argument into a set.

    Returns None when the caller requested 'all' or provided nothing.
    """
    if raw is None:
        return None

    raw_clean = raw.strip().lower()
    if raw_clean == "" or raw_clean == "all":
        return None

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return set(parts)


app = typer.Typer(add_completion=False)


@app.command()
def cli(
    select: str | None = typer.Option(
        None,
        "--select",
        "-s",
        help=(
            "Comma-separated list of datasets to process: 'clean', 'nn', 'llm'. "
            "Use 'all' or omit to process all datasets."
        ),
    ),
) -> None:
    """Command-line entrypoint for preprocessing.

    Example: `python main.py --select clean,nn` or `python main.py` for all.
    """
    logger.info("Starting dataset preprocessing script")
    selected = _parse_selection_arg(select)
    main(selected)
    logger.info("Dataset preprocessing script completed successfully")


if __name__ == "__main__":
    app()
