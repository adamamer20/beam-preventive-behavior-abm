from __future__ import annotations

import pandas as pd


def validate_processed_dataframe(
    *,
    df: pd.DataFrame,
    final_name_refs: set[str],
    logger,
) -> None:
    if final_name_refs:
        missing_final = sorted([name for name in final_name_refs if name not in df.columns])
        if missing_final:
            raise ValueError(
                "Final-name references were present in the spec but missing after processing/renames: "
                + ", ".join(missing_final)
            )

    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns
    logger.info(
        f"Final dataframe composition: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical, {len(datetime_cols)} datetime columns"
    )

    total_missing = int(df.isnull().sum().sum())
    if total_missing > 0:
        logger.warning(f"Final dataframe has {total_missing} missing values")
        missing_by_col = df.isnull().sum()
        cols_with_missing = missing_by_col[missing_by_col > 0]
        logger.debug(f"Missing values by column: {cols_with_missing.to_dict()}")
    else:
        logger.info("No missing values in final dataframe")
