from __future__ import annotations

import os
from warnings import warn

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from beam_abm.common.logging import get_logger

from .models import (
    BinningTransformation,
    Column,
    ColumnSubstitutionTransformation,
    MergedTransformation,
    OutliersTransformation,
    RemovalTransformation,
    RenamedTransformation,
    RenamingTransformation,
    ScalingTransformation,
    SplittedTransformation,
)

logger = get_logger(__name__)


def process_outliers(context, col: Column, transformation: OutliersTransformation) -> None:
    logger.info(f"Processing outliers for column '{col.id}' (min: {transformation.min}, max: {transformation.max})")

    if col.id not in context.df.columns:
        logger.warning(f"Column '{col.id}' not found in dataframe, skipping outliers")
        return

    original_dtype = context.df[col.id].dtype
    logger.debug(f"Original dtype: {original_dtype}")

    context.df[col.id] = pd.to_numeric(context.df[col.id], errors="coerce")

    na_count = context.df[col.id].isna().sum()
    if na_count > 0:
        logger.warning(f"Conversion to numeric resulted in {na_count} NA values")

    if transformation.max is not None:
        max_outliers = (context.df[col.id] > transformation.max).sum()
        if max_outliers > 0:
            logger.info(f"Capping {max_outliers} values above {transformation.max}")
            context.df.loc[context.df[col.id] > transformation.max, col.id] = transformation.max
        else:
            logger.debug(f"No values found above max threshold {transformation.max}")

    if transformation.min is not None:
        min_outliers = (context.df[col.id] < transformation.min).sum()
        if min_outliers > 0:
            logger.info(f"Capping {min_outliers} values below {transformation.min}")
            context.df.loc[context.df[col.id] < transformation.min, col.id] = transformation.min
        else:
            logger.debug(f"No values found below min threshold {transformation.min}")

    final_stats = context.df[col.id].describe()
    logger.debug(
        f"Final column statistics: min={final_stats['min']}, max={final_stats['max']}, mean={final_stats['mean']:.2f}"
    )
    logger.info(f"Completed outlier processing for column '{col.id}'")


def process_removal(context, col: Column, transformation: RemovalTransformation) -> None:
    logger.info(f"Removing column '{col.id}' (reason: {transformation.removal_reason})")

    if col.id not in context.df.columns:
        logger.warning(f"Column '{col.id}' not found in dataframe, skipping removal")
        return

    original_shape = context.df.shape
    context.df.drop(columns=[col.id], inplace=True)
    new_shape = context.df.shape

    logger.info(f"Successfully removed column '{col.id}' - dataframe shape: {original_shape} -> {new_shape}")

    if transformation.removal_redundant_col:
        logger.debug(f"Redundant column: {transformation.removal_redundant_col}")
    if transformation.removal_distribution:
        logger.debug(f"Distribution reason: {transformation.removal_distribution}")


def process_merge(context, col: Column, transformation: MergedTransformation) -> None:
    logger.info(f"Processing merge for column '{col.id}' using method '{transformation.merging_type}'")
    resolved_sources = [context._resolve_ref(src) for src in transformation.merged_from]
    logger.debug(f"Merging from columns: {transformation.merged_from} (resolved: {resolved_sources})")

    missing_cols = [source for source in resolved_sources if source not in context.df.columns]
    if missing_cols:
        logger.error(f"Missing source columns for merge: {missing_cols}")
        raise ValueError(f"Source columns not found: {missing_cols}")

    original_shape = context.df.shape

    match transformation.merging_type:
        case "mean":
            logger.debug(f"Computing mean across {len(transformation.merged_from)} columns")
            for src_col in resolved_sources:
                na_count = context.df[src_col].isna().sum()
                if na_count > 0:
                    logger.debug(f"Source column '{src_col}' has {na_count} NA values")

            context.df[col.id] = context.df[resolved_sources].mean(axis=1)
            context.df[col.id] = context.df[col.id].astype("float")

            result_stats = context.df[col.id].describe()
            logger.info(
                f"Mean merge completed - result mean: {result_stats['mean']:.2f}, std: {result_stats['std']:.2f}"
            )

        case "missing":
            logger.debug(f"Filling missing values using first-available from {len(transformation.merged_from)} columns")

            context.df[col.id] = context.df[resolved_sources[0]]
            initial_na_count = context.df[col.id].isna().sum()
            logger.debug(f"Starting with {initial_na_count} NA values from '{resolved_sources[0]}'")

            for merged_col in resolved_sources[1:]:
                before_na = context.df[col.id].isna().sum()
                context.df[col.id] = context.df[col.id].combine_first(context.df[merged_col])
                after_na = context.df[col.id].isna().sum()
                filled_count = before_na - after_na

                if filled_count > 0:
                    logger.debug(f"Filled {filled_count} missing values using column '{merged_col}'")

            final_na_count = context.df[col.id].isna().sum()
            logger.info(
                f"Missing value merge completed - filled {initial_na_count - final_na_count} values, {final_na_count} still missing"
            )

        case "sum":
            logger.debug(f"Computing sum across {len(transformation.merged_from)} columns")
            for src_col in resolved_sources:
                na_count = context.df[src_col].isna().sum()
                if na_count > 0:
                    logger.debug(f"Source column '{src_col}' has {na_count} NA values")

            context.df[col.id] = context.df[resolved_sources].sum(axis=1)
            context.df[col.id] = context.df[col.id].astype("float")

            result_stats = context.df[col.id].describe()
            logger.info(
                f"Sum merge completed - result mean: {result_stats['mean']:.2f}, max: {result_stats['max']:.2f}"
            )

        case "concat":
            logger.debug(
                f"Concatenating {len(transformation.merged_from)} columns with format: {transformation.concat_string}"
            )

            try:
                context.df[col.id] = context.df.apply(
                    lambda row: transformation.concat_string.format(
                        **{src_ref: row[context._resolve_ref(src_ref)] for src_ref in transformation.merged_from}
                    ),
                    axis=1,
                )

                sample_results = context.df[col.id].head(3).tolist()
                logger.debug(f"Sample concatenation results: {sample_results}")
                logger.info("Concatenation merge completed")

            except Exception as e:
                logger.error(f"Error during concatenation merge: {e}")
                raise

    new_shape = context.df.shape
    logger.info(f"Merge processing completed for column '{col.id}' - dataframe shape: {original_shape} -> {new_shape}")


def process_split(context, col: Column, transformation: SplittedTransformation) -> None:
    resolved_source = context._resolve_ref(transformation.split_from)
    logger.info(
        f"Processing split for column '{col.id}' from source '{transformation.split_from}' (resolved: '{resolved_source}')"
    )
    logger.debug(f"Split values: {transformation.values}")

    if resolved_source not in context.df.columns:
        logger.error(f"Source column '{resolved_source}' not found for split operation")
        raise ValueError(f"Source column '{transformation.split_from}' not found")

    context.df[col.id] = ""
    logger.debug(f"Initialized target column '{col.id}' with empty strings")

    source_series = context.df[resolved_source].astype("string").fillna("")

    for value in transformation.values:
        logger.debug(f"Processing split value: '{value}'")
        matches = source_series.str.contains(value, regex=False, na=False)
        match_count = matches.sum()
        logger.debug(f"Found {match_count} matches for value '{value}'")
        context.df.loc[matches, col.id] += value

    unique_combinations = context.df[col.id].value_counts()
    logger.info(f"Split processing completed for column '{col.id}': {len(unique_combinations)} unique combinations")
    logger.debug(f"Value distribution: {unique_combinations.head().to_dict()}")


def process_scaling(context, col: Column, transformation: ScalingTransformation) -> None:
    logger.info(f"Processing scaling for column '{col.id}' using method: {transformation.scaling_type}")

    if col.id not in context.df.columns:
        logger.warning(f"Column '{col.id}' not found in dataframe, skipping scaling")
        return

    original_stats = context.df[col.id].describe()
    logger.debug(
        f"Original statistics - mean: {original_stats['mean']:.3f}, std: {original_stats['std']:.3f}, min: {original_stats['min']:.3f}, max: {original_stats['max']:.3f}"
    )

    missing_count = context.df[col.id].isnull().sum()
    if missing_count > 0:
        logger.warning(f"Column '{col.id}' has {missing_count} missing values before scaling")

    if transformation.scaling_type == "robust":
        scaler = RobustScaler()
        logger.debug("Using RobustScaler")
    elif transformation.scaling_type == "minmax":
        scaler = MinMaxScaler()
        logger.debug("Using MinMaxScaler")
    else:
        logger.error(f"Unknown scaling type: {transformation.scaling_type}")
        raise ValueError(f"Unknown scaling type: {transformation.scaling_type}")

    try:
        original_values = context.df[col.id].values.reshape(-1, 1)
        scaled_values = scaler.fit_transform(original_values)
        context.df[col.id] = scaled_values.flatten()

        scaled_stats = context.df[col.id].describe()
        logger.info(
            f"Scaling completed - new mean: {scaled_stats['mean']:.3f}, std: {scaled_stats['std']:.3f}, min: {scaled_stats['min']:.3f}, max: {scaled_stats['max']:.3f}"
        )

    except Exception as e:
        logger.error(f"Error during scaling of column '{col.id}': {e}")
        raise

    logger.info(f"Scaling processing completed for column '{col.id}'")


def process_binning(context, col: Column, transformation: BinningTransformation) -> None:
    logger.info(f"Processing binning for column '{col.id}' using bins: {transformation.bins}")

    if col.id not in context.df.columns:
        logger.warning(f"Column '{col.id}' not found in dataframe, skipping binning")
        return

    if transformation.bins not in context.model.bins:
        logger.error(f"Bins configuration '{transformation.bins}' not found in model")
        raise ValueError(f"Bins configuration '{transformation.bins}' not found")

    bins = context.model.bins[transformation.bins]
    logger.debug(f"Using {len(bins)} bin definitions")

    original_stats = context.df[col.id].describe()
    logger.debug(f"Original column range: {original_stats['min']:.3f} to {original_stats['max']:.3f}")

    minimums: list[float] = []
    labels: list[int | float | str] = []
    for i, bin_def in enumerate(bins):
        minimums.append(bin_def.min)
        labels.append(bin_def.value)
        logger.debug(f"Bin {i + 1}: min={bin_def.min}, label={bin_def.value}")

    minimums.append(float("inf"))
    logger.debug(f"Bin boundaries: {minimums}")

    min_value = context.df[col.id].min()
    if min_value < minimums[0]:
        logger.warning(f"Some values ({min_value}) are below the lowest bin boundary ({minimums[0]})")

    try:
        context.df[col.id] = pd.cut(context.df[col.id], bins=minimums, labels=labels, include_lowest=True)

        bin_counts = context.df[col.id].value_counts()
        logger.info(f"Binning completed - distribution: {bin_counts.to_dict()}")

        nan_count = context.df[col.id].isnull().sum()
        if nan_count > 0:
            logger.warning(f"Binning resulted in {nan_count} NaN values")

    except Exception as e:
        logger.error(f"Error during binning of column '{col.id}': {e}")
        raise

    logger.info(f"Binning processing completed for column '{col.id}'")


def process_column_substitution(context, col: Column, transformation: ColumnSubstitutionTransformation) -> None:
    """Replace a column with data from an external file."""
    logger.info(f"Processing column substitution for column '{col.id}' from file: {transformation.df_file}")

    if not os.path.exists(transformation.df_file):
        logger.error(f"Substitution file not found: {transformation.df_file}")
        raise FileNotFoundError(f"Substitution file not found: {transformation.df_file}")

    try:
        logger.debug(f"Loading substitution data from: {transformation.df_file}")
        sub_df = pd.read_csv(transformation.df_file)
        logger.info(f"Loaded substitution dataframe with shape: {sub_df.shape}")

        if col.id not in sub_df.columns:
            logger.error(f"Column '{col.id}' not found in substitution file columns: {list(sub_df.columns)}")
            raise ValueError(f"Column {col.id} not found in substitution file {transformation.df_file}")

        if "index" in sub_df.columns or sub_df.index.name == "index":
            logger.debug("Setting 'index' column as dataframe index")
            sub_df = sub_df.set_index("index")

        if len(sub_df) != len(context.df):
            logger.warning(
                f"Substitution dataframe length ({len(sub_df)}) doesn't match the original dataframe ({len(context.df)}). Attempting alignment by index."
            )
            warn(
                f"Substitution dataframe length ({len(sub_df)}) doesn't match the original dataframe ({len(context.df)}). Attempting alignment by index.",
                stacklevel=2,
            )

        sub_col_stats = (
            sub_df[col.id].describe() if sub_df[col.id].dtype in ["int64", "float64"] else sub_df[col.id].value_counts()
        )
        logger.debug(f"Substitution column statistics: {sub_col_stats}")

        original_dtype = context.df[col.id].dtype if col.id in context.df.columns else None
        context.df[col.id] = sub_df[col.id]
        new_dtype = context.df[col.id].dtype

        logger.info(f"Column substitution completed - dtype changed from {original_dtype} to {new_dtype}")

        missing_count = context.df[col.id].isnull().sum()
        if missing_count > 0:
            logger.warning(f"Substituted column '{col.id}' has {missing_count} missing values")

    except Exception as e:
        logger.error(f"Error during column substitution for '{col.id}': {e}")
        raise

    logger.info(f"Column substitution processing completed for column '{col.id}'")


def process_renaming(context, col: Column, transformation: RenamingTransformation) -> None:
    """Rename the column in place to its canonical alias."""
    rename_targets = transformation.rename_to or []

    if not rename_targets:
        logger.debug(f"No rename targets declared for column '{col.id}', skipping in-place rename")
        return

    if len(rename_targets) != 1:
        logger.error(
            "Renaming for column '{col}' expects exactly one target, received {targets}",
            col=col.id,
            targets=rename_targets,
        )
        raise ValueError(f"Renaming column '{col.id}' requires exactly one target")

    target = rename_targets[0]

    if col.id not in context.df.columns:
        if target in context.df.columns:
            logger.debug(f"Column '{col.id}' already renamed to '{target}', skipping")
            return
        logger.warning(f"Column '{col.id}' not present during rename; leaving unchanged")
        return

    if target in context.df.columns and target != col.id:
        logger.warning(
            "Target column '{target}' already exists before renaming '{col}', overwriting",
            target=target,
            col=col.id,
        )

    context.df.rename(columns={col.id: target}, inplace=True)
    logger.info("Renamed column '{col}' -> '{target}'", col=col.id, target=target)


def process_renamed(context, col: Column, transformation: RenamedTransformation) -> None:
    """Materialize an analysis-friendly alias column."""
    src = transformation.rename_from

    if col.id in context.df.columns:
        logger.debug("Alias column '{}' already exists; skipping copy from '{}'", col.id, src)
        return

    if src not in context.df.columns:
        logger.warning("Source column '{}' not found; cannot materialize alias '{}'", src, col.id)
        return

    context.df[col.id] = context.df[src]
    logger.info("Materialized alias column '{}' from '{}'", col.id, src)
