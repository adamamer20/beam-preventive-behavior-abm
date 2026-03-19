from __future__ import annotations

from warnings import warn

import pandas as pd

from beam_abm.common.logging import get_logger

from . import derivations as derived_transforms
from . import ppp as ppp_transforms
from . import transforms as custom_transforms
from .imputation import compute_imputation_by_prediction
from .models import Column, EncodingTransformation, ImputationTransformation, RenamingTransformation

logger = get_logger(__name__)


def _lookup_custom_encoding_function(name: str):
    for module in (custom_transforms, ppp_transforms, derived_transforms):
        func = getattr(module, name, None)
        if callable(func):
            return func
    return None


def process_encoding(context, col: Column, transformation: EncodingTransformation) -> None:
    logger.info(f"Processing encoding for column '{col.id}' with encoding type: {transformation.encoding}")

    if col.id not in context.df.columns:
        logger.warning(f"Column '{col.id}' not found in dataframe, skipping encoding")
        return

    encoding = transformation.encoding
    original_dtype = context.df[col.id].dtype
    logger.debug(f"Original column dtype: {original_dtype}")

    if isinstance(encoding, str) and encoding.startswith("$"):
        logger.debug(f"Processing special encoding: {encoding}")

        if encoding == "$ONE-HOT":
            logger.info(f"Applying one-hot encoding to column '{col.id}'")
            unique_vals = context.df[col.id].nunique()
            logger.debug(f"Column has {unique_vals} unique values")

            one_hot_encoded = pd.get_dummies(context.df[col.id], prefix=col.id)
            logger.info(f"Created {len(one_hot_encoded.columns)} one-hot encoded columns")

            original_shape = context.df.shape
            context.df = pd.concat([context.df, one_hot_encoded], axis=1)
            context.df.drop(columns=[col.id], inplace=True)
            new_shape = context.df.shape
            logger.info(f"One-hot encoding completed - dataframe shape: {original_shape} -> {new_shape}")

        elif encoding == "$EMBEDDING":
            logger.info(f"Applying embedding encoding to column '{col.id}'")
            embedding_prompt = transformation.embedding_prompt or ""
            logger.debug(f"Embedding prompt (raw): {embedding_prompt[:100] + '...' if embedding_prompt else '<default>'}")

            question_text = col.new_question or col.question
            context.df[col.id] = context.generate_embeddings(
                context.df[col.id],
                embedding_prompt,
                question_text,
            )
            logger.info(f"Embedding encoding completed for column '{col.id}'")

        elif encoding == "$NUM":
            logger.info(f"Extracting numbers from column '{col.id}'")
            context.df[col.id] = context.extract_number(context.df[col.id])
            logger.info(f"Number extraction completed for column '{col.id}'")

        elif encoding == "$COUNTING":
            logger.info(f"Applying counting encoding to column '{col.id}'")
            context.df[col.id] = context.count_answers(col)
            logger.info(f"Counting encoding completed for column '{col.id}'")

        elif encoding == "$DATETIME":
            logger.info(f"Converting column '{col.id}' to datetime with format: {transformation.datetime_format}")
            sample_values = context.df[col.id].dropna().head(5).tolist()
            logger.debug(f"Sample values before datetime conversion: {sample_values}")

            context.df[col.id] = pd.to_datetime(
                context.df[col.id].str.replace(".0", ""),
                errors="coerce",
                format=transformation.datetime_format,
            )

            na_count = context.df[col.id].isna().sum()
            successful_count = len(context.df) - na_count
            logger.info(f"Datetime conversion completed: {successful_count} successful, {na_count} failed")

            if na_count > 0:
                logger.warning(f"Some datetime conversions failed, resulting in {na_count} NA values")

        elif encoding == "$DATETIME_TO_INTERVAL":
            logger.info(f"Converting datetime column '{col.id}' to interval from {transformation.from_timestamp}")
            logger.debug(f"Using unit: {transformation.unit}")

            context.df[col.id] = getattr(
                (
                    context.df[col.id]
                    - pd.to_datetime(
                        transformation.from_timestamp,
                        format=transformation.datetime_format,
                    )
                ).dt,
                transformation.unit,
            )

            interval_stats = context.df[col.id].describe()
            logger.info(
                f"Interval conversion completed - min: {interval_stats['min']}, max: {interval_stats['max']}, mean: {interval_stats['mean']:.2f}"
            )

        elif encoding == "$INTERVAL_TO_DATETIME":
            logger.info(f"Converting interval column '{col.id}' to datetime from {transformation.from_timestamp}")
            logger.debug(f"Using unit: {transformation.unit}")

            context.df[col.id] = (
                pd.to_datetime(
                    transformation.from_timestamp,
                    format=transformation.datetime_format,
                )
                + pd.to_timedelta(context.df[col.id], unit=transformation.unit)
            ).dt.strftime(transformation.datetime_format)

            logger.info(f"Interval to datetime conversion completed for column '{col.id}'")

        else:
            logger.info(f"Applying custom encoding function: {encoding}")
            encoding_fun = _lookup_custom_encoding_function(encoding[1:])
            if not encoding_fun:
                logger.error(f"Encoding function '{encoding}' not found in globals")
                raise ValueError(f"Encoding function '{encoding}' not found")

            try:
                context.df[col.id] = encoding_fun(context.df[col.id], df=context.df, column_meta=col)
            except TypeError:
                context.df[col.id] = encoding_fun(context.df[col.id])

            logger.info(f"Custom encoding function '{encoding}' applied successfully")

        return

    logger.info(f"Processing dictionary/string-based encoding for column '{col.id}'")

    if isinstance(encoding, str):
        if context.model.encodings and encoding in context.model.encodings:
            encoding = context.model.encodings[encoding].copy()
            logger.debug(f"Using predefined encoding '{transformation.encoding}' with {len(encoding)} mappings")
        else:
            logger.warning(f"String encoding '{encoding}' not found in model encodings")
    elif isinstance(encoding, dict):
        encoding = encoding.copy()
        logger.debug(f"Using direct dictionary encoding with {len(encoding)} mappings")
    else:
        logger.error(f"Unknown encoding type: {type(encoding)}")
        raise ValueError(f"Unknown encoding type: {encoding}")

    if transformation.key_type:
        logger.debug(f"Converting encoding keys to type: {transformation.key_type}")

        if transformation.key_type == "int":
            encoding = {int(k): v for k, v in encoding.items()}
        elif transformation.key_type == "float":
            encoding = {float(k): v for k, v in encoding.items()}
        elif transformation.key_type == "bool":
            if "True" in encoding:
                encoding[True] = encoding.pop("True")
            if "False" in encoding:
                encoding[False] = encoding.pop("False")

        logger.debug("Key type conversion completed")

    values_not_in_encoding = context.df[col.id][~context.df[col.id].isin(encoding.keys())].unique()
    if len(values_not_in_encoding) > 0:
        logger.warning(
            f"Found {len(values_not_in_encoding)} values not in encoding: {values_not_in_encoding[:5]}{'...' if len(values_not_in_encoding) > 5 else ''}"
        )

    if "$DEFAULT" in encoding:
        default_value = encoding["$DEFAULT"]
        logger.info(f"Using default value '{default_value}' for {len(values_not_in_encoding)} unmapped values")

        for value in values_not_in_encoding:
            encoding[value] = default_value
        encoding.pop("$DEFAULT")
    else:
        logger.debug(f"No default value specified, keeping {len(values_not_in_encoding)} values unchanged")
        for value in values_not_in_encoding:
            encoding[value] = value

    logger.debug(f"Applying encoding mapping with {len(encoding)} entries")
    context.df[col.id] = context.df[col.id].map(encoding)

    unique_values = context.df[col.id].unique().tolist()
    logger.debug(f"After encoding, column has {len(unique_values)} unique values")

    if not set(unique_values).issubset({True, False, None}):
        try:
            logger.debug("Attempting to convert encoded column to numeric")
            context.df[col.id] = pd.to_numeric(context.df[col.id])

            if (context.df[col.id] % 1 == 0).all():
                context.df[col.id] = context.df[col.id].astype(int)
                logger.debug("Converted column to integer type")
            else:
                logger.debug("Column kept as float type")

            logger.info(f"Successfully converted encoded column '{col.id}' to numeric")

        except ValueError as e:
            logger.warning(f"Column {col.id} could not be converted to numeric after encoding: {e}")
            warn(
                f"Column {col.id} could not be converted to bool or numeric after encoding {encoding}. Make sure the column will be converted to numeric.",
                stacklevel=2,
            )
    else:
        logger.debug("Column contains only boolean values, skipping numeric conversion")

    logger.info(f"Encoding processing completed for column '{col.id}'")


def process_imputation(context, col: Column, transformation: ImputationTransformation) -> None:
    logger.info(f"Processing imputation for column '{col.id}' with value: {transformation.imputation_value}")

    if col.id not in context.df.columns:
        logger.warning(f"Column '{col.id}' not found in dataframe, skipping imputation")
        return

    missing_before = context.df[col.id].isnull().sum()
    total_values = len(context.df[col.id])
    logger.info(
        f"Column '{col.id}' has {missing_before}/{total_values} missing values ({missing_before / total_values * 100:.1f}%)"
    )

    if missing_before == 0:
        logger.info(f"No missing values found in column '{col.id}', skipping imputation")
        return

    missing_mask = context.df[col.id].isnull()

    if isinstance(transformation.imputation_value, str) and transformation.imputation_value.startswith("$"):
        logger.debug(f"Processing special imputation value: {transformation.imputation_value}")

        if transformation.imputation_value == "$PREDICTION":
            logger.info(f"Using prediction-based imputation for column '{col.id}'")
            context.imputed_values[col.id] = context.df[col.id].isnull()
            compute_imputation_by_prediction(context, col, logger=logger)

        elif transformation.imputation_value == "$MAX":
            max_value = context.df[col.id].max()
            imputation_value = max_value * 1.25
            logger.info(f"Using max-based imputation: max={max_value}, imputation_value={imputation_value}")

            context.df[col.id] = context.df[col.id].fillna(imputation_value)

        elif transformation.imputation_value.lstrip("$"):
            source_col_raw = transformation.imputation_value.lstrip("$")
            source_col = context._resolve_ref(source_col_raw)
            if source_col not in context.df.columns:
                logger.error(
                    "Imputation reference '{raw}' resolved to '{resolved}', but column not found",
                    raw=source_col_raw,
                    resolved=source_col,
                )
                raise ValueError(
                    f"Imputation source column '{source_col_raw}' (resolved to '{source_col}') not found"
                )
            logger.info(f"Using combine_first imputation from column '{source_col}'")

            source_missing = context.df[source_col].isnull().sum()
            logger.debug(f"Source column '{source_col}' has {source_missing} missing values")

            context.df[col.id] = context.df[col.id].combine_first(context.df[source_col])

        else:
            logger.error(f"Unknown special imputation value: {transformation.imputation_value}")
            raise ValueError(f"Unknown special imputation value: {transformation.imputation_value}")
    else:
        logger.info(f"Using fixed value imputation: {transformation.imputation_value}")
        context.df[col.id] = context.df[col.id].fillna(transformation.imputation_value)

    missing_after = context.df[col.id].isnull().sum()
    filled_count = missing_before - missing_after

    logger.info(
        f"Imputation completed for column '{col.id}': filled {filled_count} values, {missing_after} still missing"
    )

    if missing_after > 0:
        logger.warning(f"Column '{col.id}' still has {missing_after} missing values after imputation")

    if context.df[col.id].dtype in ["int64", "float64"]:
        stats = context.df[col.id].describe()
        logger.debug(f"Post-imputation statistics for '{col.id}': mean={stats['mean']:.3f}, std={stats['std']:.3f}")

    if filled_count > 0:
        alias_target = None
        if col.transformations:
            for transform in col.transformations:
                if isinstance(transform, RenamingTransformation):
                    if transform.rename_to and len(transform.rename_to) == 1:
                        alias_target = transform.rename_to[0]
                    else:
                        logger.debug(
                            "Renaming for column '{col}' has %d targets; using current id for misflag",
                            len(transform.rename_to) if transform.rename_to else 0,
                            col=col.id,
                        )
                    break

        flag_base = alias_target or col.id
        flag_col = f"{flag_base}_misflag"
        logger.info(f"Updating missingness flag column: '{flag_col}'")
        if flag_col not in context.df.columns:
            context.df[flag_col] = 0
        context.df.loc[missing_mask, flag_col] = 1
