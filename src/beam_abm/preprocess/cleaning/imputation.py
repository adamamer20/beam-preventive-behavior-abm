from __future__ import annotations

import os
from collections.abc import Collection

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


def compute_imputation_by_prediction(processor, col, *, logger) -> None:
    logger.info(f"Computing imputation by prediction for column '{col.id}'")

    # Count values to be imputed
    values_to_impute = processor.imputed_values[col.id].sum()
    total_values = len(processor.imputed_values[col.id])
    logger.info(f"Imputing {values_to_impute}/{total_values} values ({values_to_impute / total_values * 100:.1f}%)")

    # Create a copy of the dataframe for encoding
    logger.debug("Creating encoded copy of dataframe for prediction")
    df_encoded = processor.df.copy()

    # Handle datetime columns by converting to numeric timestamps
    datetime_cols = df_encoded.select_dtypes(include=["datetime64"]).columns.tolist()
    datetime_col_mapping = {}

    if datetime_cols:
        logger.debug(f"Found {len(datetime_cols)} datetime columns: {datetime_cols}")

        for dt_col in datetime_cols:
            # Store original column name to restore later
            datetime_col_mapping[f"{dt_col}_timestamp"] = dt_col
            # Convert to numeric timestamp (seconds since epoch)
            df_encoded[f"{dt_col}_timestamp"] = df_encoded[dt_col].astype(np.int64) // 10**9
            # Drop original datetime column
            df_encoded.drop(columns=[dt_col], inplace=True)
            logger.debug(f"Converted datetime column '{dt_col}' to timestamp")

    # Explode list type columns horizontally for prediction
    list_cols = [
        c
        for c in df_encoded.columns
        if isinstance(df_encoded[c].iloc[0], Collection) and not isinstance(df_encoded[c].iloc[0], str)
    ]

    if list_cols:
        logger.debug(f"Found {len(list_cols)} list-type columns: {list_cols}")

        for col_id in list_cols:
            logger.debug(f"Exploding list column '{col_id}'")
            embedding_df = df_encoded[col_id].apply(pd.Series).add_prefix(f"{col_id}_")
            logger.debug(f"Created {len(embedding_df.columns)} exploded columns for '{col_id}'")

            df_encoded = pd.concat([df_encoded, embedding_df], axis=1)
            df_encoded.drop(columns=[col_id], inplace=True)

    # Track encoded target columns and their encoders for later inverse transform
    target_encoders = {}

    # Identify target column(s)
    all_cols = df_encoded.columns.tolist()
    logger.debug(f"Total columns after preprocessing: {len(all_cols)}")

    # Check if target is a datetime column that was converted
    target_is_datetime = False
    target_col = col.id
    if col.id in datetime_cols:
        target_col = f"{col.id}_timestamp"
        target_is_datetime = True
        logger.debug(f"Target column '{col.id}' is datetime, using timestamp version")

    y_cols = [
        c for c in all_cols if c == target_col or (c.startswith(f"{target_col}_") and c.partition("_")[0] in list_cols)
    ]
    X_cols = [c for c in all_cols if c not in y_cols]

    logger.info(f"Identified {len(y_cols)} target columns and {len(X_cols)} feature columns")
    logger.debug(f"Target columns: {y_cols[:5]}{'...' if len(y_cols) > 5 else ''}")

    # One-hot encode categorical X columns
    X_categorical = df_encoded[X_cols].select_dtypes(exclude=["number"]).columns.tolist()
    if X_categorical:
        logger.info(f"One-hot encoding {len(X_categorical)} categorical feature columns")

        # Fill NA values before encoding
        for cat_col in X_categorical:
            if df_encoded[cat_col].isnull().any():
                na_count = df_encoded[cat_col].isnull().sum()
                logger.debug(f"Filling {na_count} NA values in categorical column '{cat_col}'")
                df_encoded[cat_col] = df_encoded[cat_col].fillna("missing")

        # Apply one-hot encoding
        X_encoded = pd.get_dummies(df_encoded[X_categorical], drop_first=False)
        logger.debug(f"Created {len(X_encoded.columns)} one-hot encoded columns")

        # Drop original categorical columns and add one-hot encoded columns
        df_encoded = df_encoded.drop(columns=X_categorical)
        df_encoded = pd.concat([df_encoded, X_encoded], axis=1)
        logger.debug(f"Updated dataframe shape after one-hot encoding: {df_encoded.shape}")

    # Update X_cols to include one-hot encoded columns
    X_cols = [c for c in df_encoded.columns if c not in y_cols]
    logger.info(f"Final feature count: {len(X_cols)} columns")

    if X_cols:
        train_mask = ~processor.imputed_values[col.id]
        X_train = df_encoded.loc[train_mask, X_cols]
        y_train = df_encoded.loc[train_mask, y_cols]
        X_pred = df_encoded.loc[processor.imputed_values[col.id], X_cols]
        X_train_matrix = X_train.to_numpy()
        X_pred_matrix = X_pred.to_numpy()

        logger.info(f"Training set size: {len(X_train)}, Prediction set size: {len(X_pred)}")

        if len(y_cols) == 1:  # Single target column case
            logger.debug("Processing single target column case")

            # Determine if classification or regression task
            y_train_series = y_train.iloc[:, 0]
            unique_values = y_train_series.dropna().unique()
            is_classification = len(unique_values) <= 10
            is_ordinal = False

            logger.debug(f"Found {len(unique_values)} unique target values")
            logger.debug(f"Initial classification decision: {is_classification}")

            # Check if values are ordinal (integers in sequence)
            if is_classification:
                sorted_values = sorted(unique_values)
                if all(
                    isinstance(val, int | np.integer) or (isinstance(val, float) and val.is_integer())
                    for val in sorted_values
                ) and not all(isinstance(val, bool) for val in sorted_values):
                    int_values = [int(val) for val in sorted_values]
                    if int_values == list(range(min(int_values), max(int_values) + 1)):
                        is_ordinal = True
                        is_classification = False  # Treat as regression with EMD loss
                        logger.debug("Detected ordinal data, switching to regression approach")

            if is_ordinal:
                logger.info(f"Using ordinal regression for target column '{y_cols[0]}'")

                # For ordinal data, use a regression approach with special handling
                model = xgb.XGBRegressor(
                    random_state=os.environ.get("SEED", 42),
                )

                logger.debug("Training XGBoost regressor for ordinal data")
                model.fit(X_train_matrix, y_train_series.to_numpy())
                predictions = model.predict(X_pred_matrix)

                # Round predictions to nearest integer for ordinal data
                predictions = np.round(predictions).astype(int)

                # Ensure predictions stay within the valid range
                min_val = int(min(unique_values))
                max_val = int(max(unique_values))
                predictions = np.clip(predictions, min_val, max_val)

                logger.info(
                    f"Ordinal predictions: range {predictions.min()}-{predictions.max()}, clipped to {min_val}-{max_val}"
                )

            elif is_classification:
                logger.info(f"Using classification for target column '{y_cols[0]}'")

                target_encoders[y_cols[0]] = LabelEncoder()
                y_train_encoded = target_encoders[y_cols[0]].fit_transform(y_train_series)

                logger.debug(f"Encoded {len(unique_values)} unique classes")

                model = xgb.XGBClassifier(random_state=os.environ.get("SEED", 42))
                model.fit(X_train_matrix, y_train_encoded)
                predictions = model.predict(X_pred_matrix)

                # Apply inverse transform if needed
                if y_cols[0] in target_encoders:
                    predictions = target_encoders[y_cols[0]].inverse_transform(predictions.astype(int))

                logger.info("Classification predictions completed")

            else:
                logger.info(f"Using regression for target column '{y_cols[0]}'")

                model = xgb.XGBRegressor(random_state=os.environ.get("SEED", 42))
                model.fit(X_train_matrix, y_train_series.to_numpy())
                predictions = model.predict(X_pred_matrix)

                pred_stats = pd.Series(predictions).describe()
                logger.info(f"Regression predictions: mean={pred_stats['mean']:.3f}, std={pred_stats['std']:.3f}")

            # Update the original dataframe
            y_col = y_cols[0]
            if target_is_datetime:
                logger.debug("Converting timestamp predictions back to datetime")
                predictions = pd.to_datetime(predictions, unit="s")
                processor.df.loc[processor.imputed_values[col.id], col.id] = predictions
            else:
                processor.df.loc[processor.imputed_values[col.id], y_col] = predictions

            logger.info(f"Updated {len(predictions)} imputed values for column '{col.id}'")

        else:
            logger.info(f"Processing multi-column target case with {len(y_cols)} targets")

            # For multi-column targets (like embeddings), use a regressor
            model = xgb.XGBRegressor(random_state=os.environ.get("SEED", 42))
            model.fit(X_train_matrix, y_train.to_numpy())
            predictions = model.predict(X_pred_matrix)

            logger.debug(f"Multi-target predictions shape: {predictions.shape}")

            # Update each target column with its prediction
            for i, y_col in enumerate(y_cols):
                pred_vals = predictions[:, i]

                # Inverse transform if needed
                if y_col in target_encoders:
                    pred_vals = target_encoders[y_col].inverse_transform(pred_vals.astype(int))

                # Map predictions back to appropriate columns
                processor.df.loc[processor.imputed_values[col.id], y_col] = pred_vals
                logger.debug(f"Updated column '{y_col}' with {len(pred_vals)} predictions")

            logger.info(f"Multi-target imputation completed for column '{col.id}'")
    else:
        logger.warning(f"No feature columns available for prediction-based imputation of column '{col.id}'")

    logger.info(f"Prediction-based imputation completed for column '{col.id}'")


def run_final_imputation_pass(processor, *, logger) -> None:
    logger.info("Starting final imputation pass")

    if processor.imputed_values.empty:
        logger.info("No imputed values to process in final pass")
        return

    columns_to_process = [
        col_id for col_id in processor.imputed_values.columns if processor.imputed_values[col_id].any()
    ]
    logger.info(f"Final imputation pass will process {len(columns_to_process)} columns")
    if not columns_to_process:
        return

    all_columns = processor.model.old_columns + (processor.model.new_columns if processor.model.new_columns else [])

    for col_id in columns_to_process:
        logger.debug(f"Processing final imputation for column '{col_id}'")
        col = next((c for c in all_columns if c.id == col_id), None)
        if col is None:
            logger.warning(f"Column '{col_id}' not found in model columns, skipping final imputation")
            continue

        values_to_reimpute = int(processor.imputed_values[col_id].sum())
        logger.debug(f"Re-imputing {values_to_reimpute} values for column '{col_id}'")
        processor._compute_imputation_by_prediction(col)

    logger.info("Final imputation pass completed")
