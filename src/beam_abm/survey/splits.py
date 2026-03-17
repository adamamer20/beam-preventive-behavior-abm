from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

from beam_abm.common.logging import get_logger

logger = get_logger(__name__)


def get_balanced_dataset_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Split dataframe into balanced train/val/test partitions using HDBSCAN."""
    logger.info(
        f"Creating balanced dataset splits with ratios - train: {train_ratio}, val: {val_ratio}, test: {test_ratio}"
    )
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) >= 1e-10:
        logger.error(f"Split ratios must sum to 1, got {ratio_sum}")
        raise AssertionError("Split ratios must sum to 1")

    np.random.seed(random_seed)
    df_result = df.copy()
    df_result["split"] = ""

    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) == 0:
        logger.warning("No numeric columns found for clustering. Using random assignment.")
        df_result["split"] = np.random.choice(
            ["train", "val", "test"],
            size=len(df),
            p=[train_ratio, val_ratio, test_ratio],
        )
        return df_result

    clustering_df = df[numeric_cols].fillna(df[numeric_cols].mean())
    clusterer = HDBSCAN(min_cluster_size=5, min_samples=1)

    try:
        cluster_labels = clusterer.fit_predict(clustering_df)
        for label in np.unique(cluster_labels):
            cluster_indices = np.where(cluster_labels == label)[0]
            np.random.shuffle(cluster_indices)
            n_samples = len(cluster_indices)
            train_end = int(n_samples * train_ratio)
            val_end = train_end + int(n_samples * val_ratio)

            df_result.iloc[cluster_indices[:train_end], df_result.columns.get_loc("split")] = "train"
            df_result.iloc[cluster_indices[train_end:val_end], df_result.columns.get_loc("split")] = "val"
            df_result.iloc[cluster_indices[val_end:], df_result.columns.get_loc("split")] = "test"

        unassigned = df_result[df_result["split"] == ""]
        if not unassigned.empty:
            unassigned_indices = unassigned.index
            df_result.loc[unassigned_indices, "split"] = np.random.choice(
                ["train", "val", "test"],
                size=len(unassigned_indices),
                p=[train_ratio, val_ratio, test_ratio],
            )

        return df_result
    except Exception as error:
        logger.error(f"Error during HDBSCAN clustering: {error}")
        logger.warning("Falling back to random assignment")
        df_result["split"] = np.random.choice(
            ["train", "val", "test"],
            size=len(df),
            p=[train_ratio, val_ratio, test_ratio],
        )
        return df_result
