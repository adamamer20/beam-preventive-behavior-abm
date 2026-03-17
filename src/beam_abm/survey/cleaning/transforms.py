from __future__ import annotations

import numpy as np
import pandas as pd

from beam_abm.common.logging import get_logger

logger = get_logger(__name__)


def knowledge_illness(column: pd.Series) -> pd.Series:
    logger.info(f"Starting knowledge_illness transformation for column with {len(column)} rows")
    logger.debug(f"Column name: {column.name}, dtype: {column.dtype}")

    new_column = pd.Series(0, index=column.index)
    correct_answers = [
        "Flu",
        "COVID-19",
        "Measles",
        "Chickenpox",
        "Hepatitis A",
        "Hepatitis B",
        "Meningitis",
    ]

    logger.debug(f"Checking for correct answers: {correct_answers}")

    for correct_answer in correct_answers:
        matches = column.str.contains(correct_answer)
        match_count = matches.sum()
        logger.debug(f"Found {match_count} matches for '{correct_answer}'")
        new_column[matches] += 1

    final_score_dist = new_column.value_counts().sort_index()
    logger.info(f"Knowledge illness scores distribution: {final_score_dist.to_dict()}")
    logger.info("Completed knowledge_illness transformation")
    return new_column


def knowledge_vax(column: pd.Series) -> pd.Series:
    logger.info(f"Starting knowledge_vax transformation for column with {len(column)} rows")
    logger.debug(f"Column name: {column.name}, dtype: {column.dtype}")

    new_column = pd.Series(0, index=column.index)
    correct_answers = [
        "Herd immunity helps reduce the spread of viruses",
        "In the past, vaccines have eliminated serious diseases",
    ]

    logger.debug(f"Checking for correct answers: {correct_answers}")

    for correct_answer in correct_answers:
        matches = column.str.contains(correct_answer)
        match_count = matches.sum()
        logger.debug(f"Found {match_count} matches for '{correct_answer}'")
        new_column[matches] += 1

    final_score_dist = new_column.value_counts().sort_index()
    logger.info(f"Knowledge vax scores distribution: {final_score_dist.to_dict()}")
    logger.info("Completed knowledge_vax transformation")
    return new_column


def reverse_1_to_7(column: pd.Series) -> pd.Series:
    """Reverse a 1–7 Likert scale: x -> 8 - x."""
    logger.info(f"Reversing 1–7 scale for column '{column.name}'")
    col = pd.to_numeric(column, errors="coerce")
    return 8 - col


def reverse_1_to_6(column: pd.Series) -> pd.Series:
    """Reverse a 1–6 Likert scale: x -> 7 - x."""
    logger.info(f"Reversing 1–6 scale for column '{column.name}'")
    col = pd.to_numeric(column, errors="coerce")
    return 7 - col


def reverse_1_to_5(column: pd.Series) -> pd.Series:
    """Reverse a 1–5 Likert scale: x -> 6 - x."""
    logger.info(f"Reversing 1–5 scale for column '{column.name}'")
    col = pd.to_numeric(column, errors="coerce")
    return 6 - col


def covax_started_any(column: pd.Series) -> pd.Series:
    """Binary ever-vaccinated flag derived from covax_num (1 if >0, else 0, preserve NaN)."""

    numeric = pd.to_numeric(column, errors="coerce")
    return pd.Series(
        np.where(numeric.isna(), np.nan, (numeric > 0).astype(float)), index=column.index, name=column.name
    )
