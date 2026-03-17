"""Analysis utilities for inferential statistics."""

from __future__ import annotations

from .inferential import (
    HighlightCriteria,
    InferentialResult,
    compute_binary_numeric_tests,
    compute_categorical_associations,
    compute_inferential_highlights,
    compute_inferential_suite,
    compute_nominal_numeric_tests,
    compute_spearman_correlations,
    generate_by_block_view,
    generate_by_outcome_view,
    generate_by_section_view,
    generate_predictor_predictor_views,
    split_inferential_by_hypothesis,
    split_inferential_by_section,
    to_canonical_raw_format,
)
from .modeling import ModelBlock, ModelResult, ModelSpec, run_model_suite, standardize_columns

__all__ = [
    "HighlightCriteria",
    "InferentialResult",
    "compute_spearman_correlations",
    "compute_categorical_associations",
    "compute_binary_numeric_tests",
    "compute_inferential_highlights",
    "split_inferential_by_section",
    "split_inferential_by_hypothesis",
    "compute_nominal_numeric_tests",
    "compute_inferential_suite",
    "to_canonical_raw_format",
    "generate_by_outcome_view",
    "generate_by_block_view",
    "generate_by_section_view",
    "generate_predictor_predictor_views",
    "ModelBlock",
    "ModelSpec",
    "ModelResult",
    "run_model_suite",
    "standardize_columns",
]
