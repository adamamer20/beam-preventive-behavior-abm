from __future__ import annotations

from collections.abc import Callable
from typing import Any

from beam_abm.common.logging import get_logger

from .handlers_encoding import process_encoding, process_imputation
from .handlers_transforms import (
    process_binning,
    process_column_substitution,
    process_merge,
    process_outliers,
    process_removal,
    process_renamed,
    process_renaming,
    process_scaling,
    process_split,
)
from .models import (
    BinningTransformation,
    Column,
    ColumnSubstitutionTransformation,
    EncodingTransformation,
    ImputationTransformation,
    MergedTransformation,
    MergingTransformation,
    OutliersTransformation,
    RemovalTransformation,
    RenamedTransformation,
    RenamingTransformation,
    ScalingTransformation,
    SplittedTransformation,
    SplittingTransformation,
    Transformation,
)

logger = get_logger(__name__)

TransformationHandler = Callable[[Any, Column, Transformation], None]


def _noop_marker_handler(context: Any, column: Column, transformation: Transformation) -> None:
    del context, column
    marker = transformation.type
    logger.debug("Skipping no-op marker transformation '{}'", marker)


TRANSFORMATION_REGISTRY: dict[type[Transformation], TransformationHandler] = {
    OutliersTransformation: process_outliers,
    RemovalTransformation: process_removal,
    EncodingTransformation: process_encoding,
    MergedTransformation: process_merge,
    RenamedTransformation: process_renamed,
    RenamingTransformation: process_renaming,
    MergingTransformation: _noop_marker_handler,
    ImputationTransformation: process_imputation,
    SplittedTransformation: process_split,
    SplittingTransformation: _noop_marker_handler,
    ScalingTransformation: process_scaling,
    BinningTransformation: process_binning,
    ColumnSubstitutionTransformation: process_column_substitution,
}


def resolve_handler(transformation: Transformation) -> TransformationHandler:
    """Return the registered handler for a transformation."""
    for transformation_cls, handler in TRANSFORMATION_REGISTRY.items():
        if isinstance(transformation, transformation_cls):
            return handler
    raise ValueError(f"Unknown transformation type: {type(transformation).__name__}")
