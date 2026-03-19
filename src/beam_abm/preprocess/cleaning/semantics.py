from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .models import (
    BinningTransformation,
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

SPECIAL_IMPUTATION_TOKENS: frozenset[str] = frozenset({"$PREDICTION", "$MAX"})

TRANSFORMATION_MODEL_BY_KIND: dict[str, type[Transformation]] = {
    "outliers": OutliersTransformation,
    "splitted": SplittedTransformation,
    "merged": MergedTransformation,
    "merging": MergingTransformation,
    "splitting": SplittingTransformation,
    "removal": RemovalTransformation,
    "encoding": EncodingTransformation,
    "imputation": ImputationTransformation,
    "scaling": ScalingTransformation,
    "binning": BinningTransformation,
    "column_substitution": ColumnSubstitutionTransformation,
    "renaming": RenamingTransformation,
    "renamed": RenamedTransformation,
}

SUPPORTED_TRANSFORM_KINDS: frozenset[str] = frozenset(TRANSFORMATION_MODEL_BY_KIND)


def parse_transformation_payload(payload: Mapping[str, Any]) -> Transformation | None:
    """Parse a raw transformation mapping into a typed model when supported."""
    transform_kind = payload.get("type")
    if not isinstance(transform_kind, str):
        return None

    model_cls = TRANSFORMATION_MODEL_BY_KIND.get(transform_kind)
    if model_cls is None:
        return None
    return model_cls.model_validate(dict(payload))


def normalize_reference_name(raw_ref: str) -> str | None:
    """Normalize a raw reference token to a column identifier when applicable."""
    if not raw_ref:
        return None

    if raw_ref.startswith("$"):
        if raw_ref in SPECIAL_IMPUTATION_TOKENS:
            return None
        normalized = raw_ref.lstrip("$")
        return normalized or None

    return raw_ref


def iter_raw_reference_names(transformation: Transformation) -> Iterable[str]:
    """Yield raw reference tokens used by a transformation."""
    if isinstance(transformation, MergedTransformation):
        yield from transformation.merged_from
        return

    if isinstance(transformation, SplittedTransformation):
        yield transformation.split_from
        return

    if isinstance(transformation, RenamedTransformation):
        yield transformation.rename_from
        return

    if isinstance(transformation, ImputationTransformation):
        if isinstance(transformation.imputation_value, str) and transformation.imputation_value.startswith("$"):
            yield transformation.imputation_value


def iter_dependency_source_ids(
    transformation: Transformation,
    *,
    resolve_ref: Callable[[str], str],
) -> list[str]:
    """Resolve dependency source ids for graph/derivation/reference workflows."""
    resolved: list[str] = []
    for raw_ref in iter_raw_reference_names(transformation):
        normalized = normalize_reference_name(raw_ref)
        if normalized is None:
            continue
        resolved.append(resolve_ref(normalized))
    return resolved
