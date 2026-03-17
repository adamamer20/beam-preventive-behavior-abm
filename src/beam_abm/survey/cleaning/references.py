from __future__ import annotations

from beam_abm.common.logging import get_logger

from .models import (
    Column,
    DataCleaningModel,
    ImputationTransformation,
    MergedTransformation,
    RenamedTransformation,
    RenamingTransformation,
    SplittedTransformation,
)

logger = get_logger(__name__)


def build_rename_to_map(columns: list[Column]) -> dict[str, str]:
    """Build mapping of final dataset column name -> spec id."""
    final_name_to_ids: dict[str, set[str]] = {}
    for col in columns:
        if not col.transformations:
            continue
        for t in col.transformations:
            if isinstance(t, RenamingTransformation):
                for final_name in t.rename_to or []:
                    if not final_name:
                        continue
                    final_name_to_ids.setdefault(str(final_name), set()).add(col.id)

    collisions = {name: ids for name, ids in final_name_to_ids.items() if len(ids) > 1}
    if collisions:
        formatted = "; ".join(f"{name}: {sorted(ids)}" for name, ids in sorted(collisions.items()))
        raise ValueError(f"Renaming collision: multiple spec ids map to the same final name: {formatted}")

    return {name: next(iter(ids)) for name, ids in final_name_to_ids.items()}


def resolve_ref(name: str, *, spec_ids: set[str], rename_to_map: dict[str, str]) -> str:
    """Resolve a reference name to a spec id."""
    if name in spec_ids:
        return name
    if name in rename_to_map:
        return rename_to_map[name]
    return name


def validate_references_before_processing(
    *,
    model: DataCleaningModel,
    spec_ids: set[str],
    rename_to_map: dict[str, str],
) -> set[str]:
    """Validate that all referenced names in the spec are resolvable."""
    all_columns = list(model.old_columns)
    if model.new_columns is not None:
        all_columns.extend(model.new_columns)

    final_name_refs: set[str] = set()
    raw_refs: list[str] = []
    for col in all_columns:
        if not col.transformations:
            continue
        for t in col.transformations:
            if isinstance(t, MergedTransformation):
                raw_refs.extend(t.merged_from)
            elif isinstance(t, SplittedTransformation):
                raw_refs.append(t.split_from)
            elif isinstance(t, RenamedTransformation):
                raw_refs.append(t.rename_from)
            elif isinstance(t, ImputationTransformation) and isinstance(t.imputation_value, str):
                if t.imputation_value.startswith("$"):
                    raw_refs.append(t.imputation_value)

    for raw_ref in raw_refs:
        if not raw_ref:
            continue
        check_name = raw_ref
        if check_name.startswith("$"):
            if check_name in {"$PREDICTION", "$MAX"}:
                continue
            check_name = check_name.lstrip("$")

        if check_name in spec_ids:
            continue
        if check_name in rename_to_map:
            final_name_refs.add(check_name)
            continue
        raise ValueError(f"Reference '{raw_ref}' not found in spec ids or renaming targets")

    return final_name_refs
