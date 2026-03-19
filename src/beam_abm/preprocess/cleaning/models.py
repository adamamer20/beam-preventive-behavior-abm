from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel

from beam_abm.common.logging import get_logger

AnyType = bool | None | int | float | str

EncodingType = dict[str, AnyType]

logger = get_logger(__name__)


class Transformation(BaseModel):
    type: Annotated[
        str,
        lambda x: x
        not in [
            "outliers",
            "splitted",
            "merged",
            "merging",
            "splitting",
            "removal",
            "encoding",
            "imputation",
        ],
    ]


class OutliersTransformation(Transformation):
    type: Literal["outliers"] = "outliers"
    min: int | None = None
    max: int | None = None


class SplittedTransformation(Transformation):
    type: Literal["splitted"] = "splitted"
    split_from: str
    values: list[str]


class MergedTransformation(Transformation):
    type: Literal["merged"] = "merged"
    merging_type: Literal["mean", "missing", "sum", "concat"]
    merged_from: list[str]
    concat_string: str | None = None


class MergingTransformation(Transformation):
    type: Literal["merging"] = "merging"
    merging_in: str


class SplittingTransformation(Transformation):
    type: Literal["splitting"] = "splitting"
    split_in: dict[str, list]


class RemovalTransformation(Transformation):
    type: Literal["removal"] = "removal"
    removal_reason: str
    removal_redundant_col: str | None = None
    removal_distribution: str | None = None


class EncodingTransformation(Transformation):
    type: Literal["encoding"] = "encoding"
    encoding: (
        Literal["$ONE-HOT"]
        | Literal["$EMBEDDING"]
        | Literal["$NUM"]
        | Literal["$COUNTING"]
        | Literal["$DATETIME"]
        | Literal["$DATETIME_TO_INTERVAL"]
        | Literal["$INTERVAL_TO_DATETIME"]
        | Literal["$REVERSE"]
        | str
        | EncodingType
    )
    embedding_prompt: str | None = None
    datetime_format: str | None = None
    from_timestamp: str | None = None
    unit: str | None = None
    key_type: Literal["str", "int", "float", "bool"] | None = None


class Bins(BaseModel):
    min: float
    value: int | float | str


class BinningTransformation(Transformation):
    type: Literal["binning"] = "binning"
    bins: str


class ImputationTransformation(Transformation):
    type: Literal["imputation"] = "imputation"
    imputation_value: Literal["$PREDICTION", "$MAX"] | int | str


class ScalingTransformation(Transformation):
    type: Literal["scaling"] = "scaling"
    scaling_type: Literal["robust", "minmax"]


class ColumnSubstitutionTransformation(Transformation):
    type: Literal["column_substitution"] = "column_substitution"
    df_file: str


class RenamingTransformation(Transformation):
    type: Literal["renaming"] = "renaming"
    rename_to: list[str]


class RenamedTransformation(Transformation):
    type: Literal["renamed"] = "renamed"
    rename_from: str


class Column(BaseModel):
    id: str
    section: str
    question: str | None = None
    question_group: str | None = None
    question_entity: str | None = None
    new_question: str | None = None
    answer: str | list[str] | None = None
    new_answer: str | list[str] | list[float] | list[int] | list[bool] | None = None
    missing_percentage: int | None = None
    n_outliers: dict[str, int] | None = None
    validation_category: str | None = None
    transformations: (
        deque[
            EncodingTransformation
            | ImputationTransformation
            | MergingTransformation
            | MergedTransformation
            | RemovalTransformation
            | SplittingTransformation
            | SplittedTransformation
            | OutliersTransformation
            | ScalingTransformation
            | BinningTransformation
            | ColumnSubstitutionTransformation
            | RenamingTransformation
            | RenamedTransformation
            | Transformation
        ]
        | None
    ) = None


class DataCleaningModel(BaseModel):
    encodings: dict[str, EncodingType] | None = None
    bins: dict[str, list[Bins]] | None = None
    question_group: dict[str, str | None] | None = None
    old_columns: list[Column]
    new_columns: list[Column] | None = None


def load_model(file_path: str) -> DataCleaningModel:
    """Load a ``DataCleaningModel`` from a JSON file."""
    logger.info(f"Loading data cleaning model from: {file_path}")
    path = Path(file_path)

    try:
        if not path.exists():
            logger.error(f"Model file not found: {file_path}")
            raise FileNotFoundError(f"Model file not found: {file_path}")

        model_json = path.read_text(encoding="utf-8")
        logger.debug(f"Model file size: {len(model_json)} characters")

        model = DataCleaningModel.model_validate_json(model_json)

        old_cols = len(model.old_columns) if model.old_columns else 0
        new_cols = len(model.new_columns) if model.new_columns else 0
        encodings = len(model.encodings) if model.encodings else 0
        bins = len(model.bins) if model.bins else 0

        logger.info(
            f"Successfully loaded model: {old_cols} old columns, {new_cols} new columns, {encodings} encodings, {bins} bin configs"
        )
        return model

    except Exception as e:
        logger.error(f"Error loading model from {file_path}: {e}")
        raise
