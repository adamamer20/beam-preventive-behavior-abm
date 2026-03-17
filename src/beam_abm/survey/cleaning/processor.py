from __future__ import annotations

import pandas as pd

from beam_abm.common.logging import get_logger

from .encoders import count_answers, extract_number, generate_embeddings
from .graph import TransformationNode, build_transformation_graph
from .handlers import DataCleaningHandlersMixin
from .imputation import run_final_imputation_pass
from .models import (
    BinningTransformation,
    Column,
    ColumnSubstitutionTransformation,
    DataCleaningModel,
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
)
from .references import build_rename_to_map, resolve_ref, validate_references_before_processing
from .validation import validate_processed_dataframe

logger = get_logger(__name__)

_TRANSFORMATION_HANDLERS: dict[type[object], str | None] = {
    OutliersTransformation: "_process_outliers",
    RemovalTransformation: "_process_removal",
    EncodingTransformation: "_process_encoding",
    MergedTransformation: "_process_merge",
    RenamedTransformation: "_process_renamed",
    RenamingTransformation: "_process_renaming",
    MergingTransformation: None,  # metadata/dependency marker only
    ImputationTransformation: "_process_imputation",
    SplittedTransformation: "_process_split",
    SplittingTransformation: None,  # metadata/dependency marker only
    ScalingTransformation: "_process_scaling",
    BinningTransformation: "_process_binning",
    ColumnSubstitutionTransformation: "_process_column_substitution",
}


class DataCleaningProcessor(DataCleaningHandlersMixin):
    def __init__(self, model: DataCleaningModel):
        logger.info("Initializing DataCleaningProcessor")
        self.model = model
        self.df: pd.DataFrame = pd.DataFrame()
        self.imputed_values = pd.DataFrame()

        all_columns = list(self.model.old_columns)
        if self.model.new_columns is not None:
            all_columns.extend(self.model.new_columns)

        self._spec_ids: set[str] = {c.id for c in all_columns}
        self._rename_to_map: dict[str, str] = self._build_rename_to_map(all_columns)

        # Log model statistics
        old_cols_count = len(model.old_columns) if model.old_columns else 0
        new_cols_count = len(model.new_columns) if model.new_columns else 0
        logger.info(f"Model has {old_cols_count} old columns and {new_cols_count} new columns")

        if model.encodings:
            logger.info(f"Model has {len(model.encodings)} encodings defined")
        if model.bins:
            logger.info(f"Model has {len(model.bins)} binning configurations defined")

        logger.info("DataCleaningProcessor initialized successfully")

    @staticmethod
    def _build_rename_to_map(columns: list[Column]) -> dict[str, str]:
        """Build mapping of final dataset column name -> spec id."""
        return build_rename_to_map(columns)

    def _resolve_ref(self, name: str) -> str:
        """Resolve a reference name to a spec id."""
        return resolve_ref(name, spec_ids=self._spec_ids, rename_to_map=self._rename_to_map)

    def _validate_references_before_processing(self) -> set[str]:
        """Validate that all referenced names in the spec are resolvable."""
        return validate_references_before_processing(
            model=self.model,
            spec_ids=self._spec_ids,
            rename_to_map=self._rename_to_map,
        )

    def extract_number(self, column: pd.Series) -> pd.Series:
        return extract_number(column, logger=logger)

    def generate_embeddings(
        self, column: pd.Series, embedding_prompt: str, question_text: str | None = None
    ) -> pd.Series:
        return generate_embeddings(self, column, embedding_prompt, question_text, logger=logger)

    def count_answers(self, col: Column) -> pd.Series:
        return count_answers(self, col, logger=logger)

    def _build_transformation_graph(self) -> list[TransformationNode]:
        return build_transformation_graph(model=self.model, resolve_ref=self._resolve_ref)

    def _all_columns(self) -> list[Column]:
        columns = list(self.model.old_columns)
        if self.model.new_columns:
            columns.extend(self.model.new_columns)
        return columns

    def process_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Starting transformation processing on dataframe with shape: {df.shape}")

        final_name_refs = self._validate_references_before_processing()

        # Initialize processing dataframe
        self.df = df.copy()
        logger.debug("Created working copy of dataframe")

        # Build transformation graph
        logger.info("Building transformation execution graph")
        topo_nodes = self._build_transformation_graph()

        # Process transformations in order
        logger.info(f"Processing {len(topo_nodes)} transformations")

        for i, node in enumerate(topo_nodes):
            logger.debug(f"Processing transformation {i + 1}/{len(topo_nodes)}: {node}")

            try:
                self._apply_transformation_node(node)

                # Log dataframe shape after each transformation
                if i % 10 == 0 or i == len(topo_nodes) - 1:  # Log every 10th transformation and the last one
                    logger.debug(f"Dataframe shape after transformation {i + 1}: {self.df.shape}")

            except Exception as e:
                logger.error(f"Error processing transformation {i + 1} ({node}): {e}")
                raise

        logger.info("All transformations completed, starting final imputation pass")

        # Add final imputation pass
        try:
            run_final_imputation_pass(self, logger=logger)
            logger.info("Final imputation pass completed")
        except Exception as e:
            logger.error(f"Error during final imputation pass: {e}")
            raise

        final_shape = self.df.shape
        logger.info(f"Transformation processing completed - final dataframe shape: {final_shape}")
        validate_processed_dataframe(df=self.df, final_name_refs=final_name_refs, logger=logger)

        return self.df

    def _apply_transformation_node(self, node: TransformationNode):
        logger.debug(f"Applying transformation node: {node}")

        # Find the column object
        col_obj = next((c for c in self._all_columns() if c.id == node.col_id), None)
        if col_obj is None:
            logger.error(f"Column '{node.col_id}' not found in model columns")
            raise ValueError(f"Column '{node.col_id}' not found")

        t = node.transformation
        transformation_type = type(t).__name__

        logger.debug(f"Applying {transformation_type} to column '{node.col_id}'")

        # Record dataframe shape before transformation
        shape_before = self.df.shape

        try:
            for transformation_cls, handler_name in _TRANSFORMATION_HANDLERS.items():
                if not isinstance(t, transformation_cls):
                    continue
                if handler_name is None:
                    marker = transformation_type.replace("Transformation", "").lower()
                    logger.debug(f"Skipping no-op '{marker}' marker for column '{node.col_id}'")
                    break
                getattr(self, handler_name)(col_obj, t)
                break
            else:
                logger.error(f"Unknown transformation type: {transformation_type}")
                raise ValueError(f"Unknown transformation type: {transformation_type}")

            # Record dataframe shape after transformation
            shape_after = self.df.shape

            if shape_before != shape_after:
                logger.debug(f"Dataframe shape changed: {shape_before} -> {shape_after}")

            logger.debug(f"Successfully applied {transformation_type} to column '{node.col_id}'")

        except Exception as e:
            logger.error(f"Error applying {transformation_type} to column '{node.col_id}': {e}")
            raise
