from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer, RobustScaler
from vllm import LLM
from vllm.outputs import EmbeddingRequestOutput

from beam_abm.settings import EMBEDDING_MODEL

if TYPE_CHECKING:
    from .models import Column


def extract_number(column: pd.Series, *, logger) -> pd.Series:
    logger.info(f"Extracting numbers from column '{column.name}' with dtype {column.dtype}")

    if column.dtype == "object":
        logger.debug("Column is object type, proceeding with number extraction")
        extracted = column.str.extract(r"(\d+\.?\d*)", expand=False).astype(float)
        successful_extractions = int(extracted.notna().sum())
        logger.info(f"Successfully extracted numbers from {successful_extractions}/{len(column)} rows")

        if successful_extractions == 0:
            logger.warning(f"No numbers could be extracted from column '{column.name}'")

        return extracted

    logger.debug("Column is not object type, attempting numeric coercion")
    coerced = pd.to_numeric(column, errors="coerce")
    successful_coercions = int(coerced.notna().sum())
    logger.info(f"Numeric coercion retained {successful_coercions}/{len(column)} non-NA values")
    return coerced


def generate_embeddings(
    processor, column: pd.Series, embedding_prompt: str, question_text: str | None, *, logger
) -> pd.Series:
    logger.info(f"Generating embeddings for column '{column.name}' with {len(column)} rows")

    unique_values = column.unique().tolist()
    logger.info(f"Found {len(unique_values)} unique values for embedding")
    logger.debug(f"Unique values sample: {unique_values[:10]}{'...' if len(unique_values) > 10 else ''}")

    if embedding_prompt and str(embedding_prompt).strip() and not str(embedding_prompt).strip().startswith("$"):
        task = str(embedding_prompt).strip()
    else:
        task = (
            f"Encode responses to the survey question: {str(question_text).strip()} "
            f"and capture behavioral and risk-related semantics."
        )
    logger.info("Using instruction-tuned inputs for embeddings (Instruct/Query)")
    processed_inputs = [
        f"Instruct: {task}\nQuery:{'' if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)}"
        for v in unique_values
    ]

    embedding_model = EMBEDDING_MODEL
    if not hasattr(processor, "_vllm_embedder") or getattr(processor, "_vllm_model", None) != embedding_model:
        logger.info(f"Initializing vLLM embedding model: {embedding_model}")
        processor._vllm_embedder = LLM(model=embedding_model, task="embed")
        processor._vllm_model = embedding_model

    request_outputs = cast(
        list[EmbeddingRequestOutput],
        processor._vllm_embedder.embed(processed_inputs),
    )

    vectors: list[list[float]] = []
    for request_output in request_outputs:
        embedding_output = request_output.outputs
        vectors.append(embedding_output.embedding)

    logger.info("Successfully generated embeddings via vLLM (offline)")

    class _Resp:
        def __init__(self, embeddings):
            self.embeddings = embeddings

    response = _Resp(vectors)
    logger.debug(
        f"Raw embeddings shape: {len(response.embeddings)}x{len(response.embeddings[0]) if response.embeddings else 0}"
    )

    logger.debug("Scaling embeddings using RobustScaler")
    scaled_embeddings = RobustScaler().fit_transform(response.embeddings)
    logger.debug(f"Scaled embeddings shape: {scaled_embeddings.shape}")

    logger.debug("Reducing dimensionality with PCA")
    pca_100 = PCA(n_components=min(100, len(scaled_embeddings)))
    pca_95 = PCA(n_components=0.95)
    reduced_embeddings_100 = pca_100.fit_transform(scaled_embeddings)
    reduced_embeddings_95 = pca_95.fit_transform(scaled_embeddings)

    logger.debug(f"PCA 100 components: {pca_100.n_components_}")
    logger.debug(f"PCA 95% variance components: {pca_95.n_components_}")

    if pca_100.n_components_ < pca_95.n_components_:
        reduced_embeddings = reduced_embeddings_100
        logger.info(f"Selected PCA with 100 components (actual: {pca_100.n_components_})")
    else:
        reduced_embeddings = reduced_embeddings_95
        logger.info(f"Selected PCA with 95% variance (components: {pca_95.n_components_})")

    logger.debug("Normalizing embeddings with L2 norm")
    normalized_embeddings = Normalizer(norm="l2").fit_transform(reduced_embeddings)
    logger.debug(f"Final embeddings shape: {normalized_embeddings.shape}")

    encoding = {}
    for value, embedding in zip(unique_values, normalized_embeddings, strict=False):
        encoding[value] = embedding

    logger.info(f"Created encoding dictionary with {len(encoding)} entries")
    result = column.map(encoding)
    logger.info("Successfully mapped embeddings to column values")
    return result


def count_answers(processor, col: Column, *, logger) -> pd.Series:
    logger.info(f"Counting answers for column '{col.id}' with {len(col.answer) if col.answer else 0} expected answers")
    logger.debug(f"Expected answers: {col.answer}")

    new_column = pd.Series(0, index=processor.df.index)

    na_count = processor.df[col.id].isna().sum()
    if na_count > 0:
        logger.debug(f"Filling {na_count} NA values with empty string")
    processor.df[col.id] = processor.df[col.id].fillna("")

    for answer in col.answer:
        matches = processor.df[col.id].str.contains(answer)
        match_count = matches.sum()
        logger.debug(f"Found {match_count} matches for answer '{answer}'")
        new_column[matches] += 1

    count_dist = new_column.value_counts().sort_index()
    logger.info(f"Answer count distribution: {count_dist.to_dict()}")
    logger.info(f"Completed answer counting for column '{col.id}'")

    return new_column
