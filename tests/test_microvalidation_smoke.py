from __future__ import annotations

import numpy as np
import pandas as pd

from beam_abm.evaluation.common.grids import (
    build_grid_for_target,
    estimate_within_stratum_nn_distances,
    fit_pca,
    knn_realize_pc,
    knn_realize_scalar,
)
from beam_abm.llm.schemas.predictions import extract_scalar_prediction


def test_extract_scalar_prediction_variants() -> None:
    assert extract_scalar_prediction({"prediction": 1.5}) == 1.5
    assert extract_scalar_prediction({"p_yes": 0.25}) == 0.25

    z = extract_scalar_prediction({"levels": ["0", "1", "2"], "probs": [0.25, 0.5, 0.25]})
    assert z is not None
    assert abs(z - 1.0) < 1e-9


def test_grid_and_realization_smoke() -> None:
    df = pd.DataFrame(
        {
            "persona": ["a"] * 300,
            "country": ["x"] * 300,
            "score": np.linspace(0, 1, 300),
            "item1": np.random.default_rng(0).normal(size=300),
            "item2": np.random.default_rng(1).normal(size=300),
        }
    )

    grid = build_grid_for_target(
        df=df,
        target_col="score",
        primary_col="persona",
        country_col="country",
        primary_value="a",
        country="x",
        min_n=200,
        is_binary=False,
        primary_label="persona",
    )
    assert len(grid.grid_points) == 3

    rng = np.random.default_rng(0)
    row = knn_realize_scalar(df=df, score_col="score", x_star=0.5, k=10, rng=rng)
    assert "item1" in row

    pca_fit = fit_pca(df=df, columns=["item1", "item2"], n_components=2)
    nn = estimate_within_stratum_nn_distances(points=pca_fit.pc_scores[:, :2], sample_n=200)
    assert nn.ndim == 1

    realized, dist = knn_realize_pc(
        df=df,
        pc_scores=pca_fit.pc_scores[:, :2],
        target_point=np.array([0.0, 0.0], dtype=float),
        k=5,
        rng=rng,
    )
    assert "item2" in realized
    assert dist >= 0.0
