from __future__ import annotations

import json
from collections import deque

import numpy as np
import pandas as pd

from beam_abm.survey.cleaning.io import load_model, process_data
from beam_abm.survey.cleaning.models import (
    Column,
    DataCleaningModel,
    EncodingTransformation,
    ImputationTransformation,
    MergedTransformation,
)
from beam_abm.survey.cleaning.ppp import income_ppp_normalized
from beam_abm.survey.cleaning.processor import DataCleaningProcessor


def test_load_model_roundtrip(tmp_path) -> None:
    model_payload = {
        "old_columns": [
            {
                "id": "score",
                "section": "demo",
                "transformations": [{"type": "encoding", "encoding": "$reverse_1_to_5"}],
            }
        ]
    }
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(model_payload), encoding="utf-8")

    model = load_model(str(model_path))
    assert model.old_columns[0].id == "score"
    assert model.old_columns[0].transformations is not None


def test_transformation_graph_orders_sources_before_merge() -> None:
    model = DataCleaningModel(
        old_columns=[
            Column(
                id="a",
                section="x",
                transformations=deque([EncodingTransformation(encoding="$NUM")]),
            ),
            Column(
                id="b",
                section="x",
                transformations=deque([EncodingTransformation(encoding="$NUM")]),
            ),
            Column(
                id="c",
                section="x",
                transformations=deque([MergedTransformation(merging_type="mean", merged_from=["a", "b"])]),
            ),
        ]
    )
    processor = DataCleaningProcessor(model)
    nodes = processor._build_transformation_graph()
    order = {(n.col_id, n.index_in_col): i for i, n in enumerate(nodes)}

    assert order[("a", 0)] < order[("c", 0)]
    assert order[("b", 0)] < order[("c", 0)]


def test_encoding_path_reverse_scale() -> None:
    model = DataCleaningModel(
        old_columns=[
            Column(
                id="score",
                section="demo",
                transformations=deque([EncodingTransformation(encoding="$reverse_1_to_5")]),
            )
        ]
    )
    df = pd.DataFrame({"score": [1, 3, 5]})
    out = process_data(model, df)
    assert out["score"].tolist() == [5, 3, 1]


def test_prediction_imputation_path_fills_missing_values() -> None:
    model = DataCleaningModel(
        old_columns=[
            Column(id="x", section="demo"),
            Column(
                id="y",
                section="demo",
                transformations=deque([ImputationTransformation(imputation_value="$PREDICTION")]),
            ),
        ]
    )
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "y": [2.0, 4.0, np.nan, 8.0, 10.0, np.nan]})
    out = process_data(model, df)
    assert out["y"].isna().sum() == 0


def test_ppp_normalization_path(monkeypatch, tmp_path) -> None:
    ppp_file = tmp_path / "ppp.csv"
    ppp_file.write_text("country_code,ppp_value\nDE,2.0\nFR,4.0\n", encoding="utf-8")

    from beam_abm.survey.cleaning import ppp as ppp_module

    monkeypatch.setattr(ppp_module, "_PPP_LOOKUP_CACHE", None)
    monkeypatch.setattr(ppp_module, "PPP_DATA_FILE", str(ppp_file))

    incomes = pd.Series(["1000 to 2000", "2000 to 3000"], index=[0, 1])
    df = pd.DataFrame({"country": ["DE", "FR"]})
    normalized = income_ppp_normalized(incomes, df=df)

    assert normalized.notna().all()
    assert normalized.iloc[0] == 750.0
    assert normalized.iloc[1] == 625.0
