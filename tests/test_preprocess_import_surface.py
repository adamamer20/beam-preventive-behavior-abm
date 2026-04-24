from __future__ import annotations

from beam_abm.preprocess.cleaning import load_model as load_model_preprocess


def test_preprocess_namespace_is_canonical() -> None:
    assert callable(load_model_preprocess)
