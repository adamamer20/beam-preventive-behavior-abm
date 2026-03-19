from __future__ import annotations

from pathlib import Path

import pandas as pd

from beam_abm.preprocess.pipeline import run_clean_pipeline
from beam_abm.preprocess.specs import clean_spec as clean_spec_module


def test_clean_spec_builder_characterization_post_general(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(clean_spec_module, "SECTION_SPEC_DIR", tmp_path / "sections")
    output_spec_path = tmp_path / "clean_spec.json"
    spec = clean_spec_module.build_clean_analysis_spec(output_path=output_spec_path, stage="post-general")

    assert output_spec_path.exists()

    old_columns = spec.get("old_columns", []) or []
    new_columns = spec.get("new_columns", []) or []
    new_ids = {c.get("id") for c in new_columns}

    # Removed-from-output policy is represented as explicit removal transforms.
    covax_col = next((c for c in old_columns if c.get("id") == "covax"), None)
    assert covax_col is not None
    assert any(t.get("type") == "removal" for t in covax_col.get("transformations", []))
    assert "covax" not in new_ids

    # Alias generation
    alias_col = next((c for c in new_columns if c.get("id") == "fivec_confidence"), None)
    assert alias_col is not None
    assert any(
        t.get("type") == "renamed" and t.get("rename_from") == "fiveC_1" for t in alias_col.get("transformations", [])
    )

    # Derived composite column
    composite_col = next((c for c in new_columns if c.get("id") == "fivec_pro_vax_idx"), None)
    assert composite_col is not None
    assert any(t.get("type") == "merged" for t in composite_col.get("transformations", []))

    # Split expansion columns
    assert any((cid or "").startswith("covax_reason__") for cid in new_ids)

    # Source removal marker for renamed aliases
    source_col = next((c for c in old_columns if c.get("id") == "fiveC_1"), None)
    assert source_col is not None
    assert any(
        t.get("type") == "removal" and t.get("removal_reason") == "renamed"
        for t in source_col.get("transformations", [])
    )

    # Section assignments
    all_columns = old_columns + new_columns
    assert all(isinstance(c.get("section"), str) and bool(c.get("section").strip()) for c in all_columns)


def test_clean_pipeline_writes_clean_output(tmp_path, monkeypatch) -> None:
    raw_df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

    def _load_raw() -> pd.DataFrame:
        return raw_df

    def _load_model(_: str):
        return object()

    def _process_data(_model, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(z=1)

    monkeypatch.setattr("beam_abm.preprocess.pipeline.load_raw_survey_df", _load_raw)
    monkeypatch.setattr("beam_abm.preprocess.pipeline.load_model", _load_model)
    monkeypatch.setattr("beam_abm.preprocess.pipeline.process_data", _process_data)

    output_csv = tmp_path / "clean.csv"
    out_df = run_clean_pipeline(model_path=Path("dummy.json"), output_csv=output_csv)

    assert output_csv.exists()
    assert list(out_df.columns) == ["x", "y", "z"]
    reloaded = pd.read_csv(output_csv)
    assert list(reloaded.columns) == ["x", "y", "z"]
