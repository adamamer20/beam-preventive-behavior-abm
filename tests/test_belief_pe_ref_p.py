"""Regression tests for PE_ref_P reference-effect generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from beam_abm.llm_microvalidation.psychological_profiles.reference_effect_tables import compute_reference_effect_table


def _write_coefficients(path: Path, rows: list[tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["term,estimate"]
    for term, estimate in rows:
        lines.append(f"{term},{estimate}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_compute_reference_effect_table_applies_threshold_transform_and_term_scaling(tmp_path: Path) -> None:
    modeling_dir = tmp_path / "modeling"
    _write_coefficients(
        modeling_dir / "P_STAKES" / "P_STAKES_covid_perceived_danger_T12_REF" / "coefficients.csv",
        [
            ("age", 0.4),
            ("exper_illness", 0.2),
            ("1.0/2.0", -1.0),
            ("2.0/3.0", 0.0),
            ("3.0/4.0", 0.0),
            ("4.0/5.0", 0.0),
        ],
    )

    profiles_df = pd.DataFrame(
        {
            "row_id": ["1", "2"],
            "country": ["IT", "IT"],
            "age": [50.0, 60.0],
            "exper_illness": [0.0, 0.0],
        }
    )
    source_df = profiles_df.copy()
    signals = [
        {
            "id": "severe_illness_vicarious",
            "text": "test",
            "shocks": [{"var": "exper_illness", "type": "binary", "from": 0, "to": 1}],
        }
    ]

    pe_df, _, status = compute_reference_effect_table(
        profiles_df=profiles_df,
        source_df=source_df,
        signals=signals,
        mutable_state=["covid_perceived_danger_T12"],
        ref_state_models={"covid_perceived_danger_T12": "P_STAKES_covid_perceived_danger_T12_REF"},
        modeling_dir=modeling_dir,
        id_col="row_id",
        country_col="country",
        alpha_secondary=0.33,
        epsilon_ref=0.03,
        top_n_primary=1,
        strict_missing_models=True,
    )

    assert status["missing"] == {}
    assert not pe_df.empty
    pe_vals = pe_df["pe_ref_p"].abs().tolist()
    assert min(pe_vals) > 0.03
