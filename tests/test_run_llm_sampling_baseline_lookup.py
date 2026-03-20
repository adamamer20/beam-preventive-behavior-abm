"""Regression tests for baseline lookup key normalization."""

from beam_abm.llm_microvalidation.shared.baselines import _lookup_baseline_value


def test_lookup_baseline_value_accepts_outcome_prefixed_baseline_ids() -> None:
    """Lookup should resolve `<outcome>__<join_id>` baseline ids for multi-outcome rows."""
    join_value = "0::DE::201"
    outcomes = [
        "flu_vaccinated_2023_2024",
        "mask_when_pressure_high",
        "mask_when_symptomatic_crowded",
        "stay_home_when_symptomatic",
        "vax_willingness_T12",
    ]
    by_id: dict[str, float | dict[str, float]] = {}
    by_id_outcome = {
        ("flu_vaccinated_2023_2024__0::DE::201", "flu_vaccinated_2023_2024"): 0.2425,
        ("mask_when_pressure_high__0::DE::201", "mask_when_pressure_high"): 4.275,
        ("mask_when_symptomatic_crowded__0::DE::201", "mask_when_symptomatic_crowded"): 4.6,
        ("stay_home_when_symptomatic__0::DE::201", "stay_home_when_symptomatic"): 4.45,
        ("vax_willingness_T12__0::DE::201", "vax_willingness_T12"): 3.025,
    }

    result = _lookup_baseline_value(
        join_value=join_value,
        outcome="flu_vaccinated_2023_2024",
        outcomes=outcomes,
        by_id=by_id,
        by_id_outcome=by_id_outcome,
    )

    assert result == {
        "flu_vaccinated_2023_2024": 0.2425,
        "mask_when_pressure_high": 4.275,
        "mask_when_symptomatic_crowded": 4.6,
        "stay_home_when_symptomatic": 4.45,
        "vax_willingness_T12": 3.025,
    }
