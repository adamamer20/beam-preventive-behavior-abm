"""Tests for ABM outcome allowlist filtering."""

from __future__ import annotations

from beam_abm.abm.data_contracts import LeversConfig, OutcomeLever, get_abm_outcome_specs
from beam_abm.abm.state_schema import ABM_ALLOWED_OUTCOMES


def test_get_abm_outcome_specs_filters_non_abm_outcomes() -> None:
    levers_cfg = LeversConfig(
        outcomes={
            "vax_willingness_T12": OutcomeLever(
                outcome="vax_willingness_T12",
                model_ref="m_vax",
                levers=("institutional_trust_avg",),
                include_cols=(),
            ),
            "mask_when_symptomatic_crowded": OutcomeLever(
                outcome="mask_when_symptomatic_crowded",
                model_ref="m_mask",
                levers=("social_norms_vax_avg",),
                include_cols=(),
            ),
            "flu_vaccinated_2023_2024": OutcomeLever(
                outcome="flu_vaccinated_2023_2024",
                model_ref="m_flu",
                levers=("vaccine_risk_avg",),
                include_cols=(),
            ),
        }
    )

    assert "flu_vaccinated_2023_2024" in levers_cfg.outcomes

    filtered = get_abm_outcome_specs(levers_cfg)

    assert "flu_vaccinated_2023_2024" not in filtered
    assert set(filtered).issubset(set(ABM_ALLOWED_OUTCOMES))
    assert "vax_willingness_T12" in filtered
    assert "mask_when_symptomatic_crowded" in filtered
