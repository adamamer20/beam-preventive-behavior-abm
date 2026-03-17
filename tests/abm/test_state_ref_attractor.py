"""Tests for survey-anchored state reference attractors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from beam_abm.abm.state_ref_attractor import build_state_ref_attractor


def _write_coefficients(path: Path, rows: list[tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["term,estimate"]
    for term, estimate in rows:
        lines.append(f"{term},{estimate}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_builds_persistent_offsets_from_baseline(tmp_path: Path) -> None:
    modeling_dir = tmp_path / "modeling"
    _write_coefficients(
        modeling_dir / "P_TRUST" / "P_TRUST_institutional_trust_REF" / "coefficients.csv",
        [
            ("Intercept", 0.0),
            ("institutional_trust_avg", 1.0),
        ],
    )
    spec_path = tmp_path / "belief_targets.json"
    spec_path.write_text(
        json.dumps(
            {
                "mutable_state": ["institutional_trust_avg"],
                "ref_state_models": {
                    "institutional_trust_avg": "P_TRUST_institutional_trust_REF",
                },
            }
        ),
        encoding="utf-8",
    )

    agents = pl.DataFrame({"institutional_trust_avg": [0.0, 1.0, -1.0]})
    attractor = build_state_ref_attractor(
        agents_t0=agents,
        modeling_dir=modeling_dir,
        spec_path=spec_path,
        strict=True,
    )
    assert attractor is not None

    predicted = attractor.predict_targets(agents)["institutional_trust_avg"]
    offsets = attractor.persistent_offsets["institutional_trust_avg"]
    observed = agents["institutional_trust_avg"].to_numpy().astype(np.float64)
    np.testing.assert_allclose(predicted + offsets, observed, atol=1e-12)


def test_social_norm_reference_targets_norm_channels(tmp_path: Path) -> None:
    modeling_dir = tmp_path / "modeling"
    _write_coefficients(
        modeling_dir / "P_NORMS" / "P_NORMS_social_norms_vax_avg_REF" / "coefficients.csv",
        [
            ("social_norms_vax_avg", 1.0),
            ("-2.0/-1.0", -2.0),
            ("-1.0/0.0", -1.0),
            ("0.0/1.0", 0.0),
            ("1.0/2.0", 1.0),
            ("2.0/3.0", 2.0),
            ("3.0/4.0", 3.0),
        ],
    )
    spec_path = tmp_path / "belief_targets.json"
    spec_path.write_text(
        json.dumps(
            {
                "mutable_state": ["social_norms_vax_avg"],
                "ref_state_models": {
                    "social_norms_vax_avg": "P_NORMS_social_norms_vax_avg_REF",
                },
            }
        ),
        encoding="utf-8",
    )
    agents = pl.DataFrame(
        {
            "social_norms_vax_avg": [4.0, 4.5],
            "social_norms_descriptive_vax": [4.2, 4.1],
        }
    )
    attractor = build_state_ref_attractor(
        agents_t0=agents,
        modeling_dir=modeling_dir,
        spec_path=spec_path,
        strict=True,
    )
    assert attractor is not None

    targets = attractor.predict_targets(agents)
    assert "social_norms_descriptive_vax" in targets
    assert "social_norms_descriptive_vax" in attractor.persistent_offsets


def test_ref_linear_models_use_standardized_continuous_terms(tmp_path: Path) -> None:
    modeling_dir = tmp_path / "modeling"
    _write_coefficients(
        modeling_dir / "P_TRUST" / "P_TRUST_institutional_trust_REF" / "coefficients.csv",
        [
            ("Intercept", 0.0),
            ("trust_onlineinfo", 1.0),
            ("exper_illness", 1.0),
        ],
    )
    spec_path = tmp_path / "belief_targets.json"
    spec_path.write_text(
        json.dumps(
            {
                "mutable_state": ["institutional_trust_avg"],
                "ref_state_models": {
                    "institutional_trust_avg": "P_TRUST_institutional_trust_REF",
                },
            }
        ),
        encoding="utf-8",
    )
    agents = pl.DataFrame(
        {
            "institutional_trust_avg": [0.0, 0.0],
            "trust_onlineinfo": [10.0, 20.0],
            "exper_illness": [0.0, 1.0],
        }
    )
    attractor = build_state_ref_attractor(
        agents_t0=agents,
        modeling_dir=modeling_dir,
        spec_path=spec_path,
        strict=True,
    )
    assert attractor is not None

    predicted = attractor.predict_targets(agents)["institutional_trust_avg"]
    np.testing.assert_allclose(predicted, np.asarray([-1.0, 2.0]), atol=1e-12)


def test_predict_targets_updates_when_immutable_precursor_changes(tmp_path: Path) -> None:
    modeling_dir = tmp_path / "modeling"
    _write_coefficients(
        modeling_dir / "P_TRUST" / "P_TRUST_institutional_trust_REF" / "coefficients.csv",
        [
            ("Intercept", 0.0),
            ("trust_gp_vax_info", 1.0),
        ],
    )
    spec_path = tmp_path / "belief_targets.json"
    spec_path.write_text(
        json.dumps(
            {
                "mutable_state": ["institutional_trust_avg"],
                "ref_state_models": {
                    "institutional_trust_avg": "P_TRUST_institutional_trust_REF",
                },
            }
        ),
        encoding="utf-8",
    )
    agents_t0 = pl.DataFrame(
        {
            "institutional_trust_avg": [0.0, 0.0],
            "trust_gp_vax_info": [1.0, 3.0],
        }
    )
    attractor = build_state_ref_attractor(
        agents_t0=agents_t0,
        modeling_dir=modeling_dir,
        spec_path=spec_path,
        strict=True,
    )
    assert attractor is not None

    agents_shifted = agents_t0.with_columns(pl.Series("trust_gp_vax_info", [3.0, 1.0]))
    baseline_targets = attractor.predict_targets(agents_t0)["institutional_trust_avg"]
    shifted_targets = attractor.predict_targets(agents_shifted)["institutional_trust_avg"]

    np.testing.assert_allclose(baseline_targets, np.asarray([-1.0, 1.0]), atol=1e-12)
    np.testing.assert_allclose(shifted_targets, np.asarray([1.0, -1.0]), atol=1e-12)


def test_ordinal_thresholds_use_statsmodels_cutpoint_transform(tmp_path: Path) -> None:
    modeling_dir = tmp_path / "modeling"
    _write_coefficients(
        modeling_dir / "P_STAKES" / "P_STAKES_covid_perceived_danger_T12_REF" / "coefficients.csv",
        [
            ("exper_illness", 0.20535517569279127),
            ("1.0/2.0", -2.11169514628266),
            ("2.0/3.0", 0.2837713619735393),
            ("3.0/4.0", 0.1941404862250914),
            ("4.0/5.0", 0.22714003307968167),
        ],
    )
    spec_path = tmp_path / "belief_targets.json"
    spec_path.write_text(
        json.dumps(
            {
                "mutable_state": ["covid_perceived_danger_T12"],
                "ref_state_models": {
                    "covid_perceived_danger_T12": "P_STAKES_covid_perceived_danger_T12_REF",
                },
            }
        ),
        encoding="utf-8",
    )
    agents = pl.DataFrame({"covid_perceived_danger_T12": [3.0], "exper_illness": [0.0]})
    attractor = build_state_ref_attractor(
        agents_t0=agents,
        modeling_dir=modeling_dir,
        spec_path=spec_path,
        strict=True,
    )
    assert attractor is not None

    low_agents = pl.DataFrame({"covid_perceived_danger_T12": [3.0], "exper_illness": [0.0]})
    high_agents = pl.DataFrame({"covid_perceived_danger_T12": [3.0], "exper_illness": [1.0]})
    low_pred = attractor.predict_targets(low_agents)["covid_perceived_danger_T12"][0]
    high_pred = attractor.predict_targets(high_agents)["covid_perceived_danger_T12"][0]

    assert high_pred > low_pred
    assert (high_pred - low_pred) > 0.05
