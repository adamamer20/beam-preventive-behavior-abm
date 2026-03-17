"""Tests for belief-update kernel effect semantics."""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from beam_abm.abm.belief_update_kernel import BeliefUpdateKernel


def _agents() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "institutional_trust_avg": [0.0, 0.1, -0.2],
            "vaccine_risk_avg": [0.0, 0.2, 0.1],
        }
    )


def test_kernel_parses_signed_numeric_effects(tmp_path) -> None:  # type: ignore[no-untyped-def]
    payload = {
        "signals": [
            {
                "id": "s1",
                "effects": {
                    "institutional_trust_avg": 2.0,
                    "vaccine_risk_avg": -0.5,
                },
            }
        ]
    }
    path = tmp_path / "targets.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    kernel = BeliefUpdateKernel.from_targets_file(path, intensity_scale=0.25)
    updates = kernel.apply(_agents(), signal_id="s1", intensity=1.0)

    trust = updates["institutional_trust_avg"]
    risk = updates["vaccine_risk_avg"]
    np.testing.assert_allclose(trust, np.array([0.5, 0.6, 0.3]), atol=1e-12)
    np.testing.assert_allclose(risk, np.array([-0.125, 0.075, -0.025]), atol=1e-12)


def test_kernel_parses_direction_and_magnitude_dict(tmp_path) -> None:  # type: ignore[no-untyped-def]
    payload = {
        "signals": [
            {
                "id": "s2",
                "effects": {
                    "institutional_trust_avg": {"direction": "up", "magnitude": 1.5},
                },
            }
        ]
    }
    path = tmp_path / "targets.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    kernel = BeliefUpdateKernel.from_targets_file(path, intensity_scale=0.2)
    updates = kernel.apply(_agents(), signal_id="s2", intensity=2.0)
    trust = updates["institutional_trust_avg"]
    np.testing.assert_allclose(trust, np.array([0.6, 0.7, 0.4]), atol=1e-12)


def test_kernel_applies_target_mask_only(tmp_path) -> None:  # type: ignore[no-untyped-def]
    payload = {"signals": [{"id": "s3", "effects": {"institutional_trust_avg": 1.0}}]}
    path = tmp_path / "targets.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    kernel = BeliefUpdateKernel.from_targets_file(path, intensity_scale=0.3)
    mask = np.array([True, False, False])
    updates = kernel.apply(_agents(), signal_id="s3", intensity=1.0, target_mask=mask)
    trust = updates["institutional_trust_avg"]
    np.testing.assert_allclose(trust, np.array([0.3, 0.1, -0.2]), atol=1e-12)
