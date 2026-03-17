"""Tests for intensity calibration utilities."""

from __future__ import annotations

import numpy as np
import polars as pl

from beam_abm.abm.intensity_calibration import (
    IntensityCalibrator,
    build_iqr_scales,
    calibrate_signal,
    load_anchor_perturbation_scales,
)
from beam_abm.abm.state_update import SignalSpec


def test_signal_intensity_matches_iqr_scale() -> None:
    values = np.linspace(-2.0, 2.0, 101, dtype=np.float64)
    agents = pl.DataFrame({"vaccine_risk_avg": values})

    scales = build_iqr_scales(agents)
    calibrator = IntensityCalibrator(global_scales=scales)
    signal = SignalSpec(target_column="vaccine_risk_avg", intensity=1.0, signal_id="test")

    calibrated = calibrate_signal(signal, calibrator)
    expected_iqr = scales["vaccine_risk_avg"]
    assert np.isclose(calibrated.intensity, expected_iqr, rtol=1e-6, atol=1e-6)


def test_zero_iqr_has_zero_scale_no_unit_fallback() -> None:
    agents = pl.DataFrame({"covax_legitimacy_scepticism_idx": [0.0] * 32})
    scales = build_iqr_scales(agents)
    assert np.isclose(scales["covax_legitimacy_scepticism_idx"], 0.0)


def test_missing_perturbation_scale_returns_zero() -> None:
    calibrator = IntensityCalibrator(global_scales={})
    got = calibrator.calibrate_intensity(
        "covax_legitimacy_scepticism_idx",
        nominal_intensity=1.0,
        effect_regime="perturbation",
    )
    assert np.isclose(got, 0.0)


def test_load_anchor_perturbation_scales_reads_shock_table(tmp_path) -> None:
    table_dir = tmp_path / "mutable_engines" / "full" / "vax_willingness_T12"
    table_dir.mkdir(parents=True, exist_ok=True)
    shock_table = table_dir / "shock_table.csv"
    shock_table.write_text(
        "\n".join(
            [
                "country,slice,lever,q25,q50,q75,delta_x,delta_z,stratum_level,stratum_key,n",
                "DE,,covax_legitimacy_scepticism_idx,0,0,0,0.5,2.5,country,DE,100",
                "DE,,institutional_trust_avg,2.0,2.8,3.5,1.5,1.6,country,DE,100",
                "UK,,covax_legitimacy_scepticism_idx,0,0,0,0.333333,2.3,country,UK,100",
                "UK,,institutional_trust_avg,2.3,2.9,3.8,1.5,1.5,country,UK,100",
            ]
        ),
        encoding="utf-8",
    )

    global_scales, country_scales = load_anchor_perturbation_scales(tmp_path)
    assert np.isclose(global_scales["covax_legitimacy_scepticism_idx"], 0.4166665)
    assert np.isclose(country_scales["DE"]["covax_legitimacy_scepticism_idx"], 0.5)
    assert np.isclose(country_scales["UK"]["institutional_trust_avg"], 1.5)
