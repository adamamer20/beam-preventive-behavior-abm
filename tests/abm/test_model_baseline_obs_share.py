"""Regression tests for baseline expected observed-share computation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl

from beam_abm.abm.engine import VaccineBehaviourModel
from beam_abm.abm.state_update import SignalSpec


def test_baseline_expected_obs_share_uses_runtime_attention_sharing_columns(
    monkeypatch,
) -> None:
    """Baseline anchor sampling should use alpha/phi columns like runtime."""
    captured: dict[str, np.ndarray] = {}

    def _fake_sampled_peer_adoption(
        network,
        outcomes,
        *,
        sample_size: int,
        rng,
        attention: np.ndarray,
        sharing: np.ndarray,
        contact_rate_base: float,
        contact_rate_multiplier: float,
        activity_score: np.ndarray,
        contact_propensity: np.ndarray,
        per_agent_sample_size: np.ndarray,
    ) -> tuple[np.ndarray, None]:
        _ = (
            network,
            outcomes,
            sample_size,
            rng,
            contact_rate_base,
            contact_rate_multiplier,
            activity_score,
            contact_propensity,
            per_agent_sample_size,
        )
        captured["attention"] = np.asarray(attention, dtype=np.float64).copy()
        captured["sharing"] = np.asarray(sharing, dtype=np.float64).copy()
        n = int(attention.shape[0])
        return np.full(n, 0.5, dtype=np.float64), None

    monkeypatch.setattr(
        "beam_abm.abm.engine.compute_sampled_peer_adoption",
        _fake_sampled_peer_adoption,
    )

    agents_df = pl.DataFrame(
        {
            "social_attention_alpha": [0.1, 0.2, 0.3],
            "social_attention_pi": [0.9, 0.9, 0.9],
            "social_sharing_phi": [0.4, 0.5, 0.6],
            "social_sharing_tau": [0.8, 0.8, 0.8],
            "activity_score": [0.0, 0.1, 0.2],
            "contact_propensity": [1.0, 1.0, 1.0],
            "vax_convo_freq_cohabitants": [1.0, 2.0, 3.0],
        }
    )
    fake_model = SimpleNamespace(
        network=object(),
        agents=SimpleNamespace(df=agents_df),
        cfg=SimpleNamespace(
            seed=42,
            norm_observation_sample_size=8,
            constants=SimpleNamespace(
                norm_observation_min_per_month=2,
                norm_observation_max_per_month=10,
                social_substeps_per_month=4,
                contact_rate_base=1.0,
            ),
            knobs=SimpleNamespace(contact_rate_multiplier=1.0),
        ),
    )
    outcomes = {"vax_willingness_T12": np.array([1.0, 2.0, 3.0], dtype=np.float64)}

    baseline = VaccineBehaviourModel._compute_baseline_expected_obs_share(
        fake_model,  # type: ignore[arg-type]
        outcomes,
        n_draws=1,
    )

    assert baseline is not None
    np.testing.assert_allclose(captured["attention"], np.array([0.1, 0.2, 0.3], dtype=np.float64))
    np.testing.assert_allclose(captured["sharing"], np.array([0.4, 0.5, 0.6], dtype=np.float64))


def test_signal_effective_activation_persists_for_perturbation_worlds() -> None:
    """Perturbation low/high worlds stay active after the start tick."""
    signal = SignalSpec(
        target_column="institutional_trust_avg",
        intensity=1.0,
        signal_type="campaign",
        start_tick=12,
        end_tick=18,
    )
    fake_model = SimpleNamespace(
        cfg=SimpleNamespace(effect_regime="perturbation", perturbation_world="low"),
    )

    assert not VaccineBehaviourModel._is_signal_effectively_active(  # type: ignore[arg-type]
        fake_model, signal, tick=11
    )
    assert VaccineBehaviourModel._is_signal_effectively_active(  # type: ignore[arg-type]
        fake_model, signal, tick=12
    )
    assert VaccineBehaviourModel._is_signal_effectively_active(  # type: ignore[arg-type]
        fake_model, signal, tick=30
    )


def test_signal_effective_activation_uses_window_outside_perturbation_worlds() -> None:
    """Non-perturbation runs should respect the original signal window."""
    signal = SignalSpec(
        target_column="institutional_trust_avg",
        intensity=1.0,
        signal_type="campaign",
        start_tick=12,
        end_tick=18,
    )
    fake_model = SimpleNamespace(
        cfg=SimpleNamespace(effect_regime="shock", perturbation_world=None),
    )

    assert not VaccineBehaviourModel._is_signal_effectively_active(  # type: ignore[arg-type]
        fake_model, signal, tick=11
    )
    assert VaccineBehaviourModel._is_signal_effectively_active(  # type: ignore[arg-type]
        fake_model, signal, tick=12
    )
    assert not VaccineBehaviourModel._is_signal_effectively_active(  # type: ignore[arg-type]
        fake_model, signal, tick=20
    )
