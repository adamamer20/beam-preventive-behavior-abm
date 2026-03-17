"""Tests for the state-update module."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from beam_abm.abm.message_contracts import NormChannelWeights
from beam_abm.abm.state_schema import CONSTRUCT_BOUNDS
from beam_abm.abm.state_update import (
    SignalSpec,
    apply_all_state_updates,
    apply_social_norm_update,
    clip_to_bounds,
    compute_observed_norm_exposure,
    compute_observed_norm_fixed_point_share,
)

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture()
def agents_df() -> pl.DataFrame:
    """Small agent DataFrame with actual mutable column names."""
    return pl.DataFrame(
        {
            "institutional_trust_avg": [0.5, -0.5, 1.0, -1.0],
            "vaccine_risk_avg": [0.0, 0.5, -0.5, 1.0],
            "fivec_confidence": [0.3, -0.2, 0.7, -0.3],
            "fivec_low_complacency": [0.1, 0.2, -0.1, 0.4],
            "disease_fear_avg": [0.0, 0.5, 1.0, -0.5],
            "covid_perceived_danger_Y": [0.2, 0.3, 0.1, 0.5],
            "covax_legitimacy_scepticism_idx": [0.5, 0.5, 0.5, 0.5],
            "fivec_collective_responsibility": [0.2, -0.1, 0.3, 0.0],
            "fivec_low_constraints": [0.0, 0.1, -0.1, 0.2],
            "fivec_pro_vax_idx": [0.1, 0.2, 0.3, 0.4],
            "social_norms_vax_avg": [0.0, 0.0, 0.0, 0.0],
            "social_norms_descriptive_vax": [4.0, 4.0, 4.0, 4.0],
            "social_norms_descriptive_npi": [4.0, 4.0, 4.0, 4.0],
            "norm_share_hat_vax": [0.5, 0.5, 0.5, 0.5],
            "norm_share_hat_npi": [0.5, 0.5, 0.5, 0.5],
            "trust_traditional": [0.3, 0.4, 0.5, 0.6],
            "trust_onlineinfo": [0.1, 0.2, 0.3, 0.4],
            "trust_social": [0.2, 0.3, 0.4, 0.5],
            "duration_socialmedia": [0.1, 0.2, 0.3, 0.4],
            "duration_infomedia": [0.1, 0.2, 0.3, 0.4],
            "infection_likelihood_avg": [0.0, 0.1, 0.2, 0.3],
            "severity_if_infected_avg": [0.1, 0.2, 0.3, 0.4],
            "flu_vaccine_risk": [0.0, 0.1, -0.1, 0.2],
            "flu_disease_fear": [0.0, 0.1, 0.2, -0.1],
            "group_trust_vax": [0.1, 0.2, 0.3, 0.4],
            "country": ["IT", "DE", "UK", "FR"],
        }
    )


# =========================================================================
# Tests: clip_to_bounds
# =========================================================================


class TestClipToBounds:
    """Tests for clip_to_bounds."""

    def test_clips_high(self) -> None:
        values = np.array([5.0, 3.0])
        clipped = clip_to_bounds(values, "institutional_trust_avg")
        bounds = CONSTRUCT_BOUNDS["institutional_trust_avg"]
        assert clipped.max() <= bounds[1]

    def test_clips_low(self) -> None:
        values = np.array([-5.0, -3.0])
        clipped = clip_to_bounds(values, "institutional_trust_avg")
        bounds = CONSTRUCT_BOUNDS["institutional_trust_avg"]
        assert clipped.min() >= bounds[0]

    def test_within_bounds_unchanged(self) -> None:
        values = np.array([0.5, -0.3])
        clipped = clip_to_bounds(values, "institutional_trust_avg")
        np.testing.assert_array_equal(values, clipped)


# =========================================================================
# Tests: SignalSpec
# =========================================================================


class TestSignalSpec:
    """Tests for SignalSpec dataclass."""

    def test_creation(self) -> None:
        spec = SignalSpec(
            target_column="vaccine_risk_avg",
            intensity=0.5,
            signal_type="pulse",
            start_tick=0,
        )
        assert spec.target_column == "vaccine_risk_avg"
        assert spec.signal_type == "pulse"

    def test_default_end_tick(self) -> None:
        spec = SignalSpec(
            target_column="vaccine_risk_avg",
            intensity=0.5,
        )
        assert spec.end_tick is None

    def test_empty_column_raises(self) -> None:
        """Empty target columns should be rejected."""
        with pytest.raises(ValueError):
            SignalSpec(
                target_column="",
                intensity=0.5,
            )

    def test_is_active(self) -> None:
        spec = SignalSpec(
            target_column="vaccine_risk_avg",
            intensity=0.5,
            start_tick=5,
            end_tick=10,
        )
        assert not spec.is_active(4)
        assert spec.is_active(5)
        assert spec.is_active(7)
        assert not spec.is_active(10)
        assert not spec.is_active(11)

    def test_target_offset_for_pulse_is_one_shot(self) -> None:
        spec = SignalSpec(
            target_column="vaccine_risk_avg",
            intensity=1.0,
            start_tick=0,
        )
        i0 = spec.target_offset(0)
        i5 = spec.target_offset(5)
        assert i0 > i5


# =========================================================================
# Tests: apply_social_norm_update
# =========================================================================


class TestApplySocialNormUpdate:
    """Tests for social norm update mechanics."""

    def test_peer_influence_direction(self) -> None:
        """Norms should move toward peer means."""
        current = np.array([0.0, 0.0, 0.0])
        peer_means = np.array([1.0, 1.0, 1.0])
        result = apply_social_norm_update(current, peer_means, gain=0.5)
        # Should move toward 1.0
        assert np.all(result > 0.0)
        assert np.all(result < 1.0)

    def test_zero_gain_near_unchanged(self) -> None:
        """Zero gain should mean only noise changes values (small)."""
        rng = np.random.default_rng(42)
        current = np.array([0.5, -0.3])
        peer_means = np.array([1.0, 1.0])
        result = apply_social_norm_update(current, peer_means, gain=0.0, noise_std=0.0, rng=rng)
        # With gain=0 and noise_std=0, should be unchanged
        np.testing.assert_allclose(result, current, atol=1e-10)

    def test_gain_is_clipped_to_unit_interval(self) -> None:
        """Peer pull gain > 1 should be clipped to 1."""
        current = np.array([0.1, 0.8, 0.4], dtype=np.float64)
        peer_means = np.array([0.9, 0.2, 0.6], dtype=np.float64)
        result = apply_social_norm_update(
            current,
            peer_means,
            gain=3.0,
            noise_std=0.0,
            rng=np.random.default_rng(1),
        )
        np.testing.assert_allclose(result, peer_means, atol=1e-12)

    def test_noise_is_mean_zero_per_update(self) -> None:
        """Bounded additive noise should preserve cross-agent mean exactly."""
        current = np.zeros(16, dtype=np.float64)
        peer_means = np.zeros(16, dtype=np.float64)
        result = apply_social_norm_update(
            current,
            peer_means,
            gain=0.0,
            noise_std=0.02,
            rng=np.random.default_rng(123),
        )
        assert float(np.mean(result)) == pytest.approx(0.0, abs=1e-12)


# =========================================================================
# Tests: apply_all_state_updates (dispatcher)
# =========================================================================


class TestApplyAllStateUpdates:
    """Tests for the master dispatcher."""

    def test_no_signals_no_change(self, agents_df: pl.DataFrame) -> None:
        """With no signals and no peer_means, state should be unchanged."""
        result = apply_all_state_updates(
            agents_df,
            signals=[],
            tick=0,
        )
        assert result.shape == agents_df.shape

    def test_single_pulse_signal(self, agents_df: pl.DataFrame) -> None:
        """Single pulse should modify the target column."""
        sig = SignalSpec(
            target_column="vaccine_risk_avg",
            intensity=1.0,
            signal_type="pulse",
            start_tick=0,
        )
        result = apply_all_state_updates(
            agents_df,
            signals=[sig],
            tick=0,
        )
        original = agents_df["vaccine_risk_avg"].to_numpy()
        updated = result["vaccine_risk_avg"].to_numpy()
        # All values should have changed
        assert not np.allclose(updated, original)

    def test_social_norm_with_peers(self, agents_df: pl.DataFrame) -> None:
        """Social norm update should respond to peer means."""
        peer_means = {
            "social_norms_vax_avg": np.array([1.0, 1.0, 1.0, 1.0]),
        }
        result = apply_all_state_updates(
            agents_df,
            signals=[],
            tick=0,
            peer_means=peer_means,
            k_social=1.0,
            social_weight_norm=0.5,
            social_weight_trust=0.0,
            social_weight_risk=0.0,
        )
        updated = result["social_norms_vax_avg"].to_numpy()
        # Composite remains bounded and defined.
        assert np.all(np.isfinite(updated))

    def test_k_social_zero_disables_peer_pull(self, agents_df: pl.DataFrame) -> None:
        peer_means = {"institutional_trust_avg": np.full(agents_df.height, 10.0, dtype=np.float64)}
        with_gain = apply_all_state_updates(
            agents_df,
            signals=[],
            tick=0,
            peer_means=peer_means,
            k_social=1.0,
            social_weight_norm=0.0,
            social_weight_trust=0.8,
            social_weight_risk=0.0,
        )
        no_gain = apply_all_state_updates(
            agents_df,
            signals=[],
            tick=0,
            peer_means=peer_means,
            k_social=0.0,
            social_weight_norm=0.0,
            social_weight_trust=0.8,
            social_weight_risk=0.0,
        )
        base = agents_df["institutional_trust_avg"].to_numpy()
        moved_with = float(np.mean(np.abs(with_gain["institutional_trust_avg"].to_numpy() - base)))
        moved_without = float(np.mean(np.abs(no_gain["institutional_trust_avg"].to_numpy() - base)))
        assert moved_with > moved_without

    def test_generic_peer_update_respects_zero_social_noise(self, agents_df: pl.DataFrame) -> None:
        peer_target = np.array([0.2, -0.1, 0.5, -0.4], dtype=np.float64)
        result = apply_all_state_updates(
            agents_df,
            signals=[],
            tick=0,
            peer_means={"institutional_trust_avg": peer_target},
            k_social=1.0,
            social_weight_norm=0.0,
            social_weight_trust=1.0,
            social_weight_risk=0.0,
            social_noise_std=0.0,
            state_relaxation_fast=0.0,
            state_relaxation_slow=0.0,
            rng=np.random.default_rng(99),
        )
        updated = result["institutional_trust_avg"].to_numpy()
        np.testing.assert_allclose(updated, peer_target, atol=1e-12)

    def test_descriptive_norm_updates_from_observed_behavior(self, agents_df: pl.DataFrame) -> None:
        observed = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        result = apply_all_state_updates(
            agents_df,
            signals=[],
            tick=0,
            observed_norm_share=observed,
            social_susceptibility=np.full(agents_df.height, 0.6),
            norm_threshold=np.full(agents_df.height, 0.4),
            social_noise_std=0.0,
            norm_channel_weights=NormChannelWeights(descriptive=0.8, injunctive=0.2),
        )
        descriptive = result["social_norms_descriptive_vax"].to_numpy()
        composite = result["social_norms_vax_avg"].to_numpy()
        assert np.all(descriptive >= 4.0)
        assert np.all(composite >= 4.0)

    def test_descriptive_norm_signal_applies_with_observed_updates(self, agents_df: pl.DataFrame) -> None:
        observed = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        sig = SignalSpec(
            target_column="social_norms_descriptive_vax",
            intensity=1.0,
            signal_type="pulse",
            start_tick=0,
        )
        no_signal = apply_all_state_updates(
            agents_df,
            signals=[],
            tick=0,
            observed_norm_share=observed,
            social_susceptibility=np.zeros(agents_df.height, dtype=np.float64),
            norm_threshold=np.full(agents_df.height, 0.5, dtype=np.float64),
            social_noise_std=0.0,
        )
        with_signal = apply_all_state_updates(
            agents_df,
            signals=[sig],
            tick=0,
            observed_norm_share=observed,
            social_susceptibility=np.zeros(agents_df.height, dtype=np.float64),
            norm_threshold=np.full(agents_df.height, 0.5, dtype=np.float64),
            social_noise_std=0.0,
        )
        baseline_desc = no_signal["social_norms_descriptive_vax"].to_numpy()
        shifted_desc = with_signal["social_norms_descriptive_vax"].to_numpy()
        assert np.allclose(baseline_desc, 4.0)
        assert np.all(shifted_desc > baseline_desc)

    def test_observed_norm_updates_latent_perceived_share(self, agents_df: pl.DataFrame) -> None:
        observed = np.ones(agents_df.height, dtype=np.float64)
        result = apply_all_state_updates(
            agents_df,
            signals=[],
            tick=0,
            observed_norm_share=observed,
            social_susceptibility=np.zeros(agents_df.height, dtype=np.float64),
            norm_threshold=np.full(agents_df.height, 0.5, dtype=np.float64),
            norm_slope=8.0,
            social_noise_std=0.0,
            state_relaxation_fast=0.0,
            state_relaxation_slow=0.5,
        )

        assert "norm_share_hat_vax" in result.columns
        share_hat = result["norm_share_hat_vax"].cast(pl.Float64).to_numpy().astype(np.float64)
        np.testing.assert_allclose(share_hat, np.full(agents_df.height, 0.75, dtype=np.float64), atol=1e-6)

    def test_observed_npi_norm_updates_separate_latent_share(self, agents_df: pl.DataFrame) -> None:
        observed_vax = np.zeros(agents_df.height, dtype=np.float64)
        observed_npi = np.ones(agents_df.height, dtype=np.float64)
        result = apply_all_state_updates(
            agents_df,
            signals=[],
            tick=0,
            observed_norm_share=observed_vax,
            observed_norm_share_npi=observed_npi,
            social_susceptibility=np.zeros(agents_df.height, dtype=np.float64),
            norm_threshold=np.full(agents_df.height, 0.5, dtype=np.float64),
            norm_slope=8.0,
            social_noise_std=0.0,
            state_relaxation_fast=0.0,
            state_relaxation_slow=0.5,
        )

        share_hat_vax = result["norm_share_hat_vax"].cast(pl.Float64).to_numpy().astype(np.float64)
        share_hat_npi = result["norm_share_hat_npi"].cast(pl.Float64).to_numpy().astype(np.float64)
        np.testing.assert_allclose(share_hat_vax, np.full(agents_df.height, 0.25, dtype=np.float64), atol=1e-6)
        np.testing.assert_allclose(share_hat_npi, np.full(agents_df.height, 0.75, dtype=np.float64), atol=1e-6)

    def test_signal_absolute_targets_pin_targeted_values(self, agents_df: pl.DataFrame) -> None:
        sig = SignalSpec(
            target_column="institutional_trust_avg",
            intensity=1.0,
            signal_type="campaign",
            start_tick=0,
            end_tick=2,
        )
        target_mask = np.array([True, False, True, False], dtype=bool)
        absolute_target = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        result = apply_all_state_updates(
            agents_df,
            signals=[sig],
            tick=0,
            target_mask=target_mask,
            signal_absolute_targets={"institutional_trust_avg": absolute_target},
            social_noise_std=0.0,
            state_relaxation_fast=0.0,
            state_relaxation_slow=0.0,
            rng=np.random.default_rng(123),
        )
        updated = result["institutional_trust_avg"].to_numpy().astype(np.float64)
        np.testing.assert_allclose(updated[target_mask], absolute_target[target_mask], atol=1e-12)

    def test_compute_observed_norm_fixed_point_share_is_inverse(self, agents_df: pl.DataFrame) -> None:
        agents_no_hat = agents_df.drop(["norm_share_hat_vax", "norm_share_hat_npi"])
        current = agents_no_hat["social_norms_descriptive_vax"].to_numpy().astype(np.float64)
        threshold = np.full(agents_df.height, 0.45, dtype=np.float64)
        fixed_share = compute_observed_norm_fixed_point_share(
            descriptive_norm=current,
            threshold=threshold,
            column="social_norms_descriptive_vax",
            slope=8.0,
        )

        fixed_point = apply_all_state_updates(
            agents_no_hat,
            signals=[],
            tick=0,
            observed_norm_share=fixed_share,
            social_susceptibility=np.ones(agents_df.height, dtype=np.float64),
            norm_threshold=threshold,
            k_social=1.0,
            social_noise_std=0.0,
            state_relaxation_fast=0.0,
            state_relaxation_slow=0.0,
        )
        updated = fixed_point["social_norms_descriptive_vax"].to_numpy().astype(np.float64)
        np.testing.assert_allclose(updated, current, atol=1e-6)

    def test_anchored_descriptive_norm_target_is_stationary_at_baseline(self, agents_df: pl.DataFrame) -> None:
        agents_no_hat = agents_df.drop(["norm_share_hat_vax", "norm_share_hat_npi"])
        current = agents_no_hat["social_norms_descriptive_vax"].to_numpy().astype(np.float64)
        threshold = np.full(agents_df.height, 0.45, dtype=np.float64)
        fixed_share = compute_observed_norm_fixed_point_share(
            descriptive_norm=current,
            threshold=threshold,
            column="social_norms_descriptive_vax",
            slope=8.0,
        )
        baseline_exposure = compute_observed_norm_exposure(
            observed_share=fixed_share,
            threshold=threshold,
            slope=8.0,
        )

        anchored = apply_all_state_updates(
            agents_no_hat,
            signals=[],
            tick=0,
            observed_norm_share=fixed_share,
            social_susceptibility=np.ones(agents_df.height, dtype=np.float64),
            norm_threshold=threshold,
            norm_slope=8.0,
            baseline_descriptive_norm=current,
            baseline_norm_exposure=baseline_exposure,
            k_social=1.0,
            social_noise_std=0.0,
            state_relaxation_fast=0.0,
            state_relaxation_slow=0.0,
        )
        updated = anchored["social_norms_descriptive_vax"].to_numpy().astype(np.float64)
        np.testing.assert_allclose(updated, current, atol=1e-6)

    def test_anchored_descriptive_norm_uses_baseline_share_hat_anchor(self, agents_df: pl.DataFrame) -> None:
        current = agents_df["social_norms_descriptive_vax"].to_numpy().astype(np.float64)
        threshold = np.full(agents_df.height, 0.5, dtype=np.float64)

        anchored = apply_all_state_updates(
            agents_df,
            signals=[],
            tick=0,
            observed_norm_share=np.full(agents_df.height, 0.5, dtype=np.float64),
            social_susceptibility=np.ones(agents_df.height, dtype=np.float64),
            norm_threshold=threshold,
            norm_slope=8.0,
            baseline_descriptive_norm=current,
            baseline_share_hat=np.full(agents_df.height, 0.5, dtype=np.float64),
            baseline_observed_share=np.full(agents_df.height, 0.8, dtype=np.float64),
            k_social=1.0,
            social_noise_std=0.0,
            state_relaxation_fast=0.0,
            state_relaxation_slow=0.0,
        )

        updated = anchored["social_norms_descriptive_vax"].to_numpy().astype(np.float64)
        np.testing.assert_allclose(updated, current, atol=1e-6)

    def test_composite_norm_not_directly_mutated(self, agents_df: pl.DataFrame) -> None:
        sig = SignalSpec(
            target_column="social_norms_vax_avg",
            intensity=2.0,
            signal_type="pulse",
            start_tick=0,
        )
        result = apply_all_state_updates(
            agents_df,
            signals=[sig],
            tick=0,
            social_noise_std=0.0,
            norm_channel_weights=NormChannelWeights(descriptive=0.7, injunctive=0.3),
        )

        desc = result["social_norms_descriptive_vax"].to_numpy()
        composite = result["social_norms_vax_avg"].to_numpy()
        expected = desc
        np.testing.assert_allclose(composite, expected)

    def test_fivec_pro_vax_recomputed_from_components(self) -> None:
        agents = pl.DataFrame(
            {
                "fivec_confidence": [7.0, 1.0],
                "fivec_low_complacency": [5.0, 3.0],
                "fivec_low_constraints": [3.0, 5.0],
                "fivec_collective_responsibility": [1.0, 7.0],
                "fivec_pro_vax_idx": [1.0, 1.0],
            }
        )
        result = apply_all_state_updates(
            agents,
            signals=[],
            tick=0,
        )
        expected = np.array([4.0, 4.0], dtype=np.float64)
        np.testing.assert_allclose(result["fivec_pro_vax_idx"].to_numpy(), expected)

    def test_fivec_pro_vax_not_directly_mutated(self) -> None:
        agents = pl.DataFrame(
            {
                "fivec_confidence": [6.0, 2.0],
                "fivec_low_complacency": [6.0, 2.0],
                "fivec_low_constraints": [2.0, 6.0],
                "fivec_collective_responsibility": [2.0, 6.0],
                "fivec_pro_vax_idx": [1.0, 7.0],
            }
        )
        sig = SignalSpec(
            target_column="fivec_pro_vax_idx",
            intensity=3.0,
            signal_type="pulse",
            start_tick=0,
        )
        result = apply_all_state_updates(
            agents,
            signals=[sig],
            tick=0,
        )
        expected = np.array([4.0, 4.0], dtype=np.float64)
        np.testing.assert_allclose(result["fivec_pro_vax_idx"].to_numpy(), expected)

    def test_pulse_recovers_toward_baseline(self, agents_df: pl.DataFrame) -> None:
        """Pulse should not accumulate indefinitely; state relaxes back toward baseline."""
        baseline = agents_df.clone()
        sig = SignalSpec(
            target_column="vaccine_risk_avg",
            intensity=1.0,
            signal_type="pulse",
            start_tick=0,
        )
        t0 = apply_all_state_updates(
            agents_df,
            signals=[sig],
            tick=0,
            baseline_agents=baseline,
        )
        t6 = apply_all_state_updates(
            t0,
            signals=[sig],
            tick=6,
            baseline_agents=baseline,
        )
        dist_t0 = np.abs(t0["vaccine_risk_avg"].to_numpy() - baseline["vaccine_risk_avg"].to_numpy()).mean()
        dist_t6 = np.abs(t6["vaccine_risk_avg"].to_numpy() - baseline["vaccine_risk_avg"].to_numpy()).mean()
        assert dist_t6 < dist_t0

    def test_impulse_only_applies_once(self, agents_df: pl.DataFrame) -> None:
        """Impulse signals should apply at start tick only."""
        baseline = agents_df.clone()
        sig = SignalSpec(
            target_column="vaccine_risk_avg",
            intensity=0.8,
            signal_type="impulse",
            start_tick=2,
        )
        t2 = apply_all_state_updates(
            agents_df,
            signals=[sig],
            tick=2,
            baseline_agents=baseline,
        )
        t3 = apply_all_state_updates(
            t2,
            signals=[sig],
            tick=3,
            baseline_agents=baseline,
        )
        delta_t2 = np.abs(t2["vaccine_risk_avg"].to_numpy() - agents_df["vaccine_risk_avg"].to_numpy()).mean()
        delta_t3 = np.abs(t3["vaccine_risk_avg"].to_numpy() - t2["vaccine_risk_avg"].to_numpy()).mean()
        assert delta_t2 > 0.0
        assert delta_t3 < delta_t2

    def test_relaxes_toward_attractor_target(self, agents_df: pl.DataFrame) -> None:
        target = np.full(agents_df.height, 1.5, dtype=np.float64)
        result = apply_all_state_updates(
            agents_df,
            signals=[],
            tick=0,
            attractor_targets={"institutional_trust_avg": target},
            state_relaxation_fast=0.2,
            state_relaxation_slow=0.2,
        )
        current = agents_df["institutional_trust_avg"].to_numpy().astype(np.float64)
        expected = current + 0.2 * (target - current)
        np.testing.assert_allclose(result["institutional_trust_avg"].to_numpy(), expected, atol=1e-12)

    def test_attractor_plus_offset_preserves_heterogeneity(self, agents_df: pl.DataFrame) -> None:
        target = np.zeros(agents_df.height, dtype=np.float64)
        offset = np.full(agents_df.height, 0.5, dtype=np.float64)
        result = apply_all_state_updates(
            agents_df,
            signals=[],
            tick=0,
            attractor_targets={"vaccine_risk_avg": target},
            endogenous_offsets={"vaccine_risk_avg": offset},
            state_relaxation_fast=0.1,
            state_relaxation_slow=0.2,
        )
        current = agents_df["vaccine_risk_avg"].to_numpy().astype(np.float64)
        expected = current + 0.2 * ((target + offset) - current)
        np.testing.assert_allclose(result["vaccine_risk_avg"].to_numpy(), expected, atol=1e-12)

    def test_endogenous_offsets_apply_without_other_active_inputs(self, agents_df: pl.DataFrame) -> None:
        """Endogenous offsets should trigger updates even with no signals/attractor."""
        offset = np.full(agents_df.height, 0.25, dtype=np.float64)
        result = apply_all_state_updates(
            agents_df,
            signals=[],
            tick=0,
            endogenous_offsets={"vaccine_risk_avg": offset},
            state_relaxation_fast=0.2,
            state_relaxation_slow=0.2,
            social_noise_std=0.0,
        )
        current = agents_df["vaccine_risk_avg"].to_numpy().astype(np.float64)
        expected = current + 0.2 * ((current + offset) - current)
        np.testing.assert_allclose(result["vaccine_risk_avg"].to_numpy(), expected, atol=1e-12)
