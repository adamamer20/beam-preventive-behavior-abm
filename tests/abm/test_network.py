"""Tests for the locality-based network module."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from beam_abm.abm.network import (
    NetworkState,
    build_contact_network,
    compute_homophily_weighted_peer_means,
    compute_pairwise_symmetric_peer_pull,
    compute_peer_means,
    compute_sampled_peer_adoption,
)
from beam_abm.abm.state_schema import MUTABLE_COLUMNS


def _make_population(n: int = 30, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    locality = ["IT_r1"] * (n // 3) + ["IT_r2"] * (n // 3) + ["DE_r1"] * (n - 2 * (n // 3))
    data: dict[str, list[float] | list[str] | list[int]] = {
        "country": ["IT"] * (2 * (n // 3)) + ["DE"] * (n - 2 * (n // 3)),
        "locality_id": locality,
        "group_size_score": rng.uniform(0.0, 1.0, n).tolist(),
        "core_degree": rng.integers(4, 9, size=n).tolist(),
        "household_size": rng.integers(1, 5, size=n).tolist(),
    }
    for col in sorted(MUTABLE_COLUMNS):
        data[col] = rng.normal(0, 1, n).tolist()
    return pl.DataFrame(data)


@pytest.fixture()
def small_population() -> pl.DataFrame:
    return _make_population(30, seed=42)


@pytest.fixture()
def network(small_population: pl.DataFrame) -> NetworkState:
    return build_contact_network(small_population, bridge_prob=0.15, seed=42)


class TestBuildContactNetwork:
    def test_core_degree_length(self, network: NetworkState) -> None:
        assert network.core_degree.shape == (30,)

    def test_locality_members_partition(self, network: NetworkState) -> None:
        all_indices: set[int] = set()
        for indices in network.locality_members.values():
            all_indices.update(indices.tolist())
        assert all_indices == set(range(30))

    def test_no_self_loops_in_core(self, network: NetworkState) -> None:
        for i, nbrs in enumerate(network.core_neighbors):
            assert i not in set(nbrs.tolist())

    def test_household_neighbors_no_self(self, network: NetworkState) -> None:
        for i, nbrs in enumerate(network.household_neighbors):
            assert i not in set(nbrs.tolist())


class TestComputePeerMeans:
    def test_returns_dict(self, network: NetworkState, small_population: pl.DataFrame) -> None:
        means = compute_peer_means(small_population, network)
        assert isinstance(means, dict)
        assert len(means) > 0

    def test_correct_shape(self, network: NetworkState, small_population: pl.DataFrame) -> None:
        means = compute_peer_means(small_population, network, columns=["institutional_trust_avg"])
        assert means["institutional_trust_avg"].shape == (30,)

    def test_uniform_values_give_same_mean(self, network: NetworkState, small_population: pl.DataFrame) -> None:
        col = "institutional_trust_avg"
        uniform_pop = small_population.with_columns(pl.lit(2.5).alias(col))
        means = compute_peer_means(uniform_pop, network, columns=[col])
        np.testing.assert_allclose(means[col], 2.5, atol=1e-6)


class TestHomophilyWeightedPeerMeans:
    def test_returns_requested_constructs(self, network: NetworkState, small_population: pl.DataFrame) -> None:
        means, avg_similarity = compute_homophily_weighted_peer_means(
            small_population,
            network,
            columns=("institutional_trust_avg", "vaccine_risk_avg"),
            homophily_gamma=1.5,
            homophily_w_trust=0.5,
            homophily_w_moral=0.5,
        )
        assert set(means) == {"institutional_trust_avg", "vaccine_risk_avg"}
        assert means["institutional_trust_avg"].shape == (30,)
        assert means["vaccine_risk_avg"].shape == (30,)
        assert 0.0 <= avg_similarity <= 1.0


class TestPairwiseSymmetricPeerPull:
    def test_preserves_population_mean(self, network: NetworkState, small_population: pl.DataFrame) -> None:
        col = "institutional_trust_avg"
        before = small_population[col].cast(pl.Float64).to_numpy().astype(np.float64)
        updated, similarity = compute_pairwise_symmetric_peer_pull(
            small_population,
            network,
            columns=(col,),
            gains_by_column={col: 0.4},
            homophily_gamma=1.5,
            homophily_w_trust=0.5,
            homophily_w_moral=0.5,
            rng=np.random.default_rng(991),
            interactions_per_agent=1.0,
        )
        assert col in updated
        after = updated[col]
        assert np.isfinite(after).all()
        assert np.isclose(float(np.mean(after)), float(np.mean(before)), atol=1e-10)
        assert 0.0 <= similarity <= 1.0

    def test_zero_gain_returns_unchanged_columns(self, network: NetworkState, small_population: pl.DataFrame) -> None:
        col = "social_norms_descriptive_vax"
        before = small_population[col].cast(pl.Float64).to_numpy().astype(np.float64)
        updated, _ = compute_pairwise_symmetric_peer_pull(
            small_population,
            network,
            columns=(col,),
            gains_by_column={col: 0.0},
            homophily_gamma=1.5,
            homophily_w_trust=0.5,
            homophily_w_moral=0.5,
            rng=np.random.default_rng(992),
            interactions_per_agent=1.0,
        )
        assert col not in updated
        np.testing.assert_allclose(before, small_population[col].cast(pl.Float64).to_numpy().astype(np.float64))

    def test_per_agent_interactions_preserve_population_mean(
        self, network: NetworkState, small_population: pl.DataFrame
    ) -> None:
        col = "institutional_trust_avg"
        before = small_population[col].cast(pl.Float64).to_numpy().astype(np.float64)
        n = small_population.height
        per_agent = np.full(n, 0.1, dtype=np.float64)
        per_agent[: n // 2] = 2.0

        updated, similarity = compute_pairwise_symmetric_peer_pull(
            small_population,
            network,
            columns=(col,),
            gains_by_column={col: 0.4},
            homophily_gamma=1.5,
            homophily_w_trust=0.5,
            homophily_w_moral=0.5,
            rng=np.random.default_rng(993),
            interactions_per_agent=0.05,
            per_agent_interactions=per_agent,
        )
        assert col in updated
        after = updated[col]
        assert np.isfinite(after).all()
        assert np.isclose(float(np.mean(after)), float(np.mean(before)), atol=1e-10)
        assert 0.0 <= similarity <= 1.0


class TestSampledPeerAdoption:
    def test_sample_size_controls_mean_contacts(self, network: NetworkState) -> None:
        n = network.locality_of_agent.shape[0]
        outcomes = {"mask_when_symptomatic_crowded": np.linspace(1.0, 7.0, n, dtype=np.float64)}
        attention = np.full(n, 2.0, dtype=np.float64)
        sharing = np.full(n, 2.0, dtype=np.float64)
        activity = np.full(n, 1.0, dtype=np.float64)

        _, diag_small = compute_sampled_peer_adoption(
            network,
            outcomes,
            sample_size=1,
            rng=np.random.default_rng(123),
            attention=attention,
            sharing=sharing,
            contact_rate_base=4.0,
            contact_rate_multiplier=1.0,
            activity_score=activity,
        )
        _, diag_large = compute_sampled_peer_adoption(
            network,
            outcomes,
            sample_size=8,
            rng=np.random.default_rng(123),
            attention=attention,
            sharing=sharing,
            contact_rate_base=4.0,
            contact_rate_multiplier=1.0,
            activity_score=activity,
        )
        assert diag_large.mean_contacts > diag_small.mean_contacts

    def test_attention_changes_observed_share(self, network: NetworkState) -> None:
        n = network.locality_of_agent.shape[0]
        base = np.linspace(1.0, 7.0, n, dtype=np.float64)
        outcomes = {"mask_when_pressure_high": base}
        sharing = np.full(n, 1.0, dtype=np.float64)
        activity = np.full(n, 1.0, dtype=np.float64)

        observed_low, _ = compute_sampled_peer_adoption(
            network,
            outcomes,
            sample_size=6,
            rng=np.random.default_rng(321),
            attention=np.full(n, 0.05, dtype=np.float64),
            sharing=sharing,
            contact_rate_base=3.5,
            contact_rate_multiplier=1.0,
            activity_score=activity,
        )
        observed_high, _ = compute_sampled_peer_adoption(
            network,
            outcomes,
            sample_size=6,
            rng=np.random.default_rng(321),
            attention=np.full(n, 5.0, dtype=np.float64),
            sharing=sharing,
            contact_rate_base=3.5,
            contact_rate_multiplier=1.0,
            activity_score=activity,
        )
        self_ref = np.clip((base - 1.0) / 6.0, 0.0, 1.0)
        delta_low = float(np.mean(np.abs(observed_low - self_ref)))
        delta_high = float(np.mean(np.abs(observed_high - self_ref)))
        assert delta_high > delta_low

    def test_per_agent_sample_caps_change_contact_intensity(self, network: NetworkState) -> None:
        n = network.locality_of_agent.shape[0]
        outcomes = {"mask_when_symptomatic_crowded": np.linspace(1.0, 7.0, n, dtype=np.float64)}
        attention = np.full(n, 2.0, dtype=np.float64)
        sharing = np.full(n, 2.0, dtype=np.float64)
        activity = np.full(n, 1.0, dtype=np.float64)

        _, diag_low_cap = compute_sampled_peer_adoption(
            network,
            outcomes,
            sample_size=8,
            rng=np.random.default_rng(777),
            attention=attention,
            sharing=sharing,
            contact_rate_base=4.0,
            contact_rate_multiplier=1.0,
            activity_score=activity,
            per_agent_sample_size=np.full(n, 1, dtype=np.int64),
        )
        _, diag_high_cap = compute_sampled_peer_adoption(
            network,
            outcomes,
            sample_size=8,
            rng=np.random.default_rng(777),
            attention=attention,
            sharing=sharing,
            contact_rate_base=4.0,
            contact_rate_multiplier=1.0,
            activity_score=activity,
            per_agent_sample_size=np.full(n, 8, dtype=np.int64),
        )

        assert diag_high_cap.mean_contacts > diag_low_cap.mean_contacts

    def test_ordinal_outcomes_are_scaled_not_clipped(self, network: NetworkState) -> None:
        n = network.locality_of_agent.shape[0]
        ordinal = np.linspace(1.0, 7.0, n, dtype=np.float64)
        observed, _ = compute_sampled_peer_adoption(
            network,
            outcomes={"mask_when_pressure_high": ordinal},
            sample_size=4,
            rng=np.random.default_rng(456),
            contact_rate_base=0.0,
            contact_rate_multiplier=1.0,
        )
        expected = np.clip((ordinal - 1.0) / 6.0, 0.0, 1.0)
        np.testing.assert_allclose(observed, expected, atol=1e-12)
