"""Tests for population initialisation."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from beam_abm.abm.data_contracts import AnchorSystem
from beam_abm.abm.population import (
    _axis_score,
    ensure_agent_schema,
    initialise_population,
    load_heterogeneity_loadings,
    sample_anchor_for_country,
    sample_donors,
)

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture()
def anchor_system() -> AnchorSystem:
    """Create a minimal anchor system for testing.

    AnchorSystem fields: metadata, n_anchors, country_shares,
    coords, neighbourhoods, memberships, features, pca_components, pca_mean.
    """
    memberships = pl.DataFrame(
        {
            "country": ["IT"] * 10 + ["DE"] * 10,
            "anchor_id": [0] * 6 + [1] * 4 + [0] * 5 + [1] * 5,
            "row_id": list(range(20)),
        }
    )

    neighbourhoods = pl.DataFrame(
        {
            "anchor_id": [0, 0, 0, 1, 1, 1] * 2,
            "country": ["IT"] * 6 + ["DE"] * 6,
            "row_id": [0, 1, 2, 6, 7, 8, 10, 11, 12, 15, 16, 17],
            "rank": [1, 2, 3, 1, 2, 3] * 2,
        }
    )

    # country_shares uses "share" column (not "pi")
    country_shares = pl.DataFrame(
        {
            "anchor_id": [0, 1, 0, 1],
            "country": ["IT", "IT", "DE", "DE"],
            "share": [0.6, 0.4, 0.5, 0.5],
        }
    )

    return AnchorSystem(
        metadata={"description": "test anchor system", "version": "0.1"},
        n_anchors=2,
        country_shares=country_shares,
        neighbourhoods=neighbourhoods,
        memberships=memberships,
    )


@pytest.fixture()
def survey_data() -> pl.DataFrame:
    """Create minimal survey data matching the anchor system row_ids."""
    n = 20
    rng = np.random.default_rng(42)
    data: dict[str, list[float] | list[str] | list[int]] = {
        "row_id": list(range(n)),
        "country": ["IT"] * 10 + ["DE"] * 10,
        "region_final": ["it_north"] * 5 + ["it_south"] * 5 + ["de_north"] * 5 + ["de_south"] * 5,
        "sex_female[T.True]": [1.0, 0.0] * 10,
        "age_bracket": [30.0] * n,
        "institutional_trust_avg": rng.normal(0, 1, n).tolist(),
        "vaccine_risk_avg": rng.normal(0, 1, n).tolist(),
        "fivec_confidence": rng.normal(0, 1, n).tolist(),
        "fivec_low_complacency": rng.normal(0, 1, n).tolist(),
        "disease_fear_avg": rng.normal(0, 1, n).tolist(),
        "covid_perceived_danger_Y": rng.normal(0, 1, n).tolist(),
        "covax_legitimacy_scepticism_idx": rng.normal(0, 1, n).tolist(),
        "fivec_collective_responsibility": rng.normal(0, 1, n).tolist(),
        "fivec_low_constraints": rng.normal(0, 1, n).tolist(),
        "fivec_pro_vax_idx": rng.normal(0, 1, n).tolist(),
        "social_norms_vax_avg": rng.uniform(1.0, 7.0, n).tolist(),
        "trust_traditional": rng.normal(0, 1, n).tolist(),
        "trust_onlineinfo": rng.normal(0, 1, n).tolist(),
        "trust_social": rng.normal(0, 1, n).tolist(),
        "duration_socialmedia": rng.normal(0, 1, n).tolist(),
        "duration_infomedia": rng.normal(0, 1, n).tolist(),
        "vax_convo_count_cohabitants": rng.integers(0, 4, n).astype(float).tolist(),
        "vax_convo_count_others": rng.integers(0, 5, n).astype(float).tolist(),
        "mfq_agreement_binding_idx": rng.normal(0, 1, n).tolist(),
        "mfq_relevance_binding_idx": rng.normal(0, 1, n).tolist(),
    }
    return pl.DataFrame(data)


# =========================================================================
# Tests: sample_anchor_for_country
# =========================================================================


class TestSampleAnchorForCountry:
    """Tests for anchor assignment sampling."""

    def test_returns_valid_anchors(self, anchor_system: AnchorSystem) -> None:
        """Should return anchor ids within the system."""
        rng = np.random.default_rng(42)
        anchors = sample_anchor_for_country(anchor_system, country="IT", n=100, rng=rng)
        assert len(anchors) == 100
        assert set(anchors.tolist()) <= {0, 1}

    def test_proportions_respect_shares(self, anchor_system: AnchorSystem) -> None:
        """Distribution should roughly match share weights."""
        rng = np.random.default_rng(42)
        anchors = sample_anchor_for_country(anchor_system, country="IT", n=10000, rng=rng)
        freq_0 = (anchors == 0).mean()
        # share for anchor 0 in IT is 0.6
        assert abs(freq_0 - 0.6) < 0.05

    def test_invalid_country_raises(self, anchor_system: AnchorSystem) -> None:
        """Unknown country should raise ValueError."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="No anchor shares"):
            sample_anchor_for_country(anchor_system, country="XX", n=10, rng=rng)


# =========================================================================
# Tests: sample_donors
# =========================================================================


class TestSampleDonors:
    """Tests for donor sampling from neighbourhoods."""

    def test_returns_dataframe(self, anchor_system: AnchorSystem, survey_data: pl.DataFrame) -> None:
        """Should return a pl.DataFrame with n rows."""
        rng = np.random.default_rng(42)
        anchor_ids = np.array([0, 0, 1, 1, 0])
        countries = np.array(["IT", "IT", "IT", "DE", "DE"])
        donors = sample_donors(anchor_ids, countries, anchor_system, survey_data, rng)
        assert isinstance(donors, pl.DataFrame)
        assert donors.shape[0] == 5

    def test_donors_have_row_id(self, anchor_system: AnchorSystem, survey_data: pl.DataFrame) -> None:
        """Returned donors should have 'row_id' column."""
        rng = np.random.default_rng(42)
        anchor_ids = np.array([0, 1])
        countries = np.array(["IT", "DE"])
        donors = sample_donors(anchor_ids, countries, anchor_system, survey_data, rng)
        assert "row_id" in donors.columns


# =========================================================================
# Tests: initialise_population
# =========================================================================


class TestInitialisePopulation:
    """Tests for full population initialisation."""

    @staticmethod
    def _spearman(x: np.ndarray, y: np.ndarray) -> float:
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            return 0.0
        x_rank = np.argsort(np.argsort(x[mask]))
        y_rank = np.argsort(np.argsort(y[mask]))
        x_rank = x_rank.astype(np.float64)
        y_rank = y_rank.astype(np.float64)
        if np.std(x_rank) <= 1e-12 or np.std(y_rank) <= 1e-12:
            return 0.0
        return float(np.corrcoef(x_rank, y_rank)[0, 1])

    def test_correct_size(self, anchor_system: AnchorSystem, survey_data: pl.DataFrame) -> None:
        """Population should have requested number of agents."""
        pop = initialise_population(
            n_agents=100,
            countries=["IT", "DE"],
            anchor_system=anchor_system,
            survey_df=survey_data,
            seed=42,
        )
        assert pop.shape[0] == 100

    def test_country_column_exists(self, anchor_system: AnchorSystem, survey_data: pl.DataFrame) -> None:
        """Population should have a 'country' column."""
        pop = initialise_population(
            n_agents=50,
            countries=["IT", "DE"],
            anchor_system=anchor_system,
            survey_df=survey_data,
            seed=42,
        )
        assert "country" in pop.columns

    def test_country_proportions_roughly_balanced(self, anchor_system: AnchorSystem, survey_data: pl.DataFrame) -> None:
        """Country proportions should reflect the share distribution."""
        pop = initialise_population(
            n_agents=1000,
            countries=["IT", "DE"],
            anchor_system=anchor_system,
            survey_df=survey_data,
            seed=42,
        )
        countries = pop["country"].to_list()
        it_frac = countries.count("IT") / len(countries)
        de_frac = countries.count("DE") / len(countries)
        # Both countries have similar total shares so expect ~50/50
        assert 0.3 < it_frac < 0.7
        assert 0.3 < de_frac < 0.7

    def test_mutable_columns_present(self, anchor_system: AnchorSystem, survey_data: pl.DataFrame) -> None:
        """Population should have mutable construct columns."""
        pop = initialise_population(
            n_agents=20,
            countries=["IT", "DE"],
            anchor_system=anchor_system,
            survey_df=survey_data,
            seed=42,
        )
        for col in [
            "institutional_trust_avg",
            "vaccine_risk_avg",
            "fivec_confidence",
            "social_norms_vax_avg",
        ]:
            assert col in pop.columns, f"Missing mutable column: {col}"

    def test_norm_channels_are_bounded(self, anchor_system: AnchorSystem, survey_data: pl.DataFrame) -> None:
        """Norm channels should be initialized within their declared bounds."""
        pop = initialise_population(
            n_agents=40,
            countries=["IT", "DE"],
            anchor_system=anchor_system,
            survey_df=survey_data,
            seed=123,
        )
        for col in ["social_norms_descriptive_vax", "social_norms_descriptive_npi"]:
            values = pop[col].to_numpy().astype(np.float64)
            assert float(values.min()) >= 1.0
            assert float(values.max()) <= 7.0

    def test_norm_share_hat_is_bounded_and_fixed_point(
        self,
        anchor_system: AnchorSystem,
        survey_data: pl.DataFrame,
    ) -> None:
        """Latent perceived adoption share should be coherent at initialisation."""
        pop = initialise_population(
            n_agents=120,
            countries=["IT", "DE"],
            anchor_system=anchor_system,
            survey_df=survey_data,
            seed=321,
        )

        share_hat = pop["norm_share_hat_vax"].cast(pl.Float64).to_numpy()
        theta = pop["norm_threshold_theta"].cast(pl.Float64).to_numpy()
        desc = pop["social_norms_descriptive_vax"].cast(pl.Float64).to_numpy()

        assert float(share_hat.min()) >= 0.0
        assert float(share_hat.max()) <= 1.0

        eps = 1e-6
        share = np.clip(share_hat, eps, 1.0 - eps)
        theta = np.clip(theta, eps, 1.0 - eps)
        share_logit = np.log(share / (1.0 - share))
        theta_logit = np.log(theta / (1.0 - theta))
        exposure = 1.0 / (1.0 + np.exp(-8.0 * (share_logit - theta_logit)))
        target = 1.0 + 6.0 * exposure
        interior = (share_hat > 1e-6) & (share_hat < 1.0 - 1e-6)
        if np.any(interior):
            np.testing.assert_allclose(target[interior], desc[interior], atol=1e-6)

        share_hat_npi = pop["norm_share_hat_npi"].cast(pl.Float64).to_numpy()
        desc_npi = pop["social_norms_descriptive_npi"].cast(pl.Float64).to_numpy()
        assert float(share_hat_npi.min()) >= 0.0
        assert float(share_hat_npi.max()) <= 1.0
        share_npi = np.clip(share_hat_npi, eps, 1.0 - eps)
        share_npi_logit = np.log(share_npi / (1.0 - share_npi))
        exposure_npi = 1.0 / (1.0 + np.exp(-8.0 * (share_npi_logit - theta_logit)))
        target_npi = 1.0 + 6.0 * exposure_npi
        interior_npi = (share_hat_npi > 1e-6) & (share_hat_npi < 1.0 - 1e-6)
        if np.any(interior_npi):
            np.testing.assert_allclose(target_npi[interior_npi], desc_npi[interior_npi], atol=1e-6)

    def test_reproducibility(self, anchor_system: AnchorSystem, survey_data: pl.DataFrame) -> None:
        """Same seed should produce identical populations."""
        pop1 = initialise_population(50, ["IT", "DE"], anchor_system, survey_data, seed=123)
        pop2 = initialise_population(50, ["IT", "DE"], anchor_system, survey_data, seed=123)
        assert pop1.equals(pop2)

    def test_heterogeneity_bounds_respected(self, anchor_system: AnchorSystem, survey_data: pl.DataFrame) -> None:
        """Structural heterogeneity traits should stay within fixed bounds."""
        pop = initialise_population(200, ["IT", "DE"], anchor_system, survey_data, seed=5)
        lam = pop["social_susceptibility_lambda"].cast(pl.Float64).to_numpy()
        assert np.isfinite(lam).all()
        assert float(lam.min()) > 0.0
        assert abs(float(lam.mean()) - 1.0) < 1e-6

        bounds = {
            "social_attention_alpha": (0.1, 4.0),
            "social_sharing_phi": (0.05, 5.0),
            "norm_threshold_theta": (0.2, 0.8),
        }
        for col, (lo, hi) in bounds.items():
            values = pop[col].cast(pl.Float64).to_numpy()
            assert float(values.min()) >= lo
            assert float(values.max()) <= hi

    def test_heterogeneity_columns_present_and_finite(
        self, anchor_system: AnchorSystem, survey_data: pl.DataFrame
    ) -> None:
        """Structural heterogeneity traits should exist and be finite."""
        pop = initialise_population(120, ["IT", "DE"], anchor_system, survey_data, seed=9)
        cols = [
            "social_susceptibility_lambda",
            "social_attention_alpha",
            "social_sharing_phi",
            "norm_threshold_theta",
        ]
        for col in cols:
            assert col in pop.columns
            values = pop[col].cast(pl.Float64).to_numpy()
            assert np.isfinite(values).all()

    def test_heterogeneity_monotonic_signs(self, anchor_system: AnchorSystem, survey_data: pl.DataFrame) -> None:
        """CDF-mapped traits should preserve expected sign relationships (Spearman)."""
        pop = initialise_population(500, ["IT", "DE"], anchor_system, survey_data, seed=8)
        loadings = load_heterogeneity_loadings()

        social = _axis_score(pop, loadings.social_activity, pop.height)
        media = _axis_score(pop, loadings.media_exposure, pop.height)
        convo_total = (
            pop["vax_convo_count_friends"].cast(pl.Float64).to_numpy()
            + pop["vax_convo_count_cohabitants"].cast(pl.Float64).to_numpy()
            + pop["vax_convo_count_others"].cast(pl.Float64).to_numpy()
        )

        lam = pop["social_susceptibility_lambda"].cast(pl.Float64).to_numpy()
        phi = pop["social_sharing_phi"].cast(pl.Float64).to_numpy()

        assert self._spearman(lam, social + media) > 0.0
        assert self._spearman(phi, convo_total) > 0.0

    def test_norm_threshold_beta_alpha_controls_dispersion(
        self, anchor_system: AnchorSystem, survey_data: pl.DataFrame
    ) -> None:
        """Larger Beta alpha should concentrate norm thresholds near the midpoint."""
        pop_low = initialise_population(
            1500,
            ["IT", "DE"],
            anchor_system,
            survey_data,
            seed=21,
            norm_threshold_beta_alpha=1.0,
        )
        pop_high = initialise_population(
            1500,
            ["IT", "DE"],
            anchor_system,
            survey_data,
            seed=21,
            norm_threshold_beta_alpha=8.0,
        )
        theta_low = pop_low["norm_threshold_theta"].cast(pl.Float64).to_numpy()
        theta_high = pop_high["norm_threshold_theta"].cast(pl.Float64).to_numpy()

        midpoint = 0.5
        assert abs(float(theta_low.mean()) - midpoint) < 0.03
        assert abs(float(theta_high.mean()) - midpoint) < 0.03
        assert float(theta_high.std()) < float(theta_low.std())

    def test_heterogeneity_cdf_knot_integrity(self) -> None:
        """CDF knots should exist for all drivers and be strictly increasing."""
        loadings = load_heterogeneity_loadings()
        knots = loadings.cdf_knots
        expected = {"lambda_driver", "alpha_driver", "phi_driver", "theta_driver"}
        assert expected.issubset(set(knots.score_knots))

        for key in expected:
            values = np.array(knots.score_knots[key], dtype=np.float64)
            assert np.isfinite(values).all()
            ordered = np.sort(values)
            assert np.all(np.diff(ordered) > 0.0)

    def test_locality_id_uses_region(self, anchor_system: AnchorSystem, survey_data: pl.DataFrame) -> None:
        """Locality IDs should be built from region-seeded country localities."""
        pop = initialise_population(
            n_agents=80,
            countries=["IT", "DE"],
            anchor_system=anchor_system,
            survey_df=survey_data,
            seed=7,
        )
        communities = set(pop["locality_id"].to_list())
        assert all(c.startswith("IT_loc") or c.startswith("DE_loc") for c in communities)

    def test_region_validation_raises_when_blank(self, anchor_system: AnchorSystem, survey_data: pl.DataFrame) -> None:
        """Population init should fail if region cannot define communities."""
        invalid = survey_data.with_columns(pl.lit("").alias("region_final"))
        with pytest.raises(ValueError, match="null/blank region"):
            initialise_population(
                n_agents=20,
                countries=["IT", "DE"],
                anchor_system=anchor_system,
                survey_df=invalid,
                seed=42,
            )

    def test_locality_sizes_bounded_for_large_country_samples(
        self,
        anchor_system: AnchorSystem,
        survey_data: pl.DataFrame,
    ) -> None:
        """For sufficiently large countries, locality sizes should be bounded."""
        pop = initialise_population(
            n_agents=1000,
            countries=["IT", "DE"],
            anchor_system=anchor_system,
            survey_df=survey_data,
            seed=123,
        )
        sizes = pop.group_by(["country", "locality_id"]).len().rename({"len": "n"})
        country_n = pop.group_by("country").len().rename({"len": "country_n"})
        checked = sizes.join(country_n, on="country", how="left").filter(pl.col("country_n") >= 100)
        if checked.height > 0:
            vals = checked["n"].to_numpy().astype(np.int64)
            assert int(vals.min()) >= 100
            assert int(vals.max()) <= 200


def test_ensure_agent_schema_backfills_norm_channels_from_baseline_norm() -> None:
    """Schema completion should not leave split norm channels at 0."""
    agents = pl.DataFrame(
        {
            "row_id": [0, 1, 2],
            "social_norms_vax_avg": [2.5, 5.5, 6.5],
        }
    )
    out = ensure_agent_schema(agents, n_agents=3)
    desc = out["social_norms_descriptive_vax"].to_numpy().astype(np.float64)
    assert np.allclose(desc, np.array([2.5, 5.5, 6.5]))
