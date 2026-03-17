"""Tests for the choice module — fixed prediction functions."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from beam_abm.abm.data_contracts import BackboneModel, LeversConfig, OutcomeLever
from beam_abm.abm.message_contracts import NormChannelWeights

# Import sigmoid directly (not in __all__ but still accessible)
from beam_abm.abm.outcomes import (
    assemble_design_matrix,
    linear_predictor,
    predict_all_outcomes,
    predict_binary,
    predict_ordinal,
    predict_outcome,
    sigmoid,
)

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture()
def binary_model() -> BackboneModel:
    """Create a minimal binary logistic model."""
    return BackboneModel(
        name="test_binary",
        model_type="binary",
        terms=("institutional_trust_avg", "fivec_confidence"),
        coefficients=np.array([0.5, -0.3]),
        thresholds=None,
        intercept=-0.2,
        blocks={
            "trust": ["institutional_trust_avg"],
            "5c": ["fivec_confidence"],
        },
    )


@pytest.fixture()
def ordinal_model() -> BackboneModel:
    """Create a minimal ordinal logistic model (5 categories)."""
    return BackboneModel(
        name="test_ordinal",
        model_type="ordinal",
        terms=(
            "institutional_trust_avg",
            "fivec_confidence",
            "vaccine_risk_avg",
        ),
        coefficients=np.array([1.0, -0.5, 0.3]),
        thresholds=np.array([-1.5, -0.5, 0.5, 1.5]),
        intercept=None,
        blocks={
            "trust": ["institutional_trust_avg"],
            "5c": ["fivec_confidence"],
            "risk": ["vaccine_risk_avg"],
        },
    )


@pytest.fixture()
def country_model() -> BackboneModel:
    """Model with country dummies."""
    return BackboneModel(
        name="test_country",
        model_type="binary",
        terms=(
            "institutional_trust_avg",
            "C(country, Treatment(reference='IT'))[T.DE]",
            "C(country, Treatment(reference='IT'))[T.UK]",
        ),
        coefficients=np.array([0.5, 0.3, -0.2]),
        thresholds=None,
        intercept=0.0,
        blocks={},
    )


@pytest.fixture()
def sample_agents() -> pl.DataFrame:
    """Create a small agent DataFrame for testing."""
    return pl.DataFrame(
        {
            "institutional_trust_avg": [0.5, -0.3, 1.0, 0.0],
            "fivec_confidence": [0.2, 0.8, -0.5, 0.0],
            "vaccine_risk_avg": [0.1, -0.1, 0.3, -0.2],
            "country": ["IT", "DE", "UK", "IT"],
        }
    )


@pytest.fixture()
def scaled_binary_model() -> BackboneModel:
    """Binary model with fit-time scaling metadata."""
    return BackboneModel(
        name="test_scaled_binary",
        model_type="binary",
        terms=("income_ppp_norm",),
        coefficients=np.array([0.05]),
        thresholds=None,
        intercept=-3.0,
        blocks={},
        term_means=np.array([2300.0]),
        term_stds=np.array([1100.0]),
    )


# =========================================================================
# Tests: sigmoid
# =========================================================================


class TestSigmoid:
    """Tests for the sigmoid function."""

    def test_sigmoid_at_zero(self) -> None:
        """sigmoid(0) = 0.5."""
        assert np.isclose(sigmoid(np.array([0.0])), 0.5)

    def test_sigmoid_large_positive(self) -> None:
        """sigmoid(large) ≈ 1."""
        assert sigmoid(np.array([100.0]))[0] > 0.999

    def test_sigmoid_large_negative(self) -> None:
        """sigmoid(-large) ≈ 0."""
        assert sigmoid(np.array([-100.0]))[0] < 0.001

    def test_sigmoid_symmetry(self) -> None:
        """sigmoid(x) + sigmoid(-x) = 1."""
        x = np.array([1.5, -2.0, 0.7])
        assert np.allclose(sigmoid(x) + sigmoid(-x), 1.0)


# =========================================================================
# Tests: linear_predictor
# =========================================================================


class TestLinearPredictor:
    """Tests for linear_predictor."""

    def test_basic(self) -> None:
        """Check manual calculation."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        beta = np.array([0.5, -0.3])
        result = linear_predictor(X, beta, intercept=0.1)
        expected = np.array([0.1 + 0.5 - 0.6, 0.1 + 1.5 - 1.2])
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_zero_intercept(self) -> None:
        """Zero intercept should give X @ beta."""
        X = np.array([[1.0, 0.0]])
        beta = np.array([2.0, 3.0])
        result = linear_predictor(X, beta, intercept=0.0)
        assert np.isclose(result[0], 2.0)


# =========================================================================
# Tests: predict_binary
# =========================================================================


class TestPredictBinary:
    """Tests for predict_binary."""

    def test_deterministic_returns_probabilities(self) -> None:
        """Deterministic mode should return P(Y=1)."""
        eta = np.array([0.0, 2.0, -2.0])
        probs = predict_binary(eta, deterministic=True)
        assert probs.shape == (3,)
        assert np.isclose(probs[0], 0.5)
        assert probs[1] > 0.8
        assert probs[2] < 0.2

    def test_stochastic_returns_binary(self) -> None:
        """Stochastic mode should return 0/1 values."""
        eta = np.zeros(1000)
        rng = np.random.default_rng(42)
        sampled = predict_binary(eta, deterministic=False, rng=rng)
        assert set(np.unique(sampled)) <= {0.0, 1.0}
        # Mean should be close to 0.5
        assert 0.4 < sampled.mean() < 0.6

    def test_reproducibility(self) -> None:
        """Same seed should give same results."""
        eta = np.linspace(-2, 2, 100)
        r1 = predict_binary(eta, rng=np.random.default_rng(99))
        r2 = predict_binary(eta, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(r1, r2)

    def test_runtime_scaling_prevents_ceiling_collapse(self, scaled_binary_model: BackboneModel) -> None:
        """Fit-time scaling metadata should be applied before scoring."""
        agents = pl.DataFrame({"income_ppp_norm": [500.0, 1500.0, 2500.0, 3500.0, 4500.0]})
        X = assemble_design_matrix(agents, scaled_binary_model)
        eta = linear_predictor(X, scaled_binary_model.coefficients, intercept=scaled_binary_model.intercept or 0.0)
        probs = predict_binary(eta, deterministic=True)
        # Without scaling this setup is near-certain (eta >> 0). With scaling
        # it remains in a non-saturated regime and preserves response spread.
        assert probs.max() < 0.5
        assert probs.min() > 0.0
        assert probs.std() > 0.002


# =========================================================================
# Tests: predict_ordinal
# =========================================================================


class TestPredictOrdinal:
    """Tests for predict_ordinal."""

    def test_deterministic_expected_value(self) -> None:
        """Expected values should be within the ordinal range."""
        thresholds = np.array([-1.0, 0.0, 1.0])  # 4 categories
        eta = np.array([0.0, 2.0, -2.0])
        ev = predict_ordinal(eta, thresholds, deterministic=True)
        assert ev.shape == (3,)
        # Expected values for 4 categories → range [1, 4]
        assert np.all(ev >= 1.0)
        assert np.all(ev <= 4.0)

    def test_stochastic_returns_integers(self) -> None:
        """Stochastic mode should return integer categories."""
        thresholds = np.array([-1.0, 0.0, 1.0])
        eta = np.zeros(1000)
        rng = np.random.default_rng(42)
        sampled = predict_ordinal(eta, thresholds, deterministic=False, rng=rng)
        assert set(np.unique(sampled)) <= {1.0, 2.0, 3.0, 4.0}

    def test_higher_eta_shifts_distribution(self) -> None:
        """Higher eta should shift expected values upward."""
        thresholds = np.array([-1.0, 0.0, 1.0, 2.0])
        eta_low = np.full(500, -2.0)
        eta_high = np.full(500, 2.0)
        ev_low = predict_ordinal(eta_low, thresholds, deterministic=True)
        ev_high = predict_ordinal(eta_high, thresholds, deterministic=True)
        assert ev_high.mean() > ev_low.mean()

    def test_probability_sum_to_one(self) -> None:
        """Category probabilities should sum to 1 with monotonic thresholds."""
        thresholds = np.array([-1.5, -0.5, 0.5, 1.5])  # Monotonically increasing
        eta = np.array([0.3])
        cum_probs = sigmoid(thresholds[np.newaxis, :] - eta[:, np.newaxis])
        n_cat = len(thresholds) + 1
        cat_probs = np.empty((1, n_cat))
        cat_probs[0, 0] = cum_probs[0, 0]
        for j in range(1, n_cat - 1):
            cat_probs[0, j] = cum_probs[0, j] - cum_probs[0, j - 1]
        cat_probs[0, n_cat - 1] = 1.0 - cum_probs[0, n_cat - 2]
        cat_probs = np.clip(cat_probs, 0, 1)
        assert np.isclose(cat_probs.sum(), 1.0, atol=0.01)


# =========================================================================
# Tests: assemble_design_matrix
# =========================================================================


class TestAssembleDesignMatrix:
    """Tests for design matrix assembly."""

    def test_basic_columns(self, binary_model: BackboneModel, sample_agents: pl.DataFrame) -> None:
        """Design matrix should have correct shape."""
        X = assemble_design_matrix(sample_agents, binary_model)
        assert X.shape == (4, 2)

    def test_country_dummies(self, country_model: BackboneModel, sample_agents: pl.DataFrame) -> None:
        """Country dummies should be correctly encoded."""
        X = assemble_design_matrix(sample_agents, country_model)
        assert X.shape == (4, 3)
        # Agent 0 is IT (reference) → dummy DE=0, UK=0
        assert X[0, 1] == 0.0 and X[0, 2] == 0.0
        # Agent 1 is DE → dummy DE=1, UK=0
        assert X[1, 1] == 1.0 and X[1, 2] == 0.0
        # Agent 2 is UK → dummy DE=0, UK=1
        assert X[2, 1] == 0.0 and X[2, 2] == 1.0

    def test_missing_column_raises(self, binary_model: BackboneModel) -> None:
        """Missing columns should default to 0 or raise."""
        bad_agents = pl.DataFrame({"institutional_trust_avg": [1.0], "x99": [0.5]})
        # fivec_confidence is missing — implementation may fill with 0
        # or raise; either is acceptable
        try:
            X = assemble_design_matrix(bad_agents, binary_model)
            # If it didn't raise, the missing col defaulted to 0
            assert X.shape == (1, 2)
        except (ValueError, KeyError):
            pass  # Also acceptable


# =========================================================================
# Tests: predict_outcome (end-to-end)
# =========================================================================


class TestPredictOutcome:
    """End-to-end prediction tests."""

    def test_binary_prediction(self, binary_model: BackboneModel, sample_agents: pl.DataFrame) -> None:
        """Binary prediction should return values in [0, 1]."""
        result = predict_outcome(sample_agents, binary_model, deterministic=True)
        assert result.shape == (4,)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_ordinal_prediction(self, ordinal_model: BackboneModel, sample_agents: pl.DataFrame) -> None:
        """Ordinal prediction should return values in category range."""
        result = predict_outcome(sample_agents, ordinal_model, deterministic=True)
        assert result.shape == (4,)
        # 5 categories → range [1, 5]
        assert np.all(result >= 1.0)
        assert np.all(result <= 5.0)


def test_norm_split_composite_reconstruction() -> None:
    agents = pl.DataFrame(
        {
            "social_norms_descriptive_vax": [6.0, 2.0],
            "fivec_confidence": [4.0, 4.0],
            "fivec_low_complacency": [4.0, 4.0],
            "fivec_low_constraints": [4.0, 4.0],
            "fivec_collective_responsibility": [4.0, 4.0],
        }
    )
    model = BackboneModel(
        name="norm_only",
        model_type="binary",
        terms=("social_norms_vax_avg",),
        coefficients=np.array([1.0]),
        thresholds=None,
        intercept=-4.0,
        blocks={},
    )
    levers = LeversConfig(
        outcomes={
            "flu_vaccinated_2023_2024": OutcomeLever(
                outcome="flu_vaccinated_2023_2024",
                model_ref="norm_only",
                levers=("social_norms_vax_avg",),
                include_cols=("social_norms_vax_avg",),
            )
        }
    )

    out_desc = predict_all_outcomes(
        agents,
        {"flu_vaccinated_2023_2024": model},
        levers,
        deterministic=True,
        norm_channel_weights=NormChannelWeights(descriptive=1.0, injunctive=0.0),
    )["flu_vaccinated_2023_2024"]

    out_inj = predict_all_outcomes(
        agents,
        {"flu_vaccinated_2023_2024": model},
        levers,
        deterministic=True,
        norm_channel_weights=NormChannelWeights(descriptive=0.0, injunctive=1.0),
    )["flu_vaccinated_2023_2024"]

    np.testing.assert_allclose(out_desc, out_inj)


def test_fivec_pro_vax_composite_recomputed_for_choice() -> None:
    agents = pl.DataFrame(
        {
            "fivec_confidence": [7.0, 1.0],
            "fivec_low_complacency": [7.0, 1.0],
            "fivec_low_constraints": [1.0, 7.0],
            "fivec_collective_responsibility": [1.0, 7.0],
            # Deliberately inconsistent with component means.
            "fivec_pro_vax_idx": [1.0, 7.0],
            "social_norms_descriptive_vax": [4.0, 4.0],
        }
    )
    model = BackboneModel(
        name="fivec_only",
        model_type="binary",
        terms=("fivec_pro_vax_idx",),
        coefficients=np.array([1.0]),
        thresholds=None,
        intercept=-4.0,
        blocks={},
    )
    levers = LeversConfig(
        outcomes={
            "flu_vaccinated_2023_2024": OutcomeLever(
                outcome="flu_vaccinated_2023_2024",
                model_ref="fivec_only",
                levers=("fivec_pro_vax_idx",),
                include_cols=("fivec_pro_vax_idx",),
            )
        }
    )

    out = predict_all_outcomes(
        agents,
        {"flu_vaccinated_2023_2024": model},
        levers,
        deterministic=True,
    )["flu_vaccinated_2023_2024"]

    # Both rows have component mean = 4.0, so predictions must match.
    assert np.isclose(out[0], out[1])
