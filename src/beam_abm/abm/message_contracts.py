"""Typed contracts for social-channel mappings and norm composition."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "NORM_CHANNEL_CONSTRUCTS",
    "TRUST_CHANNEL_CONSTRUCTS",
    "RISK_CHANNEL_CONSTRUCTS",
    "ENDOGENOUS_RISK_CONSTRUCTS",
    "ALL_SOCIAL_CHANNEL_CONSTRUCTS",
    "norm_channel_columns",
    "trust_channel_columns",
    "risk_channel_columns",
    "peer_pull_risk_columns",
    "NormChannelWeights",
]

NORM_CHANNEL_CONSTRUCTS: tuple[str, ...] = (
    "social_norms_vax_avg",
    "social_norms_descriptive_vax",
    "social_norms_descriptive_npi",
    "group_trust_vax",
)

TRUST_CHANNEL_CONSTRUCTS: tuple[str, ...] = (
    "institutional_trust_avg",
    "covax_legitimacy_scepticism_idx",
    "trust_traditional",
    "trust_onlineinfo",
    "trust_social",
)

RISK_CHANNEL_CONSTRUCTS: tuple[str, ...] = (
    "vaccine_risk_avg",
    "covid_perceived_danger_T12",
    "infection_likelihood_avg",
    "disease_fear_avg",
    "severity_if_infected_avg",
)

# Endogenous risk constructs: driven ONLY by the incidence feedback loop,
# never by peer-pull gossip.  Danger, fear, infection-likelihood and
# severity are epidemiological perceptions that should respond to local
# case counts, not to conversational averaging.
ENDOGENOUS_RISK_CONSTRUCTS: frozenset[str] = frozenset(
    {
        "covid_perceived_danger_T12",
        "infection_likelihood_avg",
        "disease_fear_avg",
        "severity_if_infected_avg",
    }
)

ALL_SOCIAL_CHANNEL_CONSTRUCTS: tuple[str, ...] = (
    *NORM_CHANNEL_CONSTRUCTS,
    *TRUST_CHANNEL_CONSTRUCTS,
    *RISK_CHANNEL_CONSTRUCTS,
)


def norm_channel_columns() -> tuple[str, ...]:
    """Constructs controlled by the social norm channel."""
    return NORM_CHANNEL_CONSTRUCTS


def trust_channel_columns() -> tuple[str, ...]:
    """Constructs controlled by the trust channel."""
    return TRUST_CHANNEL_CONSTRUCTS


def risk_channel_columns() -> tuple[str, ...]:
    """Constructs controlled by the risk/stakes channel."""
    return RISK_CHANNEL_CONSTRUCTS


def peer_pull_risk_columns() -> tuple[str, ...]:
    """Risk constructs eligible for pairwise peer-pull gossip.

    Excludes endogenous-only constructs (danger, fear, infection-likelihood,
    severity) which are driven solely by the incidence feedback loop.
    """
    return tuple(c for c in RISK_CHANNEL_CONSTRUCTS if c not in ENDOGENOUS_RISK_CONSTRUCTS)


@dataclass(frozen=True, slots=True)
class NormChannelWeights:
    """Legacy norm-composition weights kept for API compatibility."""

    descriptive: float = 0.65
    injunctive: float = 0.35

    @classmethod
    def from_descriptive_share(cls, w_d: float) -> NormChannelWeights:
        """Build weights from a single descriptive-share parameter."""
        clipped = min(1.0, max(0.0, float(w_d)))
        return cls(descriptive=clipped, injunctive=1.0 - clipped)

    def normalized(self) -> NormChannelWeights:
        total = self.descriptive + self.injunctive
        if total <= 0.0:
            return NormChannelWeights()
        return NormChannelWeights(
            descriptive=self.descriptive / total,
            injunctive=self.injunctive / total,
        )
