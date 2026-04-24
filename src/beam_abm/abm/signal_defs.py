"""Canonical signal catalogue for ABM runtime and LLM prompting metadata.

This module is the Python source of truth for signal metadata.
The LLM prompt uses only `event_card_template` content; expected
footprints are diagnostics metadata and are not prompt instructions.
"""

from __future__ import annotations

INSTANTANEOUS_DELTA_PROMPT_LINE = "Decay/persistence handled externally; output instantaneous deltas only."


def _event_card_template(*, signal_id: str, channel: str, content: str, notes: str) -> dict[str, str]:
    return {
        "signal_id": signal_id,
        "channel": channel,
        "content": f"{content} {INSTANTANEOUS_DELTA_PROMPT_LINE}",
        "notes": notes,
    }


SIGNAL_CATALOGUE: dict[str, dict[str, object]] = {
    "local_outbreak": {
        "mode": "dual",
        "label": "Local outbreak salience",
        "event_card_template": _event_card_template(
            signal_id="local_outbreak",
            channel="local news + personal network",
            content=(
                "Reports of a local COVID outbreak increase salience: "
                "people are reminded of severe illness, hospital strain, and nearby cases."
            ),
            notes="Focus on disease stakes/salience and perceived infection likelihood. Do not directly change access constraints.",
        ),
        "output_validation": {"allowed_keys": "mutable_state_exact", "value_type": "float"},
        "intended_footprint": {
            "covid_perceived_danger_T12": "up",
            "infection_likelihood_avg": "up",
            "disease_fear_avg": "up",
        },
    },
    "misinfo_side_effects": {
        "mode": "dual",
        "label": "Side-effect misinformation wave",
        "event_card_template": _event_card_template(
            signal_id="misinfo_side_effects",
            channel="social media + messaging apps",
            content=(
                "A wave of misinformation claims vaccines cause severe side "
                "effects and that authorities are hiding risks."
            ),
            notes="Allow heterogeneity: low-trust agents may update strongly; high-trust agents may discount.",
        ),
        "output_validation": {"allowed_keys": "mutable_state_exact", "value_type": "float"},
        "intended_footprint": {
            "vaccine_risk_avg": "up",
            "covax_legitimacy_scepticism_idx": "up",
            "institutional_trust_avg": "ambiguous",
        },
    },
    "credible_authority_safety": {
        "mode": "dual",
        "label": "Credible authority reassurance campaign",
        "event_card_template": _event_card_template(
            signal_id="credible_authority_safety",
            channel="public health institutions + local clinicians",
            content=(
                "A coordinated campaign communicates vaccine safety and "
                "effectiveness using trusted clinicians, transparent data, and clear explanations."
            ),
            notes="Keep boomerang/backfire effects as LLM-only signals.",
        ),
        "output_validation": {"allowed_keys": "mutable_state_exact", "value_type": "float"},
        "intended_footprint": {
            "institutional_trust_avg": "up",
            "covax_legitimacy_scepticism_idx": "down",
            "vaccine_risk_avg": "down",
        },
    },
    "peer_norms_pro_vax": {
        "mode": "dual",
        "label": "Pro-prevention peer norm surge",
        "event_card_template": _event_card_template(
            signal_id="peer_norms_pro_vax",
            channel="peers",
            content="The social environment becomes more supportive of vaccination and protective behavior.",
            notes="Norm-content shock only; diffusion speed is controlled by network parameters.",
        ),
        "output_validation": {"allowed_keys": "mutable_state_exact", "value_type": "float"},
        "intended_footprint": {
            "social_norms_vax_avg": "up",
            "group_trust_vax": "up",
        },
    },
    "peer_norms_anti_vax": {
        "mode": "dual",
        "label": "Anti-prevention peer norm surge",
        "event_card_template": _event_card_template(
            signal_id="peer_norms_anti_vax",
            channel="peers",
            content="The social environment becomes more skeptical of vaccination and mandates.",
            notes="Heterogeneous reactions expected by social conformity and trust state.",
        ),
        "output_validation": {"allowed_keys": "mutable_state_exact", "value_type": "float"},
        "intended_footprint": {
            "social_norms_vax_avg": "down",
            "group_trust_vax": "down",
        },
    },
    "trust_scandal": {
        "mode": "dual",
        "label": "Institutional trust scandal",
        "event_card_template": _event_card_template(
            signal_id="trust_scandal",
            channel="news + social amplification",
            content=(
                "A credibility scandal questions transparency and fairness in vaccine-related institutional decisions."
            ),
            notes="Primary footprint is lower institutional trust and higher process scepticism.",
        ),
        "output_validation": {"allowed_keys": "mutable_state_exact", "value_type": "float"},
        "intended_footprint": {
            "institutional_trust_avg": "down",
            "covax_legitimacy_scepticism_idx": "up",
        },
    },
    "trust_repair": {
        "mode": "dual",
        "label": "Institutional trust repair",
        "event_card_template": _event_card_template(
            signal_id="trust_repair",
            channel="institutions + clinicians",
            content=(
                "A transparent repair campaign addresses errors, publishes evidence,"
                " and rebuilds procedural legitimacy."
            ),
            notes="Repair effects are usually slower and smaller than scandal shocks.",
        ),
        "output_validation": {"allowed_keys": "mutable_state_exact", "value_type": "float"},
        "intended_footprint": {
            "institutional_trust_avg": "up",
            "covax_legitimacy_scepticism_idx": "down",
        },
    },
    "placebo_containment": {
        "mode": "dual",
        "label": "Placebo containment",
        "event_card_template": _event_card_template(
            signal_id="placebo_containment",
            channel="low-salience informational background",
            content="A very low salience informational event with minimal expected behavioral relevance.",
            notes="Used for containment diagnostics; expected near-zero movement.",
        ),
        "output_validation": {"allowed_keys": "mutable_state_exact", "value_type": "float"},
        "intended_footprint": {
            "covid_perceived_danger_T12": "small_up",
        },
    },
    "norm_cue_campaign": {
        "mode": "dual",
        "label": "Localized descriptive norm cue",
        "event_card_template": _event_card_template(
            signal_id="norm_cue_campaign",
            channel="community communication",
            content=("A local campaign highlights that most peers are adopting protective behavior and vaccination."),
            notes="Designed to trigger descriptive norm effects and potential threshold cascades.",
        ),
        "output_validation": {"allowed_keys": "mutable_state_exact", "value_type": "float"},
        "intended_footprint": {
            "social_norms_vax_avg": "up",
            "group_trust_vax": "up",
        },
    },
    "boomerang_campaign": {
        "mode": "llm_only",
        "label": "Boomerang authority messaging",
        "event_card_template": _event_card_template(
            signal_id="boomerang_campaign",
            channel="government + mainstream media",
            content="A strong institutional pro-vaccine campaign is perceived as pushy and moralizing.",
            notes="State-dependent responses expected; this is not representable as uniform parametric shock.",
        ),
        "output_validation": {"allowed_keys": "mutable_state_exact", "value_type": "float"},
        "intended_footprint": {
            "institutional_trust_avg": "polarized",
            "covax_legitimacy_scepticism_idx": "polarized",
            "vaccine_risk_avg": "ambiguous",
        },
    },
    "debunking_backfire": {
        "mode": "llm_only",
        "label": "Debunking with backfire risk",
        "event_card_template": _event_card_template(
            signal_id="debunking_backfire",
            channel="fact-checkers + platforms",
            content="A prominent fact-check labels circulating claims as false and provides corrections.",
            notes="Trust-conditional correction vs reactance expected.",
        ),
        "output_validation": {"allowed_keys": "mutable_state_exact", "value_type": "float"},
        "intended_footprint": {
            "vaccine_risk_avg": "ambiguous",
            "covax_legitimacy_scepticism_idx": "up for reactant",
            "institutional_trust_avg": "polarized",
        },
    },
}
