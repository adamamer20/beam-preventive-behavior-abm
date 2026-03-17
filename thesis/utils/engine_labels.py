"""Shared engine label dictionaries for chapter figures and tables."""

from __future__ import annotations

CHOICE_ENGINE_BLOCK_LABELS: dict[str, str] = {
    "stakes_engine": "Stakes",
    "vaccine_cost_engine": "Vaccine cost",
    "disposition_engine": "Disposition (5C)",
    "institutional_trust_engine": "Institutional trust",
    "group_trust_engine": "Group trust",
    "trust_engine": "Institutional trust",
    "legitimacy_engine": "CoVax Legitimacy",
    "norms_engine": "Norms",
    "moral_engine": "Moral",
    "habit_engine": "Habit",
    "health_history_engine": "Health history",
    "social_interaction_engine": "Social interaction",
    "information_environment_engine": "Information environment",
    "controls_demography_SES": "Controls",
    "controls_demography_health": "Controls (demography/health)",
    "controls_health_history": "Controls + health history",
    "controls_country": "Country context (fixed effect)",
}

CHOICE_ENGINE_BLOCK_ALIASES: dict[str, str] = {
    "vax_disposition_engine": "disposition_engine",
    "psych_5c_no_index": "disposition_engine",
    "norms_engine_prox": "norms_engine",
    "norms_moral": "norms_engine",
    "risk_no_fear": "stakes_engine",
    "trust_engine_with_group": "trust_engine",
    "institutional_trust": "trust_engine",
    "cost_engine": "vaccine_cost_engine",
    "information_exposure_engine": "information_environment_engine",
}

P_REF_OUTCOME_ORDER: tuple[str, ...] = (
    "institutional_trust_block",
    "group_trust_block",
    "legitimacy_block",
    "norms_block",
    "stakes_block",
    "disposition_block",
)

P_REF_OUTCOME_BLOCK_MAP: dict[str, str] = {
    "institutional_trust_avg": "institutional_trust_block",
    "group_trust_vax": "group_trust_block",
    "covax_legitimacy_scepticism_idx": "legitimacy_block",
    "social_norms_vax_avg": "norms_block",
    "covid_perceived_danger_T12": "stakes_block",
    "fivec_pro_vax_idx": "disposition_block",
}

P_REF_OUTCOME_BLOCK_LABELS: dict[str, str] = {
    "institutional_trust_block": "Institutional trust",
    "group_trust_block": "Group trust",
    "legitimacy_block": "CoVax Legitimacy",
    "norms_block": "Norms",
    "stakes_block": "Stakes",
    "disposition_block": "Disposition (5C)",
}

P_REF_BLOCK_ORDER: tuple[str, ...] = (
    "institutional_trust_engine",
    "group_trust_engine",
    "trust_engine",
    "norms_engine",
    "moral_engine",
    "ideology_engine",
    "social_interaction_engine",
    "health_history_engine",
    "information_environment_engine",
    "controls_demography_SES",
)

P_REF_BLOCK_LABELS: dict[str, str] = {
    "institutional_trust_engine": "Institutional trust",
    "group_trust_engine": "Group trust",
    "trust_engine": "Information environment credibility",
    "norms_engine": "Norms",
    "moral_engine": "Moral",
    "ideology_engine": "Ideology",
    "social_interaction_engine": "Social interaction",
    "health_history_engine": "Health history",
    "information_environment_engine": "Information environment",
    "controls_demography_SES": "Controls (Demo/SES)",
}

BELIEF_OUTCOME_ORDER: tuple[str, ...] = (
    "institutional_trust_avg",
    "group_trust_vax",
    "social_norms_vax_avg",
    "covax_legitimacy_scepticism_idx",
    "covid_perceived_danger_T12",
    "fivec_low_constraints",
)

BELIEF_OUTCOME_ENGINE_LABELS: dict[str, str] = {
    "institutional_trust_avg": P_REF_OUTCOME_BLOCK_LABELS["institutional_trust_block"],
    "group_trust_vax": P_REF_OUTCOME_BLOCK_LABELS["group_trust_block"],
    "social_norms_vax_avg": P_REF_OUTCOME_BLOCK_LABELS["norms_block"],
    "covax_legitimacy_scepticism_idx": P_REF_OUTCOME_BLOCK_LABELS["legitimacy_block"],
    "covid_perceived_danger_T12": P_REF_OUTCOME_BLOCK_LABELS["stakes_block"],
    "fivec_low_constraints": P_REF_OUTCOME_BLOCK_LABELS["disposition_block"],
}

BELIEF_SIGNAL_MECHANISM_BUCKET_MAP: dict[str, str] = {
    "trust_who_vax_info_shift": "institutional_credibility",
    "trust_who_vax_info_shift_down": "institutional_credibility",
    "trust_gp_vax_info_shift": "institutional_credibility",
    "trust_gp_vax_info_shift_down": "institutional_credibility",
    "trust_national_gov_vax_info_shift": "institutional_credibility",
    "trust_national_gov_vax_info_shift_down": "institutional_credibility",
    "trust_traditional_shift": "information_environment_credibility",
    "trust_traditional_shift_down": "information_environment_credibility",
    "trust_onlineinfo_shift": "information_environment_credibility",
    "trust_onlineinfo_shift_down": "information_environment_credibility",
    "trust_social_shift": "interpersonal_channel",
    "trust_social_shift_down": "interpersonal_channel",
    "vax_conversation_freq_shift": "interpersonal_channel",
    "vax_conversation_freq_drop": "interpersonal_channel",
    "booking_access_improves": "constraints_access",
    "booking_access_worsens": "constraints_access",
    "vax_hub_access_improves": "constraints_access",
    "vax_hub_access_worsens": "constraints_access",
    "impediment_on": "constraints_access",
    "impediment_off": "constraints_access",
    "side_effects_self": "vaccine_risk_cost_salience",
    "side_effects_other_vicarious": "vaccine_risk_cost_salience",
    "health_deterioration": "disease_stakes_salience",
    "severe_illness_vicarious": "disease_stakes_salience",
}

BELIEF_SIGNAL_MECHANISM_BUCKET_ORDER: tuple[str, ...] = (
    "institutional_credibility",
    "information_environment_credibility",
    "interpersonal_channel",
    "constraints_access",
    "vaccine_risk_cost_salience",
    "disease_stakes_salience",
)

BELIEF_SIGNAL_MECHANISM_BUCKET_LABELS: dict[str, str] = {
    "institutional_credibility": "Institutional credibility",
    "information_environment_credibility": "Information environment credibility",
    "interpersonal_channel": "Interpersonal channel",
    "constraints_access": "Constraints and access",
    "vaccine_risk_cost_salience": "Vaccine risk and cost salience",
    "disease_stakes_salience": "Disease stakes salience",
}


def label_belief_engine(state_var: str) -> str:
    """Return the display label for a psychological profile variable."""

    key = str(state_var)
    return BELIEF_OUTCOME_ENGINE_LABELS.get(key, key)


def label_belief_signal_bucket(bucket: str) -> str:
    """Return display label for a belief signal-mechanism bucket."""

    key = str(bucket).strip()
    return BELIEF_SIGNAL_MECHANISM_BUCKET_LABELS.get(key, key)


def canonical_choice_engine_block(block: str) -> str:
    """Return canonical block id for a choice-engine block alias."""

    raw = str(block).strip()
    return CHOICE_ENGINE_BLOCK_ALIASES.get(raw, raw)


def label_choice_engine_block(block: str) -> str:
    """Return the strict display label for a choice-engine block."""

    key = canonical_choice_engine_block(block)
    return CHOICE_ENGINE_BLOCK_LABELS[key]


__all__ = [
    "BELIEF_SIGNAL_MECHANISM_BUCKET_LABELS",
    "BELIEF_SIGNAL_MECHANISM_BUCKET_MAP",
    "BELIEF_SIGNAL_MECHANISM_BUCKET_ORDER",
    "BELIEF_OUTCOME_ENGINE_LABELS",
    "BELIEF_OUTCOME_ORDER",
    "CHOICE_ENGINE_BLOCK_ALIASES",
    "CHOICE_ENGINE_BLOCK_LABELS",
    "P_REF_BLOCK_LABELS",
    "P_REF_BLOCK_ORDER",
    "P_REF_OUTCOME_BLOCK_LABELS",
    "P_REF_OUTCOME_BLOCK_MAP",
    "P_REF_OUTCOME_ORDER",
    "canonical_choice_engine_block",
    "label_belief_engine",
    "label_belief_signal_bucket",
    "label_choice_engine_block",
]
