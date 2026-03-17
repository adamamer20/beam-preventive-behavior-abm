"""Agent state columns, bounds, and Polars dtypes.

Centralises every column name the ABM reads or writes so that typos
are caught at import time rather than mid-simulation.  The module
also defines valid bounds for each mutable construct (used by the
update module to clip after each tick) and provides a Polars schema
dict for zero-copy DataFrame construction.

Examples
--------
>>> "institutional_trust_avg" in MUTABLE_COLUMNS
True
>>> CONSTRUCT_BOUNDS["fivec_pro_vax_idx"]
(-4.0, 4.0)
>>> len(get_all_required_columns()) > 0
True
"""

from __future__ import annotations

from typing import Final

import polars as pl

__all__ = [
    "MUTABLE_COLUMNS",
    "IMMUTABLE_DEMOGRAPHIC_COLUMNS",
    "ABM_ALLOWED_OUTCOMES",
    "OUTCOME_COLUMNS",
    "CONSTRUCT_BOUNDS",
    "POLARS_SCHEMA",
    "get_all_required_columns",
]

# =========================================================================
# 1.  Mutable state columns  – updated by engines each tick
# =========================================================================

MUTABLE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        # --- institutional / social trust ---
        "institutional_trust_avg",
        # --- 5C vaccine-hesitancy dimensions ---
        "fivec_pro_vax_idx",
        "fivec_confidence",
        "fivec_low_constraints",
        "fivec_low_complacency",
        "fivec_collective_responsibility",
        # --- perceived risk ---
        "vaccine_risk_avg",
        # --- disease salience ---
        "disease_fear_avg",
        "infection_likelihood_avg",
        "severity_if_infected_avg",
        # --- COVID danger ---
        "covid_perceived_danger_T12",
        "covid_perceived_danger_Y",
        # --- legitimacy / scepticism ---
        "covax_legitimacy_scepticism_idx",
        # --- social norms ---
        "social_norms_vax_avg",
        "social_norms_descriptive_vax",
        "social_norms_descriptive_npi",
        "norm_share_hat_vax",
        "norm_share_hat_npi",
        "group_trust_vax",
    }
)

# =========================================================================
# 2.  Immutable demographic / background columns
# =========================================================================
#
# These columns are set at population initialisation and never modified
# during the simulation.  They include raw demographics, vaccination
# history, progressive-norms indices, moral-foundations scores,
# interaction-group variables, and missingness flags used by the
# regression models.

IMMUTABLE_DEMOGRAPHIC_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        # --- core demographics ---
        "country",
        "sex_female[T.True]",
        "age",
        "M_education",
        "income_ppp_norm",
        "minority",
        "M_religion",
        # --- health / COVID experience ---
        "medical_conditions_c",
        "health_poor",
        # Legacy alias retained until all backbone artifacts migrate.
        "health_good",
        "covid_num",
        "covax_num",
        "exper_illness",
        "side_effects",
        "side_effects_other",
        # --- flu vaccination history ---
        "flu_vaccinated_pre_pandemic",
        "flu_vaccinated_2020_2021",
        "flu_vaccinated_2021_2022",
        "flu_vaccinated_2022_2023",
        # --- progressive social norms (immutable anchors) ---
        "progressive_attitudes_nonvax_idx",
        "friends_progressive_norms_nonvax_idx",
        "family_progressive_norms_nonvax_idx",
        "colleagues_progressive_norms_nonvax_idx",
        "moral_progressive_orientation",
        # --- specific progressive-norm items ---
        "friends_climate_harm",
        "friends_immigration_tolerance",
        "friends_lgbt_adoption_rights",
        "family_climate_harm",
        "family_immigration_tolerance",
        "family_lgbt_adoption_rights",
        "colleagues_climate_harm",
        "colleagues_immigration_tolerance",
        "colleagues_lgbt_adoption_rights",
        # --- moral foundations ---
        "mfq_agreement_binding_idx",
        "mfq_relevance_binding_idx",
        # --- social-network structure ---
        "size_friend_group",
        "size_colleagues_group",
        "contact_group_cohabitant",
        "contact_group_noncohabitant",
        "contact_group_friends",
        "contact_group_colleagues",
        "contact_group_others",
        "discussion_group_cohabitant",
        "discussion_group_noncohabitant",
        "discussion_group_friends",
        "discussion_group_colleagues",
        "discussion_group_others",
        # --- vaccine conversation frequency ---
        "vax_convo_freq_cohabitants",
        "vax_convo_freq_friends",
        "vax_convo_freq_others",
        "vax_convo_count_cohabitants",
        "vax_convo_count_friends",
        "vax_convo_count_others",
        # --- other model predictors ---
        "att_vaccines_importance",
        "born_in_residence_country[T.True]",
        "parent_born_abroad[T.True]",
        "flu_infection_likelihood_current",
        "flu_severity_if_infected",
        "flu_vaccine_fear",
        "vaccine_fear_avg",
        "trust_gp_vax_info",
        "trust_local_health_authorities_vax_info",
        "trust_national_gov_vax_info",
        "trust_national_health_institutions_vax_info",
        "trust_who_vax_info",
        "trust_traditional",
        "trust_onlineinfo",
        "trust_social",
        "duration_socialmedia",
        "duration_infomedia",
        # --- missingness indicators used by models ---
        "M_religion_missing",
        "exper_illness_missing",
        "side_effects_missing",
        "side_effects_other_missing",
        "health_poor_missing",
        # Legacy alias retained until all backbone artifacts migrate.
        "health_good_missing",
        "income_ppp_norm_missing",
        "minority_missing",
        "medical_conditions_c_missing",
        "group_trust_vax_missing",
        "social_norms_vax_avg_missing",
        "trust_traditional_missing",
        "trust_onlineinfo_missing",
        "trust_social_missing",
        "severity_if_infected_avg_missing",
        "flu_vaccinated_pre_pandemic_missing",
        "flu_vaccinated_2020_2021_missing",
        "flu_vaccinated_2021_2022_missing",
        "flu_vaccinated_2022_2023_missing",
        "flu_severity_if_infected_missing",
        "friends_climate_harm_missing",
        "friends_immigration_tolerance_missing",
        "friends_lgbt_adoption_rights_missing",
        "friends_progressive_norms_nonvax_idx_missing",
        "family_climate_harm_missing",
        "family_immigration_tolerance_missing",
        "family_lgbt_adoption_rights_missing",
        "family_progressive_norms_nonvax_idx_missing",
        "colleagues_climate_harm_missing",
        "colleagues_immigration_tolerance_missing",
        "colleagues_lgbt_adoption_rights_missing",
        "colleagues_progressive_norms_nonvax_idx_missing",
    }
)

# =========================================================================
# 3.  Outcome (dependent-variable) columns
# =========================================================================

OUTCOME_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "vax_willingness_T12",
        "mask_when_symptomatic_crowded",
        "stay_home_when_symptomatic",
        "mask_when_pressure_high",
    }
)

# Canonical ABM runtime outcomes. This can be narrower than levers.json.
ABM_ALLOWED_OUTCOMES: Final[tuple[str, ...]] = tuple(sorted(OUTCOME_COLUMNS))

# =========================================================================
# 4.  Construct bounds  –  (lower, upper) for clipping after updates
# =========================================================================
#
# Most constructs are z-standardised composites centred near 0; bounds of
# ±4 cover > 99.99 % of a standard normal.  A few items retain their
# original Likert scale.

CONSTRUCT_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    # --- continuous standardised (z-score) constructs ---
    "institutional_trust_avg": (-4.0, 4.0),
    "fivec_pro_vax_idx": (1.0, 7.0),
    "fivec_confidence": (1.0, 7.0),
    "fivec_low_constraints": (1.0, 7.0),
    "fivec_low_complacency": (1.0, 7.0),
    "fivec_collective_responsibility": (1.0, 7.0),
    "vaccine_risk_avg": (-4.0, 4.0),
    "disease_fear_avg": (-4.0, 4.0),
    "infection_likelihood_avg": (-4.0, 4.0),
    "severity_if_infected_avg": (-4.0, 4.0),
    "covid_perceived_danger_T12": (-4.0, 4.0),
    "covid_perceived_danger_Y": (-4.0, 4.0),
    "covax_legitimacy_scepticism_idx": (-4.0, 4.0),
    "social_norms_vax_avg": (1.0, 7.0),
    "social_norms_descriptive_vax": (1.0, 7.0),
    "social_norms_descriptive_npi": (1.0, 7.0),
    "norm_share_hat_vax": (0.0, 1.0),
    "norm_share_hat_npi": (0.0, 1.0),
    "group_trust_vax": (1.0, 7.0),
    "trust_traditional": (1.0, 7.0),
    "trust_onlineinfo": (1.0, 7.0),
    "trust_social": (1.0, 7.0),
    # --- media-duration composites (standardised) ---
    "duration_socialmedia": (-4.0, 4.0),
    "duration_infomedia": (-4.0, 4.0),
}

# =========================================================================
# 5.  Polars dtype schema
# =========================================================================
#
# Provides a ``{column_name: polars_dtype}`` mapping suitable for
# ``pl.DataFrame(data, schema=POLARS_SCHEMA)`` construction or
# ``pl.read_csv(..., schema_overrides=POLARS_SCHEMA)``.


def _build_polars_schema() -> dict[str, pl.DataType]:
    """Construct the master Polars schema dict.

    Returns
    -------
    dict[str, pl.DataType]
        Mapping from column name to Polars data type.

    Notes
    -----
    * ``country`` → ``pl.Categorical``
    * Boolean-coded dummies (``[T.True]`` suffix, ``_missing`` suffix,
      binary vaccination flags) → ``pl.Float32`` (kept numeric for
      matrix multiplication speed).
    * All other numeric columns → ``pl.Float32`` for memory efficiency.
    """
    schema: dict[str, pl.DataType] = {}

    # -- categorical --
    schema["country"] = pl.Categorical

    # -- mutable constructs: all continuous float --
    for col in sorted(MUTABLE_COLUMNS):
        if col == "country":
            continue  # already handled
        schema[col] = pl.Float32

    # -- immutable demographics --
    for col in sorted(IMMUTABLE_DEMOGRAPHIC_COLUMNS):
        if col == "country":
            continue
        schema[col] = pl.Float32

    # -- outcome columns --
    for col in sorted(OUTCOME_COLUMNS):
        schema[col] = pl.Float32

    return schema


POLARS_SCHEMA: Final[dict[str, pl.DataType]] = _build_polars_schema()

# =========================================================================
# 6.  Helper: union of all required columns
# =========================================================================


def get_all_required_columns() -> frozenset[str]:
    """Return the union of mutable, immutable, and outcome columns.

    Returns
    -------
    frozenset[str]
        Every column name the ABM expects in the agent DataFrame.

    Examples
    --------
    >>> cols = get_all_required_columns()
    >>> "country" in cols and "vax_willingness_T12" in cols
    True
    """
    return MUTABLE_COLUMNS | IMMUTABLE_DEMOGRAPHIC_COLUMNS | OUTCOME_COLUMNS
