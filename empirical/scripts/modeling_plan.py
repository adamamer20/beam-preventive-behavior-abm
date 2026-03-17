"""Curated structural backbone model specifications derived from modeling.md.

This lives alongside the modeling CLI (not in src/) to keep the plan
easy to read/edit for analysis workflows.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import replace

from beartype import beartype

from beam_abm.empirical.modeling import ModelBlock, ModelSpec

_PEER_PROGRESSIVE_NORMS_NONVAX_IDX_TO_ITEMS: dict[str, tuple[str, ...]] = {
    "family_progressive_norms_nonvax_idx": (
        "family_climate_harm",
        "family_lgbt_adoption_rights",
        "family_immigration_tolerance",
    ),
    "friends_progressive_norms_nonvax_idx": (
        "friends_climate_harm",
        "friends_lgbt_adoption_rights",
        "friends_immigration_tolerance",
    ),
    "colleagues_progressive_norms_nonvax_idx": (
        "colleagues_climate_harm",
        "colleagues_lgbt_adoption_rights",
        "colleagues_immigration_tolerance",
    ),
}


def _spec_uses_peer_norm_indices_as_predictors(spec: ModelSpec) -> bool:
    if spec.outcome in _PEER_PROGRESSIVE_NORMS_NONVAX_IDX_TO_ITEMS:
        return False
    predictors = set(spec.predictor_list())
    return any(idx in predictors for idx in _PEER_PROGRESSIVE_NORMS_NONVAX_IDX_TO_ITEMS)


def _expand_interaction_items(interaction: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """Expand an interaction tuple by replacing any peer norm index with its items."""
    options_per_term: list[tuple[str, ...]] = []
    for term in interaction:
        options_per_term.append(_PEER_PROGRESSIVE_NORMS_NONVAX_IDX_TO_ITEMS.get(term, (term,)))
    expanded = [tuple(combo) for combo in itertools.product(*options_per_term)]
    # Preserve order while dropping duplicates.
    seen: set[tuple[str, ...]] = set()
    deduped: list[tuple[str, ...]] = []
    for combo in expanded:
        if combo not in seen:
            seen.add(combo)
            deduped.append(combo)
    return tuple(deduped)


def _with_peer_norm_items(spec: ModelSpec) -> ModelSpec:
    """Return a spec variant where peer progressive norms indices are replaced by their items."""
    blocks: list[ModelBlock] = []
    for block in spec.blocks:
        predictors: list[str] = []
        for predictor in block.predictors:
            predictors.extend(_PEER_PROGRESSIVE_NORMS_NONVAX_IDX_TO_ITEMS.get(predictor, (predictor,)))
        blocks.append(ModelBlock(block.name, tuple(predictors), block.sample_strategy))

    interactions: list[tuple[str, ...]] = []
    for interaction in spec.interactions:
        interactions.extend(_expand_interaction_items(interaction))

    suffix = "_PN_ITEMS"
    description = spec.description
    if description:
        description = f"{description} (peer norms item-level)"
    else:
        description = "Peer norms item-level"

    return replace(
        spec,
        name=f"{spec.name}{suffix}",
        blocks=tuple(blocks),
        interactions=tuple(interactions),
        description=description,
    )


def _without_country(spec: ModelSpec) -> ModelSpec:
    """Drop country predictors from blocks; keep fixed effects via add_country."""
    cleaned_blocks: list[ModelBlock] = []
    for block in spec.blocks:
        predictors = tuple(p for p in block.predictors if p != "country")
        if block.name in {"country", "controls_country"}:
            continue
        if not predictors:
            continue
        cleaned_blocks.append(ModelBlock(block.name, predictors, block.sample_strategy))
    return replace(spec, blocks=tuple(cleaned_blocks))


_CANONICAL_BLOCK_TO_COLUMNS: dict[str, tuple[str, ...]] = {
    "controls_demography_SES": (
        "M_education",
        "M_religion",
        "age",
        "born_in_residence_country",
        "income_ppp_norm",
        "minority",
        "parent_born_abroad",
        "sex_female",
    ),
    "habit_engine": (
        "covax_num",
        "flu_vaccinated_2020_2021",
        "flu_vaccinated_2021_2022",
        "flu_vaccinated_2022_2023",
        "flu_vaccinated_pre_pandemic",
    ),
    "disposition_engine": (
        "fivec_collective_responsibility",
        "fivec_confidence",
        "fivec_low_complacency",
        "fivec_low_constraints",
        "fivec_pro_vax_idx",
    ),
    "trust_engine": (
        "group_trust_vax",
        "institutional_trust_avg",
        "trust_gp_vax_info",
        "trust_local_health_authorities_vax_info",
        "trust_national_gov_vax_info",
        "trust_national_health_institutions_vax_info",
        "trust_who_vax_info",
    ),
    "legitimacy_engine": ("covax_legitimacy_scepticism_idx",),
    "moral_engine": (
        "mfq_agreement_binding_idx",
        "mfq_relevance_binding_idx",
        "moral_progressive_orientation",
    ),
    "norms_engine": (
        "colleagues_climate_harm",
        "colleagues_immigration_tolerance",
        "colleagues_lgbt_adoption_rights",
        "colleagues_progressive_norms_nonvax_idx",
        "family_climate_harm",
        "family_immigration_tolerance",
        "family_lgbt_adoption_rights",
        "family_progressive_norms_nonvax_idx",
        "friends_climate_harm",
        "friends_immigration_tolerance",
        "friends_lgbt_adoption_rights",
        "friends_progressive_norms_nonvax_idx",
        "social_norms_vax_avg",
    ),
    "stakes_engine": (
        "covid_perceived_danger_Y",
        "covid_perceived_danger_T12",
        "disease_fear_avg",
        "flu_disease_fear",
        "flu_infection_likelihood_current",
        "flu_severity_if_infected",
        "infection_likelihood_avg",
        "severity_if_infected_avg",
    ),
    "health_history_engine": (
        "covid_num",
        "exper_illness",
        "health_poor",
        "medical_conditions_c",
        "side_effects",
        "side_effects_other",
    ),
    "vaccine_cost_engine": (
        "att_vaccines_importance",
        "flu_vaccine_fear",
        "flu_vaccine_risk",
        "vaccine_fear_avg",
        "vaccine_risk_avg",
    ),
    "information_environment_engine": (
        "duration_infomedia",
        "duration_socialmedia",
        "trust_onlineinfo",
        "trust_social",
        "trust_traditional",
    ),
    "social_interaction_engine": (
        "contact_group_cohabitant",
        "contact_group_colleagues",
        "contact_group_friends",
        "contact_group_noncohabitant",
        "contact_group_others",
        "discussion_group_cohabitant",
        "discussion_group_colleagues",
        "discussion_group_friends",
        "discussion_group_noncohabitant",
        "discussion_group_others",
        "size_colleagues_group",
        "size_friend_group",
        "vax_convo_count_cohabitants",
        "vax_convo_count_friends",
        "vax_convo_count_others",
        "vax_convo_freq_cohabitants",
        "vax_convo_freq_friends",
        "vax_convo_freq_others",
    ),
    "accessibility_engine": (
        "booking",
        "vax_hub",
        "impediment",
    ),
    "ideology_engine": ("politics",),
}

_COLUMN_TO_CANONICAL_BLOCK: dict[str, str] = {}
for _block_name, _columns in _CANONICAL_BLOCK_TO_COLUMNS.items():
    for _column in _columns:
        prior = _COLUMN_TO_CANONICAL_BLOCK.get(_column)
        if prior is not None and prior != _block_name:
            raise ValueError(f"Column {_column!r} assigned to multiple canonical blocks: {prior!r}, {_block_name!r}")
        _COLUMN_TO_CANONICAL_BLOCK[_column] = _block_name


def _canonicalize_blocks(spec: ModelSpec) -> ModelSpec:
    """Canonicalize block names using the global predictor-to-block mapping."""
    predictors_by_block: dict[str, list[str]] = {}
    strategy_by_block: dict[str, str] = {}

    for block in spec.blocks:
        for predictor in block.predictors:
            canonical = _COLUMN_TO_CANONICAL_BLOCK.get(predictor)
            if canonical is None:
                if predictor.endswith("_missing"):
                    canonical = "missingness"
                else:
                    raise ValueError(f"Predictor {predictor!r} in spec {spec.name!r} has no canonical block mapping")

            existing = predictors_by_block.setdefault(canonical, [])
            if predictor not in existing:
                existing.append(predictor)

            current = strategy_by_block.get(canonical)
            if current is None:
                strategy_by_block[canonical] = block.sample_strategy
            elif current != block.sample_strategy:
                strategy_by_block[canonical] = "conditional"

    remapped = tuple(
        ModelBlock(name, tuple(predictors), strategy_by_block.get(name, "core"))
        for name, predictors in predictors_by_block.items()
        if predictors
    )
    return replace(spec, blocks=remapped)


def _compose_columns(
    *,
    from_blocks: tuple[str, ...],
    include: tuple[str, ...] | None = None,
    exclude: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Build ordered predictor lists from canonical blocks using include/exclude selectors."""
    universe: list[str] = []
    seen: set[str] = set()
    for block_name in from_blocks:
        base = _CANONICAL_BLOCK_TO_COLUMNS.get(block_name)
        if base is None:
            raise ValueError(f"Unknown canonical block: {block_name}")
        for col in base:
            if col not in seen:
                seen.add(col)
                universe.append(col)

    if include is None:
        selected = list(universe)
    else:
        missing = [col for col in include if col not in seen]
        if missing:
            raise ValueError(f"Columns not available in selected canonical blocks {from_blocks}: {missing}")
        selected = list(include)

    if exclude:
        excluded = set(exclude)
        selected = [col for col in selected if col not in excluded]

    return tuple(selected)


def _select_block(
    name: str,
    *,
    from_blocks: tuple[str, ...] | None = None,
    include: tuple[str, ...] | None = None,
    exclude: tuple[str, ...] = (),
    sample_strategy: str = "core",
) -> ModelBlock:
    """Create a ModelBlock by selecting columns from canonical block definitions."""
    source_blocks = from_blocks or (name,)
    predictors = _compose_columns(from_blocks=source_blocks, include=include, exclude=exclude)
    return ModelBlock(name, predictors, sample_strategy)


def _base_blocks() -> dict[str, ModelBlock]:
    return {
        "demography": _select_block(
            "controls_demography_SES",
            include=("age", "sex_female", "M_education", "income_ppp_norm", "minority", "M_religion"),
        ),
        "migration": _select_block(
            "controls_demography_SES",
            include=("born_in_residence_country", "parent_born_abroad"),
        ),
        "health": _select_block(
            "health_history_engine",
            from_blocks=("health_history_engine", "habit_engine"),
            include=(
                "medical_conditions_c",
                "health_poor",
                "covid_num",
                "covax_num",
                "flu_vaccinated_pre_pandemic",
                "exper_illness",
                "side_effects",
                "side_effects_other",
            ),
        ),
        "flu_habit": _select_block(
            "habit_engine",
            include=(
                "flu_vaccinated_2022_2023",
                "flu_vaccinated_2021_2022",
                "flu_vaccinated_2020_2021",
                "flu_vaccinated_pre_pandemic",
            ),
        ),
        "psych_5c": _select_block(
            "disposition_engine",
            include=(
                "fivec_pro_vax_idx",
                "fivec_confidence",
                "fivec_low_constraints",
                "fivec_low_complacency",
                "fivec_collective_responsibility",
            ),
        ),
        "risk": _select_block(
            "stakes_engine",
            from_blocks=("stakes_engine", "vaccine_cost_engine"),
            include=(
                "disease_fear_avg",
                "infection_likelihood_avg",
                "severity_if_infected_avg",
                "vaccine_risk_avg",
                "vaccine_fear_avg",
            ),
        ),
        "attitudes": _select_block("vaccine_cost_engine", include=("att_vaccines_importance",)),
        "trust": _select_block(
            "trust_engine",
            include=(
                "institutional_trust_avg",
                "trust_national_health_institutions_vax_info",
                "trust_who_vax_info",
                "trust_gp_vax_info",
                "trust_national_gov_vax_info",
                "trust_local_health_authorities_vax_info",
                "group_trust_vax",
            ),
        ),
        "norms": _select_block(
            "norms_engine",
            from_blocks=("norms_engine", "moral_engine"),
            include=(
                "family_progressive_norms_nonvax_idx",
                "friends_progressive_norms_nonvax_idx",
                "colleagues_progressive_norms_nonvax_idx",
                "social_norms_vax_avg",
                "moral_progressive_orientation",
                "mfq_agreement_binding_idx",
                "mfq_relevance_binding_idx",
            ),
            sample_strategy="conditional",
        ),
        "info": _select_block(
            "information_environment_engine",
            include=(
                "trust_traditional",
                "trust_onlineinfo",
                "trust_social",
                "duration_socialmedia",
                "duration_infomedia",
            ),
            sample_strategy="conditional",
        ),
        "contact_behaviour": _select_block(
            "social_interaction_engine",
            include=(
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
            ),
        ),
        "vax_conversation": _select_block(
            "social_interaction_engine",
            include=(
                "vax_convo_freq_cohabitants",
                "vax_convo_freq_friends",
                "vax_convo_freq_others",
                "vax_convo_count_cohabitants",
                "vax_convo_count_friends",
                "vax_convo_count_others",
            ),
        ),
        "contact_capacity": _select_block(
            "social_interaction_engine",
            include=("size_friend_group", "size_colleagues_group"),
        ),
        "accessibility": _select_block(
            "accessibility_engine",
            include=("booking", "vax_hub", "impediment"),
        ),
    }


def _health_without_vax_behaviours_block(health_block: ModelBlock) -> ModelBlock:
    predictors = tuple(
        predictor
        for predictor in health_block.predictors
        if predictor not in {"covax_num", "flu_vaccinated_pre_pandemic"}
    )
    return ModelBlock("health_history_engine", predictors, health_block.sample_strategy)


def _flu_block() -> ModelBlock:
    return _select_block(
        "stakes_engine",
        from_blocks=("stakes_engine", "vaccine_cost_engine"),
        include=(
            "flu_disease_fear",
            "flu_infection_likelihood_current",
            "flu_severity_if_infected",
            "flu_vaccine_risk",
            "flu_vaccine_fear",
        ),
    )


def _country_block() -> ModelBlock:
    return ModelBlock("country", ("country",))


def _politics_block() -> ModelBlock:
    return _select_block("ideology_engine")


def _attitude_chain_block() -> ModelBlock:
    return _select_block(
        "trust_engine",
        from_blocks=("trust_engine", "disposition_engine", "vaccine_cost_engine"),
        include=("fivec_pro_vax_idx", "vaccine_risk_avg", "institutional_trust_avg"),
    )


@beartype
def build_structural_backbone_plan(include_5c_components: bool = True) -> Iterable[ModelSpec]:
    blocks = _base_blocks()
    country_block = _country_block()
    politics_block = _politics_block()
    norms_no_peer_norms = _select_block("moral_engine")
    health_no_vax_behaviour = _health_without_vax_behaviours_block(blocks["health"])
    controls_demography_SES = blocks["demography"]

    institutional_trust_no_group = _select_block(
        "trust_engine",
        include=(
            "trust_national_health_institutions_vax_info",
            "trust_who_vax_info",
            "trust_gp_vax_info",
            "trust_national_gov_vax_info",
            "trust_local_health_authorities_vax_info",
        ),
    )
    fivec_no_index = _select_block(
        "disposition_engine",
        exclude=("fivec_pro_vax_idx",),
    )
    # Exclude vaccine_fear_avg from the trimmed risk block for backbone models
    risk_no_fear = _select_block(
        "stakes_engine",
        from_blocks=("stakes_engine", "vaccine_cost_engine"),
        include=("disease_fear_avg", "infection_likelihood_avg", "severity_if_infected_avg", "vaccine_risk_avg"),
    )

    legitimacy_block = _select_block("legitimacy_engine")
    covid_habit_block = _select_block("habit_engine", include=("covax_num",))
    trust_engine = ModelBlock("trust_engine", ("institutional_trust_avg",))
    trust_engine_with_group = ModelBlock(
        "trust_engine",
        ("institutional_trust_avg", "group_trust_vax"),
    )

    vax_disposition_engine = ModelBlock("disposition_engine", ("fivec_pro_vax_idx",))
    cost_engine = ModelBlock("vaccine_cost_engine", ("vaccine_risk_avg",))
    cost_engine_fear = ModelBlock("vaccine_cost_engine", ("vaccine_fear_avg",))
    stakes_engine = ModelBlock("stakes_engine", ("disease_fear_avg",))
    stakes_engine_danger = ModelBlock("stakes_engine", ("covid_perceived_danger_Y",))
    stakes_engine_danger_T12 = ModelBlock("stakes_engine", ("covid_perceived_danger_T12",))
    norms_engine_prox = ModelBlock(
        "norms_engine",
        ("social_norms_vax_avg",),
        sample_strategy="conditional",
    )
    norms_activation = ModelBlock(
        "social_interaction_engine",
        (
            "discussion_group_cohabitant",
            "discussion_group_noncohabitant",
            "discussion_group_friends",
            "discussion_group_colleagues",
            "discussion_group_others",
            "vax_convo_freq_cohabitants",
            "vax_convo_freq_friends",
            "vax_convo_freq_others",
            "vax_convo_count_cohabitants",
            "vax_convo_count_friends",
            "vax_convo_count_others",
        ),
    )
    demography_trust_ref = _select_block(
        "controls_demography_SES",
        include=("income_ppp_norm", "sex_female", "M_education"),
    )
    disposition_engine = vax_disposition_engine
    disposition_engine_no_index = fivec_no_index
    stakes_engine_split = ModelBlock(
        "stakes_engine",
        ("disease_fear_avg", "infection_likelihood_avg", "severity_if_infected_avg"),
        risk_no_fear.sample_strategy,
    )
    norms_engine_split = ModelBlock(
        "norms_engine",
        (
            "family_progressive_norms_nonvax_idx",
            "friends_progressive_norms_nonvax_idx",
            "colleagues_progressive_norms_nonvax_idx",
            "social_norms_vax_avg",
        ),
        blocks["norms"].sample_strategy,
    )
    moral_engine_split = ModelBlock(
        "moral_engine",
        ("moral_progressive_orientation", "mfq_agreement_binding_idx", "mfq_relevance_binding_idx"),
        blocks["norms"].sample_strategy,
    )
    norms_engine = norms_engine_prox
    stakes_engine_renamed = stakes_engine_danger
    stakes_engine_renamed_T12 = stakes_engine_danger_T12
    legitimacy_engine = legitimacy_block
    trust_engine_with_group_renamed = trust_engine_with_group
    demography_stakes_ref = _select_block("controls_demography_SES", include=("age", "M_education"))
    demography_norms_ref = _select_block("controls_demography_SES", include=("age", "M_education", "M_religion"))
    demography_5c_ref = _select_block("controls_demography_SES", include=("age", "M_religion", "income_ppp_norm"))
    demography_5c_constraints_ref = _select_block(
        "controls_demography_SES",
        include=("age", "income_ppp_norm", "minority", "M_religion"),
    )
    health_no_vax_trust_ref = _select_block(
        "health_history_engine",
        include=("side_effects_other", "health_poor", "exper_illness"),
    )
    health_no_vax_risk_ref = _select_block(
        "health_history_engine",
        include=("side_effects", "side_effects_other", "health_poor", "exper_illness", "covid_num"),
    )
    health_no_vax_stakes_ref = _select_block(
        "health_history_engine",
        include=("health_poor", "exper_illness", "side_effects", "covid_num"),
    )
    info_trust_ref = _select_block(
        "information_environment_engine",
        include=("trust_traditional", "trust_onlineinfo"),
        sample_strategy="conditional",
    )
    info_social_ref = _select_block(
        "information_environment_engine",
        include=("trust_social",),
        sample_strategy="conditional",
    )
    info_norms_ref = _select_block(
        "information_environment_engine",
        include=("trust_traditional", "trust_onlineinfo", "trust_social"),
        sample_strategy="conditional",
    )
    info_group_trust_ref = _select_block(
        "information_environment_engine",
        include=("trust_social", "trust_traditional", "trust_onlineinfo"),
        sample_strategy="conditional",
    )
    info_5c_constraints_ref = _select_block(
        "information_environment_engine",
        include=("trust_social", "trust_traditional", "duration_socialmedia", "duration_infomedia"),
        sample_strategy="conditional",
    )
    accessibility_5c_constraints_ref = _select_block(
        "accessibility_engine",
        include=("booking", "vax_hub", "impediment"),
        sample_strategy="conditional",
    )
    norms_trust_ref = _select_block(
        "norms_engine",
        from_blocks=("norms_engine", "moral_engine"),
        include=("social_norms_vax_avg", "moral_progressive_orientation"),
        sample_strategy="conditional",
    )
    norms_5c_ref = _select_block(
        "norms_engine",
        from_blocks=("norms_engine", "moral_engine"),
        include=("social_norms_vax_avg", "moral_progressive_orientation", "mfq_agreement_binding_idx"),
        sample_strategy="conditional",
    )
    norms_5c_constraints_ref = _select_block(
        "norms_engine",
        from_blocks=("norms_engine", "moral_engine"),
        include=(
            "social_norms_vax_avg",
            "moral_progressive_orientation",
            "mfq_agreement_binding_idx",
            "mfq_relevance_binding_idx",
        ),
        sample_strategy="conditional",
    )
    norms_no_peer_ref = _select_block(
        "moral_engine",
        include=("moral_progressive_orientation", "mfq_agreement_binding_idx", "mfq_relevance_binding_idx"),
    )
    norms_activation_ref = _select_block("social_interaction_engine", include=("vax_convo_freq_cohabitants",))
    psych_5c_risk_ref = _select_block(
        "disposition_engine",
        include=("fivec_confidence", "fivec_low_complacency", "fivec_collective_responsibility"),
    )
    trust_stakes_ref = _select_block("trust_engine", include=("trust_national_gov_vax_info",))
    trust_5c_ref = _select_block(
        "trust_engine",
        include=("trust_gp_vax_info", "trust_who_vax_info", "trust_national_gov_vax_info"),
    )
    trust_group_ref = _select_block(
        "trust_engine",
        include=("trust_gp_vax_info", "trust_local_health_authorities_vax_info", "trust_national_gov_vax_info"),
    )
    specs: list[ModelSpec] = []

    interaction_core = (
        ("institutional_trust_avg", "fivec_pro_vax_idx"),
        ("family_progressive_norms_nonvax_idx", "fivec_collective_responsibility"),
        ("trust_social", "duration_socialmedia"),
    )
    interaction_core_no_index = (
        ("family_progressive_norms_nonvax_idx", "fivec_collective_responsibility"),
        ("trust_social", "duration_socialmedia"),
    )
    mediation_backbone = _select_block(
        "trust_engine",
        from_blocks=("trust_engine", "disposition_engine", "vaccine_cost_engine", "norms_engine"),
        include=(
            "institutional_trust_avg",
            "fivec_pro_vax_idx",
            "vaccine_risk_avg",
            "family_progressive_norms_nonvax_idx",
        ),
    )

    specs.append(
        ModelSpec(
            name="B_COVID_MAIN_vax_willingness_Y",
            outcome="vax_willingness_Y",
            model_type="ordinal",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                fivec_no_index,
                risk_no_fear,
                blocks["trust"],
                blocks["norms"],
                blocks["info"],
                blocks["contact_behaviour"],
                blocks["vax_conversation"],
            ),
            interactions=interaction_core_no_index,
            description="Full decision rule for current COVID booster willingness (ABM backbone)",
            group="B_COVID",
        )
    )

    specs.append(
        ModelSpec(
            name="B_COVID_MAIN_vax_willingness_Y_HABIT",
            outcome="vax_willingness_Y",
            model_type="ordinal",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                covid_habit_block,
                fivec_no_index,
                risk_no_fear,
                blocks["trust"],
                blocks["norms"],
                blocks["info"],
                blocks["contact_behaviour"],
                blocks["vax_conversation"],
            ),
            interactions=interaction_core_no_index,
            description="Full decision rule + COVID vaccination history inertia (covax_num)",
            group="B_COVID",
        )
    )

    specs.append(
        ModelSpec(
            name="B_COVID_MAIN_vax_willingness_Y_LEGIT",
            outcome="vax_willingness_Y",
            model_type="ordinal",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                fivec_no_index,
                risk_no_fear,
                blocks["trust"],
                blocks["norms"],
                blocks["info"],
                blocks["contact_behaviour"],
                blocks["vax_conversation"],
                legitimacy_block,
            ),
            interactions=interaction_core_no_index,
            description="Full decision rule + legitimacy/scepticism engine predictor",
            group="B_COVID",
        )
    )

    specs.append(
        ModelSpec(
            name="B_COVID_MAIN_vax_willingness_Y_ENGINES",
            outcome="vax_willingness_Y",
            model_type="ordinal",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                trust_engine,
                vax_disposition_engine,
                cost_engine,
                stakes_engine,
                norms_engine_prox,
                legitimacy_block,
            ),
            interactions=(),
            description="Compressed ABM backbone: one block per engine (trust, disposition, costs, stakes, norms, legitimacy)",
            group="B_COVID",
        )
    )

    specs.append(
        ModelSpec(
            name="B_COVID_MAIN_vax_willingness_Y_ENGINES_STAKESdanger",
            outcome="vax_willingness_Y",
            model_type="ordinal",
            blocks=(
                controls_demography_SES,
                trust_engine,
                disposition_engine,
                stakes_engine_renamed,
                norms_engine,
                legitimacy_engine,
            ),
            interactions=(),
            description="Compressed ABM backbone (engines) with stakes operationalized as perceived danger (covid_perceived_danger_Y)",
            group="B_COVID",
        )
    )

    specs.append(
        ModelSpec(
            name="B_COVID_MAIN_vax_willingness_T12_ENGINES_STAKESdanger",
            outcome="vax_willingness_T12",
            model_type="ordinal",
            blocks=(
                controls_demography_SES,
                trust_engine,
                disposition_engine,
                stakes_engine_renamed_T12,
                norms_engine,
                legitimacy_engine,
            ),
            interactions=(),
            description="Compressed ABM backbone (engines) for initial-offer willingness (T12), with stakes as perceived danger (covid_perceived_danger_T12)",
            group="B_COVID",
        )
    )

    specs.append(
        ModelSpec(
            name="B_COVID_MAIN_vax_willingness_Y_ENGINES_COSTfear",
            outcome="vax_willingness_Y",
            model_type="ordinal",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                trust_engine,
                vax_disposition_engine,
                cost_engine_fear,
                stakes_engine,
                norms_engine_prox,
                legitimacy_block,
            ),
            interactions=(),
            description="Compressed ABM backbone (engines) with costs operationalized as fear (vaccine_fear_avg)",
            group="B_COVID",
        )
    )

    specs.append(
        ModelSpec(
            name="B_COVID_MAIN_vax_willingness_Y_ENGINES_HABIT",
            outcome="vax_willingness_Y",
            model_type="ordinal",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                covid_habit_block,
                trust_engine,
                vax_disposition_engine,
                cost_engine,
                stakes_engine,
                norms_engine_prox,
                legitimacy_block,
            ),
            interactions=(),
            description="Compressed ABM backbone + COVID vaccination history inertia (covax_num)",
            group="B_COVID",
        )
    )

    specs.append(
        ModelSpec(
            name="B_COVID_MAIN_vax_willingness_Y_ENGINES_STAKESdanger_HABIT",
            outcome="vax_willingness_Y",
            model_type="ordinal",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                covid_habit_block,
                trust_engine,
                vax_disposition_engine,
                cost_engine,
                stakes_engine_danger,
                norms_engine_prox,
                legitimacy_block,
            ),
            interactions=(),
            description="Compressed ABM backbone (engines) + COVID inertia, with stakes as perceived danger (covid_perceived_danger_Y)",
            group="B_COVID",
        )
    )

    specs.append(
        ModelSpec(
            name="B_COVID_MAIN_vax_willingness_Y_ENGINES_COSTfear_HABIT",
            outcome="vax_willingness_Y",
            model_type="ordinal",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                covid_habit_block,
                trust_engine,
                vax_disposition_engine,
                cost_engine_fear,
                stakes_engine,
                norms_engine_prox,
                legitimacy_block,
            ),
            interactions=(),
            description="Compressed ABM backbone (engines) + COVID inertia, with costs as fear (vaccine_fear_avg)",
            group="B_COVID",
        )
    )

    specs.append(
        ModelSpec(
            name="B_COVID_MAIN_vax_willingness_Y_MIGRATION",
            outcome="vax_willingness_Y",
            model_type="ordinal",
            blocks=(
                blocks["demography"],
                blocks["migration"],
                health_no_vax_behaviour,
                fivec_no_index,
                risk_no_fear,
                blocks["trust"],
                blocks["norms"],
                blocks["info"],
                blocks["contact_behaviour"],
                blocks["vax_conversation"],
            ),
            interactions=interaction_core_no_index,
            description="Current willingness decision rule + migration/birthplace robustness",
            group="B_COVID",
        )
    )
    specs.append(
        ModelSpec(
            name="B_COVID_MAIN_vax_willingness_T12",
            outcome="vax_willingness_T12",
            model_type="ordinal",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                fivec_no_index,
                risk_no_fear,
                blocks["trust"],
                blocks["norms"],
                blocks["info"],
                blocks["contact_behaviour"],
                blocks["vax_conversation"],
                blocks["accessibility"],
            ),
            interactions=interaction_core_no_index,
            description="Full decision rule for willingness when offered 1st/2nd dose (robustness)",
            group="B_COVID",
        )
    )
    specs.append(
        ModelSpec(
            name="B_COVID_MAIN_vax_willingness_T3",
            outcome="vax_willingness_T3",
            model_type="ordinal",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                fivec_no_index,
                risk_no_fear,
                blocks["trust"],
                blocks["norms"],
                blocks["info"],
                blocks["contact_behaviour"],
                blocks["vax_conversation"],
            ),
            interactions=interaction_core_no_index,
            description="Full decision rule for willingness when offered 3rd dose (robustness)",
            group="B_COVID",
        )
    )
    specs.append(
        ModelSpec(
            name="B_COVID_MAIN_vax_willingness_T4",
            outcome="vax_willingness_T4",
            model_type="ordinal",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                fivec_no_index,
                risk_no_fear,
                blocks["trust"],
                blocks["norms"],
                blocks["info"],
                blocks["contact_behaviour"],
                blocks["vax_conversation"],
            ),
            interactions=interaction_core_no_index,
            description="Full decision rule for willingness when offered 4th dose (selective sample)",
            group="B_COVID",
        )
    )
    specs.append(
        ModelSpec(
            name="B_COVID_vaccinated_any",
            outcome="covax_0_1",
            model_type="logit",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                fivec_no_index,
                risk_no_fear,
                blocks["trust"],
                blocks["norms"],
                blocks["info"],
            ),
            interactions=interaction_core_no_index,
            description="Ever vaccinated against COVID-19 (1=ever, 0=never) as historical decision rule",
            group="B_COVID",
        )
    )

    specs.append(
        ModelSpec(
            name="B_COVID_vaccinated_any_MIGRATION",
            outcome="covax_0_1",
            model_type="logit",
            blocks=(
                blocks["demography"],
                blocks["migration"],
                health_no_vax_behaviour,
                fivec_no_index,
                risk_no_fear,
                blocks["trust"],
                blocks["norms"],
                blocks["info"],
            ),
            interactions=interaction_core_no_index,
            description="Ever vaccinated (0/1) + migration/birthplace robustness",
            group="B_COVID",
        )
    )
    specs.append(
        ModelSpec(
            name="B_COVID_MEDIATION_vax_willingness_Y",
            outcome="vax_willingness_Y",
            model_type="ordinal",
            blocks=(blocks["demography"], health_no_vax_behaviour, mediation_backbone),
            interactions=(),
            description="Trust -> 5C -> risk chain on current willingness (clean mediation variant)",
            group="B_COVID",
        )
    )
    specs.append(
        ModelSpec(
            name="B_COVID_MEDIATION_vax_willingness_T4",
            outcome="vax_willingness_T4",
            model_type="ordinal",
            blocks=(blocks["demography"], health_no_vax_behaviour, mediation_backbone),
            interactions=(),
            description="Trust -> 5C -> risk chain on willingness when offered 4th dose",
            group="B_COVID",
        )
    )
    specs.append(
        ModelSpec(
            name="B_COVID_CONVERSATION_vax_willingness_Y",
            outcome="vax_willingness_Y",
            model_type="ordinal",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                fivec_no_index,
                blocks["norms"],
                blocks["contact_behaviour"],
                blocks["vax_conversation"],
                blocks["info"],
            ),
            interactions=(("friends_progressive_norms_nonvax_idx", "vax_convo_freq_friends"),),
            description="Network and conversation amplification on willingness (norms × conversation)",
            group="B_COVID",
        )
    )
    specs.append(
        ModelSpec(
            name="B_FLU_vaccinated_2023_2024",
            outcome="flu_vaccinated_2023_2024",
            model_type="logit",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                blocks["psych_5c"],
                risk_no_fear,
                blocks["trust"],
                blocks["norms"],
                blocks["info"],
                _flu_block(),
            ),
            interactions=interaction_core,
            description="Flu vaccination uptake contrast",
            group="B_FLU",
        )
    )

    specs.append(
        ModelSpec(
            name="B_FLU_vaccinated_2023_2024_HABIT",
            outcome="flu_vaccinated_2023_2024",
            model_type="logit",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                blocks["flu_habit"],
                blocks["psych_5c"],
                risk_no_fear,
                blocks["trust"],
                blocks["norms"],
                blocks["info"],
                _flu_block(),
            ),
            interactions=interaction_core,
            description="Flu uptake with explicit inertia/state-dependence (prior seasons + pre-pandemic)",
            group="B_FLU",
        )
    )

    flu_cost_engine = ModelBlock("vaccine_cost_engine", ("flu_vaccine_risk",))
    flu_stakes_engine = ModelBlock("stakes_engine", ("flu_disease_fear",))
    habit_engine = blocks["flu_habit"]
    flu_stakes_engine_renamed = flu_stakes_engine

    specs.append(
        ModelSpec(
            name="B_FLU_vaccinated_2023_2024_ENGINES",
            outcome="flu_vaccinated_2023_2024",
            model_type="logit",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                trust_engine,
                vax_disposition_engine,
                flu_cost_engine,
                flu_stakes_engine,
                norms_engine_prox,
            ),
            interactions=(),
            description="Compressed ABM backbone for flu uptake (engines)",
            group="B_FLU",
        )
    )

    specs.append(
        ModelSpec(
            name="B_FLU_vaccinated_2023_2024_ENGINES_HABIT",
            outcome="flu_vaccinated_2023_2024",
            model_type="logit",
            blocks=(
                controls_demography_SES,
                habit_engine,
                trust_engine,
                disposition_engine,
                flu_stakes_engine_renamed,
                norms_engine,
            ),
            interactions=(),
            description="Compressed ABM backbone for flu uptake (engines) + explicit inertia/state-dependence",
            group="B_FLU",
        )
    )
    specs.append(
        ModelSpec(
            name="B_PROTECT_protective_behaviour",
            outcome="protective_behaviour_avg",
            model_type="linear",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                blocks["psych_5c"],
                risk_no_fear,
                blocks["trust"],
                blocks["norms"],
                blocks["info"],
                blocks["contact_behaviour"],
                blocks["vax_conversation"],
            ),
            interactions=(("progressive_attitudes_nonvax_idx", "vax_convo_freq_friends"),),
            description="Non-vaccine protective behaviours",
            group="B_PROTECT",
        )
    )

    specs.append(
        ModelSpec(
            name="B_PROTECT_protective_behaviour_ENGINES",
            outcome="protective_behaviour_avg",
            model_type="linear",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                trust_engine,
                vax_disposition_engine,
                cost_engine,
                stakes_engine,
                norms_engine_prox,
            ),
            interactions=(),
            description="Compressed ABM backbone for non-vaccine protective behaviours (engines)",
            group="B_PROTECT",
        )
    )
    for suffix, outcome, desc in (
        (
            "support_mandatory_vaccination",
            "support_mandatory_vaccination",
            "Support for mandatory vaccination (component)",
        ),
        (
            "mask_when_symptomatic_crowded",
            "mask_when_symptomatic_crowded",
            "Mask use in crowds when symptomatic",
        ),
        (
            "stay_home_when_symptomatic",
            "stay_home_when_symptomatic",
            "Staying home when symptomatic",
        ),
        (
            "mask_when_pressure_high",
            "mask_when_pressure_high",
            "Mask use during high epidemic pressure",
        ),
    ):
        specs.append(
            ModelSpec(
                name=f"B_PROTECT_{suffix}",
                outcome=outcome,
                model_type="ordinal",
                blocks=(
                    blocks["demography"],
                    health_no_vax_behaviour,
                    blocks["psych_5c"],
                    risk_no_fear,
                    blocks["trust"],
                    blocks["norms"],
                    blocks["info"],
                    blocks["contact_behaviour"],
                    blocks["vax_conversation"],
                ),
                interactions=(),
                description=f"{desc} (B3 item-level)",
                group="B_PROTECT",
            )
        )

    for suffix, outcome, desc, extra_blocks in (
        (
            "mask_when_symptomatic_crowded",
            "mask_when_symptomatic_crowded",
            "Mask use in crowds when symptomatic",
            (),
        ),
        (
            "stay_home_when_symptomatic",
            "stay_home_when_symptomatic",
            "Staying home when symptomatic",
            (),
        ),
        (
            "mask_when_pressure_high",
            "mask_when_pressure_high",
            "Mask use during high epidemic pressure",
            (),
        ),
    ):
        specs.append(
            ModelSpec(
                name=f"B_PROTECT_{suffix}_ENGINES_REF",
                outcome=outcome,
                model_type="ordinal",
                blocks=(
                    controls_demography_SES,
                    disposition_engine_no_index,
                    stakes_engine_split,
                    trust_engine_with_group_renamed,
                    norms_engine_split,
                    moral_engine_split,
                    *extra_blocks,
                ),
                interactions=(),
                description=f"{desc} (ref engines: LOBO/BIC-positive blocks only)",
                group="B_PROTECT",
            )
        )

    specs.append(
        ModelSpec(
            name="B_PROTECT_support_mandatory_vaccination_LEGIT",
            outcome="support_mandatory_vaccination",
            model_type="ordinal",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                blocks["psych_5c"],
                risk_no_fear,
                blocks["trust"],
                blocks["norms"],
                blocks["info"],
                blocks["contact_behaviour"],
                blocks["vax_conversation"],
                legitimacy_block,
            ),
            interactions=(),
            description="Support for mandatory vaccination + legitimacy/scepticism engine predictor",
            group="B_PROTECT",
        )
    )
    specs.append(
        ModelSpec(
            name="P_5C_fivec_pro_vax",
            outcome="fivec_pro_vax_idx",
            model_type="linear",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                blocks["trust"],
                blocks["info"],
                blocks["norms"],
            ),
            interactions=(),
            description="Determinants of 5C pro-vaccine orientation",
            group="P_5C",
        )
    )
    specs.append(
        ModelSpec(
            name="P_5C_fivec_pro_vax_REF",
            outcome="fivec_pro_vax_idx",
            model_type="linear",
            blocks=(
                demography_5c_ref,
                trust_5c_ref,
                info_social_ref,
                norms_5c_ref,
            ),
            interactions=(),
            description="Determinants of 5C pro-vaccine orientation (ref: reduced blocks)",
            group="P_5C",
        )
    )
    if include_5c_components:
        for name_suffix, outcome in (
            ("fivec_confidence", "fivec_confidence"),
            ("fivec_low_complacency_rev", "fivec_low_complacency"),
            ("fivec_low_constraints_rev", "fivec_low_constraints"),
            ("fivec_collective_responsibility", "fivec_collective_responsibility"),
        ):
            component_blocks = (
                blocks["demography"],
                health_no_vax_behaviour,
                blocks["trust"],
                blocks["info"],
                blocks["norms"],
            )
            if name_suffix == "fivec_low_constraints_rev":
                component_blocks = component_blocks + (blocks["accessibility"],)
            specs.append(
                ModelSpec(
                    name=f"P_5C_{name_suffix}",
                    outcome=outcome,
                    model_type="linear",
                    blocks=component_blocks,
                    interactions=(),
                    description=f"{outcome} component",
                    group="P_5C",
                )
            )
            if name_suffix == "fivec_low_constraints_rev":
                # LOBO for P_5C_fivec_low_constraints_rev assigns strong weight to
                # controls_demography_SES and accessibility_engine, followed by
                # moral/trust/info; health_history_engine is comparatively small.
                specs.append(
                    ModelSpec(
                        name="P_5C_fivec_low_constraints_rev_REF",
                        outcome=outcome,
                        model_type="linear",
                        blocks=(
                            demography_5c_constraints_ref,
                            trust_group_ref,
                            info_5c_constraints_ref,
                            accessibility_5c_constraints_ref,
                            norms_5c_constraints_ref,
                        ),
                        interactions=(),
                        description="fivec_low_constraints component (ref: reduced blocks; LOBO-guided)",
                        group="P_5C",
                    )
                )

    specs.append(
        ModelSpec(
            name="P_TRUST_institutional_trust",
            outcome="institutional_trust_avg",
            model_type="linear",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                country_block,
                politics_block,
                blocks["norms"],
                blocks["info"],
            ),
            interactions=(),
            description="Drivers of institutional trust",
            group="P_TRUST",
        )
    )
    specs.append(
        ModelSpec(
            name="P_TRUST_institutional_trust_REF",
            outcome="institutional_trust_avg",
            model_type="linear",
            blocks=(
                demography_trust_ref,
                health_no_vax_trust_ref,
                politics_block,
                norms_trust_ref,
                info_trust_ref,
            ),
            interactions=(),
            description="Drivers of institutional trust (ref: reduced blocks)",
            group="P_TRUST",
        )
    )

    specs.append(
        ModelSpec(
            name="P_TRUST_group_trust_vax",
            outcome="group_trust_vax",
            model_type="linear",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                country_block,
                politics_block,
                norms_no_peer_norms,
                blocks["info"],
                norms_activation,
                blocks["contact_capacity"],
                institutional_trust_no_group,
            ),
            interactions=(),
            description="Trust in vaccine info from social contacts (peer epistemic trust)",
            group="P_TRUST",
        )
    )
    specs.append(
        ModelSpec(
            name="P_TRUST_group_trust_vax_REF",
            outcome="group_trust_vax",
            model_type="linear",
            blocks=(
                trust_group_ref,
                norms_no_peer_ref,
                norms_activation_ref,
                info_group_trust_ref,
            ),
            interactions=(),
            description="Trust in vaccine info from social contacts (ref: reduced blocks)",
            group="P_TRUST",
        )
    )

    specs.append(
        ModelSpec(
            name="P_LEGIT_covax_legitimacy",
            outcome="covax_legitimacy_scepticism_idx",
            model_type="linear",
            blocks=(
                blocks["demography"],
                blocks["migration"],
                country_block,
                politics_block,
                blocks["trust"],
                blocks["info"],
                norms_no_peer_norms,
            ),
            interactions=(),
            description="Determinants of legitimacy / scepticism engine (covax reasons composite)",
            group="P_LEGIT",
        )
    )
    specs.append(
        ModelSpec(
            name="P_LEGIT_covax_legitimacy_REF",
            outcome="covax_legitimacy_scepticism_idx",
            model_type="linear",
            blocks=(
                trust_5c_ref,
                politics_block,
            ),
            interactions=(),
            description="Determinants of legitimacy / scepticism engine (ref: trust-only reduced blocks)",
            group="P_LEGIT",
        )
    )

    specs.append(
        ModelSpec(
            name="P_RISK_vaccine_risk",
            outcome="vaccine_risk_avg",
            model_type="linear",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                blocks["trust"],
                blocks["psych_5c"],
                blocks["info"],
            ),
            interactions=(),
            description="Perceived vaccine risk",
            group="P_RISK",
        )
    )
    specs.append(
        ModelSpec(
            name="P_RISK_vaccine_risk_REF",
            outcome="vaccine_risk_avg",
            model_type="linear",
            blocks=(
                psych_5c_risk_ref,
                health_no_vax_risk_ref,
                info_social_ref,
            ),
            interactions=(),
            description="Perceived vaccine risk (ref: reduced blocks)",
            group="P_RISK",
        )
    )
    specs.append(
        ModelSpec(
            name="P_RISK_vaccine_fear",
            outcome="vaccine_fear_avg",
            model_type="linear",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                blocks["trust"],
                blocks["psych_5c"],
                blocks["info"],
            ),
            interactions=(),
            description="Perceived vaccine fear",
            group="P_RISK",
        )
    )

    specs.append(
        ModelSpec(
            name="P_NORMS_family_norms",
            outcome="family_progressive_norms_nonvax_idx",
            model_type="linear",
            blocks=(blocks["demography"], country_block, politics_block, norms_no_peer_norms),
            interactions=(),
            description="Family progressive norms (non-vaccine items)",
            group="P_NORMS",
        )
    )

    specs.append(
        ModelSpec(
            name="P_NORMS_family_norms_ACTIVATION",
            outcome="family_progressive_norms_nonvax_idx",
            model_type="linear",
            blocks=(
                blocks["demography"],
                country_block,
                politics_block,
                norms_no_peer_norms,
                norms_activation,
            ),
            interactions=(),
            description="Family progressive norms + activation/embeddedness predictors",
            group="P_NORMS",
        )
    )
    specs.append(
        ModelSpec(
            name="P_NORMS_friends_norms",
            outcome="friends_progressive_norms_nonvax_idx",
            model_type="linear",
            blocks=(
                blocks["demography"],
                country_block,
                politics_block,
                norms_no_peer_norms,
                blocks["contact_capacity"],
            ),
            interactions=(),
            description="Friends progressive norms (non-vaccine items)",
            group="P_NORMS",
        )
    )

    specs.append(
        ModelSpec(
            name="P_NORMS_friends_norms_ACTIVATION",
            outcome="friends_progressive_norms_nonvax_idx",
            model_type="linear",
            blocks=(
                blocks["demography"],
                country_block,
                politics_block,
                norms_no_peer_norms,
                norms_activation,
                blocks["contact_capacity"],
            ),
            interactions=(),
            description="Friends progressive norms + activation/embeddedness predictors",
            group="P_NORMS",
        )
    )
    specs.append(
        ModelSpec(
            name="P_NORMS_colleagues_norms",
            outcome="colleagues_progressive_norms_nonvax_idx",
            model_type="linear",
            blocks=(
                blocks["demography"],
                country_block,
                politics_block,
                norms_no_peer_norms,
                blocks["contact_capacity"],
            ),
            interactions=(),
            description="Colleagues progressive norms (non-vaccine items)",
            group="P_NORMS",
        )
    )

    specs.append(
        ModelSpec(
            name="P_NORMS_colleagues_norms_ACTIVATION",
            outcome="colleagues_progressive_norms_nonvax_idx",
            model_type="linear",
            blocks=(
                blocks["demography"],
                country_block,
                politics_block,
                norms_no_peer_norms,
                norms_activation,
                blocks["contact_capacity"],
            ),
            interactions=(),
            description="Colleagues progressive norms + activation/embeddedness predictors",
            group="P_NORMS",
        )
    )

    specs.append(
        ModelSpec(
            name="P_NORMS_social_norms_vax_avg",
            outcome="social_norms_vax_avg",
            model_type="linear",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                country_block,
                politics_block,
                norms_no_peer_norms,
                blocks["info"],
                norms_activation,
                blocks["contact_capacity"],
            ),
            interactions=(),
            description="Perceived vaccination social norms (avg)",
            group="P_NORMS",
        )
    )
    specs.append(
        ModelSpec(
            name="P_NORMS_social_norms_vax_avg_REF",
            outcome="social_norms_vax_avg",
            model_type="linear",
            blocks=(
                demography_norms_ref,
                norms_no_peer_ref,
                info_norms_ref,
                norms_activation_ref,
            ),
            interactions=(),
            description="Perceived vaccination social norms (avg) (ref: reduced blocks)",
            group="P_NORMS",
        )
    )

    specs.append(
        ModelSpec(
            name="P_COVDANGER_PRE_covid_perceived_danger_pre",
            outcome="covid_perceived_danger_pre",
            model_type="ordinal",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                politics_block,
                blocks["norms"],
                blocks["info"],
            ),
            interactions=(),
            description="Baseline disease risk perception prior to COVID vaccine rollout",
            group="P_COVDANGER_PRE",
        )
    )

    specs.append(
        ModelSpec(
            name="P_STAKES_covid_perceived_danger_Y",
            outcome="covid_perceived_danger_Y",
            model_type="ordinal",
            blocks=(
                blocks["demography"],
                blocks["migration"],
                health_no_vax_behaviour,
                politics_block,
                blocks["trust"],
                blocks["info"],
                norms_no_peer_norms,
            ),
            interactions=(),
            description="Infection stakes: perceived COVID danger (current, Y)",
            group="P_STAKES",
        )
    )
    specs.append(
        ModelSpec(
            name="P_STAKES_covid_perceived_danger_Y_REF",
            outcome="covid_perceived_danger_Y",
            model_type="ordinal",
            blocks=(
                demography_stakes_ref,
                health_no_vax_stakes_ref,
                politics_block,
                trust_stakes_ref,
            ),
            interactions=(),
            description="Infection stakes: perceived COVID danger (ref: reduced blocks)",
            group="P_STAKES",
        )
    )
    specs.append(
        ModelSpec(
            name="P_STAKES_covid_perceived_danger_T12",
            outcome="covid_perceived_danger_T12",
            model_type="ordinal",
            blocks=(
                blocks["demography"],
                blocks["migration"],
                health_no_vax_behaviour,
                politics_block,
                blocks["trust"],
                blocks["info"],
                norms_no_peer_norms,
            ),
            interactions=(),
            description="Infection stakes: perceived COVID danger (T12)",
            group="P_STAKES",
        )
    )
    specs.append(
        ModelSpec(
            name="P_STAKES_covid_perceived_danger_T12_REF",
            outcome="covid_perceived_danger_T12",
            model_type="ordinal",
            blocks=(
                demography_stakes_ref,
                health_no_vax_stakes_ref,
                politics_block,
                trust_stakes_ref,
            ),
            interactions=(),
            description="Infection stakes: perceived COVID danger T12 (ref: reduced blocks)",
            group="P_STAKES",
        )
    )

    specs.append(
        ModelSpec(
            name="P_STAKES_covid_disease_fear",
            outcome="covid_disease_fear",
            model_type="ordinal",
            blocks=(
                blocks["demography"],
                blocks["migration"],
                health_no_vax_behaviour,
                politics_block,
                blocks["trust"],
                blocks["info"],
                norms_no_peer_norms,
            ),
            interactions=(),
            description="Infection stakes: COVID disease fear",
            group="P_STAKES",
        )
    )

    specs.append(
        ModelSpec(
            name="S_ATTCHAIN_att_vaccines_importance",
            outcome="att_vaccines_importance",
            model_type="ordinal",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                _attitude_chain_block(),
                blocks["norms"],
                blocks["info"],
            ),
            interactions=(),
            description="Trust/5C/risk to vaccine importance attitudes",
            group="S_ATT_CHAIN",
        )
    )
    specs.append(
        ModelSpec(
            name="S_ATTCHAIN_vaccine_fear",
            outcome="vaccine_fear_avg",
            model_type="linear",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                _attitude_chain_block(),
                blocks["norms"],
                blocks["info"],
            ),
            interactions=(),
            description="Trust/5C/risk to vaccine fear",
            group="S_ATT_CHAIN",
        )
    )
    specs.append(
        ModelSpec(
            name="S_COUNTRY_vax_willingness_Y",
            outcome="vax_willingness_Y",
            model_type="ordinal",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                blocks["psych_5c"],
                blocks["risk"],
                blocks["trust"],
                blocks["norms"],
                country_block,
            ),
            interactions=(
                ("fivec_pro_vax_idx", "country"),
                ("institutional_trust_avg", "country"),
                ("family_progressive_norms_nonvax_idx", "country"),
            ),
            description="Country-heterogeneous slopes for current willingness (Y)",
            group="S_COUNTRY_HETEROGENEITY",
        )
    )
    specs.append(
        ModelSpec(
            name="S_COUNTRY_vax_willingness_T4",
            outcome="vax_willingness_T4",
            model_type="ordinal",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                blocks["psych_5c"],
                blocks["risk"],
                blocks["trust"],
                blocks["norms"],
                country_block,
            ),
            interactions=(
                ("fivec_pro_vax_idx", "country"),
                ("institutional_trust_avg", "country"),
                ("family_progressive_norms_nonvax_idx", "country"),
            ),
            description="Supporting heterogeneity at 4th-dose willingness (selective sample)",
            group="S_COUNTRY_HETEROGENEITY",
        )
    )
    specs.append(
        ModelSpec(
            name="S_COUNTRY_flu_vaccinated",
            outcome="flu_vaccinated_2023_2024",
            model_type="logit",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                blocks["psych_5c"],
                blocks["risk"],
                blocks["trust"],
                blocks["norms"],
                country_block,
                _flu_block(),
            ),
            interactions=(("fivec_pro_vax_idx", "country"), ("institutional_trust_avg", "country")),
            description="Country-heterogeneous slopes for flu",
            group="S_COUNTRY_HETEROGENEITY",
        )
    )

    specs.append(
        ModelSpec(
            name="S_COUNTRY_flu_vaccinated_HABIT",
            outcome="flu_vaccinated_2023_2024",
            model_type="logit",
            blocks=(
                blocks["demography"],
                health_no_vax_behaviour,
                blocks["flu_habit"],
                blocks["psych_5c"],
                blocks["risk"],
                blocks["trust"],
                blocks["norms"],
                country_block,
                _flu_block(),
            ),
            interactions=(
                ("fivec_pro_vax_idx", "country"),
                ("institutional_trust_avg", "country"),
            ),
            description="Country-heterogeneous slopes for flu with explicit inertia/state-dependence",
            group="S_COUNTRY_HETEROGENEITY",
        )
    )

    specs.append(
        ModelSpec(
            name="S_PORTABILITY_vax_willingness_T12_ENGINES_REF",
            outcome="vax_willingness_T12",
            model_type="ordinal",
            blocks=(
                controls_demography_SES,
                trust_engine,
                disposition_engine,
                stakes_engine_renamed_T12,
                norms_engine,
                legitimacy_engine,
            ),
            interactions=(
                ("covid_perceived_danger_T12", "country"),
                ("fivec_pro_vax_idx", "country"),
                ("institutional_trust_avg", "country"),
            ),
            description="Portability diagnostic for T12 willingness: country-varying slopes on top ref engines",
            group="S_PORTABILITY",
        )
    )

    specs.append(
        ModelSpec(
            name="S_PORTABILITY_flu_vaccinated_2023_2024_ENGINES_HABIT",
            outcome="flu_vaccinated_2023_2024",
            model_type="logit",
            blocks=(
                controls_demography_SES,
                habit_engine,
                trust_engine,
                disposition_engine,
                flu_stakes_engine_renamed,
                norms_engine,
            ),
            interactions=(
                ("flu_vaccinated_2022_2023", "country"),
                ("flu_disease_fear", "country"),
                ("fivec_pro_vax_idx", "country"),
            ),
            description="Portability diagnostic for flu uptake: country-varying slopes on habit and appraisal engines",
            group="S_PORTABILITY",
        )
    )

    for suffix, outcome, desc in (
        (
            "mask_when_symptomatic_crowded",
            "mask_when_symptomatic_crowded",
            "masking when symptomatic in crowded places",
        ),
        (
            "stay_home_when_symptomatic",
            "stay_home_when_symptomatic",
            "staying home when symptomatic",
        ),
        (
            "mask_when_pressure_high",
            "mask_when_pressure_high",
            "masking under high epidemic pressure",
        ),
    ):
        specs.append(
            ModelSpec(
                name=f"S_PORTABILITY_{suffix}_ENGINES_REF",
                outcome=outcome,
                model_type="ordinal",
                blocks=(
                    controls_demography_SES,
                    disposition_engine_no_index,
                    stakes_engine_split,
                    trust_engine_with_group_renamed,
                    norms_engine_split,
                    moral_engine_split,
                ),
                interactions=(
                    ("moral_progressive_orientation", "country"),
                    ("disease_fear_avg", "country"),
                    ("institutional_trust_avg", "country"),
                ),
                description=f"Portability diagnostic for {desc}: country-varying slopes on top ref engines",
                group="S_PORTABILITY",
            )
        )

    prevention_backbone = (
        blocks["demography"],
        blocks["health"],
        blocks["psych_5c"],
        blocks["risk"],
        blocks["attitudes"],
        blocks["trust"],
        blocks["norms"],
        blocks["info"],
        blocks["contact_capacity"],
    )
    specs.append(
        ModelSpec(
            name="S_PREVENTION_flu_joint",
            outcome="flu_vaccinated_2023_2024",
            model_type="logit",
            blocks=prevention_backbone,
            interactions=(),
            description="Joint prevention backbone (S_PREVENTION): flu uptake",
            group="S_PREVENTION",
        )
    )
    health_no_covax = ModelBlock(
        "health_history_engine",
        tuple(item for item in blocks["health"].predictors if item != "covax_num"),
        blocks["health"].sample_strategy,
    )
    prevention_backbone_no_covax = (
        blocks["demography"],
        health_no_covax,
        blocks["psych_5c"],
        blocks["risk"],
        blocks["attitudes"],
        blocks["trust"],
        blocks["norms"],
        blocks["info"],
        blocks["contact_capacity"],
    )
    specs.append(
        ModelSpec(
            name="S_PREVENTION_covax_joint",
            outcome="covax_num",
            model_type="ordinal",
            blocks=prevention_backbone_no_covax,
            interactions=(),
            description="Joint prevention backbone (S_PREVENTION): COVID doses",
            group="S_PREVENTION",
        )
    )
    specs.append(
        ModelSpec(
            name="S_PREVENTION_support_mandates_joint",
            outcome="support_mandatory_vaccination",
            model_type="ordinal",
            blocks=prevention_backbone,
            interactions=(),
            description="Joint prevention backbone (S_PREVENTION): mandate support",
            group="S_PREVENTION",
        )
    )

    specs.append(
        ModelSpec(
            name="S_PREVENTION_support_mandates_joint_LEGIT",
            outcome="support_mandatory_vaccination",
            model_type="ordinal",
            blocks=prevention_backbone + (legitimacy_block,),
            interactions=(),
            description="Joint prevention backbone (S_PREVENTION): mandate support + legitimacy/scepticism engine predictor",
            group="S_PREVENTION",
        )
    )

    prevention_engines = (
        blocks["demography"],
        blocks["health"],
        trust_engine,
        vax_disposition_engine,
        cost_engine,
        stakes_engine,
        norms_engine_prox,
        legitimacy_block,
    )
    specs.append(
        ModelSpec(
            name="S_PREVENTION_support_mandates_joint_ENGINES",
            outcome="support_mandatory_vaccination",
            model_type="ordinal",
            blocks=prevention_engines,
            interactions=(),
            description="Compressed S_PREVENTION mandate support backbone (engines)",
            group="S_PREVENTION",
        )
    )

    expanded: list[ModelSpec] = []
    for spec in specs:
        expanded.append(_canonicalize_blocks(_without_country(spec)))
        if _spec_uses_peer_norm_indices_as_predictors(spec):
            expanded.append(_canonicalize_blocks(_without_country(_with_peer_norm_items(spec))))

    return expanded
