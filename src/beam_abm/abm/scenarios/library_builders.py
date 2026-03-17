"""Scenario library builders split from scenario_defs."""

from __future__ import annotations


def build_upstream_signal_scenario_library(
    *,
    ScenarioConfig: type,
    SignalSpec: type,
    IncidenceSpec: type,
    months,
) -> dict[str, object]:
    """Construct canonical scenario suite only."""
    return {
        "baseline": ScenarioConfig(
            name="baseline",
            display_label="Baseline",
            family="baseline",
            description="No exogenous shocks; communication and endogenous loop enabled.",
            state_signals=(),
            choice_signals=(),
            communication_enabled=True,
            peer_pull_enabled=True,
            observed_norms_enabled=True,
            enable_endogenous_loop=True,
            incidence_mode="none",
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "null_world": ScenarioConfig(
            name="null_world",
            display_label="Null World",
            family="diagnostic",
            description="Everything off: no shocks, no communication, no endogenous loop.",
            state_signals=(),
            choice_signals=(),
            communication_enabled=False,
            peer_pull_enabled=False,
            observed_norms_enabled=False,
            enable_endogenous_loop=False,
            state_updates_enabled=False,
            incidence_mode="none",
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "social_only": ScenarioConfig(
            name="social_only",
            display_label="Social Only",
            family="diagnostic",
            description="Social communication only with endogenous incidence loop disabled.",
            state_signals=(),
            choice_signals=(),
            communication_enabled=True,
            peer_pull_enabled=True,
            observed_norms_enabled=True,
            enable_endogenous_loop=False,
            incidence_mode="none",
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "baseline_no_communication": ScenarioConfig(
            name="baseline_no_communication",
            display_label="Baseline No Communication",
            family="baseline",
            description="No exogenous shocks; communication disabled for clean network-effects comparison.",
            state_signals=(),
            choice_signals=(),
            communication_enabled=False,
            peer_pull_enabled=False,
            observed_norms_enabled=False,
            enable_endogenous_loop=True,
            incidence_mode="none",
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "baseline_no_peer_pull": ScenarioConfig(
            name="baseline_no_peer_pull",
            display_label="Baseline No Peer Pull",
            family="baseline",
            description=(
                "No exogenous shocks; communication enabled with observed norms only (peer-state pull disabled)."
            ),
            state_signals=(),
            choice_signals=(),
            communication_enabled=True,
            peer_pull_enabled=False,
            observed_norms_enabled=True,
            enable_endogenous_loop=True,
            incidence_mode="none",
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "baseline_no_observed_norms": ScenarioConfig(
            name="baseline_no_observed_norms",
            display_label="Baseline No Observed Norms",
            family="baseline",
            description=(
                "No exogenous shocks; communication enabled with peer-state pull only "
                "(observed descriptive norms disabled)."
            ),
            state_signals=(),
            choice_signals=(),
            communication_enabled=True,
            peer_pull_enabled=True,
            observed_norms_enabled=False,
            enable_endogenous_loop=True,
            incidence_mode="none",
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "baseline_importation": ScenarioConfig(
            name="baseline_importation",
            display_label="Baseline Importation",
            family="baseline",
            description="No exogenous shocks; communication and endogenous loop enabled with importation forcing.",
            state_signals=(),
            choice_signals=(),
            communication_enabled=True,
            peer_pull_enabled=True,
            observed_norms_enabled=True,
            enable_endogenous_loop=True,
            incidence_mode="importation",
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "baseline_importation_no_communication": ScenarioConfig(
            name="baseline_importation_no_communication",
            display_label="Baseline Importation No Communication",
            family="baseline",
            description="No exogenous shocks; communication disabled with importation forcing.",
            state_signals=(),
            choice_signals=(),
            communication_enabled=False,
            peer_pull_enabled=False,
            observed_norms_enabled=False,
            enable_endogenous_loop=True,
            incidence_mode="importation",
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "baseline_importation_no_peer_pull": ScenarioConfig(
            name="baseline_importation_no_peer_pull",
            display_label="Baseline Importation No Peer Pull",
            family="baseline",
            description=(
                "No exogenous shocks; importation forcing with observed norms only (peer-state pull disabled)."
            ),
            state_signals=(),
            choice_signals=(),
            communication_enabled=True,
            peer_pull_enabled=False,
            observed_norms_enabled=True,
            enable_endogenous_loop=True,
            incidence_mode="importation",
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "baseline_importation_no_observed_norms": ScenarioConfig(
            name="baseline_importation_no_observed_norms",
            display_label="Baseline Importation No Observed Norms",
            family="baseline",
            description=(
                "No exogenous shocks; importation forcing with peer-state pull only "
                "(observed descriptive norms disabled)."
            ),
            state_signals=(),
            choice_signals=(),
            communication_enabled=True,
            peer_pull_enabled=True,
            observed_norms_enabled=False,
            enable_endogenous_loop=True,
            incidence_mode="importation",
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "outbreak_salience_global": ScenarioConfig(
            name="outbreak_salience_global",
            display_label="Outbreak salience (C)",
            family="information",
            description=(
                "Global outbreak salience proxy via illness-experience and vulnerability campaigns "
                "(no additional incidence-wave forcing)."
            ),
            where="global",
            targeting="uniform",
            state_signals=(
                SignalSpec(
                    target_column="exper_illness",
                    intensity=1.0,
                    signal_id="severe_illness_vicarious",
                    signal_type="campaign",
                ),
                SignalSpec(
                    target_column="health_poor", intensity=1.0, signal_id="health_deterioration", signal_type="campaign"
                ),
            ),
            incidence_mode="importation",
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "outbreak_global_incidence_moderate": ScenarioConfig(
            name="outbreak_global_incidence_moderate",
            display_label="Outbreak Global Incidence Moderate",
            family="information",
            description="Global incidence-wave forcing variant (moderate target, 6-month active window).",
            where="global",
            targeting="uniform",
            state_signals=(),
            incidence_mode="importation",
            importation_multiplier=3.0,
            incidence_schedule=(
                IncidenceSpec(
                    component="importation_multiplier",
                    start_tick=months(11),
                    end_tick=months(17),
                ),
            ),
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "outbreak_global_incidence_strong": ScenarioConfig(
            name="outbreak_global_incidence_strong",
            display_label="Outbreak wave (S)",
            family="information",
            description="Global incidence-wave forcing variant (strong target, 6-month active window).",
            where="global",
            targeting="uniform",
            state_signals=(),
            incidence_mode="importation",
            importation_multiplier=5.0,
            incidence_schedule=(
                IncidenceSpec(
                    component="importation_multiplier",
                    start_tick=months(11),
                    end_tick=months(17),
                ),
            ),
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "outbreak_local_incidence_moderate": ScenarioConfig(
            name="outbreak_local_incidence_moderate",
            display_label="Outbreak Local Incidence Moderate",
            family="information",
            description="Localized incidence-wave forcing variant (moderate target, 6-month active window).",
            where="local",
            targeting="targeted",
            seed_rule="community_targeted",
            state_signals=(),
            incidence_mode="importation",
            importation_multiplier=3.0,
            incidence_schedule=(
                IncidenceSpec(
                    component="importation_multiplier",
                    start_tick=months(11),
                    end_tick=months(17),
                ),
            ),
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "outbreak_local_incidence_strong": ScenarioConfig(
            name="outbreak_local_incidence_strong",
            display_label="Outbreak Local Incidence Strong",
            family="information",
            description="Localized incidence-wave forcing variant (strong target, 6-month active window).",
            where="local",
            targeting="targeted",
            seed_rule="community_targeted",
            state_signals=(),
            incidence_mode="importation",
            importation_multiplier=5.0,
            incidence_schedule=(
                IncidenceSpec(
                    component="importation_multiplier",
                    start_tick=months(11),
                    end_tick=months(17),
                ),
            ),
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "outbreak_local_salience": ScenarioConfig(
            name="outbreak_local_salience",
            display_label="Outbreak Local Salience",
            family="information",
            description="Localized illness salience/vulnerability campaign only (no local incidence bump).",
            where="local",
            targeting="targeted",
            seed_rule="community_targeted",
            state_signals=(
                SignalSpec(
                    target_column="exper_illness",
                    intensity=1.0,
                    signal_id="severe_illness_vicarious",
                    signal_type="campaign",
                ),
                SignalSpec(
                    target_column="health_poor", intensity=1.0, signal_id="health_deterioration", signal_type="campaign"
                ),
            ),
            incidence_mode="importation",
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "trust_scandal_institutional": ScenarioConfig(
            name="trust_scandal_institutional",
            display_label="Institutional trust scandal (P)",
            family="information",
            description="One-time negative institutional credibility shock.",
            where="global",
            state_signals=(
                SignalSpec(
                    target_column="institutional_trust_avg",
                    intensity=-1.0,
                    signal_id="pe_mutable_engine_institutional_trust_shift",
                    signal_type="pulse",
                ),
            ),
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "trust_repair_institutional": ScenarioConfig(
            name="trust_repair_institutional",
            display_label="Institutional trust repair (P)",
            family="information",
            description="One-time positive institutional credibility repair shock.",
            where="global",
            state_signals=(
                SignalSpec(
                    target_column="institutional_trust_avg",
                    intensity=1.0,
                    signal_id="pe_mutable_engine_institutional_trust_shift",
                    signal_type="pulse",
                ),
            ),
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "trust_scandal_legitimacy": ScenarioConfig(
            name="trust_scandal_legitimacy",
            display_label="Legitimacy scandal (P)",
            family="information",
            description="One-time process-legitimacy scandal (higher scepticism).",
            where="global",
            state_signals=(
                SignalSpec(
                    target_column="covax_legitimacy_scepticism_idx",
                    intensity=1.0,
                    signal_id="pe_mutable_engine_legitimacy_scepticism_shift",
                    signal_type="pulse",
                ),
            ),
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "trust_repair_legitimacy": ScenarioConfig(
            name="trust_repair_legitimacy",
            display_label="Legitimacy repair (P)",
            family="information",
            description="One-time legitimacy repair shock (lower scepticism).",
            where="global",
            state_signals=(
                SignalSpec(
                    target_column="covax_legitimacy_scepticism_idx",
                    intensity=-1.0,
                    signal_id="pe_mutable_engine_legitimacy_scepticism_shift",
                    signal_type="pulse",
                ),
            ),
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "trust_scandal_group_vax": ScenarioConfig(
            name="trust_scandal_group_vax",
            display_label="Trust scandal: group trust (P)",
            family="information",
            description="One-time negative group-trust shock in vaccination context.",
            where="global",
            state_signals=(
                SignalSpec(
                    target_column="group_trust_vax",
                    intensity=-1.0,
                    signal_id="pe_mutable_engine_group_trust_shift",
                    signal_type="pulse",
                ),
            ),
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "trust_repair_group_vax": ScenarioConfig(
            name="trust_repair_group_vax",
            display_label="Trust repair: group trust (P)",
            family="information",
            description="One-time positive group-trust repair shock in vaccination context.",
            where="global",
            state_signals=(
                SignalSpec(
                    target_column="group_trust_vax",
                    intensity=1.0,
                    signal_id="pe_mutable_engine_group_trust_shift",
                    signal_type="pulse",
                ),
            ),
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "trust_scandal_socialinfo": ScenarioConfig(
            name="trust_scandal_socialinfo",
            display_label="Trust scandal: social information (P)",
            family="information",
            description="One-time decrease in trust toward social information channels.",
            where="global",
            state_signals=(
                SignalSpec(
                    target_column="trust_social",
                    intensity=-1.0,
                    signal_id="trust_social_shift_down",
                    signal_type="pulse",
                ),
            ),
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "trust_repair_socialinfo": ScenarioConfig(
            name="trust_repair_socialinfo",
            display_label="Trust repair: social information (P)",
            family="information",
            description="One-time increase in trust toward social information channels.",
            where="global",
            state_signals=(
                SignalSpec(
                    target_column="trust_social",
                    intensity=1.0,
                    signal_id="trust_social_shift",
                    signal_type="pulse",
                ),
            ),
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "trust_scandal_onlineinfo": ScenarioConfig(
            name="trust_scandal_onlineinfo",
            display_label="Trust scandal: online information (P)",
            family="information",
            description="One-time decrease in trust toward online information channels.",
            where="global",
            state_signals=(
                SignalSpec(
                    target_column="trust_onlineinfo",
                    intensity=-1.0,
                    signal_id="trust_onlineinfo_shift_down",
                    signal_type="pulse",
                ),
            ),
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "trust_repair_onlineinfo": ScenarioConfig(
            name="trust_repair_onlineinfo",
            display_label="Trust repair: online information (P)",
            family="information",
            description="One-time increase in trust toward online information channels.",
            where="global",
            state_signals=(
                SignalSpec(
                    target_column="trust_onlineinfo",
                    intensity=1.0,
                    signal_id="trust_onlineinfo_shift",
                    signal_type="pulse",
                ),
            ),
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "trust_scandal_bundle": ScenarioConfig(
            name="trust_scandal_bundle",
            display_label="Trust scandal (P)",
            family="information",
            description="Bundled trust scandal: institutional credibility down, legitimacy scepticism up, group trust down.",
            where="global",
            state_signals=(
                SignalSpec(
                    target_column="institutional_trust_avg",
                    intensity=-1.0,
                    signal_id="pe_mutable_engine_institutional_trust_shift",
                    signal_type="pulse",
                ),
                SignalSpec(
                    target_column="covax_legitimacy_scepticism_idx",
                    intensity=1.0,
                    signal_id="pe_mutable_engine_legitimacy_scepticism_shift",
                    signal_type="pulse",
                ),
                SignalSpec(
                    target_column="group_trust_vax",
                    intensity=-1.0,
                    signal_id="pe_mutable_engine_group_trust_shift",
                    signal_type="pulse",
                ),
            ),
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "trust_repair_bundle": ScenarioConfig(
            name="trust_repair_bundle",
            display_label="Trust repair (P)",
            family="information",
            description="Bundled trust repair: institutional credibility up, legitimacy scepticism down, group trust up.",
            where="global",
            state_signals=(
                SignalSpec(
                    target_column="institutional_trust_avg",
                    intensity=1.0,
                    signal_id="pe_mutable_engine_institutional_trust_shift",
                    signal_type="pulse",
                ),
                SignalSpec(
                    target_column="covax_legitimacy_scepticism_idx",
                    intensity=-1.0,
                    signal_id="pe_mutable_engine_legitimacy_scepticism_shift",
                    signal_type="pulse",
                ),
                SignalSpec(
                    target_column="group_trust_vax",
                    intensity=1.0,
                    signal_id="pe_mutable_engine_group_trust_shift",
                    signal_type="pulse",
                ),
            ),
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "misinfo_shock": ScenarioConfig(
            name="misinfo_shock",
            display_label="Misinfo Shock",
            family="information",
            description="Misinformation shock via institutional credibility erosion and side-effect activation.",
            where="global",
            state_signals=(
                SignalSpec(
                    target_column="trust_traditional",
                    intensity=1.0,
                    signal_id="trust_traditional_shift_down",
                    signal_type="pulse",
                ),
                SignalSpec(
                    target_column="trust_gp_vax_info",
                    intensity=1.0,
                    signal_id="trust_gp_vax_info_shift_down",
                    signal_type="pulse",
                ),
                SignalSpec(
                    target_column="trust_who_vax_info",
                    intensity=1.0,
                    signal_id="trust_who_vax_info_shift_down",
                    signal_type="pulse",
                ),
                SignalSpec(
                    target_column="trust_national_gov_vax_info",
                    intensity=1.0,
                    signal_id="trust_national_gov_vax_info_shift_down",
                    signal_type="pulse",
                ),
                SignalSpec(
                    target_column="side_effects_other",
                    intensity=1.0,
                    signal_id="side_effects_other_vicarious",
                    signal_type="campaign",
                ),
            ),
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "misinfo_trust_erosion_only": ScenarioConfig(
            name="misinfo_trust_erosion_only",
            display_label="Misinfo (trust erosion) (P)",
            family="information",
            description="Misinformation decomposition: trust erosion only (no side-effect activation).",
            where="global",
            state_signals=(
                SignalSpec(
                    target_column="trust_traditional",
                    intensity=1.0,
                    signal_id="trust_traditional_shift_down",
                    signal_type="pulse",
                ),
                SignalSpec(
                    target_column="trust_gp_vax_info",
                    intensity=1.0,
                    signal_id="trust_gp_vax_info_shift_down",
                    signal_type="pulse",
                ),
                SignalSpec(
                    target_column="trust_who_vax_info",
                    intensity=1.0,
                    signal_id="trust_who_vax_info_shift_down",
                    signal_type="pulse",
                ),
                SignalSpec(
                    target_column="trust_national_gov_vax_info",
                    intensity=1.0,
                    signal_id="trust_national_gov_vax_info_shift_down",
                    signal_type="pulse",
                ),
            ),
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "side_effects_activation_only": ScenarioConfig(
            name="side_effects_activation_only",
            display_label="Misinfo: side-effect salience (+, campaign) [Campaign]",
            family="information",
            description="Misinformation decomposition: side-effect activation only (no trust erosion).",
            where="global",
            state_signals=(
                SignalSpec(
                    target_column="side_effects_other",
                    intensity=1.0,
                    signal_id="side_effects_other_vicarious",
                    signal_type="campaign",
                ),
            ),
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "misinfo_shock_targeted": ScenarioConfig(
            name="misinfo_shock_targeted",
            display_label="Misinfo Shock Targeted",
            family="information",
            description="Targeted misinformation shock (high-exposure subset) to demonstrate social diffusion.",
            where="local",
            targeting="targeted",
            seed_rule="high_exposure",
            state_signals=(
                SignalSpec(
                    target_column="trust_traditional",
                    intensity=1.0,
                    signal_id="trust_traditional_shift_down",
                    signal_type="pulse",
                ),
                SignalSpec(
                    target_column="trust_gp_vax_info",
                    intensity=1.0,
                    signal_id="trust_gp_vax_info_shift_down",
                    signal_type="pulse",
                ),
                SignalSpec(
                    target_column="trust_who_vax_info",
                    intensity=1.0,
                    signal_id="trust_who_vax_info_shift_down",
                    signal_type="pulse",
                ),
                SignalSpec(
                    target_column="trust_national_gov_vax_info",
                    intensity=1.0,
                    signal_id="trust_national_gov_vax_info_shift_down",
                    signal_type="pulse",
                ),
                SignalSpec(
                    target_column="side_effects_other",
                    intensity=1.0,
                    signal_id="side_effects_other_vicarious",
                    signal_type="campaign",
                ),
            ),
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "access_intervention": ScenarioConfig(
            name="access_intervention",
            display_label="Access facilitation (C)",
            family="access",
            description="Improves access precursors for booking and vaccination-hub reachability.",
            where="global",
            state_signals=(
                SignalSpec(
                    target_column="booking", intensity=1.0, signal_id="booking_access_improves", signal_type="campaign"
                ),
                SignalSpec(
                    target_column="vax_hub", intensity=1.0, signal_id="vax_hub_access_improves", signal_type="campaign"
                ),
            ),
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "access_barrier_shock": ScenarioConfig(
            name="access_barrier_shock",
            display_label="Access barriers (C)",
            family="access",
            description=(
                "Access barrier shock: booking and vaccination-hub reachability worsen "
                "for a bounded 6-month campaign window."
            ),
            where="global",
            state_signals=(
                SignalSpec(
                    target_column="booking", intensity=1.0, signal_id="booking_access_worsens", signal_type="campaign"
                ),
                SignalSpec(
                    target_column="vax_hub", intensity=1.0, signal_id="vax_hub_access_worsens", signal_type="campaign"
                ),
            ),
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "norm_campaign": ScenarioConfig(
            name="norm_campaign",
            display_label="Norms: pro-vax shift (+, campaign, moderate) [Campaign]",
            family="norm",
            description=(
                "Short norm-shift intervention: direct descriptive-social-norm pulse (2 months) with long decay."
            ),
            where="global",
            state_signals=(
                SignalSpec(
                    target_column="social_norms_descriptive_vax",
                    intensity=1.0,
                    signal_id="pe_mutable_engine_social_norms_descriptive_shift",
                    signal_type="pulse",
                ),
            ),
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "norm_shift_campaign": ScenarioConfig(
            name="norm_shift_campaign",
            display_label="Norm shift (C)",
            family="norm",
            description=(
                "Sustained norm-shift intervention: direct descriptive-social-norm "
                "campaign (12 months, no in-window decay)."
            ),
            where="global",
            state_signals=(
                SignalSpec(
                    target_column="social_norms_descriptive_vax",
                    intensity=1.0,
                    signal_id="pe_mutable_engine_social_norms_descriptive_shift",
                    signal_type="campaign",
                ),
            ),
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "targeted_norm_seed": ScenarioConfig(
            name="targeted_norm_seed",
            display_label="Targeted Norm Seed",
            family="norm",
            description="Small high-degree conversation-frequency pulse to trigger norm cascades.",
            where="local",
            targeting="targeted",
            seed_rule="high_degree",
            state_signals=(
                SignalSpec(
                    target_column="vax_convo_freq_cohabitants",
                    intensity=1.0,
                    signal_id="vax_conversation_freq_shift",
                    signal_type="pulse",
                ),
            ),
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
        "outbreak_misinfo_combined": ScenarioConfig(
            name="outbreak_misinfo_combined",
            display_label="Outbreak Misinfo Combined",
            family="combined",
            description="Outbreak proxy and misinformation precursors jointly active.",
            where="global",
            targeting="uniform",
            state_signals=(
                SignalSpec(
                    target_column="exper_illness",
                    intensity=1.0,
                    signal_id="severe_illness_vicarious",
                    signal_type="campaign",
                ),
                SignalSpec(
                    target_column="health_poor", intensity=1.0, signal_id="health_deterioration", signal_type="campaign"
                ),
                SignalSpec(
                    target_column="trust_traditional",
                    intensity=1.0,
                    signal_id="trust_traditional_shift_down",
                    signal_type="pulse",
                ),
                SignalSpec(
                    target_column="trust_gp_vax_info",
                    intensity=1.0,
                    signal_id="trust_gp_vax_info_shift_down",
                    signal_type="pulse",
                ),
                SignalSpec(
                    target_column="trust_who_vax_info",
                    intensity=1.0,
                    signal_id="trust_who_vax_info_shift_down",
                    signal_type="pulse",
                ),
                SignalSpec(
                    target_column="trust_national_gov_vax_info",
                    intensity=1.0,
                    signal_id="trust_national_gov_vax_info_shift_down",
                    signal_type="pulse",
                ),
                SignalSpec(
                    target_column="side_effects_other",
                    intensity=1.0,
                    signal_id="side_effects_other_vicarious",
                    signal_type="campaign",
                ),
            ),
            incidence_mode="importation",
            supported_kernels=("parametric", "surrogate", "llm"),
        ),
    }


def build_mutable_latent_engine_scenario_library(
    *,
    ScenarioConfig: type,
    SignalSpec: type,
) -> dict[str, object]:
    """Construct PE-style scenarios that directly move mutable latent engines."""
    return {
        "pe_mutable_engine_institutional_trust": ScenarioConfig(
            name="pe_mutable_engine_institutional_trust",
            display_label="Pe Mutable Engine Institutional Trust",
            family="latent_engine_pe",
            description="Direct PE-style shift of institutional trust latent engine.",
            where="global",
            state_signals=(
                SignalSpec(
                    target_column="institutional_trust_avg",
                    intensity=1.0,
                    signal_id="pe_mutable_engine_institutional_trust_shift",
                    signal_type="pulse",
                ),
            ),
            supported_kernels=("parametric",),
        ),
        "pe_mutable_engine_legitimacy_scepticism": ScenarioConfig(
            name="pe_mutable_engine_legitimacy_scepticism",
            display_label="Pe Mutable Engine Legitimacy Scepticism",
            family="latent_engine_pe",
            description="Direct PE-style shift of legitimacy scepticism latent engine.",
            where="global",
            state_signals=(
                SignalSpec(
                    target_column="covax_legitimacy_scepticism_idx",
                    intensity=1.0,
                    signal_id="pe_mutable_engine_legitimacy_scepticism_shift",
                    signal_type="pulse",
                ),
            ),
            supported_kernels=("parametric",),
        ),
        "pe_mutable_engine_vaccine_risk": ScenarioConfig(
            name="pe_mutable_engine_vaccine_risk",
            display_label="Pe Mutable Engine Vaccine Risk",
            family="latent_engine_pe",
            description="Direct PE-style shift of perceived vaccine-risk latent engine.",
            where="global",
            state_signals=(
                SignalSpec(
                    target_column="vaccine_risk_avg",
                    intensity=1.0,
                    signal_id="pe_mutable_engine_vaccine_risk_shift",
                    signal_type="pulse",
                ),
            ),
            supported_kernels=("parametric",),
        ),
        "pe_mutable_engine_covid_danger": ScenarioConfig(
            name="pe_mutable_engine_covid_danger",
            display_label="Pe Mutable Engine Covid Danger",
            family="latent_engine_pe",
            description="Direct PE-style shift of COVID danger latent engine.",
            where="global",
            state_signals=(
                SignalSpec(
                    target_column="covid_perceived_danger_T12",
                    intensity=1.0,
                    signal_id="pe_mutable_engine_covid_danger_shift",
                    signal_type="pulse",
                ),
            ),
            supported_kernels=("parametric",),
        ),
        "pe_mutable_engine_social_norms": ScenarioConfig(
            name="pe_mutable_engine_social_norms",
            display_label="Pe Mutable Engine Social Norms",
            family="latent_engine_pe",
            description="Direct PE-style shift of social-norm latent engine.",
            where="global",
            state_signals=(
                SignalSpec(
                    target_column="social_norms_descriptive_vax",
                    intensity=1.0,
                    signal_id="pe_mutable_engine_social_norms_descriptive_shift",
                    signal_type="pulse",
                ),
            ),
            supported_kernels=("parametric",),
        ),
        "pe_mutable_engine_fivec_constraints": ScenarioConfig(
            name="pe_mutable_engine_fivec_constraints",
            display_label="Pe Mutable Engine Fivec Constraints",
            family="latent_engine_pe",
            description="Direct PE-style shift of 5C low-constraints latent engine.",
            where="global",
            state_signals=(
                SignalSpec(
                    target_column="fivec_low_constraints",
                    intensity=1.0,
                    signal_id="pe_mutable_engine_fivec_constraints_shift",
                    signal_type="pulse",
                ),
            ),
            supported_kernels=("parametric",),
        ),
        "pe_mutable_engine_group_trust": ScenarioConfig(
            name="pe_mutable_engine_group_trust",
            display_label="Pe Mutable Engine Group Trust",
            family="latent_engine_pe",
            description="Direct PE-style shift of group-trust latent engine.",
            where="global",
            state_signals=(
                SignalSpec(
                    target_column="group_trust_vax",
                    intensity=1.0,
                    signal_id="pe_mutable_engine_group_trust_shift",
                    signal_type="pulse",
                ),
            ),
            supported_kernels=("parametric",),
        ),
    }


__all__ = ["build_upstream_signal_scenario_library", "build_mutable_latent_engine_scenario_library"]
