"""Narrative/persona rendering for prompt templates."""

from __future__ import annotations

import math

import pandas as pd

from beam_abm.decision_function.missingness import SPECIAL_STRUCTURAL_PREFIXES


class PromptNarrativeRendererMixin:
    def build_narrative(
        self,
        attrs: dict[str, object],
        driver_cols: list[str] | None,
        strategy: str,
        *,
        perturbed_target: str | None = None,
        perturbed_grid_point: str | None = None,
        perturbed_other_value: object | None = None,
    ) -> str:
        cols = set(driver_cols) if driver_cols else set(attrs.keys())
        first = strategy == "first_person"
        ctx = {
            "subject": "I" if first else "They",
            "subject_lc": "I" if first else "they",
            "be": "am" if first else "are",
            "possessive": "my" if first else "their",
        }

        def _normalize_grid_point(raw: str | None) -> str | None:
            if raw is None:
                return None
            gp = str(raw).strip().lower()
            if gp in {"low", "high"}:
                return gp
            return None

        perturbed_gp = _normalize_grid_point(perturbed_grid_point)
        perturbed_col = str(perturbed_target) if perturbed_target is not None else None
        if perturbed_col is not None and perturbed_col not in cols:
            # If the perturbed column isn't part of the rendered driver set, do nothing.
            perturbed_col = None

        def _append_micro_modifier_if_needed(*, col: str, phrase: str, other_phrase: str | None) -> str:
            """Append a tiny disambiguator iff low/high would otherwise render identical.

            This keeps value→bucket compression intact while guaranteeing low/high are separable.
            """

            if perturbed_gp is None or perturbed_col is None or col != perturbed_col:
                return phrase
            if other_phrase is None:
                return phrase

            if phrase.strip() != other_phrase.strip():
                return phrase

            suffix = "(toward the lower end)" if perturbed_gp == "low" else "(toward the higher end)"
            s = phrase.rstrip()
            if s.endswith("."):
                return s[:-1] + f" {suffix}."
            return s + f" {suffix}"

        narrative_cols = {
            "sex_female",
            "age",
            "M_education",
            "country",
            "income_ppp_norm",
            "minority",
            "M_religion",
            "health_poor",
            "medical_conditions",
            "medical_conditions_c",
            "covid_num",
            "exper_illness",
            "side_effects",
            "side_effects_other",
            "institutional_trust_avg",
            "group_trust_vax",
            "trust_traditional",
            "trust_onlineinfo",
            "trust_social",
            "trust_gp_vax_info",
            "trust_who_vax_info",
            "trust_national_gov_vax_info",
            "trust_national_health_institutions_vax_info",
            "trust_local_health_authorities_vax_info",
            "covax_legitimacy_scepticism_idx",
            "fivec_pro_vax_idx",
            "fivec_confidence",
            "fivec_low_constraints",
            "fivec_low_complacency",
            "fivec_collective_responsibility",
            "vaccine_risk_avg",
            "vaccine_fear_avg",
            "flu_vaccine_risk",
            "flu_disease_fear",
            "covid_perceived_danger_Y",
            "covid_perceived_danger_T12",
            "disease_fear_avg",
            "infection_likelihood_avg",
            "severity_if_infected_avg",
            "social_norms_vax_avg",
            "family_progressive_norms_nonvax_idx",
            "friends_progressive_norms_nonvax_idx",
            "colleagues_progressive_norms_nonvax_idx",
            "moral_progressive_orientation",
            "mfq_agreement_binding_idx",
            "mfq_relevance_binding_idx",
            "flu_vaccinated_pre_pandemic",
            "flu_vaccinated_2020_2021",
            "flu_vaccinated_2021_2022",
            "flu_vaccinated_2022_2023",
            "duration_socialmedia",
            "duration_infomedia",
            "booking",
            "vax_hub",
            "impediment",
            "vax_convo_freq_cohabitants",
            "politics",
        }

        def _is_missing(val: object) -> bool:
            if val is None:
                return True
            if isinstance(val, float) and pd.isna(val):
                return True
            if isinstance(val, str) and val.strip().lower() in {"missing", "nan", "none"}:
                return True
            return False

        blocks: list[str] = []

        def _round_half_up(num: float) -> int:
            return int(math.floor(num + 0.5))

        def _bucket_scale_half_up(val: object, lo: int, hi: int) -> int | None:
            num = self._as_numeric(val)
            if num is None:
                return None
            rounded = _round_half_up(num)
            return max(lo, min(hi, rounded))

        def _mean(values: list[float]) -> float | None:
            if not values:
                return None
            return sum(values) / len(values)

        def _scale_extreme(val: object, hi: int) -> str | None:
            num = self._as_numeric(val)
            if num is None:
                return None
            if hi == 7:
                low_thr, high_thr = 2, 6
            elif hi == 6:
                low_thr, high_thr = 2, 5
            elif hi == 5:
                low_thr, high_thr = 2, 4
            else:
                return None
            if num <= low_thr:
                return "low"
            if num >= high_thr:
                return "high"
            return "mid"

        def _join_sentence(base: str, clauses: list[str]) -> str:
            parts = [base] + [c for c in clauses if c]
            if not parts:
                return ""
            main = parts[0].rstrip(".")
            extras = [p.rstrip(".") for p in parts[1:]]
            if not extras:
                return f"{main}."
            return f"{main}; " + "; ".join(extras) + "."

        def _covax_legitimacy_clause(val: object) -> str | None:
            num = self._as_numeric(val)
            if num is None:
                return None
            if num <= 0.2:
                return "I have almost no doubts about how COVID vaccines are approved"
            if num <= 0.4:
                return "I have only a few doubts about how COVID vaccines are approved"
            if num <= 0.6:
                return "I have some doubts about how COVID vaccines are approved"
            if num <= 0.8:
                return "I have many doubts about how COVID vaccines are approved"
            return "I have very strong doubts about how COVID vaccines are approved"

        def _media_time_clause() -> str | None:
            parts: list[str] = []
            if "duration_socialmedia" in cols:
                level = _bucket_scale_half_up(attrs.get("duration_socialmedia"), 1, 5)
                if level is not None:
                    sm_map = {
                        1: "I spend very little time on social media",
                        2: "I spend a little time on social media",
                        3: "I spend some time on social media",
                        4: "I spend quite a lot of time on social media",
                        5: "I spend a lot of time on social media",
                    }
                    phrase = sm_map[level]
                    other_phrase = None
                    if perturbed_col == "duration_socialmedia" and perturbed_other_value is not None:
                        other_level = _bucket_scale_half_up(perturbed_other_value, 1, 5)
                        other_phrase = sm_map.get(other_level) if other_level is not None else None
                    parts.append(
                        _append_micro_modifier_if_needed(
                            col="duration_socialmedia", phrase=phrase, other_phrase=other_phrase
                        )
                    )
            if "duration_infomedia" in cols:
                level = _bucket_scale_half_up(attrs.get("duration_infomedia"), 1, 5)
                if level is not None:
                    info_map = {
                        1: "I follow news and information very little",
                        2: "I follow news and information a little",
                        3: "I follow news and information sometimes",
                        4: "I follow news and information quite a lot",
                        5: "I follow news and information a lot",
                    }
                    phrase = info_map[level]
                    other_phrase = None
                    if perturbed_col == "duration_infomedia" and perturbed_other_value is not None:
                        other_level = _bucket_scale_half_up(perturbed_other_value, 1, 5)
                        other_phrase = info_map.get(other_level) if other_level is not None else None
                    parts.append(
                        _append_micro_modifier_if_needed(
                            col="duration_infomedia", phrase=phrase, other_phrase=other_phrase
                        )
                    )
            if not parts:
                return None
            if len(parts) == 2:
                return f"{parts[0]} and {parts[1]}"
            return parts[0]

        def _summarize_trust_info() -> str | None:
            source_cols = [
                ("institutional_trust_avg", "institutions and public authorities"),
                ("trust_traditional", "traditional media"),
                ("trust_onlineinfo", "online sources"),
                ("trust_social", "social media"),
                ("group_trust_vax", "people I know"),
                ("trust_gp_vax_info", "general practitioners"),
                ("trust_who_vax_info", "the WHO"),
                ("trust_national_gov_vax_info", "the national government"),
                ("trust_national_health_institutions_vax_info", "national health institutions"),
                ("trust_local_health_authorities_vax_info", "local health authorities"),
            ]
            contrast_candidates = {
                "trust_traditional",
                "trust_onlineinfo",
                "trust_social",
                "group_trust_vax",
                "trust_gp_vax_info",
                "trust_who_vax_info",
                "trust_national_gov_vax_info",
                "trust_national_health_institutions_vax_info",
                "trust_local_health_authorities_vax_info",
            }
            sources: list[tuple[str, float]] = []
            for col, label in source_cols:
                if col not in cols:
                    continue
                num = self._as_numeric(attrs.get(col))
                if num is None:
                    continue
                sources.append((label, num))
            if not sources:
                return None
            mean_val = _mean([v for _, v in sources])
            level = _bucket_scale_half_up(mean_val, 1, 5)
            overall_map = {
                1: "Overall, I do not really trust vaccine information sources",
                2: "Overall, I am cautious about vaccine information",
                3: "Overall, I feel neutral about vaccine information",
                4: "Overall, I generally trust vaccine information sources",
                5: "Overall, I trust vaccine information sources a lot",
            }
            overall = overall_map[level]
            inst_clause = None
            for col, label in source_cols:
                if col != "institutional_trust_avg":
                    continue
                if col not in cols:
                    continue
                inst_level = _bucket_scale_half_up(attrs.get(col), 1, 5)
                if inst_level is not None:
                    inst_map = {
                        1: f"I do not really trust {label}",
                        2: f"I am cautious about {label}",
                        3: f"I feel neutral about {label}",
                        4: f"I generally trust {label}",
                        5: f"I trust {label} a lot",
                    }
                    phrase = inst_map[inst_level]
                    other_phrase = None
                    if perturbed_col == col and perturbed_other_value is not None:
                        other_level = _bucket_scale_half_up(perturbed_other_value, 1, 5)
                        other_phrase = inst_map.get(other_level) if other_level is not None else None
                    inst_clause = _append_micro_modifier_if_needed(col=col, phrase=phrase, other_phrase=other_phrase)
            contrast_sources = [
                (label, val, col)
                for col, label in source_cols
                for (lab, val) in sources
                if lab == label and col in contrast_candidates
            ]
            if contrast_sources:
                top_label, top_val, _ = max(contrast_sources, key=lambda kv: kv[1])
                bottom_label, bottom_val, _ = min(contrast_sources, key=lambda kv: kv[1])
            else:
                top_label, top_val = max(sources, key=lambda kv: kv[1])
                bottom_label, bottom_val = min(sources, key=lambda kv: kv[1])
            contrast = None
            spread = top_val - bottom_val
            if spread < 0.5:
                contrast = "I do not have a clear preference across sources"
            elif spread < 1.0:
                contrast = f"I lean slightly toward {top_label} over {bottom_label}"
            else:
                contrast = f"I trust {top_label} much more than {bottom_label}"
            clauses = [inst_clause, contrast]

            # For per-source trust levers, add a tiny explicit clause so the perturbed target
            # is always represented, but keep value→bucket compression.
            trust_source_labels = {
                "trust_traditional": "traditional media",
                "trust_onlineinfo": "online sources",
                "trust_social": "social media",
                "group_trust_vax": "people I know",
                "trust_gp_vax_info": "general practitioners",
                "trust_who_vax_info": "the WHO",
                "trust_national_gov_vax_info": "the national government",
                "trust_national_health_institutions_vax_info": "national health institutions",
                "trust_local_health_authorities_vax_info": "local health authorities",
            }
            if perturbed_col in trust_source_labels and perturbed_col in cols:
                level = _bucket_scale_half_up(attrs.get(perturbed_col), 1, 5)
                if level is not None:
                    label = trust_source_labels[perturbed_col]
                    m = {
                        1: f"I do not really trust {label}",
                        2: f"I am cautious about {label}",
                        3: f"I feel neutral about {label}",
                        4: f"I generally trust {label}",
                        5: f"I trust {label} a lot",
                    }
                    phrase = m[level]
                    other_phrase = None
                    if perturbed_other_value is not None:
                        other_level = _bucket_scale_half_up(perturbed_other_value, 1, 5)
                        other_phrase = m.get(other_level) if other_level is not None else None
                    clauses.insert(
                        0,
                        _append_micro_modifier_if_needed(
                            col=perturbed_col,
                            phrase=phrase,
                            other_phrase=other_phrase,
                        ),
                    )
            covax_clause = None
            if "covax_legitimacy_scepticism_idx" in cols:
                phrase = _covax_legitimacy_clause(attrs.get("covax_legitimacy_scepticism_idx"))
                other_phrase = None
                if perturbed_col == "covax_legitimacy_scepticism_idx" and perturbed_other_value is not None:
                    other_phrase = _covax_legitimacy_clause(perturbed_other_value)
                covax_clause = (
                    _append_micro_modifier_if_needed(
                        col="covax_legitimacy_scepticism_idx",
                        phrase=phrase,
                        other_phrase=other_phrase,
                    )
                    if phrase is not None
                    else None
                )
            if covax_clause and overall == "Overall, I feel neutral about vaccine information":
                if contrast == "Trust is fairly similar across sources":
                    clauses = [f"{contrast}, but {covax_clause}"]
                    covax_clause = None
            if covax_clause:
                clauses.append(covax_clause)
            clauses.append(_media_time_clause())
            sentence = _join_sentence(overall, clauses)
            return self._apply_pronoun_context(sentence, ctx)

        def _summarize_access_discussion() -> str | None:
            def _fmt_num(num: float) -> str:
                return f"{num:.2f}".rstrip("0").rstrip(".")

            clauses: list[str] = []

            if "booking" in cols:
                level = _bucket_scale_half_up(attrs.get("booking"), 1, 5)
                if level is not None:
                    booking_map = {
                        1: "Booking a vaccination appointment would be very difficult for me",
                        2: "Booking a vaccination appointment would be somewhat difficult for me",
                        3: "Booking a vaccination appointment would be manageable for me",
                        4: "Booking a vaccination appointment would be fairly easy for me",
                        5: "Booking a vaccination appointment would be very easy for me",
                    }
                    phrase = booking_map[level]
                    other_phrase = None
                    if perturbed_col == "booking" and perturbed_other_value is not None:
                        other_level = _bucket_scale_half_up(perturbed_other_value, 1, 5)
                        other_phrase = booking_map.get(other_level) if other_level is not None else None
                    clauses.append(
                        _append_micro_modifier_if_needed(
                            col="booking",
                            phrase=phrase,
                            other_phrase=other_phrase,
                        )
                    )

            if "vax_hub" in cols:
                level = _bucket_scale_half_up(attrs.get("vax_hub"), 1, 5)
                if level is not None:
                    hub_map = {
                        1: "Reaching a vaccination hub would be very difficult for me",
                        2: "Reaching a vaccination hub would be somewhat difficult for me",
                        3: "Reaching a vaccination hub would be manageable for me",
                        4: "Reaching a vaccination hub would be fairly easy for me",
                        5: "Reaching a vaccination hub would be very easy for me",
                    }
                    phrase = hub_map[level]
                    other_phrase = None
                    if perturbed_col == "vax_hub" and perturbed_other_value is not None:
                        other_level = _bucket_scale_half_up(perturbed_other_value, 1, 5)
                        other_phrase = hub_map.get(other_level) if other_level is not None else None
                    clauses.append(
                        _append_micro_modifier_if_needed(
                            col="vax_hub",
                            phrase=phrase,
                            other_phrase=other_phrase,
                        )
                    )

            if "impediment" in cols:
                val = self._as_bool(attrs.get("impediment"))
                if val is not None:
                    phrase = (
                        "I currently face practical impediments to getting vaccinated"
                        if val
                        else "I do not currently face major practical impediments to getting vaccinated"
                    )
                    other_phrase = None
                    if perturbed_col == "impediment" and perturbed_other_value is not None:
                        other_val = self._as_bool(perturbed_other_value)
                        if other_val is not None:
                            other_phrase = (
                                "I currently face practical impediments to getting vaccinated"
                                if other_val
                                else "I do not currently face major practical impediments to getting vaccinated"
                            )
                    clauses.append(
                        _append_micro_modifier_if_needed(
                            col="impediment",
                            phrase=phrase,
                            other_phrase=other_phrase,
                        )
                    )

            if "vax_convo_freq_cohabitants" in cols:
                level = _bucket_scale_half_up(attrs.get("vax_convo_freq_cohabitants"), 1, 5)
                if level is not None:
                    convo_map = {
                        1: "Vaccination almost never comes up in conversations at home",
                        2: "Vaccination comes up occasionally in conversations at home",
                        3: "Vaccination comes up sometimes in conversations at home",
                        4: "Vaccination comes up often in conversations at home",
                        5: "Vaccination comes up very often in conversations at home",
                    }
                    phrase = convo_map[level]
                    other_phrase = None
                    if perturbed_col == "vax_convo_freq_cohabitants" and perturbed_other_value is not None:
                        other_level = _bucket_scale_half_up(perturbed_other_value, 1, 5)
                        other_phrase = convo_map.get(other_level) if other_level is not None else None
                    clauses.append(
                        _append_micro_modifier_if_needed(
                            col="vax_convo_freq_cohabitants",
                            phrase=phrase,
                            other_phrase=other_phrase,
                        )
                    )

            if "politics" in cols:
                num = self._as_numeric(attrs.get("politics"))
                if num is not None:
                    phrase = f"On the survey's politics scale, I place myself around {_fmt_num(num)}"
                    other_phrase = None
                    if perturbed_col == "politics" and perturbed_other_value is not None:
                        other_num = self._as_numeric(perturbed_other_value)
                        if other_num is not None:
                            other_phrase = (
                                f"On the survey's politics scale, I place myself around {_fmt_num(other_num)}"
                            )
                    clauses.append(
                        _append_micro_modifier_if_needed(
                            col="politics",
                            phrase=phrase,
                            other_phrase=other_phrase,
                        )
                    )

            if not clauses:
                return None
            sentence = _join_sentence(clauses[0], clauses[1:])
            return self._apply_pronoun_context(sentence, ctx)

        def _summarize_vaccine_stance() -> str | None:
            core_cols = [
                "fivec_pro_vax_idx",
                "fivec_confidence",
                "fivec_low_complacency",
                "fivec_collective_responsibility",
            ]
            core_vals: list[float] = []
            for col in core_cols:
                if col not in cols:
                    continue
                num = self._as_numeric(attrs.get(col))
                if num is not None:
                    core_vals.append(num)
            if not core_vals:
                return None
            mean_val = _mean(core_vals)
            level = _bucket_scale_half_up(mean_val, 1, 7)
            overall_map = {
                1: "Overall, I am strongly against vaccines",
                2: "Overall, I am strongly against vaccines",
                3: "Overall, I lean against vaccines",
                4: "Overall, I feel mixed about vaccines",
                5: "Overall, I lean in favor of vaccines",
                6: "Overall, I am strongly in favor of vaccines",
                7: "Overall, I am strongly in favor of vaccines",
            }
            overall = overall_map[level]
            clauses: list[str] = []
            if "fivec_pro_vax_idx" in cols:
                pro_level = _bucket_scale_half_up(attrs.get("fivec_pro_vax_idx"), 1, 7)
                if pro_level is not None:
                    if pro_level <= 2:
                        phrase = "In general, I am skeptical of vaccines"
                    elif pro_level == 3:
                        phrase = "In general, I lean against vaccines"
                    elif pro_level == 4:
                        phrase = "In general, I am on the fence about vaccines"
                    elif pro_level == 5:
                        phrase = "In general, I lean toward vaccines"
                    else:
                        phrase = "In general, I am pro-vaccine"
                    other_phrase = None
                    if perturbed_col == "fivec_pro_vax_idx" and perturbed_other_value is not None:
                        other_level = _bucket_scale_half_up(perturbed_other_value, 1, 7)
                        if other_level is not None:
                            if other_level <= 2:
                                other_phrase = "In general, I am skeptical of vaccines"
                            elif other_level == 3:
                                other_phrase = "In general, I lean against vaccines"
                            elif other_level == 4:
                                other_phrase = "In general, I am on the fence about vaccines"
                            elif other_level == 5:
                                other_phrase = "In general, I lean toward vaccines"
                            else:
                                other_phrase = "In general, I am pro-vaccine"
                    clauses.append(
                        _append_micro_modifier_if_needed(
                            col="fivec_pro_vax_idx",
                            phrase=phrase,
                            other_phrase=other_phrase,
                        )
                    )
            if "fivec_confidence" in cols:
                conf_level = _bucket_scale_half_up(attrs.get("fivec_confidence"), 1, 7)
                if conf_level is not None:
                    if conf_level <= 2:
                        phrase = "I am not very confident vaccines are safe"
                    elif conf_level == 3:
                        phrase = "I am only a little confident vaccines are safe"
                    elif conf_level == 4:
                        phrase = "I am somewhat confident vaccines are safe"
                    elif conf_level == 5:
                        phrase = "I am fairly confident vaccines are safe"
                    else:
                        phrase = "I am very confident vaccines are safe"
                    other_phrase = None
                    if perturbed_col == "fivec_confidence" and perturbed_other_value is not None:
                        other_level = _bucket_scale_half_up(perturbed_other_value, 1, 7)
                        if other_level is not None:
                            if other_level <= 2:
                                other_phrase = "I am not very confident vaccines are safe"
                            elif other_level == 3:
                                other_phrase = "I am only a little confident vaccines are safe"
                            elif other_level == 4:
                                other_phrase = "I am somewhat confident vaccines are safe"
                            elif other_level == 5:
                                other_phrase = "I am fairly confident vaccines are safe"
                            else:
                                other_phrase = "I am very confident vaccines are safe"
                    clauses.append(
                        _append_micro_modifier_if_needed(
                            col="fivec_confidence",
                            phrase=phrase,
                            other_phrase=other_phrase,
                        )
                    )
            if "fivec_low_constraints" in cols:
                constraint_level = _bucket_scale_half_up(attrs.get("fivec_low_constraints"), 1, 7)
                if constraint_level is None:
                    pass
                else:
                    if constraint_level <= 2:
                        phrase = "Everyday stress and practical barriers make it hard for me to get vaccinated"
                    elif constraint_level == 3:
                        phrase = "Stress and barriers make it somewhat hard for me to get vaccinated"
                    elif constraint_level == 4:
                        phrase = "Getting vaccinated is neither easy nor hard for me"
                    elif constraint_level == 5:
                        phrase = "Getting vaccinated is fairly manageable for me"
                    else:
                        phrase = "Even when I am busy or stressed, I can follow through on getting vaccinated"
                    other_phrase = None
                    if perturbed_col == "fivec_low_constraints" and perturbed_other_value is not None:
                        other_level = _bucket_scale_half_up(perturbed_other_value, 1, 7)
                        if other_level is not None:
                            if other_level <= 2:
                                other_phrase = (
                                    "Everyday stress and practical barriers make it hard for me to get vaccinated"
                                )
                            elif other_level == 3:
                                other_phrase = "Stress and barriers make it somewhat hard for me to get vaccinated"
                            elif other_level == 4:
                                other_phrase = "Getting vaccinated is neither easy nor hard for me"
                            elif other_level == 5:
                                other_phrase = "Getting vaccinated is fairly manageable for me"
                            else:
                                other_phrase = (
                                    "Even when I am busy or stressed, I can follow through on getting vaccinated"
                                )
                    clauses.append(
                        _append_micro_modifier_if_needed(
                            col="fivec_low_constraints",
                            phrase=phrase,
                            other_phrase=other_phrase,
                        )
                    )
            if "fivec_collective_responsibility" in cols:
                coll_level = _bucket_scale_half_up(attrs.get("fivec_collective_responsibility"), 1, 7)
                if coll_level is None:
                    pass
                else:
                    if coll_level <= 2:
                        phrase = "Protecting others is not a big factor for me here"
                    elif coll_level == 3:
                        phrase = "Protecting others matters only a little to me here"
                    elif coll_level == 4:
                        phrase = "Protecting others matters to me, but it is not the main driver"
                    elif coll_level == 5:
                        phrase = "Protecting others matters a lot to me here"
                    else:
                        phrase = "Protecting others is very important to me here"
                    other_phrase = None
                    if perturbed_col == "fivec_collective_responsibility" and perturbed_other_value is not None:
                        other_level = _bucket_scale_half_up(perturbed_other_value, 1, 7)
                        if other_level is not None:
                            if other_level <= 2:
                                other_phrase = "Protecting others is not a big factor for me here"
                            elif other_level == 3:
                                other_phrase = "Protecting others matters only a little to me here"
                            elif other_level == 4:
                                other_phrase = "Protecting others matters to me, but it is not the main driver"
                            elif other_level == 5:
                                other_phrase = "Protecting others matters a lot to me here"
                            else:
                                other_phrase = "Protecting others is very important to me here"
                    clauses.append(
                        _append_micro_modifier_if_needed(
                            col="fivec_collective_responsibility",
                            phrase=phrase,
                            other_phrase=other_phrase,
                        )
                    )
            if "vaccine_risk_avg" in cols:
                risk_level = _bucket_scale_half_up(attrs.get("vaccine_risk_avg"), 1, 5)
                if risk_level is not None:
                    risk_map = {
                        1: "I think vaccine risks are very low",
                        2: "I think vaccine risks are low",
                        3: "I think vaccine risks are moderate",
                        4: "I think vaccine risks are high",
                        5: "I think vaccine risks are very high",
                    }
                    phrase = risk_map[risk_level]
                    other_phrase = None
                    if perturbed_col == "vaccine_risk_avg" and perturbed_other_value is not None:
                        other_level = _bucket_scale_half_up(perturbed_other_value, 1, 5)
                        other_phrase = risk_map.get(other_level) if other_level is not None else None
                    clauses.append(
                        _append_micro_modifier_if_needed(
                            col="vaccine_risk_avg",
                            phrase=phrase,
                            other_phrase=other_phrase,
                        )
                    )
            if "vaccine_fear_avg" in cols:
                fear_level = _bucket_scale_half_up(attrs.get("vaccine_fear_avg"), 1, 5)
                if fear_level is not None:
                    fear_map = {
                        1: "Emotionally, I feel very calm about side effects",
                        2: "Emotionally, I feel calm about side effects",
                        3: "Emotionally, I feel somewhat uneasy about side effects",
                        4: "Emotionally, I feel afraid about side effects",
                        5: "Emotionally, I feel very afraid about side effects",
                    }
                    phrase = fear_map[fear_level]
                    other_phrase = None
                    if perturbed_col == "vaccine_fear_avg" and perturbed_other_value is not None:
                        other_level = _bucket_scale_half_up(perturbed_other_value, 1, 5)
                        other_phrase = fear_map.get(other_level) if other_level is not None else None
                    clauses.append(
                        _append_micro_modifier_if_needed(
                            col="vaccine_fear_avg",
                            phrase=phrase,
                            other_phrase=other_phrase,
                        )
                    )
            if "vaccine_risk_avg" not in cols and "vaccine_fear_avg" not in cols and "flu_vaccine_risk" in cols:
                risk_level = _bucket_scale_half_up(attrs.get("flu_vaccine_risk"), 1, 5)
                if risk_level is not None:
                    flu_map = {
                        1: "I do not see the flu vaccine as risky",
                        2: "I see the flu vaccine as a little risky",
                        3: "I have some concerns about the flu vaccine",
                        4: "I see the flu vaccine as fairly risky",
                        5: "I see the flu vaccine as very risky",
                    }
                    phrase = flu_map[risk_level]
                    other_phrase = None
                    if perturbed_col == "flu_vaccine_risk" and perturbed_other_value is not None:
                        other_level = _bucket_scale_half_up(perturbed_other_value, 1, 5)
                        other_phrase = flu_map.get(other_level) if other_level is not None else None
                    clauses.append(
                        _append_micro_modifier_if_needed(
                            col="flu_vaccine_risk",
                            phrase=phrase,
                            other_phrase=other_phrase,
                        )
                    )
            sentence = _join_sentence(overall, clauses)
            return self._apply_pronoun_context(sentence, ctx)

        def _summarize_disease_stakes() -> str | None:
            dim_cols = [
                "disease_fear_avg",
                "infection_likelihood_avg",
                "severity_if_infected_avg",
                "covid_perceived_danger_Y",
                "covid_perceived_danger_T12",
            ]
            dim_vals: list[tuple[str, float]] = []
            for col in dim_cols:
                if col not in cols:
                    continue
                num = self._as_numeric(attrs.get(col))
                if num is None:
                    continue
                dim_vals.append((col, num))
            if not dim_vals:
                return None
            clauses: list[str] = []
            fear_level = None
            severity_level = None
            covid_level = None
            likelihood_clause = None
            comp_level = None
            for col, num in dim_vals:
                if col == "infection_likelihood_avg":
                    phrase = self._infection_likelihood_phrase(num, ctx)
                    other_phrase = None
                    if perturbed_col == col and perturbed_other_value is not None:
                        other_phrase = self._infection_likelihood_phrase(perturbed_other_value, ctx)
                    likelihood_clause = (
                        _append_micro_modifier_if_needed(col=col, phrase=phrase, other_phrase=other_phrase)
                        if phrase is not None
                        else None
                    )
                elif col == "disease_fear_avg":
                    fear_level = _bucket_scale_half_up(num, 1, 5)
                elif col == "severity_if_infected_avg":
                    severity_level = _bucket_scale_half_up(num, 1, 5)
                elif col in {"covid_perceived_danger_Y", "covid_perceived_danger_T12"}:
                    covid_level = _bucket_scale_half_up(num, 1, 5)
            if "fivec_low_complacency" in cols:
                comp_level = _bucket_scale_half_up(attrs.get("fivec_low_complacency"), 1, 7)

            comp_phrase = None
            if comp_level is not None:
                if comp_level <= 2:
                    comp_phrase = "I do not see vaccine-preventable diseases like COVID, flu, measles, and meningitis as a big deal"
                elif comp_level == 3:
                    comp_phrase = "I think vaccine-preventable diseases matter, but they are not a major concern for me"
                elif comp_level == 4:
                    comp_phrase = "I am somewhat concerned about vaccine-preventable diseases"
                elif comp_level == 5:
                    comp_phrase = "I am quite concerned about vaccine-preventable diseases"
                else:
                    comp_phrase = "I am very concerned about vaccine-preventable diseases"
            if comp_phrase:
                other_phrase = None
                if perturbed_col == "fivec_low_complacency" and perturbed_other_value is not None:
                    other_comp = _bucket_scale_half_up(perturbed_other_value, 1, 7)
                    if other_comp is not None:
                        if other_comp <= 2:
                            other_phrase = "I do not see vaccine-preventable diseases like COVID, flu, measles, and meningitis as a big deal"
                        elif other_comp == 3:
                            other_phrase = (
                                "I think vaccine-preventable diseases matter, but they are not a major concern for me"
                            )
                        elif other_comp == 4:
                            other_phrase = "I am somewhat concerned about vaccine-preventable diseases"
                        elif other_comp == 5:
                            other_phrase = "I am quite concerned about vaccine-preventable diseases"
                        else:
                            other_phrase = "I am very concerned about vaccine-preventable diseases"
                clauses.append(
                    _append_micro_modifier_if_needed(
                        col="fivec_low_complacency",
                        phrase=comp_phrase,
                        other_phrase=other_phrase,
                    )
                )

            if likelihood_clause:
                clauses.append(likelihood_clause)
            if severity_level is not None:
                severity_map = {
                    1: "If I were infected, I would expect it to be mild",
                    2: "If I were infected, I would expect it to be relatively mild",
                    3: "If I were infected, I would expect it to be moderately severe",
                    4: "If I were infected, I would expect it to be quite severe",
                    5: "If I were infected, I would expect it to be very severe",
                }
                phrase = severity_map[severity_level]
                other_phrase = None
                if perturbed_col == "severity_if_infected_avg" and perturbed_other_value is not None:
                    other_level = _bucket_scale_half_up(perturbed_other_value, 1, 5)
                    other_phrase = severity_map.get(other_level) if other_level is not None else None
                clauses.append(
                    _append_micro_modifier_if_needed(
                        col="severity_if_infected_avg",
                        phrase=phrase,
                        other_phrase=other_phrase,
                    )
                )
            if covid_level is not None:
                covid_map = {
                    1: "Overall, I am not very worried about COVID specifically",
                    2: "Overall, I am only a little worried about COVID specifically",
                    3: "Overall, I feel moderately worried about COVID specifically",
                    4: "Overall, I am quite worried about COVID specifically",
                    5: "Overall, I am very worried about COVID specifically",
                }
                phrase = covid_map[covid_level]
                other_phrase = None
                if (
                    perturbed_col in {"covid_perceived_danger_Y", "covid_perceived_danger_T12"}
                    and perturbed_other_value is not None
                ):
                    other_level = _bucket_scale_half_up(perturbed_other_value, 1, 5)
                    other_phrase = covid_map.get(other_level) if other_level is not None else None
                clauses.append(
                    _append_micro_modifier_if_needed(
                        col=perturbed_col
                        if perturbed_col in {"covid_perceived_danger_Y", "covid_perceived_danger_T12"}
                        else "covid_perceived_danger_Y",
                        phrase=phrase,
                        other_phrase=other_phrase,
                    )
                )
            if fear_level is not None:
                fear_map = {
                    1: "Overall, I feel calm about COVID and flu",
                    2: "Overall, I feel only a little worried about COVID and flu",
                    3: "Overall, I feel moderately worried about COVID and flu",
                    4: "Overall, I feel quite worried about COVID and flu",
                    5: "Overall, I feel very afraid of COVID and flu",
                }
                phrase = fear_map[fear_level]
                other_phrase = None
                if perturbed_col == "disease_fear_avg" and perturbed_other_value is not None:
                    other_level = _bucket_scale_half_up(perturbed_other_value, 1, 5)
                    other_phrase = fear_map.get(other_level) if other_level is not None else None
                clauses.append(
                    _append_micro_modifier_if_needed(
                        col="disease_fear_avg",
                        phrase=phrase,
                        other_phrase=other_phrase,
                    )
                )
            if "flu_disease_fear" in cols:
                flu_level = _bucket_scale_half_up(attrs.get("flu_disease_fear"), 1, 5)
                if flu_level is not None:
                    flu_map = {
                        1: "I am not very worried about the flu",
                        2: "I am only a little worried about the flu",
                        3: "I feel moderately worried about the flu",
                        4: "I am quite worried about the flu",
                        5: "I am very afraid of the flu",
                    }
                    phrase = flu_map[flu_level]
                    other_phrase = None
                    if perturbed_col == "flu_disease_fear" and perturbed_other_value is not None:
                        other_level = _bucket_scale_half_up(perturbed_other_value, 1, 5)
                        other_phrase = flu_map.get(other_level) if other_level is not None else None
                    clauses.append(
                        _append_micro_modifier_if_needed(
                            col="flu_disease_fear",
                            phrase=phrase,
                            other_phrase=other_phrase,
                        )
                    )
            if not clauses:
                return None
            sentence = _join_sentence(clauses[0], clauses[1:])
            return self._apply_pronoun_context(sentence, ctx)

        def _progressive_self_phrase() -> str | None:
            if "moral_progressive_orientation" not in cols:
                return None
            level = _bucket_scale_half_up(attrs.get("moral_progressive_orientation"), 1, 7)
            if level is None:
                return None
            if level <= 2:
                return "I lean away from those progressive positions"
            if level == 3:
                return "I slightly lean away from those progressive positions"
            if level == 4:
                return "I feel mixed about those progressive positions"
            if level == 5:
                return "I lean somewhat toward those progressive positions"
            return "I strongly endorse those progressive positions"

        def _vax_norms_phrase() -> str | None:
            if "social_norms_vax_avg" not in cols:
                return None
            level = _bucket_scale_half_up(attrs.get("social_norms_vax_avg"), 1, 7)
            if level is None:
                return None
            if level <= 2:
                phrase = "People around me generally think vaccines are not important"
            elif level == 3:
                phrase = "People around me lean against vaccines"
            elif level == 4:
                phrase = "People around me are mixed about vaccines"
            elif level == 5:
                phrase = "People around me lean in favor of vaccines"
            else:
                phrase = "People around me strongly value vaccines"

            other_phrase = None
            if perturbed_col == "social_norms_vax_avg" and perturbed_other_value is not None:
                other_level = _bucket_scale_half_up(perturbed_other_value, 1, 7)
                if other_level is not None:
                    if other_level <= 2:
                        other_phrase = "People around me generally think vaccines are not important"
                    elif other_level == 3:
                        other_phrase = "People around me lean against vaccines"
                    elif other_level == 4:
                        other_phrase = "People around me are mixed about vaccines"
                    elif other_level == 5:
                        other_phrase = "People around me lean in favor of vaccines"
                    else:
                        other_phrase = "People around me strongly value vaccines"

            return _append_micro_modifier_if_needed(
                col="social_norms_vax_avg",
                phrase=phrase,
                other_phrase=other_phrase,
            )

        def _progressive_circles_clause() -> str | None:
            group_cols = [
                ("family", "family_progressive_norms_nonvax_idx"),
                ("friends", "friends_progressive_norms_nonvax_idx"),
                ("colleagues", "colleagues_progressive_norms_nonvax_idx"),
            ]
            values: list[tuple[str, float]] = []
            for label, col in group_cols:
                if col not in cols:
                    continue
                num = self._as_numeric(attrs.get(col))
                if num is None:
                    continue
                values.append((label, num))
            if not values:
                return None
            mean_val = _mean([v for _, v in values])
            level = _bucket_scale_half_up(mean_val, 1, 7)
            issue_label = "climate harm, LGBT adoption rights, and immigration tolerance"
            overall_map = {
                1: f"My circles are generally conservative on {issue_label}",
                2: f"My circles lean away from progressive positions on {issue_label}",
                3: f"My circles lean slightly away from progressive positions on {issue_label}",
                4: f"My circles are mixed on {issue_label}",
                5: f"My circles lean somewhat progressive on {issue_label}",
                6: f"My circles are clearly progressive on {issue_label}",
                7: f"My circles are strongly progressive on {issue_label}",
            }
            overall = overall_map[level]
            top_label, top_val = max(values, key=lambda kv: kv[1])
            bottom_label, bottom_val = min(values, key=lambda kv: kv[1])
            contrast = None
            if top_val - bottom_val >= 2:
                contrast = f"my {top_label} are more progressive than my {bottom_label}"
            elif level != 4:
                top_extreme = _scale_extreme(top_val, 7)
                bottom_extreme = _scale_extreme(bottom_val, 7)
                if top_extreme == "high" and bottom_extreme != "high":
                    contrast = f"my {top_label} are especially progressive"
                elif bottom_extreme == "low" and top_extreme != "low":
                    contrast = f"my {bottom_label} lean more conservative"
                else:
                    contrast = "views are fairly similar across groups"
            else:
                contrast = None
            if contrast:
                return f"{overall}, {contrast}"
            return overall

        def _summarize_norms_moral() -> str | None:
            clauses: list[str] = []
            self_phrase = _progressive_self_phrase()
            if self_phrase:
                clauses.append(self_phrase)
            vax_norms = _vax_norms_phrase()
            if vax_norms:
                clauses.append(vax_norms)
            circles_clause = _progressive_circles_clause()
            if circles_clause:
                clauses.append(circles_clause)
            mfq_clause = None
            if "mfq_agreement_binding_idx" in cols or "mfq_relevance_binding_idx" in cols:
                mfq_clause = self._mfq_binding_joint_phrase(
                    attrs.get("mfq_agreement_binding_idx"),
                    attrs.get("mfq_relevance_binding_idx"),
                    ctx,
                )
            if mfq_clause:
                clauses.append(mfq_clause)
            if not clauses:
                return None
            sentence = _join_sentence(clauses[0], clauses[1:])
            return self._apply_pronoun_context(sentence, ctx)

        demo_bits: list[str] = []
        if "sex_female" in cols:
            val = self._as_bool(attrs.get("sex_female"))
            if val is not None:
                demo_bits.append("a woman" if val else "a man")
        age = self._age_phrase(attrs.get("age"), ctx) if "age" in cols else None
        if age:
            demo_bits.append(age)
        if "M_education" in cols:
            edu = self._bucket_scale(attrs.get("M_education"), 0, 2)
            if edu is not None:
                edu_map = {
                    0: "with a lower level of formal education",
                    1: "with a medium level of formal education",
                    2: "with a higher level of formal education",
                }
                demo_bits.append(edu_map[edu])
        country = attrs.get("country")
        if country:
            demo_bits.append(f"living in {self._country_label(country)}")
        demo_clauses: list[str] = []
        if "minority" in cols:
            val = self._as_bool(attrs.get("minority"))
            if val is not None:
                demo_clauses.append(
                    "I consider myself part of a minority group"
                    if val
                    else "I do not consider myself part of a minority group"
                )
        if "M_religion" in cols:
            val = self._as_bool(attrs.get("M_religion"))
            if val is not None:
                if val:
                    demo_clauses.append("I am either Christian or atheist/agnostic (not another religion)")
                else:
                    demo_clauses.append(
                        "I am outside the Christianity/atheist-agnostic bucket (e.g., Orthodox, Muslim, Jewish, "
                        "Eastern, or another religion)"
                    )
        blocks.append(self._apply_pronoun_context("It is 2024.", ctx))
        if demo_bits:
            base = f"I am {', '.join(demo_bits)}"
            sentence = _join_sentence(base, demo_clauses)
            blocks.append(self._apply_pronoun_context(sentence, ctx))
        elif demo_clauses:
            sentence = _join_sentence(demo_clauses[0], demo_clauses[1:])
            blocks.append(self._apply_pronoun_context(sentence, ctx))
        income = self._income_phrase(attrs.get("income_ppp_norm"), country, ctx) if "income_ppp_norm" in cols else None
        if income:
            blocks.append(income)

        health_clauses: list[str] = []
        if "health_poor" in cols:
            health_map = {
                1: "Overall, my health is very good",
                2: "My health is good",
                3: "My health is okay",
                4: "My health is not great",
                5: "Overall, my health is poor",
            }
            phrase = self._scale_phrase(attrs.get("health_poor"), {k: f"{v}." for k, v in health_map.items()}, ctx)
            if phrase:
                health_clauses.append(phrase.rstrip("."))
        if "medical_conditions" in cols:
            val = self._as_bool(attrs.get("medical_conditions"))
            if val:
                health_clauses.append("I have at least one underlying medical condition")
        if "medical_conditions_c" in cols:
            val = self._as_bool(attrs.get("medical_conditions_c"))
            if val:
                health_clauses.append("Someone in my household has an underlying medical condition")
        if "covid_num" in cols:
            num = self._as_numeric(attrs.get("covid_num"))
            if num is not None and num >= 0.5:
                if num < 1.5:
                    health_clauses.append("I have had COVID once")
                elif num < 2.5:
                    health_clauses.append("I have had COVID twice")
                else:
                    health_clauses.append("I have had COVID multiple times")
        if "exper_illness" in cols:
            val = self._as_bool(attrs.get("exper_illness"))
            if val:
                health_clauses.append(
                    "I personally know someone who was hospitalized or died due to a serious infectious illness (not COVID-19)"
                )
        side_self = self._as_bool(attrs.get("side_effects")) if "side_effects" in cols else None
        side_other = self._as_bool(attrs.get("side_effects_other")) if "side_effects_other" in cols else None
        if side_self is not None or side_other is not None:
            if side_self and side_other:
                health_clauses.append(
                    "Vaccine side effects (beyond temporary arm pain) have affected me and people I know"
                )
            elif side_self:
                health_clauses.append("I have experienced vaccine side effects (beyond temporary arm pain)")
            elif side_other:
                health_clauses.append(
                    "I know someone who has experienced vaccine side effects (beyond temporary arm pain)"
                )
            else:
                health_clauses.append(
                    "Vaccine side effects (beyond temporary arm pain) have not been a major issue for me or people around me"
                )
        flu_bits: list[tuple[str, bool]] = []
        single_flu_phrase: str | None = None
        flu_map = {
            "flu_vaccinated_pre_pandemic": "Before 2020, I did get the flu vaccine",
            "flu_vaccinated_2020_2021": "I got a flu shot in the 2020/2021 season",
            "flu_vaccinated_2021_2022": "I got a flu shot in the 2021/2022 season",
            "flu_vaccinated_2022_2023": "I got a flu shot in the 2022/2023 season",
        }
        for key, yes_phrase in flu_map.items():
            if key not in cols:
                continue
            val = self._as_bool(attrs.get(key))
            if val is None:
                continue
            flu_bits.append((key, val))
            if len(flu_bits) == 1:
                phrase = (
                    yes_phrase
                    if val
                    else yes_phrase.replace("I got", "I did not get").replace("I did get", "I did not usually get")
                )
                single_flu_phrase = phrase
            else:
                single_flu_phrase = None
        if len(flu_bits) == 1 and single_flu_phrase:
            health_clauses.append(single_flu_phrase)
        elif len(flu_bits) >= 2:
            yes_count = sum(1 for _, v in flu_bits if v)
            no_count = sum(1 for _, v in flu_bits if not v)
            if yes_count >= 2 and yes_count > no_count:
                health_clauses.append("I have been fairly consistent about getting the flu vaccine")
            elif no_count >= 2 and no_count > yes_count:
                health_clauses.append("I rarely get the flu vaccine")
            else:
                health_clauses.append("My flu vaccination is occasional rather than routine")
        if health_clauses:
            sentence = _join_sentence(health_clauses[0], health_clauses[1:])
            blocks.append(self._apply_pronoun_context(sentence, ctx))

        trust_sentence = _summarize_trust_info()
        if trust_sentence:
            blocks.append(trust_sentence)

        access_sentence = _summarize_access_discussion()
        if access_sentence:
            blocks.append(access_sentence)

        vacc_sentence = _summarize_vaccine_stance()
        if vacc_sentence:
            blocks.append(vacc_sentence)

        stakes_sentence = _summarize_disease_stakes()
        if stakes_sentence:
            blocks.append(stakes_sentence)

        norms_sentence = _summarize_norms_moral()
        if norms_sentence:
            blocks.append(norms_sentence)

        missing = sorted(
            col
            for col in cols
            if col in attrs
            and not col.startswith(SPECIAL_STRUCTURAL_PREFIXES)
            and not _is_missing(attrs.get(col))
            and col not in narrative_cols
        )
        if missing:
            raise ValueError(f"Unmapped narrative attributes: {', '.join(missing)}")

        narrative = "\n".join(blocks)
        if first:
            commitment = "These statements describe me; I stand by them for the next question."
            if commitment not in narrative:
                narrative = (narrative.rstrip() + "\n" + commitment).strip() if narrative.strip() else commitment
        return narrative


__all__ = ["PromptNarrativeRendererMixin"]
