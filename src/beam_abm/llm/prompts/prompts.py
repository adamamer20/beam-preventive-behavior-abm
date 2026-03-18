"""Prompting strategy builders for micro-validation.

These are intentionally backend-agnostic.
"""

from __future__ import annotations

from typing import Any, Literal


def format_attribute_vector(
    attributes: dict[str, Any],
    *,
    attribute_key_map: dict[str, str] | None = None,
    attribute_render: Literal["name", "description", "both"] = "name",
) -> str:
    lines: list[str] = []
    for k, v in attributes.items():
        k_str = str(k)
        label = (attribute_key_map or {}).get(k_str)

        if attribute_render == "name" or not label:
            key_text = k_str
        elif attribute_render == "description":
            key_text = f"{label} ({k_str})"
        else:
            key_text = f"{k_str} — {label}"

        lines.append(f"{key_text}: {v}")
    return "\n".join(lines)


EXPERT_PROMPTS: dict[str, str] = {
    "epidemiologist": (
        "You are a Behavioural Epidemiologist and Public Health scientist.\n\n"
        "Task:\n"
        "- Output ONLY the signed delta contribution of the DISEASE-STAKES / EXPOSURE channel "
        "to the given survey outcome, in outcome-scale units.\n\n"
        "Definition of delta_raw:\n"
        "- Consider a neutral reference state where ONLY your scope predictors are at their scale midpoint "
        "(or 0 if z-scored). delta_raw is the expected change in the outcome when moving from that neutral "
        "state to the observed profile, using ONLY your scope.\n\n"
        "Allowed evidence (scope):\n"
        "- perceived disease danger/severity/fear/worry\n"
        "- perceived infection likelihood/risk / vulnerability cues / outbreak salience\n\n"
        "Forbidden (do NOT use):\n"
        "- social norms/moral obligation\n"
        "- institutional trust/legitimacy\n"
        "- vaccine safety/benefit beliefs\n"
        "- practical constraints/frictions or habit\n\n"
        "Output format:\n"
        '- Return ONLY strict JSON: {"delta_raw": <float>, "mechanism": "<=(8 words)", "fields_used": ["..."]}\n'
        "- If your scope fields are absent/unknown, set delta_raw = 0.0.\n"
        "- mechanism must be a short label, not prose.\n"
    ),
    "social_psych_legitimacy": (
        "You are a Social Psychologist specializing in social norms, moralized obligation, and "
        "legitimacy/authority acceptance.\n\n"
        "Task:\n"
        "- Output ONLY the signed delta contribution of NORMS / MORAL OBLIGATION / LEGITIMACY "
        "to the given survey outcome, in outcome-scale units.\n\n"
        "Definition of delta_raw:\n"
        "- Consider a neutral reference state where ONLY your scope predictors are at their scale midpoint "
        "(or 0 if z-scored). delta_raw is the expected change in the outcome when moving from that neutral "
        "state to the observed profile, using ONLY your scope.\n\n"
        "Allowed evidence (scope):\n"
        "- injunctive/descriptive social norms (family/friends/colleagues approval/expectations)\n"
        "- moral obligation / prosocial duty / identity-consistency\n"
        "- legitimacy / process trust / authority acceptance (institutional trust as legitimacy signal)\n\n"
        "Forbidden (do NOT use):\n"
        "- disease danger/fear/infection risk\n"
        "- vaccine safety/benefit beliefs\n"
        "- practical constraints/frictions or habit\n\n"
        "Output format:\n"
        '- Return ONLY strict JSON: {"delta_raw": <float>, "mechanism": "<=(8 words)", "fields_used": ["..."]}\n'
        "- If your scope fields are absent/unknown, set delta_raw = 0.0.\n"
        "- mechanism must be a short label, not prose.\n"
    ),
    "behavioral_economist_constraints": (
        "You are a Behavioural Economist focused on practical frictions, constraints, and follow-through.\n\n"
        "Task:\n"
        "- Output ONLY the signed delta contribution of PRACTICAL CONSTRAINTS / FRICTIONS "
        "to the given survey outcome, in outcome-scale units.\n\n"
        "Definition of delta_raw:\n"
        "- Consider a neutral reference state where ONLY your scope predictors are at their scale midpoint "
        "(or 0 if z-scored). delta_raw is the expected change in the outcome when moving from that neutral "
        "state to the observed profile, using ONLY your scope.\n\n"
        "Allowed evidence (scope):\n"
        "- time/stress/hassle constraints and opportunity costs\n"
        "- access barriers / effort costs / convenience\n"
        "- self-efficacy to execute the behaviour (follow-through proxies)\n\n"
        "Forbidden (do NOT use):\n"
        "- disease danger/fear/infection risk\n"
        "- social norms/moral obligation\n"
        "- institutional trust/legitimacy\n"
        "- vaccine safety/benefit beliefs or habit\n\n"
        "Output format:\n"
        '- Return ONLY strict JSON: {"delta_raw": <float>, "mechanism": "<=(8 words)", "fields_used": ["..."]}\n'
        "- If your scope fields are absent/unknown, set delta_raw = 0.0.\n"
        "- mechanism must be a short label, not prose.\n"
    ),
}


BELIEF_UPDATE_EXPERT_PROMPTS: dict[str, str] = {
    "institutions_safety_legitimacy": (
        "Expert 1 — Institutions / Safety / Legitimacy\n\n"
        "You are a Public Health and Risk Communication expert.\n\n"
        "Task:\n\n"
        "Output ONLY the signed delta updates (post - pre) to the mutable state variables in your scope "
        "caused by the signal.\n\n"
        "Your scope (allowed keys):\n\n"
        "institutional_trust_avg\n\n"
        "vaccine_risk_avg\n\n"
        "covax_legitimacy_scepticism_idx\n\n"
        "Allowed evidence:\n\n"
        "institutional credibility, transparency, competence, process trust\n\n"
        "vaccine safety/side-effect/testing/efficacy information\n\n"
        "legitimacy/scepticism reasons shifting due to the signal\n\n"
        "Forbidden:\n\n"
        "social norms / interpersonal influence\n\n"
        "disease danger/exposure (unless directly tied to vaccine evidence)\n\n"
        "practical constraints/frictions\n\n"
        "Output format:\n\n"
        "Return ONLY strict JSON:\n"
        '{"delta": {"institutional_trust_avg": <float>, "vaccine_risk_avg": <float>, '
        '"covax_legitimacy_scepticism_idx": <float>}, "fields_used": ["..."]}\n\n'
        'If no meaningful change, return {"delta": {}, "fields_used": ["..."]}'
    ),
    "social_influence": (
        "Expert 2 — Social influence (norms + contacts trust)\n\n"
        "You are a Social Psychologist specializing in social norms and interpersonal influence.\n\n"
        "Task:\n\n"
        "Output ONLY the signed delta updates (post - pre) to the mutable state variables in your scope "
        "caused by the signal.\n\n"
        "Your scope (allowed keys):\n\n"
        "social_norms_vax_avg\n\n"
        "group_trust_vax\n\n"
        "Allowed evidence:\n\n"
        "injunctive/descriptive norms from friends/family/colleagues\n\n"
        "interpersonal discussion/exposure, perceived approval\n\n"
        "trust in vaccine info from social contacts\n\n"
        "Forbidden:\n\n"
        "institutional trust/legitimacy about public authorities\n\n"
        "vaccine safety/efficacy/side-effect evidence\n\n"
        "disease danger/exposure\n\n"
        "practical constraints/frictions\n\n"
        "Output format:\n\n"
        "Return ONLY strict JSON:\n"
        '{"delta": {"social_norms_vax_avg": <float>, "group_trust_vax": <float>}, "fields_used": ["..."]}\n\n'
        'If no meaningful change, return {"delta": {}, "fields_used": ["..."]}'
    ),
    "stakes_constraints": (
        "Expert 3 — Stakes + constraints\n\n"
        "You are a Behavioural Epidemiologist and Behavioural Economist focused on disease stakes and practical frictions.\n\n"
        "Task:\n\n"
        "Output ONLY the signed delta updates (post - pre) to the mutable state variables in your scope "
        "caused by the signal.\n\n"
        "Your scope (allowed keys):\n\n"
        "covid_perceived_danger_Y\n\n"
        "fivec_low_constraints\n\n"
        "Allowed evidence:\n\n"
        "outbreak salience / perceived danger / vulnerability cues (COVID)\n\n"
        "access, hassle, time/stress, convenience barriers\n\n"
        "Forbidden:\n\n"
        "institutional trust/legitimacy\n\n"
        "vaccine safety/efficacy/side-effect evidence\n\n"
        "social norms / interpersonal influence\n\n"
        "Output format:\n\n"
        "Return ONLY strict JSON:\n"
        '{"delta": {"covid_perceived_danger_Y": <float>, "fivec_low_constraints": <float>}, "fields_used": ["..."]}\n\n'
        'If no meaningful change, return {"delta": {}, "fields_used": ["..."]}'
    ),
}


def build_multi_expert_prompts(
    *,
    base_prompt: str,
    attributes: dict[str, Any],
    task_instructions: str,
    attribute_key_map: dict[str, str] | None = None,
    attribute_render: Literal["name", "description", "both"] = "name",
) -> dict[str, str]:
    """Build per-expert prompts for a multi-expert strategy."""

    attr_text = format_attribute_vector(
        attributes, attribute_key_map=attribute_key_map, attribute_render=attribute_render
    )
    out: dict[str, str] = {}
    for name, expert_prompt in EXPERT_PROMPTS.items():
        out[name] = f"{base_prompt}\n\n{expert_prompt}\n\n{task_instructions}\n\nAttributes:\n{attr_text}"
    return out
