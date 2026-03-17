"""Prompting strategy builders for micro-validation.

These are intentionally backend-agnostic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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


def build_data_only_prompt(
    *,
    base_prompt: str,
    attributes: dict[str, Any],
    task_instructions: str,
    attribute_key_map: dict[str, str] | None = None,
    attribute_render: Literal["name", "description", "both"] = "name",
) -> str:
    return (
        f"{base_prompt}\n\n{task_instructions}\n\nAttributes:\n"
        f"{format_attribute_vector(attributes, attribute_key_map=attribute_key_map, attribute_render=attribute_render)}"
    )


def build_persona_card(
    attributes: dict[str, Any],
    *,
    attribute_key_map: dict[str, str] | None = None,
    attribute_render: Literal["name", "description", "both"] = "name",
) -> str:
    # Minimal narrative layer: keep it deterministic and compact.
    # We avoid hard-coding dataset-specific fields; instead, we prefer generic phrasing.
    lines = []
    for key, value in attributes.items():
        k_str = str(key)
        label = (attribute_key_map or {}).get(k_str)
        if attribute_render == "name" or not label:
            key_text = k_str
        elif attribute_render == "description":
            key_text = f"{label} ({k_str})"
        else:
            key_text = f"{k_str} — {label}"
        lines.append(f"- {key_text}: {value}")
    return "Persona profile (from survey data):\n" + "\n".join(lines)


def build_persona_card_prompt(
    *,
    base_prompt: str,
    attributes: dict[str, Any],
    task_instructions: str,
    attribute_key_map: dict[str, str] | None = None,
    attribute_render: Literal["name", "description", "both"] = "name",
) -> str:
    persona = build_persona_card(attributes, attribute_key_map=attribute_key_map, attribute_render=attribute_render)
    return f"{base_prompt}\n\n{task_instructions}\n\n{persona}"


def build_prototype_card(
    attributes: dict[str, Any],
    *,
    attribute_key_map: dict[str, str] | None = None,
    attribute_render: Literal["name", "description", "both"] = "name",
) -> str:
    lines = []
    for key, value in attributes.items():
        k_str = str(key)
        label = (attribute_key_map or {}).get(k_str)
        if attribute_render == "name" or not label:
            key_text = k_str
        elif attribute_render == "description":
            key_text = f"{label} ({k_str})"
        else:
            key_text = f"{k_str} — {label}"
        lines.append(f"- {key_text}: {value}")
    return "Prototype profile (anchor summary):\n" + "\n".join(lines)


def build_prototype_card_prompt(
    *,
    base_prompt: str,
    attributes: dict[str, Any],
    task_instructions: str,
    attribute_key_map: dict[str, str] | None = None,
    attribute_render: Literal["name", "description", "both"] = "name",
) -> str:
    proto = build_prototype_card(attributes, attribute_key_map=attribute_key_map, attribute_render=attribute_render)
    return f"{base_prompt}\n\n{task_instructions}\n\n{proto}"


def build_framed_prompt(
    *,
    base_prompt: str,
    attributes: dict[str, Any],
    task_instructions: str,
    perspective: Literal["first_person", "third_person"],
    attribute_key_map: dict[str, str] | None = None,
    attribute_render: Literal["name", "description", "both"] = "name",
) -> str:
    if perspective == "first_person":
        framing = "Write as if you are the respondent (first-person)."
    else:
        framing = "Write as an analyst describing the respondent (third-person)."

    return (
        f"{base_prompt}\n\n{framing}\n{task_instructions}\n\nAttributes:\n"
        f"{format_attribute_vector(attributes, attribute_key_map=attribute_key_map, attribute_render=attribute_render)}"
    )


@dataclass(frozen=True)
class PairedProfilePrompt:
    prompt: str
    order: Literal["A_then_B", "B_then_A"]


def build_paired_profile_prompt(
    *,
    base_prompt: str,
    attributes_low: dict[str, Any],
    attributes_high: dict[str, Any],
    task_instructions: str,
    randomized_order: bool = True,
    rng_seed: int | None = None,
) -> PairedProfilePrompt:
    """Paired-profile comparative prompting.

    Presents two profiles (A/B) in a single prompt.
    The caller is responsible for parsing two outputs or a delta, depending on schema.
    """

    import random

    rng = random.Random(rng_seed)
    order = "A_then_B"
    if randomized_order and rng.random() < 0.5:
        order = "B_then_A"

    a_text = json.dumps(attributes_low, ensure_ascii=False)
    b_text = json.dumps(attributes_high, ensure_ascii=False)

    if order == "A_then_B":
        a_label, b_label = "A", "B"
        first, second = a_text, b_text
    else:
        a_label, b_label = "B", "A"
        first, second = b_text, a_text

    prompt = (
        f"{base_prompt}\n\n"
        f"{task_instructions}\n\n"
        f"Profile {a_label} (JSON):\n{first}\n\n"
        f"Profile {b_label} (JSON):\n{second}\n\n"
        "Return ONLY JSON with top-level keys 'A' and 'B'. "
        "Each of A and B must be a JSON object that matches the declared outcome scale schema "
        "(e.g., continuous: {'prediction': <float>}; binary: {'prediction': <float in [0,1]>}; "
        "ordinal: {'levels': [...], 'probs': [...]}, etc.)."
    )

    return PairedProfilePrompt(prompt=prompt, order=order)


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
