from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

# Store and read specs from the versioned preprocessing/specs directory
DATA_DIR = Path("preprocess") / "specs"
GENERAL_SPEC = DATA_DIR / "5_general_transformed_questions.json"
DEFAULT_OUTPUT_RAW = DATA_DIR / "5_clean_transformed_questions.json"
DEFAULT_OUTPUT_POST_GENERAL = DATA_DIR / "5_clean_post_general_transformed_questions.json"
SECTION_SPEC_DIR = DATA_DIR / "clean"

# Analysis helpers / defaults
BASELINE_DATE = "2019-01-01"  # for date deltas
NUTS_NAME_COLUMNS = {"nuts1", "nuts2", "nuts3"}
NUTS_CODE_COLUMNS = {"nuts1_code_2024", "nuts2_code_2024", "nuts3_code_2024"}

REGION_SOURCE_DATA = Path("data") / "1_CROSSCOUNTRY_translated.csv"
REGION_ROLLUP_LOG = Path("output") / "preprocess" / "region_final_rollup.json"
REGION_LEVEL_ORDER = ("nuts3", "nuts2", "nuts1", "country")
REGION_THRESHOLDS = {
    "nuts3": 75,
    "nuts2": 150,
    "nuts1": 300,
    "country": 300,
}
REGION_CODE_COLUMNS = {
    "nuts3": "nuts3_code_2024",
    "nuts2": "nuts2_code_2024",
    "nuts1": "nuts1_code_2024",
    "country": "country",
}
REGION_LABEL_COLUMNS = {
    "nuts3": "nuts3",
    "nuts2": "nuts2",
    "nuts1": "nuts1",
    "country": "country",
}
REGION_SOURCE_COLUMNS = (
    "country",
    "nuts1",
    "nuts1_code_2024",
    "nuts2",
    "nuts2_code_2024",
    "nuts3",
    "nuts3_code_2024",
)


# Allowed transform filtering rules
ALLOWED_ENCODING_NAMES = {
    # human-interpretable encodings
    "categorical",
    "binary",
    "yesno",
    "moral",
    "moral2",
    "opinion",
    "opinion_change",
    "vax_freq",
    "duration_socialmedia",
    "months",
    "years",
    "$covax_started_any",
    "$born_in_residence_country",
    "$parent_born_abroad",
    # numeric cast
    "$NUM",
}

EXCLUDED_ENCODING_SPECIAL = {
    "$ONE-HOT",
    "$EMBEDDING",
    "$DATETIME_TO_INTERVAL",
    "$INTERVAL_TO_DATETIME",
    "$DATETIME",
    "$COUNTING",
}

ALLOWED_REMOVAL_REASONS = {
    "merging",
    "redundant",
    "survey metadata",
    "dropped from clean spec",
    "missing",
    "renamed",
}

# Aggregations we want to STOP materializing in CLEAN
DISALLOWED_MERGE_TARGETS = {"occupation_full", "origin"}

# Explicit column IDs to remove from the clean spec (no downstream usage expected)
FORCED_REMOVAL_IDS = {
    "occupation_cat_22_TEXT",
    "pregnant_woman",
    "ethnicity_UK",
    "covax",
}

# New-column IDs we never want to materialize in the clean spec output
EXCLUDED_NEW_COLUMN_IDS = {
    "origin",
    "occupation_full",
    "info_traditional",
    "info_online",
    "info_social",
    "covid_date_1",
    "covax_date_1",
    "covax_date_2",
    "covax_reason",
}

REDUNDANT_CANONICAL_COLUMNS: dict[str, str] = {
    "vaccine_safety_avg": "vaccine_risk_avg",
    "vaccine_comfort_avg": "vaccine_fear_avg",
}

VAX_NO_CHILD_OPTIONS = [
    "I am opposed to mandatory vaccination for children",
    "I believe that the vaccines are not effective for children",
    "I believe that the vaccines are not safe for children",
    "I feel that vaccinating children is a waste of money for the benefit of the big pharmaceutical companies",
    "I lack sufficient information about pediatric vaccines",
    "I never saw pediatric illnesses as a serious health risk for my child/children",
    "I think it is better for children to develop natural immunity to pediatric illnesses rather than to get vaccinated",
    "I was personally advised against them by a health care professional - please specify the reason:",
    "I was personally advised against them by a trusted person - please specify the reason:",
    "I worry about the vaccines' side effects for children",
    "My child/children don't need the vaccines because they take other precautions (e.g. a good hygiene)",
    "My child/children have a medical condition that prevents them from getting the vaccine \u2013 please specify:",
    "Other:",
    "The vaccines were not offered to my child/children at a convenient place/time",
]

POLITICS_FAMILY_DEFAULT = "No Affiliation / Prefer not to answer"
POLITICS_PARTY_TO_FAMILY: dict[str, str] = {
    # Germany
    "CDU/CSU": "Conservative / Christian-Democratic",
    "SPD": "Social-Democratic / Labour",
    "Alliance 90 / The Greens": "Green / Environmentalist",
    "Alliance 90/The Greens": "Green / Environmentalist",
    "The Left (Die Linke)": "Radical-Left / Populist-Left",
    "Die Linke": "Radical-Left / Populist-Left",
    "AfD": "Radical-Right / Nationalist / Populist-Right",
    "NPD": "Radical-Right / Nationalist / Populist-Right",
    "Republikaner": "Radical-Right / Nationalist / Populist-Right",
    "The Right": "Radical-Right / Nationalist / Populist-Right",
    "NPD / Republikaner / The Right": "Radical-Right / Nationalist / Populist-Right",
    "FDP": "Liberal / Centrist",
    # Spain
    "PP (Partido Popular)": "Conservative / Christian-Democratic",
    "PP": "Conservative / Christian-Democratic",
    "PSOE": "Social-Democratic / Labour",
    "Sumar": "Radical-Left / Populist-Left",
    "Vox": "Radical-Right / Nationalist / Populist-Right",
    "PNV": "Regionalist / Minority",
    "EH Bildu": "Regionalist / Minority",
    "Junts": "Regionalist / Minority",
    "Junts per Catalunya": "Regionalist / Minority",
    "CC (Coalicion Canaria)": "Regionalist / Minority",
    "CC (Coalici\u00f3n Canaria)": "Regionalist / Minority",
    "Coalicion Canaria": "Regionalist / Minority",
    "Coalici\u00f3n Canaria": "Regionalist / Minority",
    "BNG": "Regionalist / Minority",
    "UPN": "Regionalist / Minority",
    # France
    "La France Insoumise - NUPES": "Radical-Left / Populist-Left",
    "La France Insoumise \u2013 NUPES": "Radical-Left / Populist-Left",
    "National Rally (RN)": "Radical-Right / Nationalist / Populist-Right",
    "Renaissance": "Liberal / Centrist",
    "The Republicans (LR)": "Conservative / Christian-Democratic",
    "The Republicans": "Conservative / Christian-Democratic",
    "Democrat (MoDem & Independents)": "Liberal / Centrist",
    "MoDem": "Liberal / Centrist",
    "Horizons and affiliated": "Liberal / Centrist",
    "Horizons": "Liberal / Centrist",
    "Ecologist - NUPES": "Green / Environmentalist",
    "Ecologist \u2013 NUPES": "Green / Environmentalist",
    "Socialists and affiliated - NUPES": "Social-Democratic / Labour",
    "Socialists and affiliated \u2013 NUPES": "Social-Democratic / Labour",
    "Democratic and Republican Left - NUPES": "Radical-Left / Populist-Left",
    "Democratic and Republican Left \u2013 NUPES": "Radical-Left / Populist-Left",
    "Liberties, Independents, Overseas and Territories": "Regionalist / Minority",
    # Hungary
    "Two-Tailed Dog Party": "Other / Satirical",
    "Momentum Movement": "Liberal / Centrist",
    "Momentum": "Liberal / Centrist",
    "Fidesz-KDNP": "Conservative / Christian-Democratic",
    "Fidesz": "Conservative / Christian-Democratic",
    "MSZP": "Social-Democratic / Labour",
    "Our Homeland Movement": "Radical-Right / Nationalist / Populist-Right",
    "Dialogue (Parbeszed)": "Green / Environmentalist",
    "Dialogue (P\u00e1rbesz\u00e9d)": "Green / Environmentalist",
    "DK (Democratic Coalition)": "Liberal / Centrist",
    "Democratic Coalition": "Liberal / Centrist",
    "Jobbik": "Radical-Right / Nationalist / Populist-Right",
    "Party of Normal Life": "Populist / Catch-All / Anti-Establishment",
    "LMP (Politics Can Be Different)": "Green / Environmentalist",
    "LMP": "Green / Environmentalist",
    "Solution Movement": "Populist / Catch-All / Anti-Establishment",
    # Italy
    "More Europe (+Europa)": "Liberal / Centrist",
    "+Europa": "Liberal / Centrist",
    "Brothers of Italy (FdI)": "Radical-Right / Nationalist / Populist-Right",
    "Fratelli d'Italia": "Radical-Right / Nationalist / Populist-Right",
    "League (Lega)": "Radical-Right / Nationalist / Populist-Right",
    "Lega": "Radical-Right / Nationalist / Populist-Right",
    "Democratic Party (PD)": "Social-Democratic / Labour",
    "Partito Democratico": "Social-Democratic / Labour",
    "Five Star Movement (M5S)": "Populist / Catch-All / Anti-Establishment",
    "M5S": "Populist / Catch-All / Anti-Establishment",
    "Greens and Left Alliance": "Green / Environmentalist",
    "Forza Italia": "Conservative / Christian-Democratic",
    "Italia Viva": "Liberal / Centrist",
    "Italexit for Italy": "Radical-Right / Nationalist / Populist-Right",
    "People's Union with De Magistris": "Radical-Left / Populist-Left",
    "People\u2019s Union with De Magistris": "Radical-Left / Populist-Left",
    "Calenda's Action (Azione)": "Liberal / Centrist",
    "Azione": "Liberal / Centrist",
    "Sovereign and Popular Italy (ISP)": "Radical-Left / Populist-Left",
    "ISP": "Radical-Left / Populist-Left",
    # United Kingdom
    "Labour": "Social-Democratic / Labour",
    "Labour Party": "Social-Democratic / Labour",
    "Green Party of England & Wales": "Green / Environmentalist",
    "Green Party": "Green / Environmentalist",
    "Conservative and Unionist Party": "Conservative / Christian-Democratic",
    "Conservative Party": "Conservative / Christian-Democratic",
    "Liberal Democrats": "Liberal / Centrist",
    "Reform UK": "Radical-Right / Nationalist / Populist-Right",
    "Scottish National Party (SNP)": "Regionalist / Minority",
    "SNP": "Regionalist / Minority",
    "UKIP": "Radical-Right / Nationalist / Populist-Right",
    # Catch-all responses
    "I haven't expressed a preference": POLITICS_FAMILY_DEFAULT,
    "I haven\u2019t expressed a preference": POLITICS_FAMILY_DEFAULT,
    "I havent expressed a preference": POLITICS_FAMILY_DEFAULT,
    "I haven't expressed a preference.": POLITICS_FAMILY_DEFAULT,
    "I haven't expressed preference": POLITICS_FAMILY_DEFAULT,
    "I haven\u2019t expressed preference": POLITICS_FAMILY_DEFAULT,
    "I do not have a preference": POLITICS_FAMILY_DEFAULT,
    "I don't have a preference": POLITICS_FAMILY_DEFAULT,
    "I don't know": POLITICS_FAMILY_DEFAULT,
    "I do not know": POLITICS_FAMILY_DEFAULT,
    "I am not sure": POLITICS_FAMILY_DEFAULT,
    "I'm not sure": POLITICS_FAMILY_DEFAULT,
    "Im not sure": POLITICS_FAMILY_DEFAULT,
    "I\u2019m not sure": POLITICS_FAMILY_DEFAULT,
    "I prefer not to answer": POLITICS_FAMILY_DEFAULT,
    "Prefer not to answer": POLITICS_FAMILY_DEFAULT,
    "Prefer not to say": POLITICS_FAMILY_DEFAULT,
    "Prefer not to disclose": POLITICS_FAMILY_DEFAULT,
    "No preference": POLITICS_FAMILY_DEFAULT,
    "No party": POLITICS_FAMILY_DEFAULT,
    "No affiliation": POLITICS_FAMILY_DEFAULT,
    "None": POLITICS_FAMILY_DEFAULT,
    "Other": POLITICS_FAMILY_DEFAULT,
    "Other party": POLITICS_FAMILY_DEFAULT,
    "Other party (please specify)": POLITICS_FAMILY_DEFAULT,
    "Other / I haven't expressed a preference / Prefer not to answer": POLITICS_FAMILY_DEFAULT,
    "Other / I haven\u2019t expressed a preference / Prefer not to answer": POLITICS_FAMILY_DEFAULT,
    "Other / Prefer not to answer": POLITICS_FAMILY_DEFAULT,
    "Other / Prefer not to say": POLITICS_FAMILY_DEFAULT,
    "Undecided": POLITICS_FAMILY_DEFAULT,
    "Not sure": POLITICS_FAMILY_DEFAULT,
    "": POLITICS_FAMILY_DEFAULT,
    "$DEFAULT": POLITICS_FAMILY_DEFAULT,
}

CUSTOM_ENCODINGS: dict[str, dict[str, Any]] = {
    "coworker_closure_ordinal": {
        "None of them": 0,
        "Very few of them": 1,
        "Some of them": 2,
        "About a half of them": 3,
        "Most of them": 4,
        "Almost all of them": 5,
        "All of them": 6,
    },
    "politics_family": POLITICS_PARTY_TO_FAMILY,
}

BIRTH_MAIN_COUNTRY_MAP = {
    "France": "France",
    "Germany": "Germany",
    "Italy": "Italy",
    "Spain": "Spain",
    "Hungary": "Hungary",
    "United Kingdom": "United Kingdom",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "England": "United Kingdom",
    "Scotland": "United Kingdom",
    "Wales": "United Kingdom",
    "Northern Ireland": "United Kingdom",
    "Great Britain": "United Kingdom",
}

OTHER_EUROPEAN_COUNTRIES = {
    "Albania",
    "Andorra",
    "Armenia",
    "Austria",
    "Azerbaijan",
    "Belarus",
    "Belgium",
    "Bosnia and Herzegovina",
    "Bulgaria",
    "Croatia",
    "Cyprus",
    "Czech Republic",
    "Czechia",
    "Denmark",
    "Estonia",
    "Faroe Islands",
    "Finland",
    "Georgia",
    "Gibraltar",
    "Greece",
    "Greenland",
    "Guernsey",
    "Holy See",
    "Iceland",
    "Ireland",
    "Isle of Man",
    "Jersey",
    "Kosovo",
    "Latvia",
    "Liechtenstein",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Moldova",
    "Republic of Moldova",
    "Monaco",
    "Montenegro",
    "Netherlands",
    "North Macedonia",
    "Norway",
    "Poland",
    "Portugal",
    "Romania",
    "Russian Federation",
    "Russia",
    "San Marino",
    "Serbia",
    "Serbia and Montenegro",
    "Slovakia",
    "Slovenia",
    "Sweden",
    "Switzerland",
    "Turkey",
    "Turkiye",
    "Ukraine",
    "Vatican City",
}

BIRTH_REGION_COLUMNS = {"birth", "birth_m", "birth_f"}


def _build_birth_region_encoding() -> dict[str, str]:
    encoding = dict(BIRTH_MAIN_COUNTRY_MAP)
    for country in sorted(OTHER_EUROPEAN_COUNTRIES):
        encoding.setdefault(country, "Other Europe")
    encoding["$DEFAULT"] = "Non-European"
    return encoding


BIRTH_REGION_ENCODING = _build_birth_region_encoding()


def _should_force_remove(col_id: str | None) -> bool:
    if not col_id:
        return False
    if col_id in FORCED_REMOVAL_IDS:
        return True
    if col_id.startswith("covid_date_") and col_id.endswith("_3"):
        return True
    if col_id.endswith("_TEXT") and col_id.startswith(("covax", "vax_no_child")):
        return True
    return False


def _ensure_removal_transform(
    col: dict[str, Any],
    reason: str = "dropped from clean spec",
    *,
    redundant_col: str | None = None,
) -> None:
    transforms = col.setdefault("transformations", [])
    already_present = any(t.get("type") == "removal" and t.get("removal_reason") == reason for t in transforms)
    if not already_present:
        t: dict[str, Any] = {"type": "removal", "removal_reason": reason}
        if redundant_col:
            t["removal_redundant_col"] = redundant_col
        transforms.append(t)


def _col_index_by_id(columns: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {c.get("id"): c for c in columns or []}


def _new_col(
    *,
    col_id: str,
    question: str,
    transformations: list[dict[str, Any]],
    section: str | None = None,
    category: str | None = None,
    subcategory: str | None = None,
    question_group: str | None = None,
    question_entity: str | None = None,
) -> dict[str, Any]:
    col: dict[str, Any] = {
        "id": col_id,
        "question": question,
        "transformations": transformations,
    }
    if section:
        col["section"] = section
    if category:
        col["category"] = category
    if subcategory:
        col["subcategory"] = subcategory
    if question_group:
        col["question_group"] = question_group
    if question_entity:
        col["question_entity"] = question_entity
    return col


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass(slots=True)
class RegionRollupResult:
    code_mapping: dict[str, str]
    label_mapping: dict[str, str]
    log_rows: list[dict[str, Any]]
    total_rows: int


def _load_region_source_rows(path: Path) -> list[dict[str, str | None]]:
    if not path.exists():
        raise FileNotFoundError(f"Regional source data not found at {path} - run preprocessing download first.")

    df = pl.read_csv(path, columns=list(REGION_SOURCE_COLUMNS), null_values=["", "NA", "N/A"])
    # Ruff: avoid unnecessary list comprehension, use list() where appropriate
    str_cols = list(REGION_SOURCE_COLUMNS)
    df = df.with_columns([pl.col(col).cast(pl.Utf8).str.strip_chars().replace("", None).alias(col) for col in str_cols])
    return df.to_dicts()


def _determine_initial_level(levels: dict[str, str | None]) -> tuple[str, str | None]:
    for level in REGION_LEVEL_ORDER:
        code = levels.get(level)
        if code:
            return level, code
    # Default to country level when nothing else is available
    return "country", levels.get("country")


def _find_next_level(levels: dict[str, str | None], current_level: str) -> tuple[str | None, str | None]:
    start_idx = REGION_LEVEL_ORDER.index(current_level)
    for next_level in REGION_LEVEL_ORDER[start_idx + 1 :]:
        candidate = levels.get(next_level)
        if candidate:
            return next_level, candidate
    return None, None


def _build_region_rollup_result() -> RegionRollupResult:
    rows = _load_region_source_rows(REGION_SOURCE_DATA)
    if not rows:
        raise ValueError("Regional source data is empty; cannot compute roll-ups.")

    assignments: list[dict[str, Any]] = []
    labels_registry: dict[tuple[str, str], str] = {}

    for row in rows:
        country = row.get("country")
        if not country:
            continue
        levels = {level: row.get(REGION_CODE_COLUMNS[level]) for level in REGION_LEVEL_ORDER}
        levels["country"] = country
        labels = {level: row.get(REGION_LABEL_COLUMNS[level]) for level in REGION_LEVEL_ORDER}
        labels["country"] = labels.get("country") or country

        initial_level, initial_code = _determine_initial_level(levels)
        if not initial_code:
            continue

        assignments.append(
            {
                "country": country,
                "levels": levels,
                "labels": labels,
                "current_level": initial_level,
                "current_code": initial_code,
                "initial_level": initial_level,
                "initial_code": initial_code,
                "small_sample": False,
            }
        )

        for level in REGION_LEVEL_ORDER:
            code = levels.get(level)
            if not code:
                continue
            label_value = labels.get(level) or code
            labels_registry.setdefault((level, code), label_value)

    rollup_log: list[dict[str, Any]] = []

    for level in REGION_LEVEL_ORDER:
        threshold = REGION_THRESHOLDS[level]
        rows_by_code: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
        for idx, assignment in enumerate(assignments):
            if assignment["current_level"] != level:
                continue
            code = assignment["current_code"]
            if not code:
                continue
            key = (assignment["country"], code)
            rows_by_code[key].append(idx)

        if not rows_by_code:
            continue

        for key, indices in rows_by_code.items():
            count = len(indices)
            if level == "country":
                if count < threshold:
                    for idx in indices:
                        assignments[idx]["small_sample"] = True
                    label_value = labels_registry.get((level, key[1]), key[1])
                    rollup_log.append(
                        {
                            "action": "mark_small_sample",
                            "country": key[0],
                            "source_code": key[1],
                            "source_label": label_value,
                            "source_level": level,
                            "source_sample": count,
                            "threshold": threshold,
                            "target_code": key[1],
                            "target_label": f"{label_value} (small sample)",
                            "target_level": level,
                        }
                    )
                continue

            if count >= threshold:
                continue

            sample_idx = indices[0]
            next_level, next_code = _find_next_level(assignments[sample_idx]["levels"], level)
            if not next_code or not next_level:
                continue

            for idx in indices:
                assignments[idx]["current_level"] = next_level
                assignments[idx]["current_code"] = next_code

            rollup_log.append(
                {
                    "action": "roll_up",
                    "country": key[0],
                    "source_code": key[1],
                    "source_label": labels_registry.get((level, key[1]), key[1]),
                    "source_level": level,
                    "source_sample": count,
                    "threshold": threshold,
                    "target_code": next_code,
                    "target_label": labels_registry.get((next_level, next_code), next_code),
                    "target_level": next_level,
                }
            )

    code_mapping: dict[str, str] = {}
    label_mapping: dict[str, str] = {}

    for assignment in assignments:
        src_code = assignment.get("initial_code")
        final_code = assignment.get("current_code")
        final_level = assignment.get("current_level")
        if not src_code or not final_code or not final_level:
            continue
        existing = code_mapping.get(src_code)
        if existing and existing != final_code:
            raise ValueError(f"Inconsistent roll-up detected for code {src_code}")

        code_mapping[src_code] = final_code
        label_value = labels_registry.get((final_level, final_code), final_code)
        if assignment.get("small_sample") and final_level == "country":
            label_value = f"{label_value} (small sample)"
        label_mapping[src_code] = label_value

    sorted_code_mapping = dict(sorted(code_mapping.items()))
    sorted_label_mapping = {key: label_mapping[key] for key in sorted(label_mapping)}

    rollup_log.sort(
        key=lambda entry: (entry["country"], REGION_LEVEL_ORDER.index(entry["source_level"]), entry["source_code"])
    )

    return RegionRollupResult(
        code_mapping=sorted_code_mapping,
        label_mapping=sorted_label_mapping,
        log_rows=rollup_log,
        total_rows=len(assignments),
    )


def _write_region_rollup_log(result: RegionRollupResult) -> None:
    REGION_ROLLUP_LOG.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source": str(REGION_SOURCE_DATA),
        "thresholds": REGION_THRESHOLDS,
        "total_rows": result.total_rows,
        "rollups": result.log_rows,
    }
    with REGION_ROLLUP_LOG.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _index_columns(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cols = {}
    for col in spec.get("old_columns", []) or []:
        cols[col["id"]] = col
    for col in spec.get("new_columns", []) or []:
        cols[col["id"]] = col
    return cols


def _filter_transformations(ts: list[dict[str, Any]], *, stage: str) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []

    for t in ts:
        ttype = t.get("type")
        # For post-general application, drop merge/split steps that were already materialized
        if stage == "post-general" and ttype in {"merged", "merging", "splitting", "splitted"}:
            continue
        if ttype == "encoding":
            enc = t.get("encoding")
            # dict_map (explicit mapping) → keep
            if isinstance(enc, dict):
                filtered.append(t)
                continue

            if isinstance(enc, str):
                if enc in EXCLUDED_ENCODING_SPECIAL:
                    continue
                if enc in ALLOWED_ENCODING_NAMES:
                    filtered.append(t)
                    continue
            # otherwise drop
            continue

        if ttype == "imputation":
            val = t.get("imputation_value")
            # Exclude model-based or function-like imputations
            if isinstance(val, str) and val.startswith("$"):
                continue
            # Exclude blank string imputations
            if isinstance(val, str) and (val.strip() == ""):
                continue
            # Keep constants (numbers or non-$ strings)
            filtered.append(t)
            continue

        if ttype == "outliers":
            filtered.append(t)
            continue

        if ttype in {"merging", "merged", "splitting", "splitted"}:
            # Drop merges that aggregate into disallowed targets (keep sources separate)
            if ttype == "merging" and t.get("merging_in") in DISALLOWED_MERGE_TARGETS:
                continue
            # If the merge target is excluded from CLEAN output, drop the merge marker.
            # (We still keep/remap any corresponding source removals below so the
            # per-person component columns don't leak into the clean dataset.)
            if ttype == "merging" and t.get("merging_in") in EXCLUDED_NEW_COLUMN_IDS:
                continue
            filtered.append(t)
            continue

        if ttype == "removal":
            reason = t.get("removal_reason")
            # After general, skip removals that were only for merge cleanup
            if stage == "post-general" and reason == "merging":
                continue
            # Skip removals used to clean up merges into disallowed targets
            if reason == "merging" and t.get("removal_redundant_col") in DISALLOWED_MERGE_TARGETS:
                continue
            # If the merge target itself is excluded from CLEAN output, keep removing
            # the source column but avoid referencing the non-materialized target.
            if reason == "merging" and t.get("removal_redundant_col") in EXCLUDED_NEW_COLUMN_IDS:
                filtered.append({"type": "removal", "removal_reason": "dropped from clean spec"})
                continue
            if reason in ALLOWED_REMOVAL_REASONS:
                filtered.append(t)
            continue

        if ttype in {"column_substitution", "renaming", "renamed"}:
            filtered.append(t)
            continue

        # Exclude: scaling, binning, others not explicitly allowed
        # (No-op)

    return filtered


def _dedupe_transformations(lst: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out: list[dict[str, Any]] = []
    for t in lst or []:
        key = json.dumps(t, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def _collect_used_encoding_names(columns: list[dict[str, Any]]) -> set[str]:
    used = set()
    for col in columns:
        for t in col.get("transformations") or []:
            if t.get("type") == "encoding":
                enc = t.get("encoding")
                if isinstance(enc, str) and not enc.startswith("$"):
                    used.add(enc)
                # dict encodings are inline; no name to collect
    return used


def _section_file_slug(section: str) -> str:
    slug = section.strip().lower().replace(" ", "_")
    cleaned = "".join(ch for ch in slug if ch.isalnum() or ch in {"_", "-"})
    return cleaned or "section"


def _write_section_specs(output_spec: dict[str, Any], stage: str) -> int:
    sections: defaultdict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"old_columns": [], "new_columns": []}
    )
    for column_key in ("old_columns", "new_columns"):
        for col in output_spec.get(column_key, []) or []:
            section = col.get("section")
            if not section:
                continue
            sections[section][column_key].append(col)

    if not sections:
        return 0

    all_encodings = output_spec.get("encodings", {})
    section_dir = SECTION_SPEC_DIR / stage
    section_dir.mkdir(parents=True, exist_ok=True)

    for section_name, data in sections.items():
        section_columns = (data["old_columns"] or []) + (data["new_columns"] or [])
        section_encoding_names = _collect_used_encoding_names(section_columns)
        section_encodings = {
            name: all_encodings[name] for name in sorted(section_encoding_names) if name in all_encodings
        }
        section_spec = {
            "encodings": section_encodings,
            "question_group": output_spec.get("question_group"),
            "old_columns": data["old_columns"],
            "new_columns": data["new_columns"],
        }
        target_path = section_dir / f"{_section_file_slug(section_name)}.json"
        with target_path.open("w", encoding="utf-8") as f:
            json.dump(section_spec, f, ensure_ascii=False, indent=2)
            f.write("\n")

    return len(sections)


def build_clean_analysis_spec(
    output_path: Path,
    general_path: Path = GENERAL_SPEC,
    *,
    stage: str = "raw",
) -> dict[str, Any]:
    """
    Build a clean, analysis-friendly transform spec derived ONLY from the
    general spec. Produces an alternate "clean" spec alongside the general one.

    stage:
      - "raw": apply to raw survey data (keep merges/splits/removals per rules)
      - "post-general": apply on top of general-processed CSV (drop merges/splits
        and merge-removals, since they're already materialized)
    """
    general = _load_json(general_path)

    # Track section metadata for all known columns so derived columns can inherit it.
    column_sections: dict[str, str] = {}

    def _seed_sections(entries: list[dict[str, Any]] | None) -> None:
        for entry in entries or []:
            col_id = entry.get("id")
            section = entry.get("section")
            if col_id and isinstance(section, str):
                section_clean = section.strip()
                if section_clean:
                    column_sections[col_id] = section_clean

    _seed_sections(general.get("old_columns"))
    _seed_sections(general.get("new_columns"))

    custom_encodings_needed: set[str] = set()
    rename_markers: dict[str, set[str]] = defaultdict(set)

    def _ensure_reverse_encoding(transforms: list[dict[str, Any]]) -> None:
        if any(t.get("type") == "encoding" and t.get("encoding") == "$reverse_1_to_7" for t in transforms):
            return
        insert_at = None
        if insert_at is None:
            transforms.append({"type": "encoding", "encoding": "$reverse_1_to_7"})
        else:
            transforms.insert(insert_at, {"type": "encoding", "encoding": "$reverse_1_to_7"})

    # Filter transformations for old_columns
    clean_old: list[dict[str, Any]] = []
    for col in general.get("old_columns", []) or []:
        base = deepcopy(col)
        base["transformations"] = _dedupe_transformations(
            _filter_transformations(deepcopy(col.get("transformations") or []), stage=stage)
        )
        if _should_force_remove(base.get("id")):
            _ensure_removal_transform(base)
        col_id = base.get("id")
        if col_id in NUTS_CODE_COLUMNS | NUTS_NAME_COLUMNS:
            base["transformations"] = [t for t in base.get("transformations", []) if t.get("type") != "removal"]
            _ensure_removal_transform(base, reason="redundant")
        if col_id == "politics":
            transforms = base.setdefault("transformations", [])
            has_family_encoding = any(
                t.get("type") == "encoding" and t.get("encoding") == "politics_family" for t in transforms
            )
            if not has_family_encoding:
                transforms.append({"type": "encoding", "encoding": "politics_family"})
            custom_encodings_needed.add("politics_family")
        if col_id == "coworker_closure":
            base.setdefault("transformations", []).append({"type": "encoding", "encoding": "coworker_closure_ordinal"})
            custom_encodings_needed.add("coworker_closure_ordinal")
        if col_id in BIRTH_REGION_COLUMNS:
            transforms = base.setdefault("transformations", [])
            has_region_encoding = any(
                t.get("type") == "encoding"
                and isinstance(t.get("encoding"), dict)
                and t["encoding"].get("$DEFAULT") == "Non-European"
                for t in transforms
            )
            if not has_region_encoding:
                transforms.append({"type": "encoding", "encoding": deepcopy(BIRTH_REGION_ENCODING)})
        if col_id in {"risk_1_4", "risk_2_4"}:
            # Keep the alias-friendly "pre" columns and ensure they're numerically encoded.
            transforms = base.setdefault("transformations", [])
            retained = [t for t in transforms if t.get("type") != "removal"]
            base["transformations"] = retained
            has_categorical = any(t.get("type") == "encoding" and t.get("encoding") == "categorical" for t in retained)
            if not has_categorical:
                base["transformations"].insert(0, {"type": "encoding", "encoding": "categorical"})
        if col_id == "income":
            transforms = base.setdefault("transformations", [])
            transforms[:] = [t for t in transforms if t.get("type") != "removal"]

            def _has_transform(check: dict[str, Any], transforms=transforms) -> bool:
                return any(all(t.get(k) == v for k, v in check.items()) for t in transforms)

            if not _has_transform({"type": "encoding", "encoding": "$income_ppp_normalized"}):
                transforms.append({"type": "encoding", "encoding": "$income_ppp_normalized"})
            # Skip rounding; keep full precision.
        if col_id == "sex":
            transforms = base.setdefault("transformations", [])
            has_boolean_encoding = any(
                t.get("type") == "encoding"
                and isinstance(t.get("encoding"), dict)
                and t["encoding"].get("Female") is True
                and t["encoding"].get("Male") is False
                for t in transforms
            )
            if not has_boolean_encoding:
                transforms.append({"type": "encoding", "encoding": {"Female": True, "Male": False}})
        if col_id and col_id.startswith(("covid_date_", "covax_date_")):
            suffix = col_id.rsplit("_", 1)[-1]
            enc = "months" if suffix == "1" else "years" if suffix == "2" else None
            if enc:
                transforms = base.setdefault("transformations", [])
                if not any(t.get("type") == "encoding" and t.get("encoding") == enc for t in transforms):
                    transforms.insert(0, {"type": "encoding", "encoding": enc})
        if col_id in {
            "fiveC_2",
            "fiveC_3",
            "fiveC_5",
            "opinion_4",
            "opinion_other_4_6",
            "opinion_other_4_family",
            "opinion_other_4_colleagues",
        }:
            # Keep reverse-code sources available; drop removals that would delete them
            base["transformations"] = [t for t in base.get("transformations", []) if t.get("type") != "removal"]
        if col_id in {"fiveC_2", "fiveC_3", "fiveC_5", "opinion_4", "opinion_other_4_6"}:
            transforms = base.setdefault("transformations", [])
            _ensure_reverse_encoding(transforms)
            # Do NOT rename in place here.
            # These sources already have analysis-friendly alias columns (via `renamed`),
            # and in-place renaming would create duplicate column names once aliases
            # are materialized.
        if col_id and col_id.startswith("info_trust_vax_"):
            transforms = [t for t in base.get("transformations", []) if t.get("type") != "removal"]
            base["transformations"] = transforms
            has_categorical_encoding = any(
                t.get("type") == "encoding" and t.get("encoding") == "categorical" for t in transforms
            )
            if not has_categorical_encoding:
                transforms.append({"type": "encoding", "encoding": "categorical"})
        clean_old.append(base)

    # Filter transformations for new_columns
    clean_new: list[dict[str, Any]] = []
    for col in general.get("new_columns", []) or []:
        if _should_force_remove(col.get("id")) or col.get("id") in EXCLUDED_NEW_COLUMN_IDS:
            continue
        base = deepcopy(col)
        base["transformations"] = _dedupe_transformations(
            _filter_transformations(deepcopy(col.get("transformations") or []), stage=stage)
        )
        removed_all_sources = False
        pending_rename_sources: list[str] = []
        for t in base.get("transformations", []):
            if t.get("type") == "renaming":
                src = t.get("rename_from")
                if src:
                    t["type"] = "renamed"
                    pending_rename_sources.append(src)
            if t.get("type") == "merged":
                merged_from = t.get("merged_from") or []
                kept_sources = [src for src in merged_from if not _should_force_remove(src)]
                if len(kept_sources) != len(merged_from):
                    removed_sources = set(merged_from) - set(kept_sources)
                    t["merged_from"] = kept_sources
                    concat_string = t.get("concat_string")
                    if concat_string:
                        for src in removed_sources:
                            concat_string = concat_string.replace(f" {{{src}}}", "")
                            concat_string = concat_string.replace(f"{{{src}}} ", "")
                            concat_string = concat_string.replace(f"{{{src}}}", "")
                        t["concat_string"] = " ".join(concat_string.split())
                if not kept_sources:
                    removed_all_sources = True
            if t.get("type") == "splitted" and _should_force_remove(t.get("split_from")):
                removed_all_sources = True
        if removed_all_sources:
            continue
        col_id = base.get("id")
        if col_id in {"opinion_other_4_family", "opinion_other_4_colleagues"}:
            transforms = base.setdefault("transformations", [])
            _ensure_reverse_encoding(transforms)
            # Same rationale as above: keep transformations on the source and
            # let the analysis-friendly alias copy the transformed values.
        for src in pending_rename_sources:
            rename_markers[src].add(base.get("id"))
        # No in-place reverse here; we will add derived columns later.
        clean_new.append(base)

    # Build encodings dictionary restricted to those actually used across both
    used_encs = _collect_used_encoding_names(clean_old + clean_new)
    merged_encodings: dict[str, Any] = {}
    general_encodings = general.get("encodings") or {}
    for enc_name in sorted(used_encs):
        if enc_name in general_encodings:
            merged_encodings[enc_name] = general_encodings[enc_name]
    for enc_name in sorted(custom_encodings_needed):
        if enc_name in CUSTOM_ENCODINGS:
            merged_encodings[enc_name] = CUSTOM_ENCODINGS[enc_name]

    # Start composing output spec
    output_spec: dict[str, Any] = {
        "encodings": merged_encodings,
        "old_columns": clean_old,
    }
    if clean_new:
        output_spec["new_columns"] = clean_new
    if general.get("question_group"):
        output_spec["question_group"] = general["question_group"]

    # Ensure current output columns are registered for section lookups.
    _seed_sections(output_spec.get("old_columns"))
    _seed_sections(output_spec.get("new_columns"))

    # Helper index for existence checks
    gen_old_idx = _col_index_by_id(general.get("old_columns") or [])
    gen_new_idx = _col_index_by_id(general.get("new_columns") or [])

    def col_exists(cid: str) -> bool:
        return cid in gen_old_idx or cid in gen_new_idx

    def _finalized_source_id(cid: str) -> str:
        col = gen_old_idx.get(cid) or gen_new_idx.get(cid)
        if not col:
            return cid
        for t in col.get("transformations") or []:
            if t.get("type") == "renaming":
                rename_to = t.get("rename_to") or []
                if rename_to:
                    return rename_to[-1]
        return cid

    def _finalized_source_id_output(cid: str) -> str:
        for bucket in ("old_columns", "new_columns"):
            for col in output_spec.get(bucket, []) or []:
                if col.get("id") != cid:
                    continue
                for t in col.get("transformations") or []:
                    if t.get("type") == "renaming":
                        rename_to = t.get("rename_to") or []
                        if rename_to:
                            return rename_to[-1]
                return cid
        return _finalized_source_id(cid)

    # Ensure we have a new_columns list to append to
    output_spec.setdefault("new_columns", [])
    additions: list[dict[str, Any]] = []
    pending_new_ids: set[str] = set()

    def _register_section(col: dict[str, Any]) -> None:
        col_id = col.get("id")
        section = col.get("section")
        if col_id and isinstance(section, str) and section.strip():
            column_sections[col_id] = section.strip()

    def _infer_section_from_sources(
        sources: Iterable[str] | None,
        default: str | None = "analysis",
    ) -> str | None:
        for candidate in sources or []:
            if not candidate:
                continue
            section = column_sections.get(candidate)
            if section:
                return section
            if "__" in candidate:
                base = candidate.split("__", 1)[0]
                section = column_sections.get(base)
                if section:
                    return section
        return default

    def _make_new_col(
        *,
        section_sources: Iterable[str] | None = None,
        default_section: str | None = "analysis",
        **kwargs: Any,
    ) -> dict[str, Any]:
        col_kwargs = dict(kwargs)
        section = col_kwargs.get("section")
        if section is None:
            inferred = _infer_section_from_sources(section_sources, default_section)
            if inferred:
                col_kwargs["section"] = inferred
        col = _new_col(**col_kwargs)
        _register_section(col)
        return col

    def _collect_source_ids(col: dict[str, Any]) -> list[str]:
        sources: list[str] = []
        for t in col.get("transformations") or []:
            ttype = t.get("type")
            if ttype in {"renamed", "renaming"}:
                src = t.get("rename_from")
                if isinstance(src, str):
                    sources.append(src)
            if ttype in {"merged", "merging"}:
                merged_from = t.get("merged_from") or []
                for src in merged_from:
                    if isinstance(src, str):
                        sources.append(src)
            if ttype in {"splitted", "splitting"}:
                src = t.get("split_from")
                if isinstance(src, str):
                    sources.append(src)
            if ttype == "column_substitution":
                src = t.get("column_substitution_from")
                if isinstance(src, str):
                    sources.append(src)
            if ttype == "imputation":
                val = t.get("imputation_value")
                if isinstance(val, str) and val.startswith("$") and len(val) > 1:
                    sources.append(val.lstrip("$"))
        return sources

    def _ensure_column_sections() -> None:
        def _ensure(col: dict[str, Any]) -> None:
            if col.get("section"):
                _register_section(col)
                return
            inferred = _infer_section_from_sources(_collect_source_ids(col))
            if inferred:
                col["section"] = inferred
                _register_section(col)

        for col in output_spec.get("old_columns", []):
            _ensure(col)
        for col in output_spec.get("new_columns", []):
            _ensure(col)

    # --------------
    # Aliases (analysis-friendly names)
    # --------------
    alias_defs: dict[str, str] = {}

    def register_alias(src: str, dst: str) -> None:
        if not col_exists(src):
            return
        if src in alias_defs:
            return
        alias_defs[src] = dst

    curated_aliases: dict[str, str] = {
        # FiveC + attitudes
        "fiveC_1": "fivec_confidence",
        "fiveC_2": "fivec_low_complacency",
        "fiveC_3": "fivec_low_constraints",
        "fiveC_5": "fivec_collective_responsibility",
        "opinion_1": "att_vaccines_importance",
        "opinion_2": "att_climate_harm",
        "opinion_3": "att_lgbt_adoption_rights",
        "opinion_4": "att_immigration_tolerance",
        # Perceived social norms (friends/family/colleagues)
        "opinion_other_2_6": "friends_climate_harm",
        "opinion_other_3_6": "friends_lgbt_adoption_rights",
        "opinion_other_4_6": "friends_immigration_tolerance",
        "opinion_other_2_family": "family_climate_harm",
        "opinion_other_3_family": "family_lgbt_adoption_rights",
        "opinion_other_4_family": "family_immigration_tolerance",
        "opinion_other_2_colleagues": "colleagues_climate_harm",
        "opinion_other_3_colleagues": "colleagues_lgbt_adoption_rights",
        "opinion_other_4_colleagues": "colleagues_immigration_tolerance",
        # Risk/willingness (time anchored)
        "risk_2_1": "vax_willingness_T12",
        "risk_1_1": "covid_perceived_danger_T12",
        # Add consistent aliases for all COVID danger timepoints
        "risk_1_2": "covid_perceived_danger_T3",
        "risk_1_3": "covid_perceived_danger_T4",
        "risk_1_4": "covid_perceived_danger_pre",
        "risk_1_5": "covid_perceived_danger_Y",
        # Add consistent aliases for all vaccination willingness timepoints
        "risk_2_2": "vax_willingness_T3",
        "risk_2_3": "vax_willingness_T4",
        "risk_2_4": "vax_willingness_pre",
        "risk_2_5": "vax_willingness_Y",
        # Current risk perception (examples)
        "risk_infection_1": "covid_infection_likelihood_current",
        "risk_infection_2": "flu_infection_likelihood_current",
        "risk_infection_3": "measles_infection_likelihood_current",
        "risk_infection_4": "meningitis_infection_likelihood_current",
        "risk_illness_1": "covid_severity_if_infected",
        # Item-level examples from spec
        "risk_vax_3": "measles_vaccine_risk",
        "emotion_vax_1": "covid_vaccine_fear",
        "emotion_illness_4": "meningitis_disease_fear",
        # Trust & behaviours
        "institut_trust_vax_3": "trust_gp_vax_info",
        "protective_behaviour_2": "mask_when_symptomatic_crowded",
        # MFQ examples
        "moral1_1": "mfq_relevance_care_emotional_suffering",
        "moral1_7": "mfq_relevance_care_protect_weak",
        "moral2_10": "mfq_agreement_authority_gender_roles",
        # Income (PPP-normalised)
        "income": "income_ppp_norm",
        # Sex indicator
        "sex": "sex_female",
    }

    for src, dst in curated_aliases.items():
        register_alias(src, dst)

    # risk_vax_* → disease-specific vaccine risk
    risk_vax_names = {1: "covid", 2: "flu", 3: "measles", 4: "meningitis"}
    for i, disease in risk_vax_names.items():
        if i == 3:
            # already covered by alias_map as measles_vaccine_risk
            continue
        register_alias(f"risk_vax_{i}", f"{disease}_vaccine_risk")

    # emotion_vax_* → vaccine fear
    for i, disease in risk_vax_names.items():
        dst = f"{disease}_vaccine_fear"
        register_alias(f"emotion_vax_{i}", dst)

    # emotion_illness_* → disease fear
    disease_names = {1: "covid", 2: "flu", 3: "measles", 4: "meningitis"}
    for i, disease in disease_names.items():
        dst = f"{disease}_disease_fear"
        register_alias(f"emotion_illness_{i}", dst)

    # risk_illness_* → severity if infected
    for i, disease in disease_names.items():
        if i == 1:
            # already aliased to covid_severity_if_infected
            continue
        register_alias(f"risk_illness_{i}", f"{disease}_severity_if_infected")

    # Flu vaccination history
    fluvax_map = {
        1: "flu_vaccinated_pre_pandemic",
        2: "flu_vaccinated_2020_2021",
        3: "flu_vaccinated_2021_2022",
        4: "flu_vaccinated_2022_2023",
        5: "flu_vaccinated_2023_2024",
    }
    for i, name in fluvax_map.items():
        register_alias(f"fluvax_{i}", name)

    # Vaccine conversation counts
    # NOTE: We intentionally do NOT keep the component items for
    # noncohabitants/colleagues (e.g. *_noncohabitants_2, *_noncohabitants_3,
    # *_colleagues_4, *_colleagues_5). Historically we kept only the aggregated
    # totals.
    vax_group_names = {
        1: "cohabitants",
        6: "friends",
        7: "others",
    }
    for i, group in vax_group_names.items():
        register_alias(f"vax_group_{i}", f"vax_convo_count_{group}")
    # Aggregated counts
    register_alias("vax_group_noncohabitant", "vax_convo_count_noncohabitants")
    register_alias("vax_group_colleagues", "vax_convo_count_colleagues")

    # Vaccine conversation frequency (if individual items exist)
    for i, group in vax_group_names.items():
        register_alias(f"vax_freq_{i}", f"vax_convo_freq_{group}")
    register_alias("vax_freq_noncohabitant", "vax_convo_freq_noncohabitants")
    register_alias("vax_freq_colleagues", "vax_convo_freq_colleagues")

    # Opinion change topics
    opinion_change_topics = {
        1: "vaccines_importance",
        2: "climate_harm",
        3: "lgbt_adoption_rights",
        4: "immigration_harms_economy",
    }
    for i, topic in opinion_change_topics.items():
        register_alias(f"opinion_change_{i}", f"opinion_change_{topic}")

    # Friends' perceived norms (friend group is index 6)
    norms_topics = dict(opinion_change_topics)
    norms_topics[4] = "immigration_tolerance"
    for j, topic in norms_topics.items():
        register_alias(f"opinion_other_{j}_6", f"friends_{topic}")

    # Family perceived norms (explicit family items)
    for j, topic in norms_topics.items():
        register_alias(f"opinion_other_{j}_family", f"family_{topic}")

    # Colleagues perceived norms (co-workers/classmates)
    for j, topic in norms_topics.items():
        register_alias(f"opinion_other_{j}_colleagues", f"colleagues_{topic}")

    # 5C calculation (item 4)
    register_alias("fiveC_4", "fivec_calculation")

    # Institutional trust (vax info)
    inst_names = {
        1: "local_gov",
        2: "national_gov",
        3: "gp",  # already in alias_map
        4: "local_health_authorities",
        5: "national_health_institutions",
        6: "who",
        7: "religious_leaders",
        8: "influencers",
    }
    for i, about in inst_names.items():
        if i == 3:
            continue
        register_alias(f"institut_trust_vax_{i}", f"trust_{about}_vax_info")

    # Protective behaviors
    register_alias("protective_behaviour_1", "support_mandatory_vaccination")
    register_alias("protective_behaviour_3", "stay_home_when_symptomatic")
    register_alias("protective_behaviour_4", "mask_when_pressure_high")

    # Duration on information/social media platforms
    platform_names = {
        1: "wikipedia",
        2: "instagram",
        3: "facebook",
        4: "telegram",
        5: "twitter",
        6: "linkedin",
        7: "tiktok",
        8: "youtube",
        9: "reddit",
    }
    for i, p in platform_names.items():
        register_alias(f"duration_socialmedia_{i}", f"time_{p}")

    info_trust_channels = {
        1: "print",
        2: "online_news",
        3: "television",
        4: "radio",
        5: "podcast",
        6: "internet_searches",
        7: "wikipedia",
        8: "instagram",
        9: "facebook",
        10: "twitter",
        11: "linkedin",
        12: "telegram",
        13: "tiktok",
        14: "youtube",
        15: "reddit",
        16: "other",
    }
    for i, channel in info_trust_channels.items():
        register_alias(f"info_trust_vax_{i}", f"info_trust_vax_{channel}")

    # MFQ items (generic aliases when present and not already covered)
    for i in range(1, 12):
        register_alias(f"moral1_{i}", f"mfq_relevance_{i}")
        register_alias(f"moral2_{i}", f"mfq_agreement_{i}")

    # Cohabitant / Noncohabitant info items
    # NOTE: We intentionally do NOT keep the individual per-person info columns
    # (age/relationship/vax/covid). We only keep aggregated summary columns like
    # cohabitant_covax_num / noncohabitant_covax_num.

    region_rollup: RegionRollupResult | None = None
    if stage == "raw":
        try:
            region_rollup = _build_region_rollup_result()
            _write_region_rollup_log(region_rollup)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "Regional source data missing; expected data/1_CROSSCOUNTRY_translated.csv when building clean spec"
            ) from exc

    existing_new_ids = {c.get("id") for c in output_spec.get("new_columns", [])}
    added_alias_columns: set[str] = set()

    # NOTE: analysis-friendly aliases are implemented as *new columns* using
    # a `renamed` transformation (rename_from=<src>). We intentionally do NOT
    # also emit an in-place `renaming` marker on the source column.
    #
    # Emitting both is redundant and can create confusing/invalid sequences
    # like `removal` followed by `renaming` on the same source column.
    for src, dst in alias_defs.items():
        if dst in added_alias_columns or dst in existing_new_ids:
            continue
        additions.append(
            _make_new_col(
                col_id=dst,
                question=f"Alias of {src} (analysis-friendly)",
                transformations=[{"type": "renamed", "rename_from": src}],
                section_sources=[src],
            )
        )
        pending_new_ids.add(dst)
        added_alias_columns.add(dst)

        # Mirror the previous in-place rename semantics: once an analysis-friendly
        # alias exists, the original source column should not remain in the final
        # clean dataset.
        #
        # Implementation detail:
        # - We add an explicit `removal` transform with reason="renamed".
        # - We only do this when the source does not already have a removal,
        #   to avoid multiple competing removal nodes for the same column.
        src_col = None
        for bucket in ("old_columns", "new_columns"):
            for col in output_spec.get(bucket, []) or []:
                if col.get("id") == src:
                    src_col = col
                    break
            if src_col is not None:
                break
        if src_col is not None:
            has_any_removal = any(t.get("type") == "removal" for t in (src_col.get("transformations") or []))
            if not has_any_removal:
                _ensure_removal_transform(src_col, reason="renamed", redundant_col=dst)

    region_source_cols = [
        REGION_CODE_COLUMNS["nuts3"],
        REGION_CODE_COLUMNS["nuts2"],
        REGION_CODE_COLUMNS["nuts1"],
        REGION_CODE_COLUMNS["country"],
    ]
    # Future option: re-enable region_final_code output if raw codes needed again.
    # if region_rollup and region_rollup.code_mapping:
    #     additions.append(
    #         _make_new_col(
    #             col_id="region_final_code",
    #             question=(
    #                 "Final respondent region code after hierarchical roll-up (NUTS3 → NUTS2 → NUTS1 → Country)"
    #             ),
    #             transformations=[
    #                 {"type": "merged", "merging_type": "missing", "merged_from": region_source_cols},
    #                 {"type": "encoding", "encoding": region_rollup.code_mapping},
    #             ],
    #             section="quotas",
    #             section_sources=region_source_cols,
    #         )
    #     )
    if region_rollup and region_rollup.label_mapping:
        if "region_final" not in existing_new_ids and "region_final" not in pending_new_ids:
            additions.append(
                _make_new_col(
                    col_id="region_final",
                    question=(
                        "Final respondent region label after hierarchical roll-up (NUTS3 → NUTS2 → NUTS1 → Country)"
                    ),
                    transformations=[
                        {"type": "merged", "merging_type": "missing", "merged_from": region_source_cols},
                        {"type": "encoding", "encoding": region_rollup.label_mapping},
                    ],
                    section="quotas",
                    section_sources=region_source_cols,
                )
            )
            pending_new_ids.add("region_final")

    def _append_renaming_marker(col_id: str, destinations: set[str]) -> None:
        if not destinations:
            return
        target = next((c for c in output_spec.get("old_columns", []) if c.get("id") == col_id), None)
        if target is None:
            target = next((c for c in output_spec.get("new_columns", []) if c.get("id") == col_id), None)
        if target is None:
            return
        transforms = target.setdefault("transformations", [])
        marker = next((t for t in transforms if t.get("type") == "renaming"), None)
        if marker is None:
            marker = {"type": "renaming", "rename_to": []}
            transforms.append(marker)
        existing = set(marker.get("rename_to") or [])
        existing.update(destinations)
        marker["rename_to"] = sorted(existing)

    for src_id, dests in rename_markers.items():
        _append_renaming_marker(src_id, dests)

    # Health status: 1–5 reverse -> health_poor
    if col_exists("health_status"):
        additions.append(
            _make_new_col(
                col_id="health_poor",
                question="Self-rated poor health (higher = poorer health)",
                transformations=[
                    {"type": "merged", "merging_type": "missing", "merged_from": ["health_status"]},
                    {"type": "encoding", "encoding": "$reverse_1_to_5"},
                ],
                section_sources=["health_status"],
            )
        )
        pending_new_ids.add("health_poor")

    if col_exists("birth") and col_exists("country"):
        additions.append(
            _make_new_col(
                col_id="born_in_residence_country",
                question="Born in country of residence (1=yes, 0=no)",
                transformations=[
                    {"type": "merged", "merging_type": "missing", "merged_from": ["birth", "country"]},
                    {"type": "encoding", "encoding": "$born_in_residence_country"},
                ],
                section_sources=["birth", "country"],
            )
        )
        pending_new_ids.add("born_in_residence_country")

    if col_exists("birth_m") and col_exists("birth_f") and col_exists("country"):
        additions.append(
            _make_new_col(
                col_id="parent_born_abroad",
                question="At least one parent born abroad (1=yes, 0=no)",
                transformations=[
                    {"type": "merged", "merging_type": "missing", "merged_from": ["birth_m", "birth_f", "country"]},
                    {"type": "encoding", "encoding": "$parent_born_abroad"},
                ],
                section_sources=["birth_m", "birth_f", "country"],
            )
        )
        pending_new_ids.add("parent_born_abroad")

    # Derived reversed single items (examples): disease-specific safety/comfort
    if col_exists("risk_vax_3"):
        additions.append(
            _make_new_col(
                col_id="measles_vaccine_safety",
                question="Measles vaccine safety (reverse of risk_vax_3)",
                transformations=[
                    {"type": "merged", "merging_type": "missing", "merged_from": ["risk_vax_3"]},
                    {"type": "encoding", "encoding": "$reverse_1_to_5"},
                ],
                section_sources=["risk_vax_3"],
            )
        )
        pending_new_ids.add("measles_vaccine_safety")
    if col_exists("emotion_vax_1"):
        additions.append(
            _make_new_col(
                col_id="covid_vaccine_comfort",
                question="COVID vaccine comfort (reverse of emotion_vax_1)",
                transformations=[
                    {"type": "merged", "merging_type": "missing", "merged_from": ["emotion_vax_1"]},
                    {"type": "encoding", "encoding": "$reverse_1_to_5"},
                ],
                section_sources=["emotion_vax_1"],
            )
        )
        pending_new_ids.add("covid_vaccine_comfort")

    # --------------
    # Composites
    # --------------
    def _source_exists_anywhere(source: str) -> bool:
        if col_exists(source):
            return True
        if source in pending_new_ids:
            return True
        if any(c.get("id") == source for c in output_spec.get("new_columns", [])):
            return True
        return False

    def add_mean(id_: str, question: str, sources: list[str]) -> None:
        def _prefer_analysis_source(source: str) -> str:
            aliased = alias_defs.get(source)
            return aliased if aliased else source

        preferred_sources = [_prefer_analysis_source(s) for s in sources]
        if not all(_source_exists_anywhere(s) for s in preferred_sources):
            return

        finalized_sources = [_finalized_source_id_output(s) for s in preferred_sources]
        additions.append(
            _make_new_col(
                col_id=id_,
                question=question,
                transformations=[
                    {"type": "merged", "merging_type": "mean", "merged_from": finalized_sources},
                ],
                section_sources=finalized_sources,
            )
        )
        pending_new_ids.add(id_)

    if col_exists("covax_num"):
        additions.append(
            _make_new_col(
                col_id="covax_0_1",
                question="Ever vaccinated (0/1 derived from covax_num)",
                transformations=[
                    {"type": "merged", "merging_type": "missing", "merged_from": ["covax_num"]},
                    {"type": "encoding", "encoding": "$covax_started_any"},
                ],
                section_sources=["covax_num"],
                question_group="covax_num",
            )
        )
        pending_new_ids.add("covax_0_1")

    # 5C pro-vax antecedents index (exclude calculation/fiveC_4)
    add_mean(
        "fivec_pro_vax_idx",
        "Pro-vaccine antecedents index (5C; excl. calculation)",
        ["fiveC_1", "fiveC_2", "fiveC_3", "fiveC_5"],
    )

    # Progressive attitudes composite
    add_mean(
        "progressive_attitudes_nonvax_idx",
        "Progressive attitudes (climate, LGBT, immigration tolerance)",
        ["opinion_2", "opinion_3", "att_immigration_tolerance"],
    )

    # Perceived social norms composites
    add_mean(
        "friends_progressive_norms_nonvax_idx",
        "Friends' progressive norms (climate, LGBT, immigration tolerance)",
        ["opinion_other_2_6", "opinion_other_3_6", "friends_immigration_tolerance"],
    )
    add_mean(
        "family_progressive_norms_nonvax_idx",
        "Family progressive norms (climate, LGBT, immigration tolerance)",
        ["opinion_other_2_family", "opinion_other_3_family", "family_immigration_tolerance"],
    )
    add_mean(
        "colleagues_progressive_norms_nonvax_idx",
        "Colleagues progressive norms (climate, LGBT, immigration tolerance)",
        [
            "opinion_other_2_colleagues",
            "opinion_other_3_colleagues",
            "colleagues_immigration_tolerance",
        ],
    )
    add_mean(
        "social_norms_vax_avg",
        "Perceived social norms: vaccines important (avg)",
        [
            "family_vaccines_importance",
            "friends_vaccines_importance",
            "colleagues_vaccines_importance",
        ],
    )

    # COVID risk & willingness (time-averaged) - removed as not informative

    # Current disease risk perceptions
    add_mean(
        "infection_likelihood_avg",
        "Perceived infection likelihood (avg across diseases)",
        ["risk_infection_1", "risk_infection_2", "risk_infection_3", "risk_infection_4"],
    )
    add_mean(
        "severity_if_infected_avg",
        "Perceived severity if infected (avg)",
        ["risk_illness_1", "risk_illness_2"],
    )

    # Vaccine risk/fear and their reverses
    add_mean(
        "vaccine_risk_avg",
        "Perceived vaccine risk (avg)",
        ["risk_vax_1", "risk_vax_2", "risk_vax_3", "risk_vax_4"],
    )

    # Reverse of vaccine risk (safety)
    additions.append(
        _make_new_col(
            col_id="vaccine_safety_avg",
            question="Perceived vaccine safety (reverse of vaccine_risk_avg)",
            transformations=[
                {"type": "merged", "merging_type": "missing", "merged_from": ["vaccine_risk_avg"]},
                {"type": "encoding", "encoding": "$reverse_1_to_5"},
            ],
            section_sources=["vaccine_risk_avg"],
        )
    )
    pending_new_ids.add("vaccine_safety_avg")
    add_mean(
        "vaccine_fear_avg",
        "Fear of vaccines (avg)",
        ["emotion_vax_1", "emotion_vax_2", "emotion_vax_3", "emotion_vax_4"],
    )
    additions.append(
        _make_new_col(
            col_id="vaccine_comfort_avg",
            question="Comfort with vaccines (reverse of vaccine_fear_avg)",
            transformations=[
                {"type": "merged", "merging_type": "missing", "merged_from": ["vaccine_fear_avg"]},
                {"type": "encoding", "encoding": "$reverse_1_to_5"},
            ],
            section_sources=["vaccine_fear_avg"],
        )
    )
    pending_new_ids.add("vaccine_comfort_avg")
    add_mean(
        "disease_fear_avg",
        "Fear of diseases (avg)",
        ["emotion_illness_1", "emotion_illness_2", "emotion_illness_3", "emotion_illness_4"],
    )

    # Media trust composites (information sources)
    add_mean(
        "trust_traditional",
        "Average trust in traditional media vaccine information",
        ["info_trust_vax_print", "info_trust_vax_television", "info_trust_vax_radio"],
    )
    add_mean(
        "trust_onlineinfo",
        "Average trust in online informational sources (news, Wikipedia, long-form video/podcasts)",
        ["info_trust_vax_online_news", "info_trust_vax_wikipedia", "info_trust_vax_youtube", "info_trust_vax_podcast"],
    )
    add_mean(
        "trust_social",
        "Average trust in social platforms for vaccine information",
        [
            "info_trust_vax_facebook",
            "info_trust_vax_instagram",
            "info_trust_vax_twitter",
            "info_trust_vax_tiktok",
            "info_trust_vax_telegram",
            "info_trust_vax_linkedin",
        ],
    )

    # Institutional trust & protective behaviours
    add_mean(
        "institutional_trust_avg",
        "Institutional trust in vaccine info (avg)",
        [
            "institut_trust_vax_1",
            "institut_trust_vax_2",
            "institut_trust_vax_3",
            "institut_trust_vax_4",
            "institut_trust_vax_5",
            "institut_trust_vax_6",
        ],
    )
    add_mean(
        "protective_behaviour_avg",
        "Protective behaviour (avg)",
        ["protective_behaviour_2", "protective_behaviour_3", "protective_behaviour_4"],
    )

    # MFQ per-foundation means (relevance)
    add_mean("mfq_relevance_care_idx", "MFQ relevance: Care/Harm", ["moral1_1", "moral1_7"])
    add_mean("mfq_relevance_fairness_idx", "MFQ relevance: Fairness/Cheating", ["moral1_2", "moral1_8"])
    add_mean("mfq_relevance_loyalty_idx", "MFQ relevance: Loyalty/Betrayal", ["moral1_3", "moral1_9"])
    add_mean("mfq_relevance_authority_idx", "MFQ relevance: Authority/Subversion", ["moral1_4", "moral1_10"])
    add_mean("mfq_relevance_purity_idx", "MFQ relevance: Purity/Degradation", ["moral1_5", "moral1_11"])
    # MFQ per-foundation means (agreement)
    add_mean("mfq_agreement_care_idx", "MFQ agreement: Care/Harm", ["moral2_1", "moral2_7"])
    add_mean("mfq_agreement_fairness_idx", "MFQ agreement: Fairness/Cheating", ["moral2_2", "moral2_8"])
    add_mean("mfq_agreement_loyalty_idx", "MFQ agreement: Loyalty/Betrayal", ["moral2_3", "moral2_9"])
    add_mean("mfq_agreement_authority_idx", "MFQ agreement: Authority/Subversion", ["moral2_4", "moral2_10"])
    add_mean("mfq_agreement_purity_idx", "MFQ agreement: Purity/Degradation", ["moral2_5", "moral2_11"])
    # MFQ higher-order (mean-based)
    add_mean(
        "mfq_relevance_individualizing_idx",
        "MFQ relevance: Individualizing (Care, Fairness)",
        ["mfq_relevance_care_idx", "mfq_relevance_fairness_idx"],
    )
    add_mean(
        "mfq_relevance_binding_idx",
        "MFQ relevance: Binding (Loyalty, Authority, Purity)",
        ["mfq_relevance_loyalty_idx", "mfq_relevance_authority_idx", "mfq_relevance_purity_idx"],
    )
    add_mean(
        "mfq_agreement_individualizing_idx",
        "MFQ agreement: Individualizing (Care, Fairness)",
        ["mfq_agreement_care_idx", "mfq_agreement_fairness_idx"],
    )
    add_mean(
        "mfq_agreement_binding_idx",
        "MFQ agreement: Binding (Loyalty, Authority, Purity)",
        ["mfq_agreement_loyalty_idx", "mfq_agreement_authority_idx", "mfq_agreement_purity_idx"],
    )
    add_mean(
        "moral_progressive_orientation",
        "Moral + progressive orientation (avg)",
        [
            "progressive_attitudes_nonvax_idx",
            "mfq_relevance_individualizing_idx",
            "mfq_agreement_individualizing_idx",
        ],
    )

    # Append all additions
    output_spec["new_columns"].extend(additions)

    # --------------
    # Add date deltas from 2019-01-01 for covid and covax month/year pairs
    # --------------
    def _add_date_delta(prefix: str, indices: range) -> None:
        for i in indices:
            m_id = f"{prefix}{i}_1"  # month
            y_id = f"{prefix}{i}_2"  # year
            if col_exists(m_id) and col_exists(y_id):
                new_id = f"{prefix}{i}_days_from_2019_01_01"
                output_spec["new_columns"].append(
                    _make_new_col(
                        col_id=new_id,
                        question=f"Days since {BASELINE_DATE} for {prefix}{i} (mid-month=15)",
                        transformations=[
                            {
                                "type": "merged",
                                "merging_type": "concat",
                                "merged_from": [y_id, m_id],
                                "concat_string": f"{{{y_id}}}-{{{m_id}}}-15",
                            },
                            {"type": "encoding", "encoding": "$DATETIME", "datetime_format": "%Y-%m-%d"},
                            {
                                "type": "encoding",
                                "encoding": "$DATETIME_TO_INTERVAL",
                                "from_timestamp": BASELINE_DATE,
                                "datetime_format": "%Y-%m-%d",
                                "unit": "days",
                            },
                        ],
                        section_sources=[m_id, y_id],
                    )
                )

    _add_date_delta("covid_date_", range(1, 11))
    _add_date_delta("covax_date_", range(1, 5))

    # --------------
    # Drop raw month/year fields once we materialize day-based deltas
    # --------------
    def _find_output_column(col_id: str | None) -> dict[str, Any] | None:
        if not col_id:
            return None
        for bucket in ("old_columns", "new_columns"):
            for col in output_spec.get(bucket, []) or []:
                if col.get("id") == col_id:
                    return col
        return None

    def _append_redundant_removal(col_id: str, redundant_col: str) -> None:
        target = _find_output_column(col_id)
        redundant_target = _find_output_column(redundant_col)
        if target is None or redundant_target is None:
            return

        trans = target.setdefault("transformations", [])
        already_present = any(
            t.get("type") == "removal"
            and t.get("removal_reason") == "redundant"
            and t.get("removal_redundant_col") == redundant_col
            for t in trans
        )
        if not already_present:
            trans.append(
                {
                    "type": "removal",
                    "removal_reason": "redundant",
                    "removal_redundant_col": redundant_col,
                }
            )

    def _remove_month_year_pair(prefix: str, indices: range) -> None:
        for idx in indices:
            day_col = f"{prefix}{idx}_days_from_2019_01_01"
            if not any(col.get("id") == day_col for col in output_spec.get("new_columns", [])):
                continue
            for suffix in ("_1", "_2"):
                _append_redundant_removal(f"{prefix}{idx}{suffix}", day_col)

    # First COVID infection and first two vaccine dose components already carry removal steps.
    _remove_month_year_pair("covid_date_", range(2, 11))
    _remove_month_year_pair("covax_date_", range(3, 5))

    # Cleanup dictionary: remove raw/redundant indicators once their canonical reverse-coded versions exist.
    for redundant_col, canonical in REDUNDANT_CANONICAL_COLUMNS.items():
        _append_redundant_removal(redundant_col, canonical)

    # Keep only the analysis-friendly health variable.
    _append_redundant_removal("health_status", "health_poor")

    # --------------
    # Multi-select explosion for covax reasons and child reasons
    # --------------
    def _get_answers(col_id: str) -> list[str]:
        src = gen_old_idx.get(col_id) or gen_new_idx.get(col_id)
        answer = src.get("answer") if src else None
        return answer if isinstance(answer, list) else []

    def _slug(s: str) -> str:
        import re

        s = s.lower()
        s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
        return s[:80]  # keep IDs short-ish

    def _mark_split_source_removed(source_id: str) -> None:
        target = next((c for c in output_spec.get("old_columns", []) if c.get("id") == source_id), None)
        if target is None:
            target = next((c for c in output_spec.get("new_columns", []) if c.get("id") == source_id), None)
        if target is None:
            return
        transforms = target.setdefault("transformations", [])
        if any(t.get("type") == "removal" for t in transforms):
            return
        transforms.append({"type": "removal", "removal_reason": "dropped from clean spec"})

    def _add_option_splits(
        source_id: str,
        options: list[str],
        question_template: str,
        *,
        drop_source: bool = True,
    ) -> None:
        if not options or not col_exists(source_id):
            return
        seen: set[str] = set()
        for opt in options:
            if not opt or opt in seen:
                continue
            seen.add(opt)
            col_bool_id = f"{source_id}__{_slug(opt)}"
            output_spec["new_columns"].append(
                _make_new_col(
                    col_id=col_bool_id,
                    question=question_template.format(option=opt),
                    transformations=[
                        {"type": "splitted", "split_from": source_id, "values": [opt]},
                        {"type": "encoding", "encoding": {"": 0, opt: 1}},
                    ],
                    section_sources=[source_id],
                )
            )
        if drop_source and seen:
            _mark_split_source_removed(source_id)

    # Build a single concatenated text column for vaccination reasons across sources
    covax_reason_sources = [
        "covax_yes",
        "covax_yes_11_TEXT",
        "covax_yesno",
        "covax_yesno_6_TEXT",
        "covax_yesno_7_TEXT",
        "covax_yesno_13_TEXT",
        "covax_no",
        "covax_no_1_TEXT",
        "covax_no_7_TEXT",
        "covax_no_8_TEXT",
        "covax_no_16_TEXT",
    ]
    covax_reason_sources = [c for c in covax_reason_sources if col_exists(c) and not _should_force_remove(c)]
    if covax_reason_sources:
        output_spec["new_columns"].append(
            _make_new_col(
                col_id="covax_reason_text",
                question="All vaccination-related reasons (concat)",
                transformations=[
                    {
                        "type": "merged",
                        "merging_type": "concat",
                        "merged_from": covax_reason_sources,
                        "concat_string": "|".join([f"{{{c}}}" for c in covax_reason_sources]),
                    }
                ],
                section_sources=covax_reason_sources,
            )
        )

        # Union of answer options from main reason questions
        covax_reason_options: list[str] = []
        for src_id in ["covax_yes", "covax_yesno", "covax_no"]:
            covax_reason_options.extend(_get_answers(src_id))
        # Deduplicate, keep order
        seen = set()
        covax_reason_options = [x for x in covax_reason_options if not (x in seen or seen.add(x))]
        # Create one boolean column per option
        for opt in covax_reason_options:
            if not opt:
                continue
            col_bool_id = f"covax_reason__{_slug(opt)}"
            output_spec["new_columns"].append(
                _make_new_col(
                    col_id=col_bool_id,
                    question=f"Reason present: {opt}",
                    transformations=[
                        {"type": "splitted", "split_from": "covax_reason_text", "values": [opt]},
                        {"type": "encoding", "encoding": {"": 0, opt: 1}},
                    ],
                    section_sources=["covax_reason_text"],
                )
            )
        _mark_split_source_removed("covax_reason_text")

        # Legitimacy / scepticism composite (COVID vaccination reasons)
        # Must run *after* covax_reason__* split columns have been created.
        covax_legitimacy_sources = [
            "covax_reason__i_believe_that_the_vaccine_is_not_safe",
            "covax_reason__i_do_not_believe_that_the_vaccine_has_been_sufficiently_tested",
            "covax_reason__i_worry_about_the_vaccine_s_side_effects",
            "covax_reason__i_feel_that_the_vaccine_is_a_waste_of_money_for_the_benefit_of_the_big_pharmaceu",
            "covax_reason__i_am_opposed_to_mandatory_vaccination",
            "covax_reason__i_think_developing_natural_immunity_is_better_than_getting_vaccinated",
        ]

        existing_output_ids = {c.get("id") for c in output_spec.get("new_columns", [])}
        if "covax_legitimacy_scepticism_idx" not in existing_output_ids:
            output_spec["new_columns"].append(
                _make_new_col(
                    col_id="covax_legitimacy_scepticism_idx",
                    question=(
                        "COVID vaccination legitimacy / scepticism index "
                        "(reasons: safety, testing, side effects, pharma, mandates, natural immunity)"
                    ),
                    transformations=[
                        {
                            "type": "merged",
                            "merging_type": "mean",
                            "merged_from": covax_legitimacy_sources,
                        },
                    ],
                    section_sources=covax_legitimacy_sources,
                )
            )
            pending_new_ids.add("covax_legitimacy_scepticism_idx")

    # Child reasons (explode per option directly from the column)
    _add_option_splits(
        "covax_yesno_child",
        _get_answers("covax_yesno_child"),
        "Child dose-gap reason present: {option}",
    )
    _add_option_splits(
        "covax_no_child",
        _get_answers("covax_no_child"),
        "Child reason present: {option}",
    )
    _add_option_splits(
        "knowledge_illness",
        _get_answers("knowledge_illness"),
        "Believes illness prevented by vaccines: {option}",
    )
    _add_option_splits(
        "knowledge_vax",
        _get_answers("knowledge_vax"),
        "Believes vaccine statement is true: {option}",
    )
    _add_option_splits(
        "vax_no_child",
        VAX_NO_CHILD_OPTIONS,
        "Childhood vaccine reason present: {option}",
    )
    _add_option_splits(
        "info_general",
        _get_answers("info_general"),
        "Information channel primarily used: {option}",
        drop_source=False,
    )

    # --------------
    # Ensure risk_1_4 and risk_2_4 are present as ordinal-categorical derived columns
    # --------------
    # NOTE: we intentionally do not create a 'risk_1_4_cat' derived column anymore.
    # The 'risk_1_4' source is available and an analysis-friendly alias exists
    # (`covid_perceived_danger_pre`), so we avoid adding a separate categorical
    # derived column to keep the clean spec compact and avoid duplicate fields.
    # NOTE: we intentionally do not create a 'risk_2_4_cat' derived column anymore.
    # The 'risk_2_4' source is available and an analysis-friendly alias exists
    # (`vax_willingness_pre`), so we avoid adding a separate categorical derived
    # column to keep the clean spec compact and avoid duplicate fields.

    # Drop MFQ controls (moral1_6, moral2_6)
    for col in output_spec.get("old_columns", []):
        if col.get("id") in {"moral1_6", "moral2_6"}:
            col.setdefault("transformations", []).append({"type": "removal", "removal_reason": "redundant"})

    # Ensure every column entry carries a section assignment when possible
    _ensure_column_sections()

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_spec, f, ensure_ascii=False, indent=2)
        f.write("\n")

    sections_written = _write_section_specs(output_spec, stage)

    print(
        f"✅ Clean spec ({stage}) written to {output_path} | old={len(clean_old)}, new={len(clean_new)}, "
        f"encodings={len(merged_encodings)}, sections={sections_written} -> {SECTION_SPEC_DIR / stage}"
    )

    return output_spec


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build clean analysis spec")
    parser.add_argument(
        "--stage",
        choices=["raw", "post-general"],
        default="raw",
        help="Apply spec to raw data or to general-processed data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output path for the generated spec",
    )
    args = parser.parse_args()

    if args.output is None:
        output = DEFAULT_OUTPUT_RAW if args.stage == "raw" else DEFAULT_OUTPUT_POST_GENERAL
    else:
        output = args.output

    build_clean_analysis_spec(output_path=output, stage=args.stage)
