"""Shared analysis taxonomy and default paths."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_COLUMN_TYPES_PATH = Path(os.getenv("ANALYSIS_COLUMN_TYPES_PATH", "empirical/specs/column_types.tsv"))
DEFAULT_DERIVATIONS_PATH = Path(os.getenv("ANALYSIS_DERIVATIONS_PATH", "preprocess/specs/derivations.tsv"))

BLOCKS: tuple[str, ...] = ("structural", "experience", "psychological", "information", "social")

SECTION_TO_BLOCK: dict[str, str] = {
    "metadata": "structural",
    "quotas": "structural",
    "socio-demo-econ-health": "structural",
    "covid_experience": "experience",
    "other_illness": "experience",
    "group": "social",
    "accessibility": "structural",
    "info_source": "information",
    "opinion": "psychological",
    "moral": "psychological",
}

OUTCOME_PREFIXES: tuple[str, ...] = ("vax_willingness_",)

OUTCOME_EXACT: frozenset[str] = frozenset(
    {
        "support_mandatory_vaccination",
        "protective_behaviour_avg",
        "covax_0_1",
        "flu_vaccinated_2023_2024",
        "mask_when_symptomatic_crowded",
        "stay_home_when_symptomatic",
        "mask_when_pressure_high",
    }
)
