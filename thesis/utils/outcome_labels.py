"""Shared outcome labels and ordering for thesis chapters."""

from __future__ import annotations

from pathlib import Path

from utils.qmd_utils import load_labels

CORE_OUTCOME_ORDER: tuple[str, ...] = (
    "vax_willingness_Y",
    "flu_vaccinated_2023_2024",
    "mask_when_symptomatic_crowded",
    "stay_home_when_symptomatic",
    "mask_when_pressure_high",
)

EXTENDED_OUTCOME_ORDER: tuple[str, ...] = (
    "vax_willingness_T12",
    "vax_willingness_Y",
    "flu_vaccinated_2023_2024",
    "flu_vaccinated_2023_2024_no_habit",
    "mask_when_symptomatic_crowded",
    "stay_home_when_symptomatic",
    "mask_when_pressure_high",
)

_DISPLAY_LABELS_PATH = Path(__file__).resolve().parents[1] / "specs" / "display_labels.json"


def _select_labels(*, variant: str) -> dict[str, str]:
    all_labels = load_labels(path=_DISPLAY_LABELS_PATH, variant=variant)
    return {key: all_labels[key] for key in EXTENDED_OUTCOME_ORDER}


OUTCOME_LABELS: dict[str, str] = _select_labels(variant="long")
OUTCOME_SHORT_LABELS: dict[str, str] = _select_labels(variant="short")


def label_outcome(outcome: str) -> str:
    raw = str(outcome)
    return OUTCOME_LABELS[raw]


def label_outcome_short(outcome: str) -> str:
    raw = str(outcome)
    return OUTCOME_SHORT_LABELS[raw]


__all__ = [
    "CORE_OUTCOME_ORDER",
    "EXTENDED_OUTCOME_ORDER",
    "OUTCOME_LABELS",
    "OUTCOME_SHORT_LABELS",
    "label_outcome",
    "label_outcome_short",
]
