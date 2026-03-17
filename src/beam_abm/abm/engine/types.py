"""State and typing contracts for the ABM engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl
from pydantic import BaseModel

from beam_abm.abm.network import ContactDiagnostics


@dataclass(slots=True)
class TimingBreakdown:
    """Wall-clock timing breakdown for a run."""

    choice_total: float = 0.0
    update_total: float = 0.0
    social_total: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "choice_total": self.choice_total,
            "update_total": self.update_total,
            "social_total": self.social_total,
            "total": self.total,
        }

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict()


class ModelStateSnapshot(BaseModel):
    """Typed continuation snapshot payload for warm-start restoration."""

    model_config: ClassVar[dict] = {  # type: ignore[misc]
        "arbitrary_types_allowed": True,
        "extra": "forbid",
        "frozen": False,
        "validate_default": True,
    }

    agents_df: pl.DataFrame
    population_t0: pl.DataFrame
    outcome_t0_means: dict[str, float]
    current_tick: int
    community_incidence: dict[str, float]
    prev_community_incidence: dict[str, float]
    importation_peak_reference: float | None = None
    importation_peak_reference_by_community: dict[str, float] | None = None
    target_mask: np.ndarray | None
    seed_mask: np.ndarray | None
    graph_distance_from_seed: np.ndarray | None
    latest_outcomes: dict[str, np.ndarray] | None
    latest_social_similarity_mean: float
    latest_contact_diagnostics: ContactDiagnostics | None
    random_state: dict[str, object] | None
