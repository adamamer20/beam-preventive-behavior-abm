from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CalibrationStatus = Literal["uncalibrated", "calibrated"]


@dataclass(frozen=True, slots=True)
class CanonicalizeStats:
    runs_seen: int
    rows_in: int
    rows_out: int
    samples_in: int
    samples_out: int
