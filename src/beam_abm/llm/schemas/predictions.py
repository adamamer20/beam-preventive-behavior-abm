"""Structured output schemas for micro-validation.

These schemas are intentionally simple and focused on enforcing the *declared outcome scale*
(binary/ordinal/continuous) so that downstream alignment metrics can operate on a consistent
scalar prediction.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContinuousPrediction(BaseModel):
    """Continuous outcome on the original scale."""

    prediction: float = Field(..., description="Point prediction on the declared continuous outcome scale")
    rationale: str | None = Field(default=None, description="Optional short rationale")


class BinaryPrediction(BaseModel):
    """Binary outcome represented by probability of Y=1."""

    p_yes: float = Field(..., ge=0.0, le=1.0, description="P(Y=1) on the declared binary outcome")
    rationale: str | None = Field(default=None, description="Optional short rationale")


class OrdinalPrediction(BaseModel):
    """Ordinal outcome represented by a categorical distribution.

    `levels` can be strings (e.g., ['strongly_disagree', ...]) or numeric-like.
    The expected value is computed downstream if levels are numeric-like; otherwise we map
    to equally spaced ranks 0..K-1.
    """

    levels: list[str] = Field(..., min_length=2)
    probs: list[float] = Field(..., min_length=2, description="Probabilities aligned with levels")
    rationale: str | None = Field(default=None, description="Optional short rationale")

    @model_validator(mode="after")
    def _validate_probs(self) -> OrdinalPrediction:
        if len(self.levels) != len(self.probs):
            raise ValueError("levels and probs must have same length")

        probs = [float(p) for p in self.probs]
        if any((p < 0.0) or (p > 1.0) for p in probs):
            raise ValueError("probs must be in [0, 1]")

        s = float(sum(probs))
        if not (s > 0.0):
            raise ValueError("sum(probs) must be > 0")

        # Enforce sum(probs)=1 with tolerance; renormalize minor drift.
        if abs(s - 1.0) > 1e-2:
            raise ValueError("sum(probs) must be approximately 1")
        if abs(s - 1.0) > 1e-6:
            self.probs = [p / s for p in probs]
        else:
            self.probs = probs

        return self


def ordinal_expected_value_and_mapping(
    *,
    levels: list[object],
    probs: list[object],
) -> tuple[float | None, dict[str, object] | None]:
    """Compute ordinal expected value and expose mapping choice.

    Mapping rules:
    - If levels are numeric-like, use their float values.
    - Otherwise map to ranks 0..K-1 in the provided level order.
    """

    if len(levels) < 2 or len(levels) != len(probs):
        return None, None

    try:
        p = [float(x) for x in probs]
    except (TypeError, ValueError):
        return None, None

    numeric_levels: list[float] | None = []
    for lv in levels:
        try:
            numeric_levels.append(float(lv))
        except (TypeError, ValueError):
            numeric_levels = None
            break

    if numeric_levels is None:
        numeric_levels = list(range(len(levels)))
        mapping: dict[str, object] = {
            "ordinal_level_mapping": "rank_index",
            "ordinal_levels": [str(x) for x in levels],
            "ordinal_numeric_levels": numeric_levels,
        }
    else:
        mapping = {
            "ordinal_level_mapping": "numeric",
            "ordinal_levels": [str(x) for x in levels],
            "ordinal_numeric_levels": numeric_levels,
        }

    z = float(sum(pi * xi for pi, xi in zip(p, numeric_levels, strict=False)))
    return z, mapping


class PairedContinuousPrediction(BaseModel):
    """Paired-profile output with low profile as A and high profile as B."""

    A: ContinuousPrediction
    B: ContinuousPrediction


class PairedBinaryPrediction(BaseModel):
    """Paired-profile output with low profile as A and high profile as B."""

    A: BinaryPrediction
    B: BinaryPrediction


class PairedOrdinalPrediction(BaseModel):
    """Paired-profile output with low profile as A and high profile as B."""

    A: OrdinalPrediction
    B: OrdinalPrediction


class BeliefUpdatePrediction(BaseModel):
    """Belief-update output schema for Phase 0 prompts.

    Fields are optional so the model can omit variables that are expected to remain unchanged.
    """

    model_config = ConfigDict(extra="ignore")

    institutional_trust_avg: float | None = Field(default=None)
    covax_legitimacy_scepticism_idx: float | None = Field(default=None)
    vaccine_risk_avg: float | None = Field(default=None)
    covid_perceived_danger_T12: float | None = Field(default=None)
    social_norms_vax_avg: float | None = Field(default=None)
    fivec_low_constraints: float | None = Field(default=None)
    group_trust_vax: float | None = Field(default=None)


class BeliefUpdatePairedPrediction(BaseModel):
    """Belief-update paired-profile output schema."""

    model_config = ConfigDict(extra="ignore")

    A: BeliefUpdatePrediction
    B: BeliefUpdatePrediction


def _coerce_scalar(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def extract_scalar_prediction(parsed: dict[str, Any] | float | int | None) -> float | None:
    """Extract a scalar prediction from a structured LLM response.

    Supports the built-in schemas above and common legacy keys.
    """

    if isinstance(parsed, int | float):
        return float(parsed)
    if not isinstance(parsed, dict):
        return None

    if len(parsed) == 1:
        lone_val = next(iter(parsed.values()))
        scalar = _coerce_scalar(lone_val)
        if scalar is not None:
            return scalar

    if "prediction" in parsed:
        try:
            return float(parsed["prediction"])
        except (TypeError, ValueError):
            return None

    if "expected" in parsed:
        try:
            return float(parsed["expected"])
        except (TypeError, ValueError):
            return None

    if "p_yes" in parsed:
        try:
            return float(parsed["p_yes"])
        except (TypeError, ValueError):
            return None

    if "probs" in parsed and "levels" in parsed:
        probs = parsed.get("probs")
        levels = parsed.get("levels")
        if not isinstance(probs, list) or not isinstance(levels, list) or len(probs) != len(levels) or len(probs) < 2:
            return None

        z, _ = ordinal_expected_value_and_mapping(levels=levels, probs=probs)
        return z

    return None


def extract_paired_scalar_predictions(
    parsed: dict[str, Any] | float | int | None,
) -> tuple[float | None, float | None] | None:
    """Extract (A, B) scalar predictions from a paired-profile response."""

    if not isinstance(parsed, dict):
        return None
    a = parsed.get("A")
    b = parsed.get("B")

    # Common shortcut: LLM outputs just numbers for each profile.
    if isinstance(a, int | float) and isinstance(b, int | float):
        return float(a), float(b)

    if not isinstance(a, dict) or not isinstance(b, dict):
        return None

    return extract_scalar_prediction(a), extract_scalar_prediction(b)


OutcomeScale = Literal["continuous", "binary", "ordinal"]


def extract_scalar_prediction_with_metadata(
    parsed: dict[str, Any] | float | int | None,
) -> tuple[float | None, dict[str, object] | None]:
    """Like extract_scalar_prediction, but also returns metadata for ordinal mappings."""

    if isinstance(parsed, int | float):
        return float(parsed), None
    if not isinstance(parsed, dict):
        return None, None

    if len(parsed) == 1:
        lone_val = next(iter(parsed.values()))
        scalar = _coerce_scalar(lone_val)
        if scalar is not None:
            return scalar, None

    if (
        "probs" in parsed
        and "levels" in parsed
        and isinstance(parsed.get("probs"), list)
        and isinstance(parsed.get("levels"), list)
    ):
        z, meta = ordinal_expected_value_and_mapping(levels=parsed["levels"], probs=parsed["probs"])
        return z, meta

    return extract_scalar_prediction(parsed), None
