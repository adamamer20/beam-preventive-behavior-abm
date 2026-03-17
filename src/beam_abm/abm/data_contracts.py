"""Data contracts for ABM input artefacts.

Strict typed contracts for every external input the ABM consumes.
Rule: fail fast if any required predictor for a choice rule is missing.

Modules
-------
BackboneModel
    Coefficients + thresholds loaded from ``empirical/output/modeling``.
AnchorSystem
    Country prevalences, neighbourhoods, and PCA metadata from
    ``empirical/output/anchors/system``.
LeversConfig
    Per-outcome lever/include column mapping from ``config/levers.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from loguru import logger

from beam_abm.abm.state_schema import ABM_ALLOWED_OUTCOMES

__all__ = [
    "BackboneModel",
    "AnchorMetadata",
    "AnchorSystem",
    "LeversConfig",
    "OutcomeLever",
    "get_abm_levers_config",
    "get_abm_outcome_specs",
    "log_non_abm_outcomes",
    "load_levers_config",
    "load_backbone_model",
    "load_all_backbone_models",
    "load_anchor_system",
]


# =========================================================================
# 1.  Levers config  (config/levers.json)
# =========================================================================


@dataclass(frozen=True, slots=True)
class OutcomeLever:
    """Specification for a single ABM target outcome.

    Parameters
    ----------
    outcome : str
        Target variable name.
    model_ref : str
        Reference model directory name under ``modeling_dir``.
    levers : tuple[str, ...]
        Mutable predictor columns the ABM may perturb.
    include_cols : tuple[str, ...]
        Additional immutable columns needed by the model (e.g. country dummies).
    """

    outcome: str
    model_ref: str
    levers: tuple[str, ...]
    include_cols: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LeversConfig:
    """Complete lever mapping for all ABM target outcomes.

    Parameters
    ----------
    outcomes : dict[str, OutcomeLever]
        Mapping from outcome name to its lever specification.
    """

    outcomes: dict[str, OutcomeLever]

    def get(self, outcome: str) -> OutcomeLever:
        """Retrieve the lever spec for *outcome*, or fail.

        Parameters
        ----------
        outcome : str
            Target variable name.

        Returns
        -------
        OutcomeLever

        Raises
        ------
        KeyError
            If *outcome* is not in the config.
        """
        if outcome not in self.outcomes:
            msg = f"Outcome '{outcome}' not found in levers config. Available: {sorted(self.outcomes)}"
            raise KeyError(msg)
        return self.outcomes[outcome]


def load_levers_config(path: Path) -> LeversConfig:
    """Load and validate ``levers.json``.

    Parameters
    ----------
    path : Path
        Path to the JSON file.

    Returns
    -------
    LeversConfig

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If a required key is missing.
    """
    if not path.is_file():
        msg = f"Levers config not found: {path}"
        raise FileNotFoundError(msg)

    raw: dict = json.loads(path.read_text(encoding="utf-8"))
    outcomes: dict[str, OutcomeLever] = {}
    for outcome_name, spec in raw.items():
        for required_key in ("model_ref", "levers"):
            if required_key not in spec:
                msg = f"Levers config for '{outcome_name}' missing required key '{required_key}'."
                raise ValueError(msg)
        outcomes[outcome_name] = OutcomeLever(
            outcome=outcome_name,
            model_ref=spec["model_ref"],
            levers=tuple(spec["levers"]),
            include_cols=tuple(spec.get("include_cols", ())),
        )
    logger.info("Loaded levers config: {n} outcomes.", n=len(outcomes))
    return LeversConfig(outcomes=outcomes)


def get_abm_outcome_specs(levers_cfg: LeversConfig) -> dict[str, OutcomeLever]:
    """Return ABM-supported outcome specs from a levers config."""
    return {outcome: spec for outcome, spec in levers_cfg.outcomes.items() if outcome in ABM_ALLOWED_OUTCOMES}


def get_abm_levers_config(levers_cfg: LeversConfig) -> LeversConfig:
    """Build a levers config restricted to ABM-supported outcomes."""
    return LeversConfig(outcomes=get_abm_outcome_specs(levers_cfg))


def log_non_abm_outcomes(levers_cfg: LeversConfig, *, context: str) -> None:
    """Log outcomes present in levers config but excluded from ABM runtime."""
    allowed = set(ABM_ALLOWED_OUTCOMES)
    ignored = sorted(outcome for outcome in levers_cfg.outcomes if outcome not in allowed)
    logger.info(
        "{context}: n_total_lever_outcomes={n_total}, n_allowed_abm_outcomes={n_allowed}, "
        "ignored_non_abm_outcomes={ignored}",
        context=context,
        n_total=len(levers_cfg.outcomes),
        n_allowed=len(levers_cfg.outcomes) - len(ignored),
        ignored=ignored,
    )


# =========================================================================
# 2.  Backbone model  (coefficients.csv per model directory)
# =========================================================================


@dataclass(frozen=True, slots=True)
class BackboneModel:
    """Fitted structural backbone for a single behavioural outcome.

    Parameters
    ----------
    name : str
        Model reference name (directory basename).
    model_type : {"binary", "ordinal"}
        Logistic model family.
    terms : tuple[str, ...]
        Ordered predictor names (excluding thresholds).
    coefficients : np.ndarray
        1-D array of log-odds coefficients aligned with *terms*.
    thresholds : np.ndarray | None
        Ordered cut-points for ordinal models; ``None`` for binary.
    intercept : float | None
        Intercept for binary models; ``None`` for ordinal.
    blocks : dict[str, list[str]]
        Mapping from block name to the terms belonging to it.
    term_means : np.ndarray | None
        Per-term centring values used at fit time.
    term_stds : np.ndarray | None
        Per-term scaling values used at fit time.
    """

    name: str
    model_type: Literal["binary", "ordinal"]
    terms: tuple[str, ...]
    coefficients: np.ndarray
    thresholds: np.ndarray | None
    intercept: float | None
    blocks: dict[str, list[str]]
    term_means: np.ndarray | None = None
    term_stds: np.ndarray | None = None

    @property
    def n_categories(self) -> int:
        """Number of response categories for ordinal models.

        Returns
        -------
        int
            ``len(thresholds) + 1`` for ordinal; ``2`` for binary.
        """
        if self.thresholds is not None:
            return len(self.thresholds) + 1
        return 2

    def validate_predictors(self, available: frozenset[str]) -> None:
        """Check that every required term is in *available*.

        Parameters
        ----------
        available : frozenset[str]
            Column names present in the agent DataFrame.

        Raises
        ------
        ValueError
            Lists all missing predictors.
        """
        missing = set(self.terms) - available
        if missing:
            msg = f"Model '{self.name}' requires predictors not available in agent state: {sorted(missing)}"
            raise ValueError(msg)


def load_backbone_model(
    modeling_dir: Path,
    model_ref: str,
) -> BackboneModel:
    """Load a single fitted model from its ``coefficients.csv``.

    Parameters
    ----------
    modeling_dir : Path
        Root modelling output directory.
    model_ref : str
        Model reference name (subdirectory under a model group).

    Returns
    -------
    BackboneModel

    Raises
    ------
    FileNotFoundError
        If the expected coefficients file does not exist.
    ValueError
        If the CSV has unexpected structure.
    """
    # Model directories are nested: group / model_ref / coefficients.csv
    # The group prefix is the first two underscore-joined tokens.
    # E.g.  B_COVID_MAIN_... -> B_COVID/B_COVID_MAIN_.../coefficients.csv
    coeff_path = _find_coefficients_csv(modeling_dir, model_ref)

    df = pl.read_csv(coeff_path)
    required_cols = {"term", "estimate"}
    if not required_cols.issubset(set(df.columns)):
        msg = f"coefficients.csv for '{model_ref}' missing columns: {required_cols - set(df.columns)}"
        raise ValueError(msg)

    # Separate thresholds from regular terms
    threshold_mask = df["term"].str.contains("/")
    thresholds_df = df.filter(threshold_mask)
    intercept_mask = df["term"] == "Intercept"
    intercept_df = df.filter(intercept_mask)
    predictors_df = df.filter(~threshold_mask & ~intercept_mask)

    # Determine model type
    if thresholds_df.height > 0:
        model_type: Literal["binary", "ordinal"] = "ordinal"
        # Parse thresholds: "1.0/2.0" -> numeric cutpoints
        threshold_vals = _parse_thresholds(thresholds_df)
        intercept_val = None
    elif intercept_df.height > 0:
        model_type = "binary"
        threshold_vals = None
        intercept_val = float(intercept_df["estimate"][0])
    else:
        msg = f"Model '{model_ref}' has neither thresholds nor an intercept row."
        raise ValueError(msg)

    terms = tuple(predictors_df["term"].to_list())
    coefficients = predictors_df["estimate"].to_numpy().astype(np.float64)

    # Build block mapping if 'block' column exists
    blocks: dict[str, list[str]] = {}
    if "block" in predictors_df.columns:
        for row in predictors_df.iter_rows(named=True):
            block_name = row.get("block", "")
            if block_name:
                blocks.setdefault(block_name, []).append(row["term"])

    logger.info(
        "Loaded {model_type} model '{name}': {n_terms} terms{thresholds}.",
        model_type=model_type,
        name=model_ref,
        n_terms=len(terms),
        thresholds=(f", {len(threshold_vals)} thresholds" if threshold_vals is not None else ""),
    )
    return BackboneModel(
        name=model_ref,
        model_type=model_type,
        terms=terms,
        coefficients=coefficients,
        thresholds=threshold_vals,
        intercept=intercept_val,
        blocks=blocks,
    )


def _find_coefficients_csv(modeling_dir: Path, model_ref: str) -> Path:
    """Locate ``coefficients.csv`` for *model_ref*.

    Parameters
    ----------
    modeling_dir : Path
        Root modelling directory.
    model_ref : str
        Model reference name.

    Returns
    -------
    Path
        Absolute path to the CSV.

    Raises
    ------
    FileNotFoundError
        If the file cannot be found.
    """
    # Strategy: iterate group directories and look for model_ref subdir
    for group_dir in sorted(modeling_dir.iterdir()):
        if not group_dir.is_dir():
            continue
        candidate = group_dir / model_ref / "coefficients.csv"
        if candidate.is_file():
            return candidate

    msg = f"No coefficients.csv found for model_ref='{model_ref}' under {modeling_dir}"
    raise FileNotFoundError(msg)


def _parse_thresholds(df: pl.DataFrame) -> np.ndarray:
    """Extract ordered cumulative-logit thresholds.

    Parameters
    ----------
    df : pl.DataFrame
        Filtered to threshold rows only.

    Returns
    -------
    np.ndarray
        1-D array of cumulative threshold values, ordered by
        the lower boundary of each cut-point.
    """
    # OrderedModel exports the first cutpoint directly and subsequent
    # cutpoints as log-increments; reconstruct cumulative thresholds.
    lower_bounds = df["term"].str.split("/").list.first().cast(pl.Float64)
    sorted_idx = lower_bounds.arg_sort()
    raw = df["estimate"].gather(sorted_idx).to_numpy().astype(np.float64)
    if raw.size == 0:
        return raw
    thresholds = np.empty_like(raw)
    thresholds[0] = raw[0]
    if raw.size > 1:
        thresholds[1:] = raw[0] + np.cumsum(np.exp(raw[1:]))
    return thresholds


def load_all_backbone_models(
    modeling_dir: Path,
    levers_cfg: LeversConfig,
    survey_df: pl.DataFrame | None = None,
) -> dict[str, BackboneModel]:
    """Load backbone models for every outcome in *levers_cfg*.

    Parameters
    ----------
    modeling_dir : Path
        Root modelling output directory.
    levers_cfg : LeversConfig
        Lever config specifying the ``model_ref`` per outcome.

    Returns
    -------
    dict[str, BackboneModel]
        Mapping from outcome name to its loaded backbone.
    """
    models: dict[str, BackboneModel] = {}
    for outcome_name, spec in levers_cfg.outcomes.items():
        model = load_backbone_model(modeling_dir, spec.model_ref)
        if survey_df is not None:
            means, stds = _compute_term_scaler_stats(model.terms, survey_df)
            model = BackboneModel(
                name=model.name,
                model_type=model.model_type,
                terms=model.terms,
                coefficients=model.coefficients,
                thresholds=model.thresholds,
                intercept=model.intercept,
                blocks=model.blocks,
                term_means=means,
                term_stds=stds,
            )
        models[outcome_name] = model
    return models


def _compute_term_scaler_stats(
    terms: tuple[str, ...],
    survey_df: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-term standardization stats matching fit-time behavior.

    Non-numeric, binary/dummy, and missing-indicator terms are left
    unscaled with mean=0 and std=1.
    """
    means = np.zeros(len(terms), dtype=np.float64)
    stds = np.ones(len(terms), dtype=np.float64)
    for i, term in enumerate(terms):
        if term.startswith("C(country"):
            continue
        if term.endswith("_missing"):
            continue
        if term not in survey_df.columns:
            continue
        series = survey_df[term]
        if not series.dtype.is_numeric():
            continue
        s = series.cast(pl.Float64).drop_nulls().drop_nans()
        if s.len() == 0:
            continue
        # Keep binary indicators in original 0/1 scale.
        unique_vals = s.unique().sort()
        if unique_vals.len() <= 2:
            vals = unique_vals.to_numpy()
            if np.isin(vals, [0.0, 1.0]).all():
                continue
        mean = float(s.mean())
        std = float(s.std(ddof=0))
        if not np.isfinite(std) or np.isclose(std, 0.0):
            continue
        means[i] = mean
        stds[i] = std
    return means, stds


# =========================================================================
# 3.  Anchor system  (anchors/system/)
# =========================================================================


@dataclass(slots=True)
class AnchorMetadata:
    """Typed metadata for anchor system."""

    input_csv: str | None = None
    country_col: str | None = None
    row_id_col: str | None = None
    normalization: str | None = None
    anchor_k: int | None = None
    features: tuple[str, ...] = ()
    pca_path: str | None = None
    extras: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict) -> AnchorMetadata:
        known_keys = {
            "input_csv",
            "country_col",
            "row_id_col",
            "normalization",
            "anchor_k",
            "features",
            "pca_path",
        }
        extras = {k: v for k, v in raw.items() if k not in known_keys}
        return cls(
            input_csv=raw.get("input_csv"),
            country_col=raw.get("country_col"),
            row_id_col=raw.get("row_id_col"),
            normalization=raw.get("normalization"),
            anchor_k=raw.get("anchor_k"),
            features=tuple(raw.get("features", [])),
            pca_path=raw.get("pca_path"),
            extras=extras,
        )


@dataclass(slots=True)
class AnchorSystem:
    """Pre-computed anchor profiles and neighbourhood mappings.

    Parameters
    ----------
    metadata : AnchorMetadata
        Parsed metadata from ``anchor_metadata.json``.
    n_anchors : int
        Number of cluster centroids.
    country_shares : pl.DataFrame
        Per-anchor per-country weight shares (``anchor_id, country, share``).
    coords : pl.DataFrame
        Anchor centroid PCA coordinates.
    neighbourhoods : pl.DataFrame
        Survey-respondent neighbours per anchor×country.
    memberships : pl.DataFrame
        Soft membership weights for all survey respondents.
    features : tuple[str, ...]
        Feature names used for PCA / anchor construction.
    pca_components : np.ndarray | None
        PCA loadings matrix.
    pca_mean : np.ndarray | None
        PCA centering vector.
    """

    metadata: AnchorMetadata = field(repr=False)
    n_anchors: int = 10
    country_shares: pl.DataFrame = field(default_factory=pl.DataFrame)
    coords: pl.DataFrame = field(default_factory=pl.DataFrame)
    neighbourhoods: pl.DataFrame = field(default_factory=pl.DataFrame)
    memberships: pl.DataFrame = field(default_factory=pl.DataFrame)
    features: tuple[str, ...] = ()
    pca_components: np.ndarray | None = None
    pca_mean: np.ndarray | None = None


def load_anchor_system(anchors_dir: Path) -> AnchorSystem:
    """Load the full anchor system from disk.

    Parameters
    ----------
    anchors_dir : Path
        Path to ``empirical/output/anchors/system/``.

    Returns
    -------
    AnchorSystem

    Raises
    ------
    FileNotFoundError
        If any required file is missing.
    """
    # --- metadata ---
    meta_path = anchors_dir / "anchor_metadata.json"
    if not meta_path.is_file():
        msg = f"Anchor metadata not found: {meta_path}"
        raise FileNotFoundError(msg)
    metadata_raw: dict = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata = AnchorMetadata.from_dict(metadata_raw)

    # --- country shares ---
    shares_path = anchors_dir / "anchor_country_share.csv"
    if not shares_path.is_file():
        msg = f"Anchor country share not found: {shares_path}"
        raise FileNotFoundError(msg)
    country_shares = pl.read_csv(shares_path)

    # --- coordinates ---
    coords_path = anchors_dir / "anchor_coords.csv"
    if not coords_path.is_file():
        msg = f"Anchor coordinates not found: {coords_path}"
        raise FileNotFoundError(msg)
    coords = pl.read_csv(coords_path)

    # --- neighbourhoods ---
    neigh_path = anchors_dir / "anchor_neighborhoods.parquet"
    if not neigh_path.is_file():
        msg = f"Anchor neighbourhoods not found: {neigh_path}"
        raise FileNotFoundError(msg)
    neighbourhoods = pl.read_parquet(neigh_path)

    # --- memberships ---
    memb_path = anchors_dir / "anchor_memberships.parquet"
    if not memb_path.is_file():
        msg = f"Anchor memberships not found: {memb_path}"
        raise FileNotFoundError(msg)
    memberships = pl.read_parquet(memb_path)

    # --- PCA ---
    pca_components = None
    pca_mean = None
    pca_path = anchors_dir / (metadata.pca_path or "anchor_pca.npz")
    if pca_path.is_file():
        pca_data = np.load(pca_path)
        pca_components = pca_data.get("components")
        pca_mean = pca_data.get("mean")

    n_anchors = int(metadata.anchor_k or 10)
    features = tuple(metadata.features)

    logger.info(
        "Loaded anchor system: {n} anchors, {f} features, {neigh} neighbourhood rows.",
        n=n_anchors,
        f=len(features),
        neigh=neighbourhoods.height,
    )
    return AnchorSystem(
        metadata=metadata,
        n_anchors=n_anchors,
        country_shares=country_shares,
        coords=coords,
        neighbourhoods=neighbourhoods,
        memberships=memberships,
        features=features,
        pca_components=pca_components,
        pca_mean=pca_mean,
    )
