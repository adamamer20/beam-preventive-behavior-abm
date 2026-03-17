"""Population initialization via anchor-country neighbourhood resampling.

Implements the methodology-aligned sampling protocol:

1. Sample anchor ``k`` by country prevalence ``pi_{k|c}``.
2. Sample donor respondent from anchor-country neighbourhood.
3. Copy donor values to ``(a_i, s_i(0), y_i(0))``, preserving
   joint covariate structure.

Examples
--------
>>> from beam_abm.abm.config import SimulationConfig
>>> cfg = SimulationConfig(n_agents=100)
>>> # pop = initialise_population(cfg)  # with real data
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from beam_abm.abm.data_contracts import AnchorSystem
from beam_abm.abm.state_schema import (
    IMMUTABLE_DEMOGRAPHIC_COLUMNS,
    MUTABLE_COLUMNS,
    OUTCOME_COLUMNS,
)
from beam_abm.abm.state_update import compute_observed_norm_fixed_point_share

__all__ = [
    "HeterogeneityAxisLoadings",
    "HeterogeneityCDFKnots",
    "HeterogeneityBounds",
    "HeterogeneityLoadings",
    "build_heterogeneity_cdf_knots",
    "load_heterogeneity_loadings",
    "ensure_agent_schema",
    "initialise_population",
    "sample_anchor_for_country",
    "sample_donors",
]


@dataclass(slots=True, frozen=True)
class HeterogeneityAxisLoadings:
    """Empirical PC1 transform metadata for one heterogeneity axis."""

    columns: tuple[str, ...]
    means: dict[str, float]
    stds: dict[str, float]
    pc1_loadings: dict[str, float]


@dataclass(slots=True, frozen=True)
class HeterogeneityLoadings:
    """Empirical loadings bundle for structural heterogeneity construction."""

    social_activity: HeterogeneityAxisLoadings
    media_exposure: HeterogeneityAxisLoadings
    identity_lock: HeterogeneityAxisLoadings
    cdf_knots: HeterogeneityCDFKnots


@dataclass(slots=True, frozen=True)
class HeterogeneityCDFKnots:
    """Empirical CDF knots for mapping heterogeneity scores to percentiles."""

    quantiles: tuple[float, ...]
    score_knots: dict[str, tuple[float, ...]]
    source: str = "artifact"


@dataclass(slots=True, frozen=True)
class HeterogeneityBounds:
    """Fixed trait bounds for deterministic percentile mapping."""

    # Used to shape raw lambda before mean-normalization to E[lambda]=1.
    lambda_min: float = 0.05
    lambda_max: float = 0.95
    alpha_min: float = 0.1
    alpha_max: float = 4.0
    phi_min: float = 0.05
    phi_max: float = 5.0
    theta_min: float = 0.2
    theta_max: float = 0.8


def _parse_axis(payload: dict[str, object], axis_name: str) -> HeterogeneityAxisLoadings:
    axis = payload.get(axis_name)
    if not isinstance(axis, dict):
        msg = f"Missing heterogeneity axis '{axis_name}' in loadings payload."
        raise ValueError(msg)
    columns_raw = axis.get("columns")
    means_raw = axis.get("means")
    stds_raw = axis.get("stds")
    pc1_raw = axis.get("pc1_loadings")
    if not isinstance(columns_raw, list) or not isinstance(means_raw, dict):
        msg = f"Axis '{axis_name}' has invalid columns/means structure."
        raise ValueError(msg)
    if not isinstance(stds_raw, dict) or not isinstance(pc1_raw, dict):
        msg = f"Axis '{axis_name}' has invalid stds/loadings structure."
        raise ValueError(msg)

    columns = tuple(str(c) for c in columns_raw)
    means = {str(k): float(v) for k, v in means_raw.items()}
    stds = {str(k): float(v) for k, v in stds_raw.items()}
    pc1 = {str(k): float(v) for k, v in pc1_raw.items()}

    missing = [c for c in columns if c not in means or c not in stds or c not in pc1]
    if missing:
        msg = f"Axis '{axis_name}' missing metadata for columns: {missing}"
        raise ValueError(msg)

    return HeterogeneityAxisLoadings(columns=columns, means=means, stds=stds, pc1_loadings=pc1)


def _parse_cdf_knots(payload: dict[str, object]) -> HeterogeneityCDFKnots:
    cdf = payload.get("cdf_knots")
    if not isinstance(cdf, dict):
        msg = "Invalid heterogeneity loadings artifact: missing 'cdf_knots'."
        raise ValueError(msg)

    quantiles_raw = cdf.get("quantiles")
    if not isinstance(quantiles_raw, list) or not quantiles_raw:
        msg = "Invalid heterogeneity CDF knots: 'quantiles' must be a non-empty list."
        raise ValueError(msg)
    quantiles = tuple(float(q) for q in quantiles_raw)
    if any((q <= 0.0 or q >= 1.0) for q in quantiles):
        msg = "Invalid heterogeneity CDF knots: quantiles must be within (0, 1)."
        raise ValueError(msg)
    if any(q2 <= q1 for q1, q2 in zip(quantiles, quantiles[1:], strict=False)):
        msg = "Invalid heterogeneity CDF knots: quantiles must be strictly increasing."
        raise ValueError(msg)

    scores_raw = cdf.get("scores", {})
    if not isinstance(scores_raw, dict):
        msg = "Invalid heterogeneity CDF knots: 'scores' must be a dict."
        raise ValueError(msg)
    score_knots: dict[str, tuple[float, ...]] = {}
    for key, values in scores_raw.items():
        if not isinstance(values, list):
            msg = f"Invalid CDF knots for '{key}': expected list of score values."
            raise ValueError(msg)
        if values and len(values) != len(quantiles):
            msg = f"CDF knot length mismatch for '{key}': expected {len(quantiles)} values."
            raise ValueError(msg)
        knots = np.array([float(v) for v in values], dtype=np.float64)
        if not np.isfinite(knots).all():
            msg = f"Invalid CDF knots for '{key}': values must be finite."
            raise ValueError(msg)
        # Keep CDF score knots strictly increasing to avoid flat/ambiguous
        # interpolation segments for discrete drivers.
        for idx in range(1, knots.size):
            if knots[idx] <= knots[idx - 1]:
                knots[idx] = np.nextafter(knots[idx - 1], np.inf)
        score_knots[str(key)] = tuple(float(v) for v in knots)

    source = str(cdf.get("source", "artifact"))
    return HeterogeneityCDFKnots(quantiles=quantiles, score_knots=score_knots, source=source)


@lru_cache(maxsize=1)
def load_heterogeneity_loadings() -> HeterogeneityLoadings:
    """Load empirical heterogeneity loadings from the committed JSON artifact."""
    path = Path(__file__).resolve().parent / "assets" / "heterogeneity_loadings.json"
    if not path.exists():
        msg = f"Missing heterogeneity loadings artifact: {path}"
        raise FileNotFoundError(msg)
    raw = json.loads(path.read_text(encoding="utf-8"))
    axes = raw.get("axes") if isinstance(raw, dict) else None
    if not isinstance(axes, dict):
        msg = "Invalid heterogeneity loadings artifact: missing top-level 'axes'."
        raise ValueError(msg)
    cdf_knots = _parse_cdf_knots(raw)
    return HeterogeneityLoadings(
        social_activity=_parse_axis(axes, "social_activity"),
        media_exposure=_parse_axis(axes, "media_exposure"),
        identity_lock=_parse_axis(axes, "identity_lock"),
        cdf_knots=cdf_knots,
    )


def _safe_column(agents: pl.DataFrame, name: str, n_agents: int) -> np.ndarray:
    if name not in agents.columns:
        return np.zeros(n_agents, dtype=np.float64)
    return np.nan_to_num(agents[name].cast(pl.Float64).to_numpy(), nan=0.0)


def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    std = float(np.std(x))
    if std <= eps:
        return np.zeros_like(x)
    return (x - float(np.mean(x))) / std


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _cdf_percentile(x: np.ndarray, cdf_knots: HeterogeneityCDFKnots, key: str) -> np.ndarray:
    values = cdf_knots.score_knots.get(key)
    if values is None or len(values) == 0:
        msg = f"Missing CDF knot values for '{key}'."
        raise ValueError(msg)
    knots = np.array(values, dtype=np.float64)
    quantiles = np.array(cdf_knots.quantiles, dtype=np.float64)
    if knots.shape[0] != quantiles.shape[0]:
        msg = f"CDF knot length mismatch for '{key}': {knots.shape[0]} vs {quantiles.shape[0]}."
        raise ValueError(msg)
    order = np.argsort(knots)
    knots_sorted = knots[order]
    quantiles_sorted = quantiles[order]
    u = np.interp(x, knots_sorted, quantiles_sorted, left=quantiles_sorted[0], right=quantiles_sorted[-1])
    return np.clip(u, 0.0, 1.0)


def _linear_from_percentile(u: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return lo + (hi - lo) * u


def _log_from_percentile(u: np.ndarray, lo: float, hi: float) -> np.ndarray:
    log_lo = np.log(lo)
    log_hi = np.log(hi)
    return np.exp(log_lo + u * (log_hi - log_lo))


def _mean_normalize_multiplier(x: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Return a positive multiplier vector with sample mean approximately 1."""
    mean_x = float(np.mean(x))
    if (not np.isfinite(mean_x)) or mean_x <= eps:
        return np.ones_like(x, dtype=np.float64)
    return x / mean_x


def _axis_score(
    agents: pl.DataFrame,
    axis: HeterogeneityAxisLoadings,
    n_agents: int,
) -> np.ndarray:
    score = np.zeros(n_agents, dtype=np.float64)
    for column in axis.columns:
        values = _safe_column(agents, column, n_agents)
        mean = float(axis.means[column])
        std = max(float(axis.stds[column]), 1e-8)
        z = (values - mean) / std
        score += float(axis.pc1_loadings[column]) * z
    return score


def build_heterogeneity_cdf_knots(
    agents: pl.DataFrame,
    loadings: HeterogeneityLoadings,
    *,
    quantiles: tuple[float, ...],
    source: str = "survey",
) -> HeterogeneityCDFKnots:
    """Build empirical CDF knots from a reference survey population."""
    n_agents = agents.height
    social_activity = _axis_score(agents, loadings.social_activity, n_agents)
    media_exposure = _axis_score(agents, loadings.media_exposure, n_agents)
    convo_total = (
        _safe_column(agents, "vax_convo_count_friends", n_agents)
        + _safe_column(agents, "vax_convo_count_cohabitants", n_agents)
        + _safe_column(agents, "vax_convo_count_others", n_agents)
    )

    drivers = {
        "lambda_driver": social_activity + media_exposure,
        "alpha_driver": media_exposure + 0.5 * social_activity,
        "phi_driver": convo_total,
        "theta_driver": -social_activity,
    }

    score_knots: dict[str, tuple[float, ...]] = {}
    q = np.array(quantiles, dtype=np.float64)
    for key, values in drivers.items():
        clean = values[np.isfinite(values)]
        if clean.size == 0:
            score_knots[key] = tuple(0.0 for _ in quantiles)
            continue
        pct = np.nanpercentile(clean, q * 100.0)
        pct = np.maximum.accumulate(pct.astype(np.float64))
        score_knots[key] = tuple(float(v) for v in pct)

    return HeterogeneityCDFKnots(quantiles=quantiles, score_knots=score_knots, source=source)


def _attach_structural_heterogeneity(
    agents: pl.DataFrame,
    loadings: HeterogeneityLoadings,
    bounds: HeterogeneityBounds,
    cdf_knots: HeterogeneityCDFKnots,
    *,
    theta_beta_alpha: float = 2.0,
    rng: np.random.Generator | None = None,
) -> pl.DataFrame:
    """Attach structural heterogeneity via empirical mappings.

    Three social traits use survey-anchored percentile mappings. The norm
    threshold uses a symmetric Beta prior on [0, 1] mapped to
    ``[theta_min, theta_max]``.
    """
    n_agents = agents.height
    if theta_beta_alpha <= 0.0:
        msg = "theta_beta_alpha must be strictly positive."
        raise ValueError(msg)
    if rng is None:
        rng = np.random.default_rng()

    convo_total = (
        _safe_column(agents, "vax_convo_count_friends", n_agents)
        + _safe_column(agents, "vax_convo_count_cohabitants", n_agents)
        + _safe_column(agents, "vax_convo_count_others", n_agents)
    )

    social_activity = _axis_score(agents, loadings.social_activity, n_agents)
    media_exposure = _axis_score(agents, loadings.media_exposure, n_agents)

    u_lambda = _cdf_percentile(social_activity + media_exposure, cdf_knots, "lambda_driver")
    u_alpha = _cdf_percentile(media_exposure + 0.5 * social_activity, cdf_knots, "alpha_driver")
    u_phi = _cdf_percentile(convo_total, cdf_knots, "phi_driver")
    u_theta = rng.beta(theta_beta_alpha, theta_beta_alpha, size=n_agents)

    lam = _linear_from_percentile(u_lambda, bounds.lambda_min, bounds.lambda_max)
    lam = _mean_normalize_multiplier(lam)
    alpha = _log_from_percentile(u_alpha, bounds.alpha_min, bounds.alpha_max)
    phi = _log_from_percentile(u_phi, bounds.phi_min, bounds.phi_max)
    theta = _linear_from_percentile(u_theta, bounds.theta_min, bounds.theta_max)

    return agents.with_columns(
        pl.Series("social_susceptibility_lambda", lam.astype(np.float64)),
        pl.Series("social_attention_alpha", alpha.astype(np.float64)),
        pl.Series("social_sharing_phi", phi.astype(np.float64)),
        pl.Series("norm_threshold_theta", theta.astype(np.float64)),
    )


def _resolve_cdf_knots(
    loadings: HeterogeneityLoadings,
    survey_df: pl.DataFrame,
) -> HeterogeneityCDFKnots:
    if loadings.cdf_knots.score_knots:
        return loadings.cdf_knots
    return build_heterogeneity_cdf_knots(
        survey_df,
        loadings,
        quantiles=loadings.cdf_knots.quantiles,
        source="survey",
    )


def _attach_locality_id(agents: pl.DataFrame) -> pl.DataFrame:
    """Attach locality IDs from region, with strict usability checks.

    Localities represent social/local clusters (region-like context),
    while ``anchor_id`` remains a latent-profile index.
    """
    if "locality_id" in agents.columns:
        return agents

    if "country" not in agents.columns:
        msg = "Cannot build region-based locality_id: missing required 'country' column."
        raise ValueError(msg)
    if "region_final" not in agents.columns:
        msg = "Cannot build region-based locality_id: missing required 'region_final' column."
        raise ValueError(msg)

    cleaned = agents.with_columns(
        pl.col("country").cast(pl.Utf8).str.strip_chars().alias("country"),
        pl.col("region_final").cast(pl.Utf8).str.strip_chars().alias("region_final"),
    )

    invalid_region_rows = cleaned.filter(pl.col("region_final").is_null() | (pl.col("region_final") == ""))
    if invalid_region_rows.height > 0:
        msg = f"Cannot build region-based locality_id: found {invalid_region_rows.height} rows with null/blank region."
        raise ValueError(msg)

    region_counts = (
        cleaned.group_by(["country", "region_final"])
        .len()
        .rename({"len": "n"})
        .sort(["country", "n"], descending=[False, True])
    )
    regions_per_country = region_counts.group_by("country").len().rename({"len": "n_regions"})
    too_few_regions = regions_per_country.filter(pl.col("n_regions") < 2)
    if too_few_regions.height > 0:
        bad = ", ".join(f"{row['country']}({int(row['n_regions'])})" for row in too_few_regions.iter_rows(named=True))
        msg = f"Cannot build region-based locality_id: some countries do not have enough regional variation: {bad}."
        raise ValueError(msg)

    dominance = (
        region_counts.join(
            cleaned.group_by("country").len().rename({"len": "country_n"}),
            on="country",
            how="left",
        )
        .with_columns((pl.col("n") / pl.col("country_n")).alias("share"))
        .group_by("country")
        .agg(pl.col("share").max().alias("max_region_share"))
    )
    overly_concentrated = dominance.filter(pl.col("max_region_share") > 0.95)
    if overly_concentrated.height > 0:
        bad = ", ".join(
            f"{row['country']}({float(row['max_region_share']):.2f})"
            for row in overly_concentrated.iter_rows(named=True)
        )
        msg = (
            "Cannot build region-based locality_id: region assignment is too "
            f"concentrated in some countries (max share > 0.95): {bad}."
        )
        raise ValueError(msg)

    seeded = cleaned.with_columns(
        pl.concat_str(
            [
                pl.col("country"),
                pl.lit("_r"),
                pl.col("region_final"),
            ],
            separator="",
        ).alias("locality_seed_id")
    )
    return _merge_seeded_localities(seeded, min_size=100, target_size=150, max_size=200)


def _merge_seeded_localities(
    agents: pl.DataFrame,
    *,
    min_size: int,
    target_size: int,
    max_size: int,
) -> pl.DataFrame:
    """Merge region-seeded localities into balanced localities per country.

    Uses ``region_final`` as locality seed order, then packs agents into
    balanced bins so locality sizes are practically usable for targeted shocks.
    """
    if min_size <= 0 or target_size <= 0 or max_size <= 0:
        msg = "Community size parameters must be strictly positive."
        raise ValueError(msg)
    if not (min_size <= target_size <= max_size):
        msg = "Community size parameters must satisfy min_size <= target_size <= max_size."
        raise ValueError(msg)

    if "country" not in agents.columns or "locality_seed_id" not in agents.columns:
        msg = "Cannot merge localities: missing required country/locality_seed_id columns."
        raise ValueError(msg)

    # Deterministic ordering: preserve region-local contiguity, then agent id.
    ordered = agents.sort(["country", "locality_seed_id", "agent_id"])

    rows: list[pl.DataFrame] = []
    for country in sorted(ordered.get_column("country").unique().to_list()):
        country_df = ordered.filter(pl.col("country") == country).with_row_index("country_rank")
        n_country = country_df.height
        if n_country == 0:
            continue

        # Select bin count to keep locality sizes near target and within bounds.
        min_bins = max(1, int(np.ceil(n_country / max_size)))
        max_bins = max(1, int(np.floor(n_country / min_size)))
        if min_bins > max_bins:
            n_bins = 1
        else:
            preferred = max(1, int(round(n_country / target_size)))
            n_bins = int(np.clip(preferred, min_bins, max_bins))

        bin_idx = (country_df["country_rank"].to_numpy() * n_bins) // n_country
        local_ids = np.array([f"{country}_loc{int(i):03d}" for i in bin_idx], dtype=object)
        rows.append(
            country_df.drop("country_rank").with_columns(
                pl.Series("locality_id", local_ids),
            )
        )

    merged = (
        pl.concat(rows, how="vertical") if rows else ordered.with_columns(pl.lit("global_loc000").alias("locality_id"))
    )

    size_check = (
        merged.group_by(["country", "locality_id"])
        .len()
        .rename({"len": "n"})
        .join(
            merged.group_by("country").len().rename({"len": "country_n"}),
            on="country",
            how="left",
        )
        .filter((pl.col("country_n") >= min_size) & ((pl.col("n") < min_size) | (pl.col("n") > max_size)))
    )
    if size_check.height > 0:
        msg = "Community balancing failed to enforce requested size bounds."
        raise ValueError(msg)

    return merged.drop("locality_seed_id")


def _attach_contact_scalars(agents: pl.DataFrame) -> pl.DataFrame:
    """Attach per-agent contact scalars used by network/contact modules."""
    n_agents = agents.height

    friend_size = _safe_column(agents, "size_friend_group", n_agents)
    colleague_size = _safe_column(agents, "size_colleagues_group", n_agents)
    group_size_raw = friend_size + colleague_size

    convo_general = (
        _safe_column(agents, "convo_general_count_friends", n_agents)
        + _safe_column(agents, "convo_general_count_cohabitants", n_agents)
        + _safe_column(agents, "convo_general_count_others", n_agents)
    )
    if np.allclose(convo_general, 0.0):
        convo_general = (
            _safe_column(agents, "vax_convo_count_friends", n_agents)
            + _safe_column(agents, "vax_convo_count_cohabitants", n_agents)
            + _safe_column(agents, "vax_convo_count_others", n_agents)
        )
    media = _safe_column(agents, "duration_socialmedia", n_agents)

    vax_convo = (
        _safe_column(agents, "vax_convo_count_friends", n_agents)
        + _safe_column(agents, "vax_convo_count_cohabitants", n_agents)
        + _safe_column(agents, "vax_convo_count_others", n_agents)
    )

    contact_physical = (
        _safe_column(agents, "contact_group_cohabitant", n_agents)
        + _safe_column(agents, "contact_group_noncohabitant", n_agents)
        + _safe_column(agents, "contact_group_friends", n_agents)
        + _safe_column(agents, "contact_group_colleagues", n_agents)
        + _safe_column(agents, "contact_group_others", n_agents)
    )
    contact_information = (
        _safe_column(agents, "discussion_group_cohabitant", n_agents)
        + _safe_column(agents, "discussion_group_noncohabitant", n_agents)
        + _safe_column(agents, "discussion_group_friends", n_agents)
        + _safe_column(agents, "discussion_group_colleagues", n_agents)
        + _safe_column(agents, "discussion_group_others", n_agents)
    )
    remote = _safe_column(agents, "yesterday_remote", n_agents)
    friend_closure = _safe_column(agents, "friend_closure", n_agents)

    group_size_score = _sigmoid(_zscore(group_size_raw))
    activity_score = _sigmoid(0.75 * _zscore(convo_general) + 0.25 * _zscore(media))
    topic_salience_score = _sigmoid(_zscore(vax_convo))

    contact_index = _zscore(contact_physical) + 0.35 * _zscore(contact_information) - 0.6 * _zscore(remote)
    contact_propensity = np.clip(0.35 + 2.65 * _sigmoid(contact_index), 0.15, 3.0)

    total_contacts = np.maximum(contact_physical, 1e-6)
    household_contact_share = np.clip(
        _safe_column(agents, "contact_group_cohabitant", n_agents) / total_contacts, 0.0, 0.7
    )
    recurring_share = np.clip(
        0.20 + 0.65 * _sigmoid(0.6 * _zscore(group_size_raw) + 0.4 * _zscore(friend_closure)), 0.05, 0.9
    )
    recurring_share = np.clip(recurring_share, 0.0, 1.0 - household_contact_share)

    core_degree = np.clip(np.rint(4.0 + 26.0 * group_size_score), 4, 30).astype(np.int64)
    has_household = _safe_column(agents, "household_size", n_agents) > 1.0

    return agents.with_columns(
        pl.Series("group_size_score", group_size_score.astype(np.float64)),
        pl.Series("activity_score", activity_score.astype(np.float64)),
        pl.Series("topic_salience_score", topic_salience_score.astype(np.float64)),
        pl.Series("contact_propensity", contact_propensity.astype(np.float64)),
        pl.Series("household_contact_share", household_contact_share.astype(np.float64)),
        pl.Series("recurring_share", recurring_share.astype(np.float64)),
        pl.Series("core_degree", core_degree),
        pl.Series("has_household", has_household),
    )


def _attach_norm_channels(agents: pl.DataFrame) -> pl.DataFrame:
    """Create and validate descriptive norm channels.

        Strict rule:
        - ``social_norms_vax_avg`` must exist.
    - if descriptive channels are missing, derive from
      ``social_norms_vax_avg`` (single deterministic source of truth);
        - if values are non-finite or out-of-range, sanitize into [1, 7].
    """
    if "social_norms_vax_avg" not in agents.columns:
        msg = "Missing required column 'social_norms_vax_avg' for norm-channel construction."
        raise ValueError(msg)

    avg = agents["social_norms_vax_avg"].cast(pl.Float64)
    avg_np_raw = avg.to_numpy()
    avg_np = np.clip(np.nan_to_num(avg_np_raw, nan=4.0, posinf=7.0, neginf=1.0), 1.0, 7.0)

    def _validated_channel(column: str) -> pl.Series:
        if column not in agents.columns:
            return pl.Series(column, avg_np.astype(np.float64))
        vals_raw = agents[column].cast(pl.Float64).to_numpy()
        vals = np.clip(np.nan_to_num(vals_raw, nan=4.0, posinf=7.0, neginf=1.0), 1.0, 7.0)
        return pl.Series(column, vals.astype(np.float64))

    return agents.with_columns(
        _validated_channel("social_norms_descriptive_vax"),
        _validated_channel("social_norms_descriptive_npi"),
    )


def _attach_norm_share_hat(agents: pl.DataFrame, *, slope: float = 8.0) -> pl.DataFrame:
    """Attach latent perceived-adoption share implied by baseline norm state.

    The latent states ``norm_share_hat_vax`` and ``norm_share_hat_npi`` are
    initialized so that feeding them through their descriptive-norm logistic
    target maps reproduces current descriptive norms (fixed-point coherence).
    """
    if "social_norms_descriptive_vax" not in agents.columns and "social_norms_descriptive_npi" not in agents.columns:
        return agents

    n_agents = agents.height
    if n_agents == 0:
        return agents.with_columns(
            pl.Series("norm_share_hat_vax", np.array([], dtype=np.float64)),
            pl.Series("norm_share_hat_npi", np.array([], dtype=np.float64)),
        )

    if "norm_threshold_theta" in agents.columns:
        threshold = agents["norm_threshold_theta"].cast(pl.Float64).to_numpy().astype(np.float64)
    else:
        threshold = np.full(n_agents, 0.5, dtype=np.float64)
    threshold = np.clip(np.nan_to_num(threshold, nan=0.5, posinf=0.5, neginf=0.5), 0.05, 0.95)

    slope_val = float(slope)
    if not np.isfinite(slope_val) or slope_val <= 0.0:
        slope_val = 8.0

    updates: list[pl.Series] = []
    if "social_norms_descriptive_vax" in agents.columns:
        descriptive = agents["social_norms_descriptive_vax"].cast(pl.Float64).to_numpy().astype(np.float64)
        descriptive = np.clip(np.nan_to_num(descriptive, nan=4.0, posinf=7.0, neginf=1.0), 1.0, 7.0)
        share_hat_vax = compute_observed_norm_fixed_point_share(
            descriptive_norm=descriptive,
            threshold=threshold,
            column="social_norms_descriptive_vax",
            slope=slope_val,
        )
        updates.append(pl.Series("norm_share_hat_vax", np.clip(share_hat_vax, 0.0, 1.0).astype(np.float64)))
    if "social_norms_descriptive_npi" in agents.columns:
        descriptive = agents["social_norms_descriptive_npi"].cast(pl.Float64).to_numpy().astype(np.float64)
        descriptive = np.clip(np.nan_to_num(descriptive, nan=4.0, posinf=7.0, neginf=1.0), 1.0, 7.0)
        share_hat_npi = compute_observed_norm_fixed_point_share(
            descriptive_norm=descriptive,
            threshold=threshold,
            column="social_norms_descriptive_npi",
            slope=slope_val,
        )
        updates.append(pl.Series("norm_share_hat_npi", np.clip(share_hat_npi, 0.0, 1.0).astype(np.float64)))
    if not updates:
        return agents
    return agents.with_columns(updates)


# =========================================================================
# 1.  Anchor sampling by country prevalence
# =========================================================================


def sample_anchor_for_country(
    anchor_system: AnchorSystem,
    country: str,
    n: int,
    rng: np.random.Generator,
    country_shares: pl.DataFrame | None = None,
) -> np.ndarray:
    """Sample *n* anchor IDs for a given country using ``pi_{k|c}``.

    Parameters
    ----------
    anchor_system : AnchorSystem
        Loaded anchor system with country shares.
    country : str
        ISO-2 country code.
    n : int
        Number of samples to draw.
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    np.ndarray
        Integer anchor IDs of shape ``(n,)``.
    """
    shares_df = country_shares if country_shares is not None else anchor_system.country_shares
    shares = shares_df.filter(pl.col("country") == country)
    if shares.height == 0:
        msg = f"No anchor shares found for country '{country}'."
        raise ValueError(msg)

    anchor_ids = shares["anchor_id"].to_numpy()
    probs = shares["share"].to_numpy().astype(np.float64)
    probs = probs / probs.sum()  # re-normalise for safety

    return rng.choice(anchor_ids, size=n, replace=True, p=probs)


# =========================================================================
# 2.  Donor sampling from neighbourhoods
# =========================================================================


def sample_donors(
    anchor_ids: np.ndarray,
    countries: np.ndarray,
    anchor_system: AnchorSystem,
    survey_df: pl.DataFrame,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Sample donor respondents from anchor-country neighbourhoods.

    For each agent, selects a random survey respondent from the
    appropriate anchor×country neighbourhood, weighted inversely
    by distance rank to favour closer neighbours.

    Parameters
    ----------
    anchor_ids : np.ndarray
        Anchor IDs assigned to each agent, shape ``(n_agents,)``.
    countries : np.ndarray
        Country codes for each agent, shape ``(n_agents,)``.
    anchor_system : AnchorSystem
        Loaded anchor system with neighbourhood data.
    survey_df : pl.DataFrame
        Clean survey micro-data (the donor pool).
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    pl.DataFrame
        Resampled rows from *survey_df* with *n_agents* rows.
    """
    n_agents = len(anchor_ids)
    donor_row_ids = np.empty(n_agents, dtype=np.int64)

    # Pre-index neighbourhoods by (anchor_id, country)
    neigh = anchor_system.neighbourhoods
    neigh_grouped: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for (aid, ctry), group_df in neigh.group_by(["anchor_id", "country"]):
        row_ids = group_df["row_id"].to_numpy()
        ranks = group_df["rank"].to_numpy().astype(np.float64)
        # Inverse-rank weighting: w_i = 1 / rank_i
        weights = 1.0 / np.maximum(ranks, 1.0)
        weights /= weights.sum()
        neigh_grouped[(int(aid), str(ctry))] = (row_ids, weights)

    for i in range(n_agents):
        key = (int(anchor_ids[i]), str(countries[i]))
        if key in neigh_grouped:
            pool_ids, pool_weights = neigh_grouped[key]
            donor_row_ids[i] = rng.choice(pool_ids, p=pool_weights)
        else:
            # Fallback: sample uniformly from same country
            country_ids = survey_df.filter(pl.col("country") == str(countries[i]))["row_id"].to_numpy()
            donor_row_ids[i] = rng.choice(country_ids)

    # Join donors from survey
    donor_df = pl.DataFrame({"row_id": donor_row_ids}).join(survey_df, on="row_id", how="left")
    return donor_df


def ensure_agent_schema(
    agents: pl.DataFrame,
    n_agents: int,
    anchor_ids: np.ndarray | None = None,
) -> pl.DataFrame:
    """Ensure required agent columns exist with expected types."""
    if "agent_id" not in agents.columns:
        agents = agents.with_columns(pl.Series("agent_id", np.arange(n_agents)))

    if "anchor_id" not in agents.columns:
        if anchor_ids is None:
            anchor_ids = np.full(n_agents, -1, dtype=np.int64)
        agents = agents.with_columns(pl.Series("anchor_id", anchor_ids.astype(np.int64)))

    split_norm_channels = {
        "social_norms_descriptive_vax",
        "social_norms_descriptive_npi",
    }
    for col in MUTABLE_COLUMNS:
        if col in split_norm_channels:
            continue
        if col not in agents.columns:
            agents = agents.with_columns(pl.lit(0.0).cast(pl.Float32).alias(col))

    for col in OUTCOME_COLUMNS:
        if col not in agents.columns:
            agents = agents.with_columns(pl.lit(np.nan).cast(pl.Float32).alias(col))

    for col in IMMUTABLE_DEMOGRAPHIC_COLUMNS:
        if col not in agents.columns:
            agents = agents.with_columns(pl.lit(0.0).cast(pl.Float32).alias(col))

    for col in set(MUTABLE_COLUMNS) | set(IMMUTABLE_DEMOGRAPHIC_COLUMNS):
        if col not in agents.columns:
            continue
        try:
            vals = agents[col].cast(pl.Float64).to_numpy()
        except pl.exceptions.InvalidOperationError:
            continue
        if np.isfinite(vals).all():
            continue
        finite = vals[np.isfinite(vals)]
        fill_value = float(np.mean(finite)) if finite.size > 0 else 0.0
        clean = np.nan_to_num(vals, nan=fill_value, posinf=fill_value, neginf=fill_value).astype(np.float32)
        agents = agents.with_columns(pl.Series(col, clean))

    if "sex_female[T.True]" not in agents.columns and "sex_female" in agents.columns:
        agents = agents.with_columns(pl.col("sex_female").cast(pl.Float32).alias("sex_female[T.True]"))

    if "household_size" not in agents.columns:
        if "contact_group_cohabitant" in agents.columns:
            source = np.nan_to_num(agents["contact_group_cohabitant"].cast(pl.Float64).to_numpy(), nan=1.0)
            hh = np.clip(np.rint(source), 1, 7)
        elif "discussion_group_cohabitant" in agents.columns:
            source = np.nan_to_num(agents["discussion_group_cohabitant"].cast(pl.Float64).to_numpy(), nan=1.0)
            hh = np.clip(np.rint(source), 1, 7)
        else:
            hh = np.ones(n_agents, dtype=np.float64)
        agents = agents.with_columns(pl.Series("household_size", hh.astype(np.int64)))

    # Norm channels must remain on their Likert bounds even when source
    # populations (e.g., cached survey/parquet artifacts) predate the split.
    agents = _attach_norm_channels(agents)
    agents = _attach_norm_share_hat(agents)

    return agents


# =========================================================================
# 3.  Full population initialisation
# =========================================================================


def initialise_population(
    n_agents: int,
    countries: list[str],
    anchor_system: AnchorSystem,
    survey_df: pl.DataFrame,
    seed: int = 42,
    country_shares: pl.DataFrame | None = None,
    heterogeneity_loadings: HeterogeneityLoadings | None = None,
    heterogeneity_bounds: HeterogeneityBounds = HeterogeneityBounds(),
    norm_threshold_beta_alpha: float = 2.0,
) -> pl.DataFrame:
    """Create the initial agent population via anchor-country resampling.

    Parameters
    ----------
    n_agents : int
        Total number of agents to create.
    countries : list[str]
        Country codes to distribute agents across.
    anchor_system : AnchorSystem
        Loaded anchor system.
    survey_df : pl.DataFrame
        Clean survey data for donor resampling.
    seed : int
        Random seed for reproducibility.
    heterogeneity_loadings : HeterogeneityLoadings or None
        Empirical PCA loadings used to build structural heterogeneity.
        If ``None``, loads the committed default artifact.
    heterogeneity_bounds : HeterogeneityBounds
        Trait bounds for structural heterogeneity mapping.
    norm_threshold_beta_alpha : float
        Symmetric Beta shape parameter used to initialise
        ``norm_threshold_theta`` on ``[theta_min, theta_max]``.

    Returns
    -------
    pl.DataFrame
        Agent DataFrame with all mutable, immutable, and outcome
        columns initialised from donor respondents.

    Notes
    -----
    Agents are distributed across countries proportionally to their
    representation in the anchor shares (summed across anchors).
    """
    rng = np.random.default_rng(seed)
    loadings = heterogeneity_loadings or load_heterogeneity_loadings()
    cdf_knots = _resolve_cdf_knots(loadings, survey_df)

    # Compute per-country agent counts proportional to total share
    shares = country_shares if country_shares is not None else anchor_system.country_shares
    country_total_share: dict[str, float] = {}
    for country in countries:
        ctry_shares = shares.filter(pl.col("country") == country)
        if ctry_shares.height > 0:
            country_total_share[country] = float(ctry_shares["share"].sum())
        else:
            country_total_share[country] = 1.0

    total = sum(country_total_share.values())
    country_counts: dict[str, int] = {}
    remaining = n_agents
    sorted_countries = sorted(countries)
    for i, country in enumerate(sorted_countries):
        if i == len(sorted_countries) - 1:
            country_counts[country] = remaining
        else:
            count = int(round(n_agents * country_total_share[country] / total))
            country_counts[country] = count
            remaining -= count

    logger.info(
        "Population distribution: {d}",
        d=", ".join(f"{c}={n}" for c, n in sorted(country_counts.items())),
    )

    # Sample anchors and donors per country
    all_anchor_ids: list[np.ndarray] = []
    all_countries: list[np.ndarray] = []

    for country, n_c in sorted(country_counts.items()):
        if n_c == 0:
            continue
        anchors = sample_anchor_for_country(
            anchor_system,
            country,
            n_c,
            rng,
            country_shares=shares,
        )
        all_anchor_ids.append(anchors)
        all_countries.append(np.full(n_c, country, dtype=object))

    anchor_ids = np.concatenate(all_anchor_ids)
    country_arr = np.concatenate(all_countries)

    # Ensure survey has row_id column
    if "row_id" not in survey_df.columns:
        survey_df = survey_df.with_row_index("row_id")

    # Sample donors
    agents = sample_donors(anchor_ids, country_arr, anchor_system, survey_df, rng)

    agents = ensure_agent_schema(agents, n_agents, anchor_ids=anchor_ids)
    agents = _attach_locality_id(agents)
    agents = _attach_norm_channels(agents)
    agents = _attach_structural_heterogeneity(
        agents,
        loadings,
        heterogeneity_bounds,
        cdf_knots,
        theta_beta_alpha=norm_threshold_beta_alpha,
        rng=rng,
    )
    agents = _attach_norm_share_hat(agents)
    agents = _attach_contact_scalars(agents)

    logger.info(
        "Initialised {n} agents ({cols} columns).",
        n=agents.height,
        cols=agents.width,
    )
    return agents
