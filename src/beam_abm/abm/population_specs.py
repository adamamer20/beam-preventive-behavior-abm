"""Population and society specifications for ABM sampling."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import polars as pl
from loguru import logger
from pydantic import BaseModel, Field, model_validator

from beam_abm.abm.data_contracts import AnchorSystem
from beam_abm.abm.population import (
    HeterogeneityBounds,
    _attach_contact_scalars,
    _attach_locality_id,
    _attach_structural_heterogeneity,
    build_heterogeneity_cdf_knots,
    ensure_agent_schema,
    initialise_population,
    load_heterogeneity_loadings,
)

__all__ = [
    "ENGINE_COLUMN_MAP",
    "PopulationSpec",
    "SocietySpec",
    "SOCIETY_LIBRARY",
    "get_society",
    "reweight_anchor_shares",
    "build_population",
]


ENGINE_COLUMN_MAP: dict[str, str] = {
    "trust": "institutional_trust_avg",
    "constraints": "fivec_low_constraints",
    "norms": "social_norms_vax_avg",
}


class PopulationSpec(BaseModel):
    """Specification for population construction."""

    model_config: ClassVar[dict] = {  # type: ignore[misc]
        "extra": "forbid",
        "frozen": False,
        "validate_default": True,
    }

    mode: Literal["survey", "resample_on_manifold", "parquet"] = "resample_on_manifold"
    n_agents: int | None = Field(default=None, gt=0)
    seed: int = Field(default=42)
    stratify_by_country: bool = Field(default=True)
    parquet_path: Path | None = None

    @model_validator(mode="after")
    def _validate_parquet(self) -> PopulationSpec:
        if self.mode == "parquet" and self.parquet_path is None:
            raise ValueError("PopulationSpec.parquet_path is required when mode='parquet'.")
        return self


class SocietySpec(BaseModel):
    """Specification for counterfactual society weighting."""

    model_config: ClassVar[dict] = {  # type: ignore[misc]
        "extra": "forbid",
        "frozen": False,
        "validate_default": True,
        "populate_by_name": True,
    }

    name: str
    description: str = ""
    type: Literal["baseline", "tilt", "polarize"] = "baseline"
    engine: Literal["trust", "constraints", "norms"] | None = None
    lambda_: float = Field(default=0.0, alias="lambda")
    country_specific: bool = False

    @property
    def engine_column(self) -> str | None:
        if self.engine is None:
            return None
        return ENGINE_COLUMN_MAP[self.engine]


SOCIETY_LIBRARY: dict[str, SocietySpec] = {
    "S0_survey": SocietySpec(
        name="Survey society",
        description="Survey population baseline.",
        type="baseline",
    ),
    "S1_high_trust": SocietySpec(
        name="High-trust society",
        description="Reweight anchors toward higher institutional trust.",
        type="tilt",
        engine="trust",
        lambda_=0.6,
    ),
    "S2_low_trust": SocietySpec(
        name="Low-trust society",
        description="Reweight anchors toward lower institutional trust.",
        type="tilt",
        engine="trust",
        lambda_=-0.6,
    ),
    "S3_high_constraints": SocietySpec(
        name="High-constraint society",
        description="Reweight anchors toward higher perceived constraints.",
        type="tilt",
        engine="constraints",
        lambda_=-0.6,
    ),
    "S4_high_norms": SocietySpec(
        name="High-norm society",
        description="Reweight anchors toward stronger pro-vaccine norms.",
        type="tilt",
        engine="norms",
        lambda_=0.6,
    ),
    "S5_polarized_trust": SocietySpec(
        name="Polarized-trust society",
        description="Overweight extreme institutional trust anchors.",
        type="polarize",
        engine="trust",
        lambda_=0.6,
    ),
}


def get_society(key: str) -> SocietySpec:
    """Retrieve a society spec by key."""
    if key not in SOCIETY_LIBRARY:
        msg = f"Society '{key}' not found. Available: {sorted(SOCIETY_LIBRARY)}"
        raise KeyError(msg)
    return SOCIETY_LIBRARY[key]


def _compute_anchor_scores(
    anchor_system: AnchorSystem,
    survey_df: pl.DataFrame,
    *,
    engine_column: str,
    country_specific: bool,
) -> pl.DataFrame:
    if engine_column not in survey_df.columns:
        msg = f"Engine column '{engine_column}' not in survey data."
        raise ValueError(msg)

    memberships = anchor_system.memberships.select(["row_id", "anchor_id", "weight"])
    joined = memberships.join(
        survey_df.select(["row_id", "country", engine_column]),
        on="row_id",
        how="left",
    ).drop_nulls(engine_column)

    if country_specific:
        scores = (
            joined.group_by(["anchor_id", "country"])
            .agg((pl.col(engine_column) * pl.col("weight")).sum() / pl.col("weight").sum())
            .rename({engine_column: "score"})
        )
    else:
        scores = (
            joined.group_by(["anchor_id"])
            .agg((pl.col(engine_column) * pl.col("weight")).sum() / pl.col("weight").sum())
            .rename({engine_column: "score"})
        )

    return scores


def reweight_anchor_shares(
    anchor_system: AnchorSystem,
    survey_df: pl.DataFrame,
    society: SocietySpec,
) -> pl.DataFrame:
    """Reweight anchor-country shares for counterfactual societies."""
    if society.type == "baseline":
        return anchor_system.country_shares

    engine_column = society.engine_column
    if engine_column is None:
        raise ValueError("SocietySpec.engine must be set for non-baseline societies.")

    scores = _compute_anchor_scores(
        anchor_system,
        survey_df,
        engine_column=engine_column,
        country_specific=society.country_specific,
    )

    shares = anchor_system.country_shares
    if society.country_specific:
        merged = shares.join(scores, on=["anchor_id", "country"], how="left")
    else:
        merged = shares.join(scores, on=["anchor_id"], how="left")

    if merged["score"].null_count() > 0:
        raise ValueError("Missing anchor scores after merge; check anchor system coverage.")

    score_expr = pl.col("score")
    if society.type == "polarize":
        score_expr = score_expr.abs()

    raw = (pl.col("share") * (pl.lit(society.lambda_) * score_expr).exp()).alias("raw_share")
    adjusted = merged.with_columns(raw)
    adjusted = adjusted.with_columns(
        (pl.col("raw_share") / pl.col("raw_share").sum().over("country")).alias("share")
    ).drop("raw_share")

    logger.info(
        "Reweighted anchor shares for society '{name}' (type={t}, engine={e}).",
        name=society.name,
        t=society.type,
        e=society.engine,
    )
    return adjusted


def _sample_survey_rows(
    survey_df: pl.DataFrame,
    *,
    n_agents: int,
    seed: int,
    stratify_by_country: bool,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    if n_agents == survey_df.height:
        return survey_df

    def _sample_rows(df: pl.DataFrame, n: int) -> pl.DataFrame:
        row_ids = df["row_id"].to_numpy()
        replace = n > len(row_ids)
        picked = rng.choice(row_ids, size=n, replace=replace)
        return pl.DataFrame({"row_id": picked}).join(survey_df, on="row_id", how="left")

    if not stratify_by_country:
        return _sample_rows(survey_df, n_agents)

    counts: dict[str, int] = {}
    total = survey_df.height
    remaining = n_agents
    countries = sorted(survey_df.get_column("country").unique().to_list())
    for i, country in enumerate(countries):
        if i == len(countries) - 1:
            counts[str(country)] = remaining
        else:
            share = float(survey_df.filter(pl.col("country") == country).height) / float(total)
            count = int(round(n_agents * share))
            counts[str(country)] = count
            remaining -= count

    samples: list[pl.DataFrame] = []
    for country, count in counts.items():
        if count <= 0:
            continue
        country_df = survey_df.filter(pl.col("country") == country)
        samples.append(_sample_rows(country_df, count))

    return pl.concat(samples, how="vertical") if samples else survey_df


def _attach_anchor_ids(
    sampled: pl.DataFrame,
    anchor_system: AnchorSystem,
) -> pl.DataFrame:
    """Attach anchor IDs from membership table using max membership weight."""
    if "anchor_id" in sampled.columns:
        return sampled
    memberships = anchor_system.memberships
    required = {"row_id", "anchor_id", "weight"}
    if not required.issubset(set(memberships.columns)):
        return sampled
    dominant = (
        memberships.select(["row_id", "anchor_id", "weight"])
        .sort(["row_id", "weight", "anchor_id"], descending=[False, True, False])
        .unique(subset=["row_id"], keep="first")
        .select(["row_id", "anchor_id"])
    )
    enriched = sampled.join(dominant, on="row_id", how="left")
    missing = int(enriched["anchor_id"].null_count()) if "anchor_id" in enriched.columns else enriched.height
    if missing > 0:
        logger.warning(
            "Anchor assignment missing for {n_missing}/{n_total} sampled rows.",
            n_missing=missing,
            n_total=enriched.height,
        )
    return enriched


def build_population(
    population: PopulationSpec,
    society: SocietySpec | None,
    anchor_system: AnchorSystem,
    survey_df: pl.DataFrame,
    *,
    default_n_agents: int | None = None,
    norm_threshold_beta_alpha: float = 2.0,
) -> pl.DataFrame:
    """Build population using population and society specifications."""

    def _enrich_for_runtime(df: pl.DataFrame) -> pl.DataFrame:
        enriched = ensure_agent_schema(df, df.height)
        if "locality_id" not in enriched.columns:
            enriched = _attach_locality_id(enriched)
        if "social_susceptibility_lambda" not in enriched.columns:
            rng = np.random.default_rng(population.seed)
            loadings = load_heterogeneity_loadings()
            cdf_knots = build_heterogeneity_cdf_knots(
                survey_df,
                loadings,
                quantiles=loadings.cdf_knots.quantiles,
            )
            enriched = _attach_structural_heterogeneity(
                enriched,
                loadings,
                HeterogeneityBounds(),
                cdf_knots,
                theta_beta_alpha=norm_threshold_beta_alpha,
                rng=rng,
            )
        if "activity_score" not in enriched.columns:
            enriched = _attach_contact_scalars(enriched)
        return enriched

    if population.mode == "parquet":
        df = pl.read_parquet(population.parquet_path)
        return _enrich_for_runtime(df)

    n_agents = population.n_agents or default_n_agents or survey_df.height

    if population.mode == "survey":
        sampled = _sample_survey_rows(
            survey_df,
            n_agents=n_agents,
            seed=population.seed,
            stratify_by_country=population.stratify_by_country,
        )
        sampled = _attach_anchor_ids(sampled, anchor_system)
        if sampled.height != n_agents:
            logger.warning(
                "Survey sampling returned {n} rows, expected {exp}.",
                n=sampled.height,
                exp=n_agents,
            )
        return _enrich_for_runtime(sampled)

    if population.mode != "resample_on_manifold":
        raise ValueError(f"Unsupported population mode: {population.mode}")

    society_spec = society or SOCIETY_LIBRARY["S0_survey"]
    shares = reweight_anchor_shares(anchor_system, survey_df, society_spec)
    return initialise_population(
        n_agents=n_agents,
        countries=sorted(survey_df.get_column("country").unique().to_list()),
        anchor_system=anchor_system,
        survey_df=survey_df,
        seed=population.seed,
        country_shares=shares,
        norm_threshold_beta_alpha=norm_threshold_beta_alpha,
    )
