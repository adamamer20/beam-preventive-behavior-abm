"""Rebuild heterogeneity CDF knots from the survey population."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from beam_abm.abm.config import SimulationConfig
from beam_abm.abm.io import load_survey_data
from beam_abm.abm.population import build_heterogeneity_cdf_knots, load_heterogeneity_loadings


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate heterogeneity CDF knots from survey data.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path (defaults to heterogeneity_loadings.json).",
    )
    args = parser.parse_args()

    cfg = SimulationConfig()
    survey_df = load_survey_data(cfg.survey_csv, cfg.countries)
    loadings = load_heterogeneity_loadings()

    cdf_knots = build_heterogeneity_cdf_knots(
        survey_df,
        loadings,
        quantiles=loadings.cdf_knots.quantiles,
        source="survey",
    )

    target_path = args.output
    if target_path is None:
        target_path = (
            Path(__file__).resolve().parents[1] / "src" / "beam_abm" / "abm" / "assets" / "heterogeneity_loadings.json"
        )

    raw = json.loads(target_path.read_text(encoding="utf-8"))
    raw.setdefault("cdf_knots", {})
    raw["cdf_knots"]["source"] = cdf_knots.source
    raw["cdf_knots"]["quantiles"] = list(cdf_knots.quantiles)
    raw["cdf_knots"]["scores"] = {key: list(values) for key, values in sorted(cdf_knots.score_knots.items())}

    target_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    logger.info("Wrote heterogeneity CDF knots to {p}", p=target_path)


if __name__ == "__main__":
    main()
