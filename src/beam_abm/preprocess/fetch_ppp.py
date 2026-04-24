from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl
import requests
from beartype import beartype

from beam_abm.common.logging import get_logger
from beam_abm.settings import PREPROCESSING_OUTPUT_DIR

API_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/prc_ppp_ind"
DEFAULT_COUNTRIES = ("IT", "FR", "DE", "ES", "HU", "UK")
DEFAULT_YEAR = 2024
DEFAULT_NA_ITEM = "PPP_EU27_2020"
DEFAULT_PPP_CATEGORY = "GDP"

logger = get_logger(__name__)


@beartype
def _build_params(year: int, countries: Sequence[str], na_item: str, ppp_category: str) -> list[tuple[str, str]]:
    params: list[tuple[str, str]] = [
        ("time", str(year)),
        ("na_item", na_item),
        ("ppp_cat", ppp_category),
    ]
    for code in countries:
        params.append(("geo", code))
    return params


@beartype
def _compute_multipliers(sizes: Sequence[int]) -> list[int]:
    multipliers: list[int] = []
    for idx in range(len(sizes)):
        product = 1
        for size in sizes[idx + 1 :]:
            product *= size
        multipliers.append(product)
    return multipliers


@beartype
def _linear_index(
    coords: Mapping[str, int],
    multipliers: Sequence[int],
    dimension_ids: Sequence[str],
) -> int:
    return int(sum(coords[dim] * multipliers[pos] for pos, dim in enumerate(dimension_ids)))


@beartype
def _extract_rows(payload: dict[str, Any], countries: Sequence[str], year: int) -> list[dict[str, Any]]:
    dimension_ids: list[str] = payload["id"]
    sizes: list[int] = payload["size"]
    dimensions: Mapping[str, Any] = payload["dimension"]

    multipliers = _compute_multipliers(sizes)
    coords: dict[str, int] = {}
    for dim in dimension_ids:
        category_index = dimensions[dim]["category"]["index"]
        if dim == "geo":
            coords[dim] = 0
            continue
        if dim == "time":
            year_key = str(year)
            if year_key not in category_index:
                raise ValueError(f"Year {year} is not available in dataset response.")
            coords[dim] = category_index[year_key]
        else:
            first_index = next(iter(category_index.values()))
            coords[dim] = first_index

    geo_index_map: Mapping[str, int] = dimensions["geo"]["category"]["index"]
    geo_labels: Mapping[str, str] = dimensions["geo"]["category"]["label"]
    na_item_code = next(iter(dimensions["na_item"]["category"]["index"].keys()))
    na_item_label = dimensions["na_item"]["category"]["label"].get(na_item_code, na_item_code)
    ppp_code = next(iter(dimensions["ppp_cat"]["category"]["index"].keys()))
    ppp_label = dimensions["ppp_cat"]["category"]["label"].get(ppp_code, ppp_code)
    freq_code = next(iter(dimensions["freq"]["category"]["index"].keys()))
    freq_label = dimensions["freq"]["category"]["label"].get(freq_code, freq_code)

    values = payload.get("value", {})
    status_map = payload.get("status", {})
    updated_at = payload.get("updated")
    dataset_id = payload.get("extension", {}).get("id", "PRC_PPP_IND")

    rows: list[dict[str, Any]] = []
    for code in countries:
        if code not in geo_index_map:
            raise ValueError(f"Country code '{code}' is missing from the Eurostat response.")

        coords["geo"] = geo_index_map[code]
        linear_idx = str(_linear_index(coords, multipliers, dimension_ids))
        value = values.get(linear_idx)
        status = status_map.get(linear_idx)

        rows.append(
            {
                "country_code": code,
                "country_name": geo_labels.get(code, code),
                "year": year,
                "frequency": freq_code,
                "frequency_label": freq_label,
                "na_item": na_item_code,
                "na_item_label": na_item_label,
                "ppp_category": ppp_code,
                "ppp_category_label": ppp_label,
                "ppp_value": value,
                "observation_status": status,
                "last_updated": updated_at,
                "source_dataset": dataset_id,
            }
        )

    return rows


@beartype
def fetch_ppp(
    year: int,
    countries: Sequence[str],
    na_item: str = DEFAULT_NA_ITEM,
    ppp_category: str = DEFAULT_PPP_CATEGORY,
) -> pl.DataFrame:
    params = _build_params(year, countries, na_item, ppp_category)
    logger.info("Requesting Eurostat PPP data for year {} and countries {}", year, countries)
    response = requests.get(API_URL, params=params, timeout=30)

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        message = exc.response.text if exc.response is not None else str(exc)
        logger.error("Eurostat request failed: {}", message)
        raise

    payload: dict[str, Any] = response.json()
    rows = _extract_rows(payload, countries, year)
    logger.info("Retrieved {} PPP observations", len(rows))
    return pl.DataFrame(rows)


@beartype
def _normalize_countries(codes: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for code in codes:
        cleaned = code.strip().upper()
        if not cleaned or cleaned in seen:
            continue
        normalized.append(cleaned)
        seen.add(cleaned)
    if not normalized:
        raise ValueError("At least one valid country code must be provided.")
    return tuple(normalized)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Purchasing Power Parity (PPP) data from Eurostat and persist it as CSV.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help="Reference year to download (default: %(default)s).",
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        default=list(DEFAULT_COUNTRIES),
        metavar="CODE",
        help="ISO alpha-2 country codes to download (default: %(default)s).",
    )
    parser.add_argument(
        "--na-item",
        default=DEFAULT_NA_ITEM,
        help="Eurostat NA_ITEM code to request (default: %(default)s).",
    )
    parser.add_argument(
        "--ppp-category",
        default=DEFAULT_PPP_CATEGORY,
        help="PPP analytical category to request (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the CSV file. Defaults to PREPROCESSING_OUTPUT_DIR/eurostat_ppp_<year>.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    countries = _normalize_countries(args.countries)
    df = fetch_ppp(
        year=args.year,
        countries=countries,
        na_item=args.na_item,
        ppp_category=args.ppp_category,
    )

    output_path = args.output or Path(PREPROCESSING_OUTPUT_DIR) / f"eurostat_ppp_{args.year}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_path)
    logger.success("Saved Eurostat PPP data to {}", output_path)


if __name__ == "__main__":
    main()
