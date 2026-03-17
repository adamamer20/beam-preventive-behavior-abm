"""Tests for ABM I/O helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from beam_abm.abm import io as io_mod


def test_load_survey_data_retries_after_shape_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Loader retries with lenient CSV options after schema mismatch."""
    csv_path = tmp_path / "survey.csv"
    csv_path.write_text("country,health_good\nIT,1.0\n", encoding="utf-8")

    calls: list[dict[str, Any]] = []

    def _fake_read_csv(path: Path, **kwargs: Any) -> pl.DataFrame:
        calls.append(kwargs)
        if len(calls) == 1:
            raise pl.exceptions.ShapeError("unable to vstack")
        return pl.DataFrame({"country": ["IT"], "health_good": [1.0]})

    monkeypatch.setattr(io_mod.pl, "read_csv", _fake_read_csv)
    df = io_mod.load_survey_data(csv_path)

    assert len(calls) == 2
    assert calls[0] == {"infer_schema_length": 5000}
    assert calls[1] == {
        "infer_schema_length": 5000,
        "ignore_errors": True,
        "truncate_ragged_lines": True,
    }
    assert "health_poor" in df.columns
    assert "row_id" in df.columns


def test_load_survey_data_sanitizes_bom_and_nul_headers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Loader strips BOM/NUL artifacts from column names."""
    csv_path = tmp_path / "survey.csv"
    csv_path.write_text("country,health_good\nIT,1.0\n", encoding="utf-8")

    bad_col = "\x00\x00region_ftime_wik"
    bom_col = "\ufefftime_wikipedia"

    def _fake_read_csv(path: Path, **kwargs: Any) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "country": ["IT"],
                "health_good": [1.0],
                bad_col: [2.0],
                bom_col: [3.0],
            }
        )

    monkeypatch.setattr(io_mod.pl, "read_csv", _fake_read_csv)
    df = io_mod.load_survey_data(csv_path)

    assert bad_col not in df.columns
    assert bom_col not in df.columns
    assert "region_ftime_wik" in df.columns
    assert "time_wikipedia" in df.columns
