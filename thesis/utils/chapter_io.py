"""Shared I/O helpers for Quarto thesis chapters."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl
from IPython.display import Markdown

from utils.table_utils import render_markdown_table


def read_csv_fast(path: str | Path) -> pl.DataFrame:
    """Read a CSV with a larger schema inference window for chapter outputs."""

    return pl.read_csv(path, infer_schema_length=10_000)


def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> Markdown:
    """Return a Markdown table from headers and rows."""

    return Markdown(render_markdown_table(headers, rows))
