"""Shared JSONL and tabular row I/O helpers for evaluation workflows."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_jsonl(path: Path, *, dicts_only: bool = False, skip_invalid: bool = False) -> list[Any]:
    """Read JSONL records from disk."""

    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if skip_invalid:
                    continue
                raise
            if dicts_only and not isinstance(row, dict):
                continue
            rows.append(row)
    return rows


def write_jsonl(
    path: Path,
    rows: Iterable[Any],
    *,
    json_default: Callable[[Any], Any] | None = None,
) -> None:
    """Write records to JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def numpy_json_default(obj: Any) -> Any:
    """Serialize common NumPy scalar and array values."""

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def read_rows(path: Path, *, skip_invalid_jsonl: bool = False) -> list[dict[str, Any]]:
    """Read parquet/csv/jsonl inputs into a list of row dicts."""

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    if suffix == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    return read_jsonl(path, dicts_only=True, skip_invalid=skip_invalid_jsonl)


def read_frame(path: Path, *, skip_invalid_jsonl: bool = False) -> pd.DataFrame:
    """Read parquet/csv/jsonl inputs into a DataFrame."""

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.DataFrame(read_jsonl(path, dicts_only=True, skip_invalid=skip_invalid_jsonl))
    raise ValueError(f"Unsupported input format: {path}")
