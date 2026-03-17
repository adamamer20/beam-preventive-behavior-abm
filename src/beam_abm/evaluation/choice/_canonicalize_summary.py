from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def write_strategies_summary(
    *,
    outcome_dir: Path,
    model: str,
    strategy_rows: list[dict[str, Any]],
) -> None:
    out_dir = outcome_dir / model / "_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "strategies_summary.csv"
    if not strategy_rows:
        out_path.write_text("", encoding="utf-8")
        return
    _write_csv(out_path, strategy_rows)


def write_models_strategies_summary(*, outcome_dir: Path, rows: list[dict[str, Any]]) -> None:
    out_dir = outcome_dir / "_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "models_strategies_summary.csv", rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)
