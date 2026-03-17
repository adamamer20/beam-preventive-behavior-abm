"""Build paired-profile records from a unified design table.

Input: design table (parquet/csv/jsonl) with low/high rows.
Output: JSONL with one row per (id, target) containing:
- attributes_low / attributes_high
- metadata about the pairing (and original point metadata)

This output is consumed by `generate_prompts.py --strategy paired_profile`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    return _read_jsonl(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = _read_rows(Path(args.in_path))
    rng = np.random.default_rng(args.seed)

    meta_cols = {
        "id",
        "eval_id",
        "row_id",
        "anchor_id",
        "slice",
        "rank",
        "dist",
        "target",
        "lever",
        "grid_point",
        "grid_value",
        "q25",
        "q50",
        "q75",
        "delta_x",
        "delta_z",
        "stratum_level",
        "stratum_key",
        "n",
        "outcome",
    }

    meta_keep = [
        "target",
        "outcome",
        "grid_point",
        "grid_value",
        "delta_x",
        "delta_z",
        "stratum_level",
        "stratum_key",
    ]

    by_key: dict[tuple[object, object, object], dict[str, dict]] = {}
    for r in rows:
        rid = r.get("id")
        tgt = r.get("target")
        outcome = r.get("outcome")
        gp = str(r.get("grid_point"))
        by_key.setdefault((rid, tgt, outcome), {})[gp] = r

    out: list[dict] = []
    for (rid, tgt, outcome), m in by_key.items():
        if "low" not in m or "high" not in m:
            continue
        low = m["low"]
        high = m["high"]

        attrs_low = {k: v for k, v in low.items() if k not in meta_cols}
        attrs_high = {k: v for k, v in high.items() if k not in meta_cols}
        meta_low = {k: low.get(k) for k in meta_keep if k in low}
        meta_high = {k: high.get(k) for k in meta_keep if k in high}

        out.append(
            {
                "id": rid,
                "target": tgt,
                "outcome": outcome,
                "attributes_low": attrs_low,
                "attributes_high": attrs_high,
                "metadata_low": meta_low,
                "metadata_high": meta_high,
                "seed": int(rng.integers(0, 2**31 - 1)),
            }
        )

    _write_jsonl(Path(args.out_path), out)


if __name__ == "__main__":
    main()
