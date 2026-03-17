from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from beam_abm.evaluation.choice._canonicalize_ids import _row_dataset_id


def samples_to_predictions_df(rows: list[dict[str, Any]], *, strategy: str) -> pd.DataFrame:
    def _extract_outcome(row: dict[str, Any]) -> str | None:
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}

        raw = row.get("outcome")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        raw = meta.get("outcome") if isinstance(meta, dict) else None
        if isinstance(raw, str) and raw.strip():
            return raw.strip()

        # Back-compat: some generators provide a single outcome via `outcomes` list.
        raw_outcomes = meta.get("outcomes") if isinstance(meta, dict) else None
        if raw_outcomes is None:
            raw_outcomes = row.get("outcomes")
        if isinstance(raw_outcomes, list) and len(raw_outcomes) == 1:
            only = raw_outcomes[0]
            if isinstance(only, str) and only.strip():
                return only.strip()

        rid = row.get("dataset_id") or row.get("id")
        if isinstance(rid, str) and rid.strip() and "__" in rid:
            return rid.split("__", 1)[0].strip() or None
        return None

    @functools.lru_cache(maxsize=4)
    def _load_outcome_types_from_column_types(path: str) -> dict[str, str]:
        p = Path(path)
        if not p.exists():
            return {}
        df = pd.read_csv(p, sep="\t")
        if "column" not in df.columns or "type" not in df.columns:
            return {}
        return {str(k): str(v).strip().lower() for k, v in df.set_index("column")["type"].to_dict().items()}

    def _default_column_types_path() -> str:
        override = os.getenv("LLM_ABM_COLUMN_TYPES_PATH")
        if override:
            return override
        root = Path(os.getenv("PROJECT_ROOT", Path.cwd())).resolve()
        return str(root / "empirical" / "specs" / "column_types.tsv")

    def _extract_outcome_type(row: dict[str, Any], *, outcome: str | None, y_true: object) -> str | None:
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}

        raw = row.get("outcome_type")
        if isinstance(raw, str) and raw.strip():
            return raw.strip().lower()
        raw = meta.get("outcome_type") if isinstance(meta, dict) else None
        if isinstance(raw, str) and raw.strip():
            return raw.strip().lower()

        if isinstance(meta, dict) and outcome:
            m = meta.get("outcome_type_by_outcome")
            if isinstance(m, dict):
                t = m.get(outcome)
                if isinstance(t, str) and t.strip():
                    return t.strip().lower()

        if outcome:
            types = _load_outcome_types_from_column_types(_default_column_types_path())
            t = types.get(outcome)
            if isinstance(t, str) and t.strip():
                return t.strip().lower()

        # Last-resort heuristic (kept minimal): infer binary for 0/1 labels.
        try:
            f = float(y_true)
        except (TypeError, ValueError):
            return None
        if np.isfinite(f) and f in (0.0, 1.0):
            return "binary"
        return None

    def _extract_y_true(row: dict[str, Any], *, outcome: str | None) -> object:
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if isinstance(meta, dict) and "y_true" in meta:
            return meta.get("y_true")
        if "y_true" in row:
            return row.get("y_true")
        if isinstance(meta, dict) and outcome:
            by_outcome = meta.get("y_true_by_outcome")
            if isinstance(by_outcome, dict) and outcome in by_outcome:
                return by_outcome.get(outcome)
        return None

    def _meta(row: dict[str, Any]) -> dict[str, Any]:
        m = row.get("metadata")
        return m if isinstance(m, dict) else {}

    out_rows: list[dict[str, Any]] = []
    for row in rows:
        dataset_id = _row_dataset_id(row)
        meta = _meta(row)
        outcome = _extract_outcome(row)
        country = meta.get("country") or row.get("country")
        y_true = _extract_y_true(row, outcome=outcome)

        outcome_type = _extract_outcome_type(row, outcome=outcome, y_true=y_true)

        if outcome is None or outcome_type is None:
            rid = row.get("id") or row.get("dataset_id")
            raise ValueError(
                "Samples row missing outcome/outcome_type "
                f"(id={rid!r}, derived_outcome={outcome!r}, derived_outcome_type={outcome_type!r}). "
                "Expected `outcome`/`outcome_type` (top-level or metadata), or a single-item `outcomes` list, "
                "or an id like '<outcome>__<...>'. For type inference, set env var LLM_ABM_COLUMN_TYPES_PATH "
                "to a column_types.tsv with 'column' and 'type' columns."
            )
        if y_true is None:
            raise ValueError("Samples row missing y_true")

        samples = row.get("samples")
        if not isinstance(samples, list):
            samples = []
        preds: list[float] = []
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            val = sample.get("prediction_scalar")
            if val is None:
                parsed = sample.get("parsed")
                if isinstance(parsed, dict):
                    val = parsed.get("prediction")
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fval):
                preds.append(fval)

        y_pred = float(np.mean(preds)) if preds else float("nan")
        out_rows.append(
            {
                "dataset_id": dataset_id,
                "model": row.get("model"),
                "outcome": str(outcome),
                "outcome_type": str(outcome_type).lower(),
                "prompt_family": row.get("prompt_family") or strategy,
                "strategy": strategy,
                "y_true": float(y_true),
                "y_pred": y_pred,
                "country": "" if country is None else str(country),
            }
        )
    return pd.DataFrame(out_rows)
