"""XGBoost surrogate belief-update kernel.

Loads a trained surrogate bundle and predicts construct deltas
conditioned on current agent state and signal intensity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from xgboost import XGBRegressor

from beam_abm.abm.state_schema import CONSTRUCT_BOUNDS

__all__ = ["BeliefUpdateSurrogate"]


@dataclass(slots=True)
class BeliefUpdateSurrogate:
    feature_columns: tuple[str, ...]
    model_files: dict[str, dict[str, Path]]
    fallback_mean_delta: dict[str, dict[str, float]]
    models: dict[tuple[str, str], XGBRegressor]

    @classmethod
    def from_bundle_dir(cls, bundle_dir: Path) -> BeliefUpdateSurrogate:
        manifest_path = bundle_dir / "manifest.json"
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))

        feature_columns = tuple(str(col) for col in raw.get("feature_columns", []))
        model_files_raw = raw.get("model_files", {})
        fallback_mean_delta = raw.get("fallback_mean_delta", {})

        model_files: dict[str, dict[str, Path]] = {}
        models: dict[tuple[str, str], XGBRegressor] = {}

        if isinstance(model_files_raw, dict):
            for signal_id, by_construct in model_files_raw.items():
                if not isinstance(by_construct, dict):
                    continue
                signal_key = str(signal_id)
                model_files[signal_key] = {}
                for construct, rel_path in by_construct.items():
                    model_path = (bundle_dir / str(rel_path)).resolve()
                    model_files[signal_key][str(construct)] = model_path
                    model = XGBRegressor()
                    model.load_model(model_path)
                    models[(signal_key, str(construct))] = model

        return cls(
            feature_columns=feature_columns,
            model_files=model_files,
            fallback_mean_delta={
                str(signal): {str(construct): float(delta) for construct, delta in constructs.items()}
                for signal, constructs in (fallback_mean_delta.items() if isinstance(fallback_mean_delta, dict) else [])
                if isinstance(constructs, dict)
            },
            models=models,
        )

    def _feature_matrix(self, agents: pl.DataFrame, *, intensity: float) -> np.ndarray:
        n = agents.height
        columns: list[np.ndarray] = []
        for feature in self.feature_columns:
            if feature == "signal_intensity":
                columns.append(np.full(n, float(intensity), dtype=np.float64))
                continue

            if feature in agents.columns:
                values = agents[feature].to_numpy()
                try:
                    arr = values.astype(np.float64)
                except (TypeError, ValueError):
                    arr = np.zeros(n, dtype=np.float64)
            else:
                arr = np.zeros(n, dtype=np.float64)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            columns.append(arr)

        if not columns:
            return np.zeros((n, 0), dtype=np.float64)
        return np.column_stack(columns)

    def apply(
        self,
        agents: pl.DataFrame,
        *,
        signal_id: str,
        intensity: float,
        target_mask: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        signal_key = str(signal_id)
        if signal_key not in self.model_files and signal_key not in self.fallback_mean_delta:
            return {}

        x = self._feature_matrix(agents, intensity=float(intensity))
        updates: dict[str, np.ndarray] = {}

        constructs = set()
        constructs.update(self.model_files.get(signal_key, {}).keys())
        constructs.update(self.fallback_mean_delta.get(signal_key, {}).keys())

        for construct in sorted(constructs):
            if construct not in agents.columns:
                continue

            current = agents[construct].to_numpy().astype(np.float64)
            updated = current.copy()

            model = self.models.get((signal_key, construct))
            if model is not None:
                delta = model.predict(x).astype(np.float64)
            else:
                fallback = self.fallback_mean_delta.get(signal_key, {}).get(construct, 0.0)
                delta = np.full(agents.height, float(fallback) * float(intensity), dtype=np.float64)

            if target_mask is None:
                updated += delta
            else:
                updated[target_mask] = updated[target_mask] + delta[target_mask]

            lo, hi = CONSTRUCT_BOUNDS.get(construct, (-np.inf, np.inf))
            updates[construct] = np.clip(updated, lo, hi)

        return updates
