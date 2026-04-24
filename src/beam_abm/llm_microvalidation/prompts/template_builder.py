"""Shared prompt templates for microvalidation prompt tournaments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .labels import PromptLabelsMixin
from .renderers import PromptRendererMixin


@dataclass(slots=True)
class PromptTemplateBuilder(PromptRendererMixin, PromptLabelsMixin):
    display_labels: dict[str, str]
    display_groups: dict[str, dict[str, Any]]
    clean_questions: dict[str, str]
    clean_answers: dict[str, str]
    encodings: dict[str, Any]
    by_id: dict[str, dict[str, Any]]
    column_types: dict[str, str]
    possible_bounds: dict[str, tuple[float, float]]
    ordinal_endpoints: dict[str, tuple[str, str]]

    _memo_base: dict[str, str | None] | None = None
    _visiting: set[str] | None = None
    _income_quintile_cache: dict[str, list[float]] | None = None

    @staticmethod
    def fmt_num(x: float) -> str:
        if float(x).is_integer():
            return str(int(x))
        return f"{x:.3f}".rstrip("0").rstrip(".")

    def _as_numeric(self, val: object) -> float | None:
        if val is None:
            return None
        if isinstance(val, bool | np.bool_):
            return 1.0 if bool(val) else 0.0
        if isinstance(val, int | float | np.integer | np.floating) and val == val:
            return float(val)
        if isinstance(val, str):
            s = val.strip().lower()
            if s in {"true", "false"}:
                return 1.0 if s == "true" else 0.0
        try:
            num = float(str(val))
            # Treat NaN or infinite values as missing
            if not math.isfinite(num):
                return None
            return num
        except Exception:
            return None

    def _base_id(self, col_id: str) -> str | None:
        if self._memo_base is None:
            self._memo_base = {}
        if self._visiting is None:
            self._visiting = set()
        cid = col_id.strip()
        if not cid:
            return None
        if cid in self._memo_base:
            return self._memo_base[cid]
        if cid in self._visiting:
            return None
        self._visiting.add(cid)
        item = self.by_id.get(cid)
        if item is None:
            self._visiting.remove(cid)
            self._memo_base[cid] = None
            return None
        transforms = item.get("transformations") or []
        if not isinstance(transforms, list):
            transforms = []
        for tr in transforms:
            if not isinstance(tr, dict) or tr.get("type") != "renamed":
                continue
            src = tr.get("rename_from")
            if isinstance(src, str) and src.strip():
                base = self._base_id(src)
                self._visiting.remove(cid)
                self._memo_base[cid] = base
                return base
        self._visiting.remove(cid)
        self._memo_base[cid] = cid
        return cid

    def _encoding_spec_for(self, col_id: str) -> object | None:
        base = self._base_id(col_id)
        if base is None:
            return None
        item = self.by_id.get(base)
        if item is None:
            return None
        transforms = item.get("transformations") or []
        if not isinstance(transforms, list):
            return None
        for tr in transforms:
            if not isinstance(tr, dict) or tr.get("type") != "encoding":
                continue
            return tr.get("encoding")
        return None

    def _mapping_from_encoding(self, enc: object | None) -> dict[str, object] | None:
        if enc is None:
            return None
        if isinstance(enc, dict):
            return enc
        if isinstance(enc, str):
            s = enc.strip()
            m = self.encodings.get(s)
            return m if isinstance(m, dict) else None
        return None

    def _normalize_value(self, col: str, val: object) -> str:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "missing"
        mapping = self._mapping_from_encoding(self._encoding_spec_for(col))
        if mapping:
            num_val = self._as_numeric(val)
            key = str(val)
            mapped = mapping.get(key)
            if mapped is None and key.lower() != key:
                mapped = mapping.get(key.lower())
            if mapped is None and key.upper() != key:
                mapped = mapping.get(key.upper())
            if mapped is None and "$DEFAULT" in mapping:
                mapped = mapping.get("$DEFAULT")
            if mapped is None:
                if num_val is not None:
                    return self.fmt_num(num_val)
                return "missing"
            num = self._as_numeric(mapped)
            if num is not None:
                return self.fmt_num(num)
            return str(mapped)
        num = self._as_numeric(val)
        if num is not None:
            return self.fmt_num(num)
        return str(val)


__all__ = ["PromptTemplateBuilder"]
