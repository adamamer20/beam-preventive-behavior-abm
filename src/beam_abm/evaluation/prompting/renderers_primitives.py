"""Prompt rendering primitives: scales, attributes, groups, and helpers."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from beam_abm.empirical.missingness import SPECIAL_STRUCTURAL_PREFIXES


class PromptRendererPrimitivesMixin:
    def maybe_survey_context(self, attrs: dict[str, object]) -> str | None:
        for key in ("country", "Country", "country_name"):
            if key in attrs and attrs[key] is not None:
                country_label = self._country_label(attrs[key])
                return f"Survey context: Country={country_label}, Year=2024"
        return None

    def _maybe_label_numeric(self, col: str, val: object) -> str:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "missing"
        if str(self.column_types.get(col, "")).strip().lower() == "binary":
            b = self._as_bool(val)
            if b is None:
                return "missing"
            return "Yes" if b else "No"
        raw = self._normalize_value(col, val)
        num = self._as_numeric(raw)
        if num is None:
            return raw
        return raw

    def _scale_descriptor(self, col: str) -> tuple[tuple[object, ...], str] | None:
        if str(self.column_types.get(col, "")).strip().lower() == "binary":
            return None
        if col == "M_education":
            return None
        if col == "covax_legitimacy_scepticism_idx":
            lo_s, hi_s = "0", "1"
            low_lab = "no scepticism reasons endorsed"
            high_lab = "endorsed all listed scepticism reasons"
            key = (lo_s, hi_s, low_lab, high_lab)
            label = f"{lo_s}–{hi_s} ({lo_s} = {low_lab}, {hi_s} = {high_lab})"
            return key, label

        def _normalize_scale_label(label: str) -> str:
            cleaned = label.strip().lower().replace("’", "").replace("'", "").replace("–", "-").replace("—", "-")
            return " ".join(cleaned.split())

        def _canonical_endpoint(label: str) -> str:
            norm = _normalize_scale_label(label)
            synonyms = {
                "strongly disagree": "strongly disagree",
                "strongly agree": "strongly agree",
                "not at all": "not at all",
                "completely": "extremely",
                "extremely": "extremely",
                "not at all relevant": "not at all relevant",
                "extremely relevant": "extremely relevant",
            }
            return synonyms.get(norm, norm)

        def _canonical_scale_label(
            lo_s: str,
            hi_s: str,
            low_raw: str,
            high_raw: str,
        ) -> tuple[tuple[object, ...], str]:
            low_norm = _canonical_endpoint(low_raw)
            high_norm = _canonical_endpoint(high_raw)
            key = (lo_s, hi_s, low_norm, high_norm)
            canonical_labels = {
                ("1", "7", "strongly disagree", "strongly agree"): ("Strongly disagree", "Strongly agree"),
                ("1", "6", "strongly disagree", "strongly agree"): ("Strongly disagree", "Strongly agree"),
                ("1", "6", "not at all relevant", "extremely relevant"): (
                    "Not at all relevant",
                    "Extremely relevant",
                ),
                ("1", "5", "not at all", "extremely"): ("Not at all", "Extremely"),
            }
            display_pair = canonical_labels.get(key)
            if display_pair is None:
                display_pair = (low_raw, high_raw)
            low_disp, high_disp = display_pair
            label = f"{lo_s}–{hi_s} ({lo_s} = {low_disp}, {hi_s} = {high_disp})"
            return key, label

        def _resolve_bits(cid: str, seen: set[str]) -> tuple[tuple[str, str] | None, tuple[float, float] | None]:
            if cid in seen:
                return None, None
            seen.add(cid)

            endpoints = self.ordinal_endpoints.get(cid)
            bounds = self.possible_bounds.get(cid)

            item = self.by_id.get(cid)
            if not isinstance(item, dict):
                return endpoints, bounds
            transforms = item.get("transformations") or []
            if not isinstance(transforms, list):
                return endpoints, bounds
            for tr in transforms:
                if not isinstance(tr, dict):
                    continue
                if tr.get("type") == "renamed":
                    src = tr.get("rename_from")
                    if isinstance(src, str):
                        src_endpoints, src_bounds = _resolve_bits(src, seen)
                        if endpoints is None and src_endpoints is not None:
                            endpoints = src_endpoints
                        if bounds is None and src_bounds is not None:
                            bounds = src_bounds
                if tr.get("type") == "merged":
                    merged_from = tr.get("merged_from")
                    if not isinstance(merged_from, list):
                        continue
                    for src in merged_from:
                        if not isinstance(src, str):
                            continue
                        src_endpoints, src_bounds = _resolve_bits(src, seen)
                        if endpoints is None and src_endpoints is not None:
                            endpoints = src_endpoints
                        if bounds is None and src_bounds is not None:
                            bounds = src_bounds
            return endpoints, bounds

        endpoints, bounds = _resolve_bits(col, set())
        if endpoints is None and bounds is None:
            return None
        if bounds is not None:
            lo, hi = bounds
            lo_s, hi_s = self.fmt_num(float(lo)), self.fmt_num(float(hi))
            if endpoints is not None:
                low_lab, high_lab = endpoints
                return _canonical_scale_label(lo_s, hi_s, low_lab, high_lab)
            key = (lo_s, hi_s)
            return key, f"{lo_s}–{hi_s}"
        low_lab, high_lab = endpoints
        key = (low_lab, high_lab)
        return key, f"{low_lab} … {high_lab}"

    def format_attr_lines(
        self,
        attrs: dict[str, object],
        driver_cols: list[str] | None,
        *,
        scale_seen: set[tuple[object, ...]],
    ) -> list[str]:
        _ = scale_seen
        if driver_cols is None:
            cols = [k for k in attrs.keys() if not k.startswith(SPECIAL_STRUCTURAL_PREFIXES)]
        else:
            cols = [c for c in driver_cols if c in attrs]
        lines: list[str] = []
        scale_items: dict[tuple[object, ...], dict[str, object]] = {}
        scale_order: list[tuple[object, ...]] = []
        non_scale_lines: list[str] = []
        grouped_cols: set[str] = set()
        for group in self.display_groups.values():
            members = group.get("members")
            if not isinstance(members, dict):
                continue
            grouped_cols.update(members.keys())

        for group in self.display_groups.values():
            members = group.get("members")
            if not isinstance(members, dict):
                continue
            member_entries: list[tuple[str, str, str]] = []
            for col, member_label in members.items():
                if col not in cols:
                    continue
                if col.startswith(SPECIAL_STRUCTURAL_PREFIXES):
                    continue
                raw_val = attrs.get(col)
                if raw_val is None or (isinstance(raw_val, float) and pd.isna(raw_val)):
                    continue
                if isinstance(raw_val, str) and raw_val.strip().lower() in {"missing", "nan", "none"}:
                    continue
                val = self._maybe_label_numeric(col, raw_val)
                member_entries.append((member_label, val, col))
            if member_entries:
                title = group.get("title")
                if not isinstance(title, str) or not title.strip():
                    continue
                group_title = title.strip()
                by_scale: dict[tuple[object, ...], list[tuple[str, str]]] = {}
                for member_label, val, col in member_entries:
                    scale = self._scale_descriptor(col)
                    if scale is None:
                        non_scale_lines.append(f"- {group_title}:")
                        non_scale_lines.append(f"  - {member_label}: {val}")
                        continue
                    key, scale_label = scale
                    by_scale.setdefault((key, scale_label), []).append((member_label, val))

                scale_inline = bool(group.get("scale_inline"))
                if scale_inline and len(by_scale) == 1:
                    (key, _scale_label), entries = next(iter(by_scale.items()))
                    lo_s, hi_s = key[0], key[1]
                    non_scale_lines.append(f"- {group_title} (scale={lo_s}-{hi_s}):")
                    for member_label, val in entries:
                        non_scale_lines.append(f"  - {member_label}: {val}")
                    continue

                for (key, scale_label), entries in by_scale.items():
                    if key not in scale_items:
                        scale_items[key] = {"label": scale_label, "items": []}
                        scale_order.append(key)
                    items = scale_items[key]["items"]
                    existing = next(
                        (item for item in items if item["type"] == "group" and item["title"] == group_title),
                        None,
                    )
                    if existing is None:
                        existing = {"type": "group", "title": group_title, "entries": []}
                        items.append(existing)
                    existing["entries"].extend(entries)

        inline_entries: list[tuple[str, str, str]] = []
        for col in cols:
            if col.startswith(SPECIAL_STRUCTURAL_PREFIXES):
                continue
            if col in grouped_cols:
                continue
            raw_val = attrs.get(col)
            if raw_val is None or (isinstance(raw_val, float) and pd.isna(raw_val)):
                continue
            if isinstance(raw_val, str) and raw_val.strip().lower() in {"missing", "nan", "none"}:
                continue
            label = self._display_label(col)
            val = self._maybe_label_numeric(col, raw_val)
            inline_entries.append((label, val, col))
        for label, val, col in inline_entries:
            scale = self._scale_descriptor(col)
            if scale is None:
                non_scale_lines.append(f"- {label}: {val}")
                continue
            key, scale_label = scale
            if key not in scale_items:
                scale_items[key] = {"label": scale_label, "items": []}
                scale_order.append(key)
            scale_items[key]["items"].append({"type": "item", "label": label, "value": val})

        for key in scale_order:
            scale_label = scale_items[key]["label"]
            items = scale_items[key]["items"]
            total_items = 0
            for item in items:
                if item["type"] == "group":
                    total_items += len(item["entries"])
                else:
                    total_items += 1
            if total_items == 1 and items and items[0]["type"] == "item":
                item = items[0]
                inline_label = f"{item['label']} (scale {scale_label})"
                lines.append(f"- {inline_label}: {item['value']}")
                continue
            lines.append(f"Scale: {scale_label}")
            for item in items:
                if item["type"] == "group":
                    lines.append(f"- {item['title']}:")
                    for member_label, value in item["entries"]:
                        lines.append(f"  - {member_label}: {value}")
                else:
                    lines.append(f"- {item['label']}: {item['value']}")
            lines.append("")

        if non_scale_lines:
            lines.extend(non_scale_lines)
        return lines

    def _bucket_scale(self, val: object, lo: int, hi: int) -> int | None:
        num = self._as_numeric(val)
        if num is None:
            return None
        rounded = int(round(num))
        return max(lo, min(hi, rounded))

    def _as_bool(self, val: object) -> bool | None:
        num = self._as_numeric(val)
        if num is None:
            return None
        return bool(round(num))

    def _income_quintiles_by_country(self) -> dict[str, list[float]]:
        if self._income_quintile_cache is not None:
            return self._income_quintile_cache
        path = Path("preprocess/output/clean_processed_survey.csv")
        if not path.exists():
            self._income_quintile_cache = {}
            return self._income_quintile_cache
        df = pd.read_csv(path, usecols=["country", "income_ppp_norm"])
        df = df.dropna(subset=["country", "income_ppp_norm"])
        if df.empty:
            self._income_quintile_cache = {}
            return self._income_quintile_cache
        out: dict[str, list[float]] = {}
        for country, g in df.groupby("country", dropna=True):
            arr = pd.to_numeric(g["income_ppp_norm"], errors="coerce").dropna()
            if arr.empty:
                continue
            qs = arr.quantile([0.2, 0.4, 0.6, 0.8]).tolist()
            out[str(country)] = [float(x) for x in qs]
        self._income_quintile_cache = out
        return self._income_quintile_cache

    def _income_phrase(self, val: object, country: object, ctx: dict[str, str]) -> str | None:
        num = self._as_numeric(val)
        if num is None:
            return None
        cuts = self._income_quintiles_by_country().get(str(country))
        if not cuts:
            return None
        labels = [
            "on the very low side",
            "on the lower side",
            "around the middle",
            "on the higher side",
            "on the very high side",
        ]
        idx = 0
        for c in cuts:
            if num > c:
                idx += 1
        idx = min(idx, 4)
        return f"{ctx['possessive'].capitalize()} income is {labels[idx]}."

    def _apply_pronoun_context(self, text: str, ctx: dict[str, str]) -> str:
        if ctx["subject"] == "I":
            return text
        replacements: list[tuple[str, str]] = [
            (r"(?:(?<=^)|(?<=[.!?]\s))I'm\b", "They are"),
            (r"(?:(?<=^)|(?<=[.!?]\s))I’m\b", "They are"),
            (r"\bI'm\b", "they are"),
            (r"\bI’m\b", "they are"),
            (r"(?:(?<=^)|(?<=[.!?]\s))I've\b", "They have"),
            (r"(?:(?<=^)|(?<=[.!?]\s))I’ve\b", "They have"),
            (r"\bI've\b", "they have"),
            (r"\bI’ve\b", "they have"),
            (r"(?:(?<=^)|(?<=[.!?]\s))I'd\b", "They would"),
            (r"(?:(?<=^)|(?<=[.!?]\s))I’d\b", "They would"),
            (r"\bI'd\b", "they would"),
            (r"\bI’d\b", "they would"),
            (r"(?:(?<=^)|(?<=[.!?]\s))I'll\b", "They will"),
            (r"(?:(?<=^)|(?<=[.!?]\s))I’ll\b", "They will"),
            (r"\bI'll\b", "they will"),
            (r"\bI’ll\b", "they will"),
            (r"(?:(?<=^)|(?<=[.!?]\s))I am\b", "They are"),
            (r"\bI am\b", "they are"),
            (r"(?:(?<=^)|(?<=[.!?]\s))I have\b", "They have"),
            (r"\bI have\b", "they have"),
            (r"(?:(?<=^)|(?<=[.!?]\s))I had\b", "They had"),
            (r"\bI had\b", "they had"),
            (r"(?:(?<=^)|(?<=[.!?]\s))I will\b", "They will"),
            (r"\bI will\b", "they will"),
        ]
        for pattern, repl in replacements:
            text = re.sub(pattern, repl, text)
        text = re.sub(r"\bmyself\b", "themselves", text)
        text = re.sub(r"\bmine\b", "theirs", text)
        text = re.sub(r"\bme\b", "them", text)
        text = re.sub(r"\bmy\b", "their", text)
        text = re.sub(r"\bMy\b", "Their", text)
        text = re.sub(r"(?:(?<=^)|(?<=[.!?]\s))I\b", "They", text)
        text = re.sub(r"\bI\b", "they", text)
        return text

    def _trust_phrase(self, val: object, obj: str, ctx: dict[str, str]) -> str | None:
        level = self._bucket_scale(val, 1, 5)
        if level is None:
            return None
        mapping = {
            1: "I don't really trust {obj} at all.",
            2: "I trust {obj} a little, but I'm cautious.",
            3: "I'm somewhat neutral — I trust {obj} in some cases, but not blindly.",
            4: "I generally trust {obj}.",
            5: "I trust {obj} a lot.",
        }
        return self._apply_pronoun_context(mapping[level].format(obj=obj), ctx)

    def _scale_phrase(self, val: object, mapping: dict[int, str], ctx: dict[str, str]) -> str | None:
        level = self._bucket_scale(val, min(mapping), max(mapping))
        if level is None:
            return None
        return self._apply_pronoun_context(mapping[level], ctx)

    def _mfq_binding_joint_phrase(
        self,
        agreement_val: object,
        relevance_val: object,
        ctx: dict[str, str],
    ) -> str | None:
        agreement_level = self._bucket_scale(agreement_val, 1, 6)
        relevance_level = self._bucket_scale(relevance_val, 1, 6)
        if agreement_level is None or relevance_level is None:
            return None
        agrees = agreement_level >= 4
        relevance_high = relevance_level >= 5
        relevance_mid = relevance_level in (3, 4)
        label = "loyalty, authority, and purity"
        if agrees and relevance_high:
            s = f"I endorse principles of {label}, and they are important in how I judge right and wrong."
        elif agrees and relevance_mid:
            s = f"I endorse principles of {label}, and they matter somewhat when I judge right and wrong."
        elif agrees:
            s = f"I endorse principles of {label}, but they do not matter much in my moral judgments."
        elif relevance_high:
            s = f"Principles of {label} matter to me, even if I do not fully endorse them."
        elif relevance_mid:
            s = (
                f"I do not fully endorse principles of {label}, but they still matter somewhat when I judge "
                "right and wrong."
            )
        else:
            s = f"I do not endorse principles of {label}, and they are not very relevant to my moral judgments."
        return self._apply_pronoun_context(s, ctx)

    def _infection_likelihood_phrase(self, val: object, ctx: dict[str, str]) -> str | None:
        num = self._as_numeric(val)
        if num is None:
            return None
        if num < 1.5:
            s = "I think it's very unlikely I'll catch an infectious disease like COVID or flu."
        elif num < 2.5:
            s = "I think it's unlikely I'll catch an infectious disease like COVID or flu."
        elif num < 3.5:
            s = "I think there's a fair chance I'll catch an infectious disease like COVID or flu."
        elif num < 4.5:
            s = "I think it's likely I'll catch an infectious disease like COVID or flu."
        else:
            s = "I think it's very likely I'll catch an infectious disease like COVID or flu."
        return self._apply_pronoun_context(s, ctx)

    def _norms_phrase(self, val: object, group_label: str, ctx: dict[str, str]) -> str | None:
        level = self._bucket_scale(val, 1, 7)
        if level is None:
            return None
        mapping = {
            1: f"My {group_label} generally don’t hold progressive views on these issues.",
            2: f"My {group_label} tend to be more conservative than progressive on these issues.",
            3: f"My {group_label} lean slightly away from progressive positions.",
            4: f"My {group_label} are mixed—no clear progressive or conservative tilt.",
            5: f"My {group_label} lean somewhat progressive on these issues.",
            6: f"My {group_label} are clearly progressive on these issues.",
            7: f"My {group_label} are strongly progressive on these issues.",
        }
        return mapping[level].replace("My ", f"{ctx['possessive'].capitalize()} ")

    def _covax_legitimacy_phrase(self, val: object, ctx: dict[str, str]) -> str | None:
        num = self._as_numeric(val)
        if num is None:
            return None
        if num < 0.2:
            s = "I have almost no doubts about how COVID vaccines are tested and approved."
        elif num < 0.4:
            s = "I have only a few doubts about how COVID vaccines are tested and approved."
        elif num < 0.6:
            s = "I have mixed feelings about how COVID vaccines are tested and approved."
        elif num < 0.8:
            s = "I have quite a lot of doubts about how COVID vaccines are tested and approved."
        else:
            s = "I have very strong doubts about how COVID vaccines are tested and approved."
        return self._apply_pronoun_context(s, ctx)

    def _age_phrase(self, val: object, ctx: dict[str, str]) -> str | None:
        num = self._as_numeric(val)
        if num is None:
            return None
        if num < 30:
            band = "in my 20s"
        elif num < 45:
            band = "in my 30s or early 40s"
        elif num < 60:
            band = "in my late 40s or 50s"
        elif num < 75:
            band = "in my 60s or early 70s"
        else:
            band = "in my mid-70s or older"
        if ctx["subject"] == "They":
            band = band.replace("my", ctx["possessive"])
        return band

    @staticmethod
    def _country_label(country: object) -> str:
        if country is None:
            return ""
        raw = str(country).strip()
        mapping = {
            "IT": "Italy",
            "FR": "France",
            "DE": "Germany",
            "ES": "Spain",
            "HU": "Hungary",
            "UK": "United Kingdom",
        }
        return mapping.get(raw, raw)


__all__ = ["PromptRendererPrimitivesMixin"]
