"""Reusable helpers for rendering modelling tables in the thesis."""

from __future__ import annotations

import csv
import difflib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from IPython.display import Markdown


@dataclass(slots=True)
class CoefficientIndex:
    """Index coefficients by (model, term) for quick lookup."""

    rows: dict[tuple[str, str], dict[str, str]]

    _TERM_ALIASES: ClassVar[Mapping[str, tuple[str, ...]]] = {
        # Older thesis text/specs refer to this name; newer coefficient exports
        # use a shorter, semantically equivalent label.
        "progressive_attitudes_nonvax_idx": ("moral_progressive_orientation",),
        # Older text referred to a single reference-group item; modelling exports now
        # use a pooled social norms index.
        "family_vaccines_importance": ("social_norms_vax_avg",),
        # Some models use the merged moral/progressive composite instead of
        # listing MFQ individualizing agreement separately.
        "mfq_agreement_individualizing_idx": ("moral_progressive_orientation",),
        "mfq_relevance_individualizing_idx": ("moral_progressive_orientation",),
    }

    def fetch_row(self, model: str, term: str) -> dict[str, str]:
        """Return the raw coefficient row, raising a helpful error on miss."""

        key = (model, term)
        row = self.rows.get(key)
        if row is not None:
            return row

        for alias in self._TERM_ALIASES.get(term, ()):
            alias_row = self.rows.get((model, alias))
            if alias_row is not None:
                return alias_row

        model_terms = sorted({t for (m, t) in self.rows.keys() if m == model})
        suggestions = difflib.get_close_matches(term, model_terms, n=8, cutoff=0.6)
        hint = f". Did you mean one of: {', '.join(suggestions)}" if suggestions else ""
        raise KeyError(f"Missing coefficient for model={model} term={term}{hint}")

    def format_entry(self, model: str, term: str) -> tuple[str, str, str]:
        """Return formatted beta, z and significance markers for a coefficient."""

        row = self.fetch_row(model, term)

        beta = float(row["estimate"])
        z_value = float(row["z_value"])
        sig = row.get("significance") or ""
        return f"{beta:+.2f}", f"{z_value:.2f}", sig

    def beta_value(self, model: str, term: str, *, signed: bool = False) -> str:
        """Return the formatted beta for inline display."""

        beta, _, _ = self.format_entry(model, term)
        return beta if signed else beta.lstrip("+")

    def z_value(self, model: str, term: str) -> str:
        """Return the formatted z statistic for inline display."""

        _, z_val, _ = self.format_entry(model, term)
        return z_val

    def beta_z(self, model: str, term: str, *, signed: bool = False) -> str:
        """Return beta and z formatted for inline narrative use."""

        return f"{self.beta_value(model, term, signed=signed)}, z = {self.z_value(model, term)}"

    def beta_z_md(self, model: str, term: str, *, signed: bool = False):
        """Return beta and z as Markdown containing inline math."""

        beta = self.beta_value(model, term, signed=signed)
        z_val = self.z_value(model, term)
        return Markdown(rf"$\beta = {beta}, z = {z_val}$")

    def beta_range_md(
        self,
        model1: str,
        term1: str,
        model2: str,
        term2: str,
        *,
        signed: bool = False,
    ):
        """Return a beta range as Markdown, e.g. $\beta \approx a–b$."""

        a = self.beta_value(model1, term1, signed=signed)
        b = self.beta_value(model2, term2, signed=signed)
        return Markdown(rf"$\beta \approx {a}–{b}$")


def load_coefficients(paths: Sequence[str | Path]) -> CoefficientIndex:
    """Load coefficient exports from one or more CSV files."""

    entries: dict[tuple[str, str], dict[str, str]] = {}
    for path_like in paths:
        path = Path(path_like)
        if not path.exists():
            raise FileNotFoundError(f"Coefficient export not found: {path}")
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                entries[(row["model"], row["term"])] = row
    return CoefficientIndex(entries)


def label_for(term: str, labels: Mapping[str, str]) -> str:
    """Return the human-readable label for a term, falling back to the raw name."""

    return labels.get(term, term)


def render_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a simple Markdown table given headers and row values."""

    def _escape_cell(value: str) -> str:
        # Markdown tables use `|` as a column delimiter. Replace it with the HTML
        # entity so headers/cells can display a literal pipe without splitting
        # into extra columns in Quarto/Markdown renderers.
        return value.replace("|", "&#124;")

    safe_headers = [_escape_cell(str(h)) for h in headers]
    header_row = "| " + " | ".join(safe_headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = [header_row, separator_row]
    for row in rows:
        safe_row = [_escape_cell(str(cell)) for cell in row]
        lines.append("| " + " | ".join(safe_row) + " |")
    return "\n".join(lines)


def effect_with_sig(beta: str, sig: str) -> str:
    """Attach significance markers to a formatted beta."""

    sig = (sig or "").strip()
    if not sig:
        return beta
    thin_space = "\u2009"
    return f"{beta}{thin_space}{sig}"


def load_pseudo_r2(path: str | Path) -> dict[str, float]:
    """Load a model -> McFadden pseudo-R² mapping from a CSV export."""

    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        return {row["model"]: float(row["pseudo_r2"]) for row in csv.DictReader(handle)}


def format_pseudo_r2(pseudo_r2: Mapping[str, float], model: str, *, digits: int = 2) -> str:
    """Format pseudo-R² for narrative display."""

    if model not in pseudo_r2:
        raise KeyError(f"Missing pseudo-R² for model '{model}'")
    return f"{pseudo_r2[model]:.{digits}f}"


def format_pseudo_ratio(
    pseudo_r2: Mapping[str, float],
    *,
    numerator_model: str,
    denominator_model: str,
) -> str:
    """Format the ratio of two pseudo-R² values as a percent."""

    numerator = pseudo_r2[numerator_model]
    denominator = pseudo_r2[denominator_model]
    return f"{(numerator / denominator) * 100:.0f}%"
