"""Plot/table render helpers for chapter 3 (decision function)."""

from __future__ import annotations

import csv
import json
import math
import textwrap
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Rectangle

from utils.chapter_io import read_csv_fast
from utils.engine_labels import canonical_choice_engine_block, label_choice_engine_block
from utils.plot_style import apply_thesis_matplotlib_style

apply_thesis_matplotlib_style()


def render_behaviour_backbone_tables(
    *,
    fetch: Callable[[str, str], tuple[str, str, str]],
    effect_with_sig: Callable[[str, str], str],
    label_for: Callable[[str], str],
    render_markdown_table: Callable[[list[str], list[list[str]]], str],
) -> tuple[str, str]:
    """Build panel-A and panel-B behaviour backbone tables for chapter 3."""

    behaviour_columns_base = [
        {
            "id": "covid_willingness",
            "label": "COVID booster willingness (Y)",
            "model": "B_COVID_MAIN_vax_willingness_Y",
        },
        {
            "id": "covid_uptake",
            "label": "Ever vaccinated against COVID",
            "model": "B_COVID_vaccinated_any",
        },
        {
            "id": "flu_vaccination",
            "label": "Flu vax (2023–24)",
            "model": "B_FLU_vaccinated_2023_2024",
        },
        {
            "id": "support_mandates",
            "label": "Support for mandates",
            "model": "B_PROTECT_support_mandatory_vaccination_LEGIT",
        },
        {
            "id": "npi_behaviour",
            "label": "NPI behaviour index",
            "model": "B_PROTECT_protective_behaviour",
        },
    ]

    behaviour_columns_state = [
        {
            "id": "covid_willingness",
            "label": "COVID booster willingness (Y)",
            "model": "B_COVID_MAIN_vax_willingness_Y_HABIT",
        },
        {
            "id": "covid_uptake",
            "label": "Ever vaccinated against COVID",
            "model": "B_COVID_vaccinated_any",
        },
        {
            "id": "flu_vaccination",
            "label": "Flu vax (2023–24)",
            "model": "B_FLU_vaccinated_2023_2024_HABIT",
        },
        {
            "id": "support_mandates",
            "label": "Support for mandates",
            "model": "B_PROTECT_support_mandatory_vaccination_LEGIT",
        },
        {
            "id": "npi_behaviour",
            "label": "NPI behaviour index",
            "model": "B_PROTECT_protective_behaviour",
        },
    ]

    def _all_terms(columns: list[dict[str, str]], term: str) -> dict[str, str]:
        return {col["id"]: term for col in columns}

    def _build_base_rows(columns: list[dict[str, str]]) -> list[dict[str, object]]:
        return [
            {"label_term": "fivec_confidence", "terms": _all_terms(columns, "fivec_confidence")},
            {"label": "Legitimacy scepticism", "terms": {"support_mandates": "covax_legitimacy_scepticism_idx"}},
            {
                "label": "Disease fear",
                "terms": {
                    "covid_willingness": "disease_fear_avg",
                    "covid_uptake": "disease_fear_avg",
                    "flu_vaccination": "flu_disease_fear",
                    "support_mandates": "disease_fear_avg",
                    "npi_behaviour": "disease_fear_avg",
                },
            },
            {
                "label": "Vaccine risk (across different diseases)",
                "terms": {
                    "covid_willingness": "vaccine_risk_avg",
                    "covid_uptake": "vaccine_risk_avg",
                    "flu_vaccination": "flu_vaccine_risk",
                    "support_mandates": "vaccine_risk_avg",
                    "npi_behaviour": "vaccine_risk_avg",
                },
            },
            {
                "label_term": "fivec_collective_responsibility",
                "terms": _all_terms(columns, "fivec_collective_responsibility"),
            },
            {
                "label_term": "progressive_attitudes_nonvax_idx",
                "terms": _all_terms(columns, "progressive_attitudes_nonvax_idx"),
            },
            {
                "label_term": "family_progressive_norms_nonvax_idx",
                "terms": _all_terms(columns, "family_progressive_norms_nonvax_idx"),
            },
            {
                "label_term": "friends_progressive_norms_nonvax_idx",
                "terms": _all_terms(columns, "friends_progressive_norms_nonvax_idx"),
            },
            {
                "label_term": "colleagues_progressive_norms_nonvax_idx",
                "terms": _all_terms(columns, "colleagues_progressive_norms_nonvax_idx"),
            },
            {"label_term": "trust_who_vax_info", "terms": _all_terms(columns, "trust_who_vax_info")},
            {
                "label_term": "mfq_agreement_individualizing_idx",
                "terms": _all_terms(columns, "mfq_agreement_individualizing_idx"),
            },
            {"label_term": "mfq_agreement_binding_idx", "terms": _all_terms(columns, "mfq_agreement_binding_idx")},
            {"label_term": "side_effects", "terms": _all_terms(columns, "side_effects")},
            {"label_term": "side_effects_other", "terms": _all_terms(columns, "side_effects_other")},
            {"label_term": "exper_illness", "terms": _all_terms(columns, "exper_illness")},
            {"label_term": "sex_female[T.True]", "terms": _all_terms(columns, "sex_female[T.True]")},
            {"label_term": "age", "terms": _all_terms(columns, "age")},
            {"label_term": "minority", "terms": _all_terms(columns, "minority")},
            {
                "label": "Flu vaccine fear",
                "terms": {
                    "covid_willingness": None,
                    "covid_uptake": None,
                    "flu_vaccination": "flu_vaccine_fear",
                    "support_mandates": None,
                    "npi_behaviour": None,
                },
            },
            {
                "label_term": "C(country, Treatment(reference='IT'))[T.FR]",
                "terms": _all_terms(columns, "C(country, Treatment(reference='IT'))[T.FR]"),
            },
            {
                "label_term": "C(country, Treatment(reference='IT'))[T.HU]",
                "terms": _all_terms(columns, "C(country, Treatment(reference='IT'))[T.HU]"),
            },
            {
                "label_term": "C(country, Treatment(reference='IT'))[T.UK]",
                "terms": _all_terms(columns, "C(country, Treatment(reference='IT'))[T.UK]"),
            },
        ]

    def _row_label(row_spec: dict[str, object]) -> str:
        if "label" in row_spec:
            return str(row_spec["label"])
        return label_for(str(row_spec["label_term"]))

    def _build_rows(columns: list[dict[str, str]], rows: list[dict[str, object]]) -> list[list[str]]:
        out_rows: list[list[str]] = []
        for row in rows:
            cells = [_row_label(row)]
            for col in columns:
                term_spec = row["terms"].get(col["id"])
                if not term_spec:
                    cells.append(" - ")
                    continue
                model_name = col["model"]
                term_name = str(term_spec)
                beta, _, sig = fetch(model_name, term_name)
                cells.append(effect_with_sig(beta, sig))
            out_rows.append(cells)
        return out_rows

    habit_rows = [
        {"label": "COVID vaccination dose history", "terms": {"covid_willingness": "covax_num"}},
        {"label": "Flu vaccination last season (2022–23)", "terms": {"flu_vaccination": "flu_vaccinated_2022_2023"}},
        {
            "label": "Flu vaccination two seasons ago (2021–22)",
            "terms": {"flu_vaccination": "flu_vaccinated_2021_2022"},
        },
    ]

    table_a = (
        "Table: Shared predictors of prevention behaviours {#tbl-appendix-predictors-estimate}\n\n"
        + render_markdown_table(
            ["Predictor"] + [col["label"] for col in behaviour_columns_base],
            _build_rows(behaviour_columns_base, _build_base_rows(behaviour_columns_base)),
        )
        + "\n\n"
        + "*Note:* Standardised coefficients are reported to support magnitude-first interpretation."
    )
    table_b = (
        "Table: Behavioural outcome equations with psychological-profile augmentation (adds habit) {#tbl-appendix-predictors-estimate-state}\n\n"
        + render_markdown_table(
            ["Predictor"] + [col["label"] for col in behaviour_columns_state],
            _build_rows(behaviour_columns_state, _build_base_rows(behaviour_columns_state) + habit_rows),
        )
        + "\n\n"
        + "*Note:* Standardised coefficients are reported to support magnitude-first interpretation."
    )
    return table_a, table_b


def render_lobo_bic_block_outcome_heatmap(
    *,
    label_outcome,
    extended_outcome_order: list[str],
    model_plan_ref_choice_path: Path | None = Path("../thesis/artifacts/empirical/modeling/model_plan_ref_choice.json"),
    model_lobo_all_path: Path = Path("../thesis/artifacts/empirical/modeling/model_lobo_all.csv"),
    extra_lobo_paths: list[Path] | None = None,
    model_name_prefixes: list[str] | None = None,
    model_name_suffix: str | None = None,
    always_include_models: set[str] | None = None,
    outcome_aliases: dict[str, str] | None = None,
    display_outcome_overrides_by_model: dict[str, str] | None = None,
    block_order: list[str] | None = None,
    block_labels: dict[str, str] | None = None,
    plot_title: str = "Relative block importance by outcome (LOBO ΔBIC weights)",
    cell_aggregation: str = "median",
    x_axis_label: str = "Outcomes",
    y_axis_label: str = "Blocks",
    colorbar_label: str = "Relative importance (ΔBIC share)",
    y_label_wrap_width: int = 24,
    highlight_cells: Sequence[tuple[str, str]] | None = None,
    highlight_color: str = "#d63a31",
    highlight_linewidth: float = 2.8,
    cmap_name: str = "cividis",
    output_path: Path | None = None,
    show: bool = True,
) -> str | None:
    default_engine_block_order = [
        "institutional_trust_engine",
        "group_trust_engine",
        "trust_engine",
        "disposition_engine",
        "stakes_engine",
        "vaccine_cost_engine",
        "norms_engine",
        "moral_engine",
        "legitimacy_engine",
        "habit_engine",
        "controls_demography_SES",
    ]
    engine_block_order = list(block_order) if block_order else list(default_engine_block_order)
    provided_block_labels = {str(k): str(v) for k, v in block_labels.items()} if block_labels else None

    def _label_block(block: str) -> str:
        canonical = canonical_choice_engine_block(block)
        if provided_block_labels is not None:
            return provided_block_labels[canonical]
        return label_choice_engine_block(canonical)

    prefixes = [str(p).strip() for p in (model_name_prefixes or []) if str(p).strip()]
    suffix = str(model_name_suffix).strip() if model_name_suffix else ""
    include_models = (
        {str(v).strip() for v in always_include_models if str(v).strip()}
        if always_include_models is not None
        else {"B_FLU_vaccinated_2023_2024_ENGINES"}
    )

    outcome_alias_map = (
        {str(k): str(v) for k, v in outcome_aliases.items()}
        if outcome_aliases is not None
        else {"vax_willingness_Y": "vax_willingness_T12"}
    )
    display_outcome_overrides = (
        {str(k): str(v) for k, v in display_outcome_overrides_by_model.items()}
        if display_outcome_overrides_by_model is not None
        else {
            "B_FLU_vaccinated_2023_2024_ENGINES": "flu_vaccinated_2023_2024_no_habit",
            "B_FLU_vaccinated_2023_2024_ENGINES_HABIT": "flu_vaccinated_2023_2024",
        }
    )
    outcome_order = list(extended_outcome_order)

    def _to_float_local(value: str | float | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        value = str(value).strip()
        if not value:
            return None
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None

    def _load_rows_local(path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    if model_plan_ref_choice_path is not None and model_plan_ref_choice_path.exists():
        ref_plan_rows = json.loads(model_plan_ref_choice_path.read_text(encoding="utf-8"))
        ref_choice_models = {
            str(row.get("model", row.get("name", ""))).strip()
            for row in ref_plan_rows
            if row.get("model", row.get("name"))
        }
    else:
        ref_choice_models = set()

    if not model_lobo_all_path.exists():
        return "No LOBO export found; skipping heatmap."
    raw_lobo_rows = _load_rows_local(model_lobo_all_path)

    all_extra = extra_lobo_paths or [
        Path("../thesis/artifacts/empirical/modeling/B_FLU_vaccinated_2023_2024_ENGINES_lobo.csv"),
    ]
    for extra_path in all_extra:
        if extra_path.exists():
            raw_lobo_rows.extend(_load_rows_local(extra_path))

    lobo_rows_local: list[dict[str, str]] = []
    for row in raw_lobo_rows:
        model_name = str(row.get("model", "")).strip()
        if ref_choice_models and model_name not in ref_choice_models and model_name not in include_models:
            continue
        if (
            prefixes
            and not any(model_name.startswith(prefix) for prefix in prefixes)
            and model_name not in include_models
        ):
            continue
        if suffix and not model_name.endswith(suffix) and model_name not in include_models:
            continue
        row = dict(row)
        outcome_name = str(row.get("outcome", "")).strip()
        mapped_outcome = outcome_alias_map.get(outcome_name, outcome_name)
        row["outcome"] = mapped_outcome
        display_outcome = display_outcome_overrides.get(model_name, mapped_outcome)
        row["display_outcome"] = str(display_outcome)
        row["block"] = canonical_choice_engine_block(str(row.get("block", "")).strip())
        lobo_rows_local.append(row)

    available_metrics_local = sorted({row.get("metric", "pseudo_r2") for row in lobo_rows_local})

    def _cell_median_lobo_bic_weight(block: str, outcome: str) -> float:
        vals: list[float] = []
        denom_by_model_outcome: dict[tuple[str, str], float] = {}
        for row in lobo_rows_local:
            if row.get("metric") != "bic":
                continue
            row_outcome = row.get("display_outcome", row.get("outcome"))
            row_model = row.get("model")
            if row_outcome != outcome:
                continue
            val = _to_float_local(row.get("lobo_bic_weight"))
            if val is None:
                key = (str(row_model), str(row_outcome))
                if key not in denom_by_model_outcome:
                    denom = 0.0
                    for other in lobo_rows_local:
                        if other.get("metric") != "bic":
                            continue
                        if other.get("model") != row_model:
                            continue
                        if other.get("display_outcome", other.get("outcome")) != row_outcome:
                            continue
                        d = _to_float_local(other.get("delta"))
                        denom += max(d or 0.0, 0.0)
                    denom_by_model_outcome[key] = denom
                d_self = _to_float_local(row.get("delta"))
                denom = denom_by_model_outcome[key]
                if d_self is not None and denom > 0.0:
                    val = max(d_self, 0.0) / denom
            if row.get("block") == block and val is not None:
                vals.append(val)
        if not vals:
            return float("nan")
        if str(cell_aggregation).strip().lower() == "mean":
            return float(sum(vals) / len(vals))
        return float(median(vals))

    if "bic" not in available_metrics_local:
        return "No LOBO BIC rows were found; skipping heatmap."

    present_blocks = {row.get("block") for row in lobo_rows_local if row.get("metric") == "bic"}
    blocks = [b for b in engine_block_order if b in present_blocks]
    outcomes = [
        o for o in outcome_order if any(r.get("display_outcome", r.get("outcome")) == o for r in lobo_rows_local)
    ]

    if not blocks or not outcomes:
        return "Insufficient LOBO BIC data to draw block × outcome heatmap."

    mat = np.full((len(blocks), len(outcomes)), np.nan, dtype=float)
    for i, block in enumerate(blocks):
        for j, outcome in enumerate(outcomes):
            mat[i, j] = _cell_median_lobo_bic_weight(block, outcome)

    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        return "LOBO BIC-weight values are all missing after filtering; skipping heatmap."

    vmin = float(np.quantile(finite, 0.02))
    vmax = float(np.quantile(finite, 0.98))
    if not (np.isfinite(vmin) and np.isfinite(vmax) and vmin < vmax):
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))

    fig, ax = plt.subplots(figsize=(13.6, 7.3), constrained_layout=True)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="#f0f0f0")
    im = ax.imshow(
        np.ma.masked_invalid(mat),
        cmap=cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks(range(len(outcomes)))
    ax.set_yticks(range(len(blocks)))
    ax.set_xticklabels([label_outcome(o) for o in outcomes], rotation=25, ha="right", fontsize=12)
    ax.set_yticklabels(
        ["\n".join(textwrap.wrap(_label_block(b), width=y_label_wrap_width)) for b in blocks],
        fontsize=12,
    )
    ax.tick_params(axis="both", which="major", labelsize=12)

    for i in range(len(blocks)):
        for j in range(len(outcomes)):
            val = mat[i, j]
            if np.isfinite(val):
                t = (float(val) - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                rgba = cmap(np.clip(t, 0.0, 1.0))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                txt_color = "black" if luminance >= 0.5 else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=12, color=txt_color)
            else:
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor="#f0f0f0",
                        edgecolor="#cfcfcf",
                        hatch="\\\\\\",
                        linewidth=0.0,
                        zorder=0,
                    )
                )
                ax.text(j, i, "NA", ha="center", va="center", fontsize=11, color="#8a8a8a")

    if highlight_cells:
        block_index = {block: idx for idx, block in enumerate(blocks)}
        outcome_index = {outcome: idx for idx, outcome in enumerate(outcomes)}
        for raw_block, raw_outcome in highlight_cells:
            block = canonical_choice_engine_block(str(raw_block).strip())
            outcome = str(raw_outcome).strip()
            i = block_index.get(block)
            j = outcome_index.get(outcome)
            if i is None or j is None:
                continue
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor=highlight_color,
                    linewidth=highlight_linewidth,
                    zorder=5,
                    joinstyle="round",
                )
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    cbar.set_label(colorbar_label)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(colorbar_label, fontsize=12)
    ax.set_xlabel(x_axis_label, fontsize=12)
    ax.set_ylabel(y_axis_label, fontsize=12)
    ax.set_title(plot_title, fontsize=12, pad=10)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return None


def render_reference_signal_magnitude(
    *,
    ref_ice_root: Path,
    context_label: str,
    label_outcome,
    ref_model_plan_path: Path,
    full_model_plan_path: Path,
    policy_blocks: set[str],
    block_labels: dict[str, str],
    block_color_order: list[str],
    block_normalize_map: dict[str, str] | None = None,
    excluded_blocks: set[str] | None = None,
    x_max_override: float | None = None,
    outcome_order: Sequence[str] | None = None,
    panel_nrows: int | None = None,
    panel_ncols: int | None = None,
    y_axis_label: str = "Block",
    x_axis_label: str = r"Standardised perturbation effect $|PE^{\mathrm{choice}}_{\mathrm{std}}|$",
    y_label_wrap_width: int = 24,
    highlight_outcomes: Sequence[str] | None = None,
    highlight_color: str = "#d63a31",
    highlight_linewidth: float = 2.8,
    output_path: Path | None = None,
    show: bool = True,
) -> str | None:
    normalize_map = dict(block_normalize_map or {})
    excluded = set(excluded_blocks or set())

    def _norm_block(block: str) -> str:
        raw = str(block).strip()
        return canonical_choice_engine_block(normalize_map.get(raw, raw))

    def _label_outcome(outcome: str) -> str:
        return label_outcome(outcome)

    if not ref_model_plan_path.exists():
        return f"No reference model plan found for {context_label} perturbation-signal plot."

    plan_obj = json.loads(ref_model_plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan_obj, list):
        return f"Reference model plan format is invalid for {context_label} perturbation-signal plot."

    entries: list[dict[str, object]] = [
        row for row in plan_obj if isinstance(row, dict) and row.get("outcome") and row.get("model")
    ]

    extra_flu_outcome = "flu_vaccinated_2023_2024_no_habit"
    extra_flu_model = "B_FLU_vaccinated_2023_2024_ENGINES"
    extra_flu_dir = ref_ice_root / extra_flu_outcome
    if extra_flu_dir.exists() and not any(str(row.get("outcome", "")).strip() == extra_flu_outcome for row in entries):
        full_plan_obj: list[dict[str, object]] = []
        if full_model_plan_path.exists():
            raw_full = json.loads(full_model_plan_path.read_text(encoding="utf-8"))
            if isinstance(raw_full, list):
                full_plan_obj = [r for r in raw_full if isinstance(r, dict)]
        extra_predictors: dict[str, object] = {}
        for row in full_plan_obj:
            if (
                str(row.get("model", "")).strip() == extra_flu_model
                and str(row.get("outcome", "")).strip() == "flu_vaccinated_2023_2024"
            ):
                cand = row.get("predictors")
                if isinstance(cand, dict):
                    extra_predictors = cand
                    break
        entries.append(
            {
                "outcome": extra_flu_outcome,
                "model": extra_flu_model,
                "predictors": extra_predictors,
            }
        )
    if not entries:
        return f"No valid reference model entries found for {context_label} perturbation-signal plot."

    target_block_rows: list[dict[str, str]] = []
    for entry in entries:
        outcome = str(entry["outcome"]).strip()
        predictors = entry.get("predictors")
        if not outcome or not isinstance(predictors, dict):
            continue
        for block, cols in predictors.items():
            if not isinstance(block, str) or not isinstance(cols, list):
                continue
            block_norm = _norm_block(block)
            for col in cols:
                target = str(col).strip()
                if target:
                    target_block_rows.append({"outcome": outcome, "target": target, "block": block_norm})

    target_block_df = (
        pl.DataFrame(target_block_rows).unique(subset=["outcome", "target"], keep="first")
        if target_block_rows
        else pl.DataFrame()
    )

    ref_frames: list[pl.DataFrame] = []
    for entry in entries:
        outcome = str(entry["outcome"]).strip()
        model_ref = str(entry["model"]).strip()
        if not outcome or not model_ref:
            continue
        outcome_dir = ref_ice_root / outcome
        candidate_paths = [
            outcome_dir / f"ice_ref__{model_ref}.csv",
            outcome_dir / "ice_ref__linear.csv",
            outcome_dir / "ice_ref.csv",
        ]
        ref_path = next((p for p in candidate_paths if p.exists()), None)
        if ref_path is None:
            continue
        df = read_csv_fast(ref_path)
        if not {"target", "ice_low", "ice_high", "delta_z"}.issubset(set(df.columns)):
            continue
        df = df.with_columns(
            pl.lit(outcome).alias("outcome"),
            pl.col("target").cast(pl.Utf8),
            (pl.col("ice_high").cast(pl.Float64) - pl.col("ice_low").cast(pl.Float64)).alias("pe_ref"),
            pl.col("delta_z").cast(pl.Float64),
        ).with_columns(
            pl.when(pl.col("delta_z").is_finite() & (pl.col("delta_z") != 0))
            .then((pl.col("pe_ref") / pl.col("delta_z")).abs())
            .otherwise(float("nan"))
            .alias("pe_ref_std")
        )
        ref_frames.append(df.select("outcome", "target", "pe_ref_std"))

    if not ref_frames:
        return f"No reference ICE exports found for {context_label} signal plot."

    ref_all = pl.concat(ref_frames, how="diagonal").filter(pl.col("pe_ref_std").is_finite())
    if ref_all.is_empty():
        return f"No finite reference PE_z values available for {context_label} signal plot."

    if not target_block_df.is_empty():
        ref_all = ref_all.join(target_block_df, on=["outcome", "target"], how="inner")
    if ref_all.is_empty():
        return f"No target-to-block mapping available for {context_label} signal plot."

    block_meta_rows: list[dict[str, str]] = []
    for entry in entries:
        predictors = entry.get("predictors")
        if not isinstance(predictors, dict):
            continue
        for block in predictors:
            if isinstance(block, str) and block.strip():
                block_meta_rows.append({"block": _norm_block(block)})
    all_blocks_from_plan = (
        pl.DataFrame(block_meta_rows).unique().get_column("block").to_list() if block_meta_rows else []
    )

    if "mutable" in context_label:
        block_universe = [b for b in all_blocks_from_plan if b in policy_blocks]
    elif "fixed" in context_label:
        block_universe = [b for b in all_blocks_from_plan if b not in policy_blocks]
    else:
        block_universe = list(all_blocks_from_plan)

    if not block_universe:
        block_universe = [str(v) for v in ref_all.get_column("block").unique().to_list()]
    block_universe = [b for b in block_universe if b not in excluded]

    presence_df = (
        ref_all.select("outcome", "block").unique().group_by("block").agg(pl.len().alias("n_outcomes_present"))
    )
    presence_map = {str(row["block"]): int(row["n_outcomes_present"]) for row in presence_df.to_dicts()}

    order_index = {b: i for i, b in enumerate(block_color_order)}
    block_universe = sorted(
        dict.fromkeys(block_universe),
        key=lambda b: (-presence_map.get(str(b), 0), order_index.get(str(b), 10_000), str(b)),
    )

    outcomes_present = {str(v) for v in ref_all.get_column("outcome").unique().to_list()}
    if outcome_order:
        outcomes = [str(o) for o in outcome_order if str(o) in outcomes_present]
        outcomes.extend(
            o for o in [str(e["outcome"]).strip() for e in entries] if o in outcomes_present and o not in set(outcomes)
        )
    else:
        outcomes = [o for o in dict.fromkeys(str(e["outcome"]).strip() for e in entries) if o in outcomes_present]
    override_ok = x_max_override is not None and np.isfinite(float(x_max_override)) and float(x_max_override) > 0
    if override_ok:
        cap = float(x_max_override)
        x_left = 0.0
    else:
        arr = ref_all.get_column("pe_ref_std").to_numpy()
        cap = float(np.quantile(np.abs(arr), 0.995)) if arr.size else 1.0
        cap = max(cap, 0.05)
        x_left = -0.02 * cap

    n_outcomes = len(outcomes)
    if panel_nrows is not None and panel_ncols is not None:
        nrows = max(1, int(panel_nrows))
        ncols = max(1, int(panel_ncols))
        if nrows * ncols < n_outcomes:
            return f"Requested panel layout {nrows}x{ncols} is too small for {n_outcomes} {context_label} outcomes."
    else:
        nrows = 2 if n_outcomes > 1 else 1
        ncols = int(np.ceil(n_outcomes / nrows))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.0 * nrows),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    default_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#4C78A8"])
    block_color_map = {b: default_cycle[i % len(default_cycle)] for i, b in enumerate(block_universe)}
    highlight_set = {str(outcome).strip() for outcome in (highlight_outcomes or [])}

    for ax, outcome in zip(axes, outcomes, strict=False):
        sub = ref_all.filter(pl.col("outcome") == outcome)
        panel_present = [str(v) for v in sub.get_column("block").unique().to_list()]
        blocks_for_panel = [b for b in block_universe if b in set(panel_present)]
        if not blocks_for_panel:
            ax.axis("off")
            continue
        positions = np.arange(1, len(blocks_for_panel) + 1, dtype=float)
        labels = [
            "\n".join(
                textwrap.wrap(
                    block_labels[canonical_choice_engine_block(b)],
                    width=y_label_wrap_width,
                )
            )
            for b in blocks_for_panel
        ]
        for y, block in zip(positions, blocks_for_panel, strict=False):
            vals = sub.filter(pl.col("block") == block).get_column("pe_ref_std").to_numpy()
            vals = np.clip(vals, 0.0, cap)
            if vals.size:
                color = block_color_map.get(block, "#4C78A8")
                q10, q25, q50, q75, q90 = np.quantile(vals, [0.10, 0.25, 0.50, 0.75, 0.90])
                ax.hlines(y, q10, q90, color=color, linewidth=1.0, alpha=0.9, zorder=3, clip_on=False)
                ax.hlines(y, q25, q75, color=color, linewidth=3.0, alpha=0.95, zorder=4, clip_on=False)
                ax.vlines(
                    [q10, q90], y - 0.08, y + 0.08, color=color, linewidth=0.9, alpha=0.9, zorder=3, clip_on=False
                )
                ax.vlines(
                    [q25, q75], y - 0.10, y + 0.10, color=color, linewidth=1.4, alpha=0.95, zorder=4, clip_on=False
                )
                ax.scatter([q50], [y], color="black", s=10, zorder=5)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=8.8)
        ax.tick_params(axis="x", labelsize=8.6)
        ax.grid(axis="x", alpha=0.2)
        ax.set_xlim(x_left, cap)
        ax.set_title(_label_outcome(outcome), fontsize=10.8)
        ax.set_xlabel("")
        if str(outcome) in highlight_set:
            ax.add_patch(
                Rectangle(
                    (0, 0),
                    1,
                    1,
                    transform=ax.transAxes,
                    fill=False,
                    edgecolor=highlight_color,
                    linewidth=highlight_linewidth,
                    zorder=20,
                    clip_on=False,
                )
            )

    for ax in axes[len(outcomes) :]:
        ax.axis("off")

    fig.supylabel(y_axis_label, fontsize=10.8)
    fig.supxlabel(x_axis_label, fontsize=10.8)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return None


__all__ = [
    "render_lobo_bic_block_outcome_heatmap",
    "render_reference_signal_magnitude",
]
