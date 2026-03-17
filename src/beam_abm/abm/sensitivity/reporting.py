"""Reporting helpers for sensitivity pipeline artefacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from .analysis import _sanitize_name


def _plot_morris_heatmaps(
    *,
    morris_per_scenario: pl.DataFrame,
    output_dir: Path,
) -> None:
    """Create per-scenario Morris heatmaps for ``mu*`` and ``sigma``."""
    if morris_per_scenario.is_empty():
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = sorted(set(morris_per_scenario.get_column("scenario").to_list()))

    for scenario_name in scenarios:
        scenario_df = morris_per_scenario.filter(pl.col("scenario") == scenario_name)
        if scenario_df.is_empty():
            continue

        mu_star_pivot = scenario_df.pivot(
            values="mu_star",
            index="parameter",
            on="output",
            aggregate_function="first",
        )
        sigma_pivot = scenario_df.pivot(
            values="sigma",
            index="parameter",
            on="output",
            aggregate_function="first",
        )

        mu_pdf = mu_star_pivot.to_pandas().set_index("parameter")
        sigma_pdf = sigma_pivot.to_pandas().set_index("parameter")

        fig, axes = plt.subplots(1, 2, figsize=(15, 10), constrained_layout=True)
        sns.heatmap(mu_pdf, cmap="viridis", ax=axes[0])
        axes[0].set_title(f"{scenario_name}: Morris mu*")
        axes[0].set_xlabel("Output")
        axes[0].set_ylabel("Parameter")

        sns.heatmap(sigma_pdf, cmap="magma", ax=axes[1])
        axes[1].set_title(f"{scenario_name}: Morris sigma")
        axes[1].set_xlabel("Output")
        axes[1].set_ylabel("Parameter")

        fig_path = output_dir / f"morris_heatmap_{_sanitize_name(scenario_name)}.png"
        fig.savefig(fig_path, dpi=180)
        plt.close(fig)


def _write_shortlist_rationale(
    *,
    shortlist_df: pl.DataFrame,
    output_path: Path,
    sigma_threshold: float,
) -> None:
    lines = [
        "# Parameter Shortlist Rationale",
        "",
        f"- Sigma-flag threshold: `{sigma_threshold:.6f}`",
        f"- Selected parameters: `{shortlist_df.height}`",
        "",
        "| rank | parameter | mu*_max | sigma_max | by_importance | by_sigma |",
        "|---:|---|---:|---:|:---:|:---:|",
    ]

    for row in shortlist_df.iter_rows(named=True):
        lines.append(
            "| {rank} | {parameter} | {mu:.4f} | {sigma:.4f} | {imp} | {sig} |".format(
                rank=row["shortlist_rank"],
                parameter=row["parameter"],
                mu=float(row["mu_star_max"]),
                sigma=float(row["sigma_max"]),
                imp="yes" if bool(row["selected_by_importance"]) else "no",
                sig="yes" if bool(row["selected_by_sigma"]) else "no",
            )
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_sobol_narrative(
    *,
    sobol_df: pl.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# Sobol Uncertainty Drivers (Flagship Scenarios)",
        "",
    ]

    if sobol_df.is_empty():
        lines.append("No retained surrogate models met the minimum fit threshold.")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return

    grouped = sobol_df.group_by("scenario", "output")
    for (scenario_name, output_name), group_df in grouped:
        top_rows = group_df.sort("S_Ti", descending=True).head(3)
        max_gap = float(group_df.select(pl.col("interaction_gap").max()).item())
        interaction_note = "interactions appear relevant" if max_gap >= 0.05 else "interactions appear limited"
        lines.append(f"## {scenario_name} · {output_name}")
        lines.append(f"- Interaction assessment: **{interaction_note}** (max S_Ti - S_i = {max_gap:.3f})")
        for row in top_rows.iter_rows(named=True):
            lines.append(
                "- {param}: S_i={si:.3f}, S_Ti={sti:.3f}, gap={gap:.3f}".format(
                    param=row["parameter"],
                    si=float(row["S_i"]),
                    sti=float(row["S_Ti"]),
                    gap=float(row["interaction_gap"]),
                )
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_non_flagship_morris_summary(
    *,
    morris_per_scenario: pl.DataFrame,
    non_flagship_scenarios: tuple[str, ...],
    output_path: Path,
) -> None:
    lines = [
        "# Non-Flagship Scenarios: Morris Robustness Summary",
        "",
        "(Morris-only reporting; no Sobol decomposition.)",
        "",
    ]

    if not non_flagship_scenarios:
        lines.append("No non-flagship scenarios available in this run.")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return

    filtered = morris_per_scenario.filter(pl.col("scenario").is_in(non_flagship_scenarios))
    if filtered.is_empty():
        lines.append("No Morris indices available for non-flagship scenarios.")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return

    for scenario_name in non_flagship_scenarios:
        scenario_df = filtered.filter(pl.col("scenario") == scenario_name)
        if scenario_df.is_empty():
            continue
        ranked = (
            scenario_df.group_by("parameter")
            .agg(
                pl.col("mu_star").median().alias("mu_star_median"),
                pl.col("sigma").max().alias("sigma_max"),
            )
            .sort("mu_star_median", descending=True)
            .head(5)
        )
        lines.append(f"## {scenario_name}")
        for row in ranked.iter_rows(named=True):
            lines.append(
                "- {param}: mu*_median={mu:.4f}, sigma_max={sigma:.4f}".format(
                    param=row["parameter"],
                    mu=float(row["mu_star_median"]),
                    sigma=float(row["sigma_max"]),
                )
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
