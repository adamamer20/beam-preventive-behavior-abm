"""ABM thesis export and canonical summary workflows."""

from __future__ import annotations

import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl

__all__ = [
    "rebuild_canonical_summary",
    "export_thesis_artifacts",
]


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _load_scenario_keys(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r'^\s*"([a-zA-Z0-9_]+)":\s*ScenarioConfig\(', text, flags=re.M))


def _parse_run_id(
    run_id: str,
    scenario_keys: set[str],
    run_dir: Path,
) -> tuple[str, str, str | None, str | None, str | None]:
    match = re.match(r"^(?P<prefix>.+)_rep\d+_seed\d+$", run_id)
    prefix = match.group("prefix") if match is not None else run_id

    scenario: str
    regime: str
    perturbation_world: str | None = None
    if prefix.endswith("_perturbation_contrast"):
        base = prefix[: -len("_perturbation_contrast")]
        if base in scenario_keys:
            scenario, regime = base, "perturbation"
            perturbation_world = "contrast"
            return _resolve_with_metadata(scenario, regime, perturbation_world, run_dir)
    if prefix.endswith("_perturbation_low"):
        base = prefix[: -len("_perturbation_low")]
        if base in scenario_keys:
            scenario, regime = base, "perturbation"
            perturbation_world = "low"
            return _resolve_with_metadata(scenario, regime, perturbation_world, run_dir)
    if prefix.endswith("_perturbation_high"):
        base = prefix[: -len("_perturbation_high")]
        if base in scenario_keys:
            scenario, regime = base, "perturbation"
            perturbation_world = "high"
            return _resolve_with_metadata(scenario, regime, perturbation_world, run_dir)
    elif prefix.endswith("_shock"):
        base = prefix[: -len("_shock")]
        if base in scenario_keys:
            scenario, regime = base, "shock"
            return _resolve_with_metadata(scenario, regime, perturbation_world, run_dir)

    if prefix in scenario_keys:
        scenario, regime = prefix, "unknown"
    else:
        scenario, regime = prefix, "unknown"
    return _resolve_with_metadata(scenario, regime, perturbation_world, run_dir)


def _parse_rep_seed(run_id: str) -> tuple[int | None, int | None]:
    match = re.match(r"^.+_rep(?P<rep>\d+)_seed(?P<seed>\d+)$", run_id)
    if match is None:
        return None, None
    return int(match.group("rep")), int(match.group("seed"))


def _resolve_with_metadata(
    scenario: str,
    effect_regime: str,
    perturbation_world: str | None,
    run_dir: Path,
) -> tuple[str, str, str | None, str | None, str | None]:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return scenario, effect_regime, perturbation_world, None, None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return scenario, effect_regime, perturbation_world, None, None

    scenario_cfg = payload.get("scenario_config")
    if isinstance(scenario_cfg, dict):
        scenario_key = scenario_cfg.get("scenario_key")
        if isinstance(scenario_key, str) and scenario_key:
            scenario = scenario_key

    raw_regime = payload.get("effect_regime")
    if isinstance(raw_regime, str):
        norm = raw_regime.strip().lower()
        if norm in {"perturbation", "shock"}:
            effect_regime = norm

    raw_world = payload.get("perturbation_world")
    if isinstance(raw_world, str):
        norm_world = raw_world.strip().lower()
        if norm_world in {"low", "high", "contrast"}:
            perturbation_world = norm_world

    contrast_type = payload.get("contrast_type") if isinstance(payload.get("contrast_type"), str) else None
    reference_scenario = (
        payload.get("reference_scenario") if isinstance(payload.get("reference_scenario"), str) else None
    )
    if effect_regime == "perturbation" and contrast_type == "high_minus_low" and perturbation_world is None:
        perturbation_world = "contrast"

    return scenario, effect_regime, perturbation_world, contrast_type, reference_scenario


def _trapz(x: list[int], y: list[float]) -> float:
    return sum((x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2.0 for i in range(len(x) - 1))


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values))


def rebuild_canonical_summary(
    *,
    runs_dir: Path = Path("abm/output/runs"),
    scenario_defs: Path = Path("src/beam_abm/abm/scenario_defs.py"),
    output: Path = Path("abm/output/report_summary_canonical_effects.csv"),
    burn_in_tick: int = 12,
) -> Path:
    """Build canonical scenario-level summary from run trajectories."""
    scenario_keys = _load_scenario_keys(scenario_defs)

    per_group: dict[tuple[str, str, str], list[tuple[float, float, float, float]]] = defaultdict(list)
    run_levels: list[dict[str, object]] = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        outcomes_path = run_dir / "trajectories" / "outcomes.parquet"
        if not outcomes_path.exists():
            continue

        try:
            frame = pl.read_parquet(outcomes_path).select(
                [
                    "tick",
                    "vax_willingness_T12_mean",
                    "mask_when_symptomatic_crowded_mean",
                    "stay_home_when_symptomatic_mean",
                    "mask_when_pressure_high_mean",
                ]
            )
        except pl.exceptions.ColumnNotFoundError:
            continue

        if frame.height == 0:
            continue

        frame = frame.sort("tick")
        ticks = [int(value) for value in frame["tick"].to_list()]
        vax = [float(value) for value in frame["vax_willingness_T12_mean"].to_list()]
        mask_sym = [float(value) for value in frame["mask_when_symptomatic_crowded_mean"].to_list()]
        stay_home = [float(value) for value in frame["stay_home_when_symptomatic_mean"].to_list()]
        mask_pressure = [float(value) for value in frame["mask_when_pressure_high_mean"].to_list()]
        npi = [(mask_sym[i] + stay_home[i] + mask_pressure[i]) / 3.0 for i in range(len(ticks))]

        v0 = vax[0]
        ms0 = mask_sym[0]
        sh0 = stay_home[0]
        mp0 = mask_pressure[0]

        vax_delta = [value - v0 for value in vax]
        mask_sym_delta = [value - ms0 for value in mask_sym]
        stay_home_delta = [value - sh0 for value in stay_home]
        mask_pressure_delta = [value - mp0 for value in mask_pressure]
        npi_delta = [(mask_sym_delta[i] + stay_home_delta[i] + mask_pressure_delta[i]) / 3.0 for i in range(len(ticks))]

        scenario, effect_regime, perturbation_world, _, reference_scenario = _parse_run_id(
            run_id, scenario_keys, run_dir
        )
        rep, seed = _parse_rep_seed(run_id)
        reference = reference_scenario
        if reference is None:
            if effect_regime == "perturbation":
                reference = "q25_to_q75_high_minus_low"
            elif effect_regime == "shock":
                reference = "baseline_importation"

        run_levels.append(
            {
                "run_id": run_id,
                "scenario": scenario,
                "effect_regime": effect_regime,
                "perturbation_world": perturbation_world,
                "reference_scenario": reference,
                "rep": rep,
                "seed": seed,
                "ticks": ticks,
                "vax": vax,
                "npi": npi,
            }
        )

        if effect_regime not in {"perturbation", "shock"}:
            continue
        if effect_regime == "perturbation" and perturbation_world != "contrast":
            continue

        per_group[(scenario, effect_regime, reference)].append(
            (
                vax_delta[-1],
                npi_delta[-1],
                _trapz(ticks, vax_delta),
                _trapz(ticks, npi_delta),
            )
        )

    baseline_index: dict[tuple[str, str, int, int], dict[str, object]] = {}
    baseline_index_fallback: dict[tuple[str, int, int], dict[str, object]] = {}
    for rec in run_levels:
        rep = rec["rep"]
        seed = rec["seed"]
        if str(rec["scenario"]) == "baseline_importation" and rep is not None and seed is not None:
            baseline_index_fallback[(str(rec["scenario"]), int(rep), int(seed))] = rec

        if str(rec["scenario"]) != "baseline_importation" or str(rec["effect_regime"]) != "shock":
            continue
        if rep is None or seed is None:
            continue
        key = (str(rec["scenario"]), str(rec["effect_regime"]), int(rep), int(seed))
        baseline_index[key] = rec

    per_group_relative: dict[tuple[str, str, str], list[tuple[float, float, float, float]]] = defaultdict(list)
    for rec in run_levels:
        scenario = str(rec["scenario"])
        effect_regime = str(rec["effect_regime"])
        reference_scenario = str(rec["reference_scenario"])
        rep = rec["rep"]
        seed = rec["seed"]
        if rep is None or seed is None:
            continue

        if effect_regime == "perturbation":
            if str(rec.get("perturbation_world")) != "contrast":
                continue
            tick_to_idx_this = {int(tick): i for i, tick in enumerate(rec["ticks"])}
            shared_post_burn = [tick for tick in sorted(tick_to_idx_this) if tick >= int(burn_in_tick)]
            if not shared_post_burn:
                continue

            dv = [float(rec["vax"][tick_to_idx_this[tick]]) for tick in shared_post_burn]
            dn = [float(rec["npi"][tick_to_idx_this[tick]]) for tick in shared_post_burn]
            final_vax_rel = dv[-1]
            final_npi_rel = dn[-1]
            auc_vax_rel = float(dv[0]) if len(shared_post_burn) == 1 else _trapz(shared_post_burn, dv)
            auc_npi_rel = float(dn[0]) if len(shared_post_burn) == 1 else _trapz(shared_post_burn, dn)
            per_group_relative[(scenario, effect_regime, reference_scenario)].append(
                (final_vax_rel, final_npi_rel, auc_vax_rel, auc_npi_rel)
            )
            continue

        base = baseline_index.get((reference_scenario, effect_regime, int(rep), int(seed)))
        if base is None:
            base = baseline_index.get(("baseline_importation", effect_regime, int(rep), int(seed)))
        if base is None:
            base = baseline_index_fallback.get((reference_scenario, int(rep), int(seed)))
        if base is None:
            base = baseline_index_fallback.get(("baseline_importation", int(rep), int(seed)))
        if base is None:
            continue

        tick_to_idx_this = {int(tick): i for i, tick in enumerate(rec["ticks"])}
        tick_to_idx_base = {int(tick): i for i, tick in enumerate(base["ticks"])}
        shared_ticks = sorted(set(tick_to_idx_this).intersection(tick_to_idx_base))
        shared_post_burn = [tick for tick in shared_ticks if tick >= int(burn_in_tick)]
        if not shared_post_burn:
            continue

        dv: list[float] = []
        dn: list[float] = []
        for tick in shared_post_burn:
            i_this = tick_to_idx_this[tick]
            i_base = tick_to_idx_base[tick]
            dv.append(float(rec["vax"][i_this]) - float(base["vax"][i_base]))
            dn.append(float(rec["npi"][i_this]) - float(base["npi"][i_base]))

        final_vax_rel = dv[-1]
        final_npi_rel = dn[-1]
        auc_vax_rel = float(dv[0]) if len(shared_post_burn) == 1 else _trapz(shared_post_burn, dv)
        auc_npi_rel = float(dn[0]) if len(shared_post_burn) == 1 else _trapz(shared_post_burn, dn)
        per_group_relative[(scenario, effect_regime, reference_scenario)].append(
            (final_vax_rel, final_npi_rel, auc_vax_rel, auc_npi_rel)
        )

    rows: list[dict[str, str | int | float]] = []
    for (scenario, effect_regime, reference_scenario), run_stats in sorted(per_group.items()):
        final_vax = [x[0] for x in run_stats]
        final_npi = [x[1] for x in run_stats]
        auc_vax = [x[2] for x in run_stats]
        auc_npi = [x[3] for x in run_stats]
        rel_stats = per_group_relative.get((scenario, effect_regime, reference_scenario), [])
        rel_final_vax = [x[0] for x in rel_stats]
        rel_final_npi = [x[1] for x in rel_stats]
        rel_auc_vax = [x[2] for x in rel_stats]
        rel_auc_npi = [x[3] for x in rel_stats]
        rows.append(
            {
                "scenario": scenario,
                "effect_regime": effect_regime,
                "n_runs": len(run_stats),
                "final_vax_effect": _mean(final_vax),
                "final_npi_effect": _mean(final_npi),
                "vax_auc": _mean(auc_vax),
                "npi_auc": _mean(auc_npi),
                "reference_scenario": reference_scenario,
                "burn_in_tick": int(burn_in_tick),
                "n_relative_runs": len(rel_stats),
                "post_burn_final_vax_vs_baseline_importation": _mean(rel_final_vax) if rel_final_vax else float("nan"),
                "post_burn_final_npi_vs_baseline_importation": _mean(rel_final_npi) if rel_final_npi else float("nan"),
                "post_burn_vax_auc_vs_baseline_importation": _mean(rel_auc_vax) if rel_auc_vax else float("nan"),
                "post_burn_npi_auc_vs_baseline_importation": _mean(rel_auc_npi) if rel_auc_npi else float("nan"),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "scenario",
        "effect_regime",
        "n_runs",
        "final_vax_effect",
        "final_npi_effect",
        "vax_auc",
        "npi_auc",
        "reference_scenario",
        "burn_in_tick",
        "n_relative_runs",
        "post_burn_final_vax_vs_baseline_importation",
        "post_burn_final_npi_vs_baseline_importation",
        "post_burn_vax_auc_vs_baseline_importation",
        "post_burn_npi_auc_vs_baseline_importation",
    ]
    pl.DataFrame(rows).select(columns).write_csv(output)
    return output


def _mean_ci_columns(col: str) -> list[pl.Expr]:
    n_col = f"n_{col}"
    mean_col = col
    sd_col = f"{col}_sd"
    return [
        pl.col(col).count().alias(n_col),
        pl.col(col).mean().alias(mean_col),
        pl.col(col).std().alias(sd_col),
    ]


def _export_run_context_and_auc_table(repo_root: Path, abm_out: Path, artifacts_root: Path) -> None:
    runs_root = abm_out / "runs"
    if not runs_root.exists():
        raise FileNotFoundError(f"Missing required ABM runs root: {runs_root}")

    sys.path.insert(0, str((repo_root / "thesis").resolve()))
    from utils.ch05_plot_tables import load_abm_context, matched_effects

    context = load_abm_context(runs_root=runs_root)
    run_ctx_root = artifacts_root / "run_context"
    run_ctx_root.mkdir(parents=True, exist_ok=True)
    context.run_index.write_parquet(run_ctx_root / "run_index.parquet")
    context.series.write_parquet(run_ctx_root / "series.parquet")
    context.grouped_outcomes.write_parquet(run_ctx_root / "grouped_outcomes.parquet")
    context.grouped_mediators.write_parquet(run_ctx_root / "grouped_mediators.parquet")

    effects = matched_effects(context)
    effects.write_parquet(run_ctx_root / "matched_effects.parquet")

    auc_table = (
        effects.group_by("scenario")
        .agg(
            *_mean_ci_columns("delta_vax_auc"),
            *_mean_ci_columns("delta_mask_auc"),
        )
        .with_columns(
            pl.when(pl.col("n_delta_vax_auc") > 1)
            .then(1.96 * pl.col("delta_vax_auc_sd") / pl.col("n_delta_vax_auc").cast(pl.Float64).sqrt())
            .otherwise(0.0)
            .alias("delta_vax_auc_ci_half"),
            pl.when(pl.col("n_delta_mask_auc") > 1)
            .then(1.96 * pl.col("delta_mask_auc_sd") / pl.col("n_delta_mask_auc").cast(pl.Float64).sqrt())
            .otherwise(0.0)
            .alias("delta_mask_auc_ci_half"),
        )
        .with_columns(
            (pl.col("delta_vax_auc") - pl.col("delta_vax_auc_ci_half")).alias("delta_vax_auc_ci_low"),
            (pl.col("delta_vax_auc") + pl.col("delta_vax_auc_ci_half")).alias("delta_vax_auc_ci_high"),
            (pl.col("delta_mask_auc") - pl.col("delta_mask_auc_ci_half")).alias("delta_mask_auc_ci_low"),
            (pl.col("delta_mask_auc") + pl.col("delta_mask_auc_ci_half")).alias("delta_mask_auc_ci_high"),
        )
        .select(
            "scenario",
            "n_delta_vax_auc",
            "delta_vax_auc",
            "delta_vax_auc_ci_low",
            "delta_vax_auc_ci_high",
            "n_delta_mask_auc",
            "delta_mask_auc",
            "delta_mask_auc_ci_low",
            "delta_mask_auc_ci_high",
        )
        .rename(
            {
                "n_delta_vax_auc": "n_vax_auc",
                "n_delta_mask_auc": "n_mask_auc",
            }
        )
        .sort("scenario")
    )
    auc_table.write_csv(artifacts_root / "scenario_delta_auc_table.csv")


def export_thesis_artifacts(
    repo_root: Path,
    *,
    refresh_canonical_summary: bool = True,
    burn_in_tick: int = 12,
) -> None:
    """Export compact ABM thesis artifacts under thesis/artifacts/abm."""
    abm_out = repo_root / "abm" / "output"
    artifacts_root = repo_root / "thesis" / "artifacts" / "abm"
    if refresh_canonical_summary:
        rebuild_canonical_summary(
            runs_dir=abm_out / "runs",
            scenario_defs=repo_root / "src" / "beam_abm" / "abm" / "scenario_defs.py",
            output=abm_out / "report_summary_canonical_effects.csv",
            burn_in_tick=burn_in_tick,
        )

    required_map: dict[Path, Path] = {
        abm_out / "report_summary_canonical_effects.csv": artifacts_root / "report_summary_canonical_effects.csv",
    }
    missing = [str(src) for src in required_map if not src.exists()]
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing required ABM artifact inputs:\n{joined}")
    for src, dst in required_map.items():
        _copy_file(src, dst)

    optional_map: dict[Path, Path] = {
        abm_out / "report_summary.csv": artifacts_root / "report_summary.csv",
        abm_out / "leaderboard_dynamics_summary.csv": artifacts_root / "leaderboard_dynamics_summary.csv",
        abm_out / "scenario_delta_auc_table.csv": artifacts_root / "scenario_delta_auc_table_runtime.csv",
        abm_out / "full_sensitivity_pipeline" / "sobol" / "indices.csv": artifacts_root
        / "full_sensitivity_pipeline"
        / "sobol"
        / "indices.csv",
        abm_out / "full_sensitivity_pipeline" / "morris" / "indices_per_scenario.csv": artifacts_root
        / "full_sensitivity_pipeline"
        / "morris"
        / "indices_per_scenario.csv",
        abm_out / "full_sensitivity_pipeline" / "lhc" / "delta.parquet": artifacts_root
        / "full_sensitivity_pipeline"
        / "lhc"
        / "delta.parquet",
    }
    for src, dst in optional_map.items():
        if src.exists():
            _copy_file(src, dst)

    _export_run_context_and_auc_table(repo_root, abm_out, artifacts_root)
