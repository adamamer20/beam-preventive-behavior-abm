"""Build canonical scenario-level summary directly from run trajectories.

The canonical summary contains one row per (scenario, effect_regime) with:
- final effects (vax + NPI composite)
- AUC over tick for vax + NPI composite

NPI composite is the mean of:
1) mask_when_symptomatic_crowded_mean
2) stay_home_when_symptomatic_mean
3) mask_when_pressure_high_mean

Also computes post-burn summary metrics:
- For shock runs: matched effects versus baseline_importation by (rep, seed).
  If no shock-tagged baseline_importation run is available, falls back to
  baseline_importation regardless of effect_regime tag.
- For perturbation contrast runs: direct post-burn contrast trajectory metrics
  from the contrast run itself
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import polars as pl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("abm/output/runs"),
        help="Directory with per-run outputs under runs/<run_id>/trajectories/outcomes.parquet.",
    )
    parser.add_argument(
        "--scenario-defs",
        type=Path,
        default=Path("src/beam_abm/abm/scenario_defs.py"),
        help="Path to scenario_defs.py for canonical scenario keys.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("abm/output/report_summary_canonical_effects.csv"),
        help="Output canonical summary CSV path.",
    )
    parser.add_argument(
        "--burn-in-tick",
        type=int,
        default=12,
        help="Post-burn start tick used for baseline-relative metrics (default: 12).",
    )
    return parser.parse_args()


def _load_scenario_keys(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r'^\s*"([a-zA-Z0-9_]+)":\s*ScenarioConfig\(', text, flags=re.M))


def _parse_run_id(
    run_id: str,
    scenario_keys: set[str],
    run_dir: Path,
) -> tuple[str, str, str | None, str | None, str | None]:
    m = re.match(r"^(?P<prefix>.+)_rep\d+_seed\d+$", run_id)
    prefix = m.group("prefix") if m is not None else run_id

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
    m = re.match(r"^.+_rep(?P<rep>\d+)_seed(?P<seed>\d+)$", run_id)
    if m is None:
        return None, None
    return int(m.group("rep")), int(m.group("seed"))


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


def main() -> None:
    args = _parse_args()
    scenario_keys = _load_scenario_keys(args.scenario_defs)

    # (scenario, effect_regime, reference_scenario) -> list[(final_vax, final_npi, auc_vax, auc_npi)]
    per_group: dict[tuple[str, str, str], list[tuple[float, float, float, float]]] = defaultdict(list)
    # Per-run level trajectories for matched baseline-relative post-burn effects.
    run_levels: list[dict[str, object]] = []

    for run_dir in sorted(args.runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        outcomes_path = run_dir / "trajectories" / "outcomes.parquet"
        if not outcomes_path.exists():
            continue

        try:
            df = pl.read_parquet(outcomes_path).select(
                [
                    "tick",
                    "vax_willingness_T12_mean",
                    "mask_when_symptomatic_crowded_mean",
                    "stay_home_when_symptomatic_mean",
                    "mask_when_pressure_high_mean",
                ]
            )
        except pl.exceptions.ColumnNotFoundError:
            # Skip legacy/incompatible runs lacking canonical outcome columns.
            continue

        if df.height == 0:
            continue

        df = df.sort("tick")
        ticks = [int(v) for v in df["tick"].to_list()]
        vax = [float(v) for v in df["vax_willingness_T12_mean"].to_list()]
        mask_sym = [float(v) for v in df["mask_when_symptomatic_crowded_mean"].to_list()]
        stay_home = [float(v) for v in df["stay_home_when_symptomatic_mean"].to_list()]
        mask_pressure = [float(v) for v in df["mask_when_pressure_high_mean"].to_list()]
        npi = [(mask_sym[i] + stay_home[i] + mask_pressure[i]) / 3.0 for i in range(len(ticks))]

        v0 = vax[0]
        ms0 = mask_sym[0]
        sh0 = stay_home[0]
        mp0 = mask_pressure[0]

        vax_delta = [v - v0 for v in vax]
        mask_sym_delta = [v - ms0 for v in mask_sym]
        stay_home_delta = [v - sh0 for v in stay_home]
        mask_pressure_delta = [v - mp0 for v in mask_pressure]
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

    # Index shock baseline runs for matched post-burn deltas vs baseline_importation.
    baseline_index: dict[tuple[str, str, int, int], dict[str, object]] = {}
    # Fallback index keyed without effect regime, for legacy runs that were
    # emitted without explicit effect_regime metadata/suffix in run_id.
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

    # (scenario, effect_regime, reference_scenario) -> list[(final_vax, final_npi, auc_vax, auc_npi)]
    per_group_relative: dict[tuple[str, str, str], list[tuple[float, float, float, float]]] = defaultdict(list)
    for rec in run_levels:
        scenario = str(rec["scenario"])
        effect_regime = str(rec["effect_regime"])
        reference_scenario = str(rec["reference_scenario"])
        rep = rec["rep"]
        seed = rec["seed"]
        if rep is None or seed is None:
            continue

        # Perturbation rows should report the post-burn contrast metric directly.
        if effect_regime == "perturbation":
            if str(rec.get("perturbation_world")) != "contrast":
                continue
            tick_to_idx_this = {int(t): i for i, t in enumerate(rec["ticks"])}
            shared_post_burn = [t for t in sorted(tick_to_idx_this) if t >= int(args.burn_in_tick)]
            if not shared_post_burn:
                continue

            dv = [float(rec["vax"][tick_to_idx_this[t]]) for t in shared_post_burn]
            dn = [float(rec["npi"][tick_to_idx_this[t]]) for t in shared_post_burn]
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

        tick_to_idx_this = {int(t): i for i, t in enumerate(rec["ticks"])}
        tick_to_idx_base = {int(t): i for i, t in enumerate(base["ticks"])}
        shared_ticks = sorted(set(tick_to_idx_this).intersection(tick_to_idx_base))
        shared_post_burn = [t for t in shared_ticks if t >= int(args.burn_in_tick)]
        if not shared_post_burn:
            continue

        dv: list[float] = []
        dn: list[float] = []
        for t in shared_post_burn:
            i_this = tick_to_idx_this[t]
            i_base = tick_to_idx_base[t]
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
                "burn_in_tick": int(args.burn_in_tick),
                "n_relative_runs": len(rel_stats),
                "post_burn_final_vax_vs_baseline_importation": (
                    _mean(rel_final_vax) if rel_final_vax else float("nan")
                ),
                "post_burn_final_npi_vs_baseline_importation": (
                    _mean(rel_final_npi) if rel_final_npi else float("nan")
                ),
                "post_burn_vax_auc_vs_baseline_importation": _mean(rel_auc_vax) if rel_auc_vax else float("nan"),
                "post_burn_npi_auc_vs_baseline_importation": _mean(rel_auc_npi) if rel_auc_npi else float("nan"),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
