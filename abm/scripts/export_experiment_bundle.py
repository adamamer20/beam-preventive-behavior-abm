"""Export a compact experiment bundle for external analysis upload.

The bundle includes:
- Runtime experiment definitions (scenarios/signals JSON).
- Benchmark metrics (if available).
- Selected run outputs (metadata + key trajectory parquet files).

By default, run selection uses the latest batch in ``abm/output/runs``,
grouped by scenario and a configurable time window.
"""

from __future__ import annotations

import argparse
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import polars as pl

RUN_NAME_RE = re.compile(r"^(?P<scenario>.+)_rep(?P<rep>\d+)_seed(?P<seed>\d+)$")

DEFAULT_PARQUET_RUN_FILES = (
    "trajectories/outcomes.parquet",
    "trajectories/outcomes_by_group.parquet",
    "trajectories/effects.parquet",
    "trajectories/mediators.parquet",
)

DEFAULT_METADATA_FILES = ("metadata.json",)

DEFAULT_CONTEXT_FILES = (
    "abm/output/runtime_artifacts/scenarios.generated.json",
    "abm/output/runtime_artifacts/signals.generated.json",
    "abm/output/benchmarks/report.json",
    "abm/output/benchmarks/report.verify.json",
)


@dataclass(frozen=True, slots=True)
class RunInfo:
    """Parsed metadata for one run directory."""

    name: str
    path: Path
    scenario: str
    mtime: float


def _parse_run_dir(run_dir: Path) -> RunInfo | None:
    match = RUN_NAME_RE.match(run_dir.name)
    if match is None:
        return None
    return RunInfo(
        name=run_dir.name,
        path=run_dir,
        scenario=match.group("scenario"),
        mtime=run_dir.stat().st_mtime,
    )


def _load_runs(runs_root: Path) -> list[RunInfo]:
    runs: list[RunInfo] = []
    for path in sorted(runs_root.iterdir()):
        if not path.is_dir():
            continue
        run = _parse_run_dir(path)
        if run is not None:
            runs.append(run)
    return runs


def _select_runs(
    runs: list[RunInfo],
    selection: str,
    scenario: str | None,
    explicit_runs: tuple[str, ...],
    batch_window_seconds: int,
) -> list[RunInfo]:
    if not runs:
        raise ValueError("No run directories found.")

    if selection == "all-scenarios":
        return sorted(runs, key=lambda run: run.name)

    if selection == "explicit":
        if not explicit_runs:
            raise ValueError("--run-name is required for --selection explicit.")
        selected = [run for run in runs if run.name in explicit_runs]
        if len(selected) != len(set(explicit_runs)):
            found = {run.name for run in selected}
            missing = [name for name in explicit_runs if name not in found]
            raise ValueError(f"Requested runs not found: {missing}")
        return sorted(selected, key=lambda run: run.name)

    if selection == "scenario":
        if not scenario:
            raise ValueError("--scenario is required for --selection scenario.")
        selected = [run for run in runs if run.scenario == scenario]
        if not selected:
            raise ValueError(f"No runs found for scenario '{scenario}'.")
        return sorted(selected, key=lambda run: run.name)

    latest = max(runs, key=lambda run: run.mtime)
    lower_bound = latest.mtime - float(batch_window_seconds)
    selected = [run for run in runs if run.scenario == latest.scenario and run.mtime >= lower_bound]
    if not selected:
        selected = [run for run in runs if run.scenario == latest.scenario]
    return sorted(selected, key=lambda run: run.name)


def _filter_baseline_runs(selected_runs: list[RunInfo], baseline_only: bool) -> list[RunInfo]:
    if not baseline_only:
        return selected_runs
    baseline_runs = [run for run in selected_runs if run.scenario.startswith("baseline")]
    if not baseline_runs:
        raise ValueError("No baseline scenarios found in selected runs (expected scenario prefix 'baseline_').")
    return baseline_runs


def _build_file_groups_by_scenario(
    selected_runs: list[RunInfo],
    metadata_files: tuple[str, ...],
    context_files: tuple[str, ...],
    manifest_path: Path,
    csv_root: Path,
) -> tuple[list[Path], dict[str, list[Path]]]:
    common_paths: list[Path] = []
    for context in context_files:
        path = Path(context)
        if path.exists():
            common_paths.append(path)
    common_paths.append(manifest_path)

    per_scenario_paths: dict[str, list[Path]] = {}
    for run in selected_runs:
        scenario_paths = per_scenario_paths.setdefault(run.scenario, [])
        for rel_file in metadata_files:
            file_path = run.path / rel_file
            if file_path.exists():
                scenario_paths.append(file_path)
        for rel_file in DEFAULT_PARQUET_RUN_FILES:
            csv_path = csv_root / run.name / rel_file.replace(".parquet", ".csv")
            if csv_path.exists():
                scenario_paths.append(csv_path)

    return common_paths, per_scenario_paths


def _convert_parquet_to_csv(selected_runs: list[RunInfo], csv_root: Path) -> None:
    for run in selected_runs:
        for rel_file in DEFAULT_PARQUET_RUN_FILES:
            parquet_path = run.path / rel_file
            if not parquet_path.exists():
                continue
            csv_rel = rel_file.replace(".parquet", ".csv")
            csv_path = csv_root / run.name / csv_rel
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            frame = pl.read_parquet(parquet_path)
            frame.write_csv(csv_path)


def _write_manifest(
    manifest_path: Path,
    selection: str,
    baseline_only: bool,
    selected_runs: list[RunInfo],
    metadata_files: tuple[str, ...],
    parquet_files: tuple[str, ...],
    context_files: tuple[str, ...],
) -> None:
    created = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        f"bundle_created_utc={created}",
        f"selection={selection}",
        f"baseline_only={str(baseline_only).lower()}",
        f"included_runs_count={len(selected_runs)}",
        "included_runs:",
    ]
    lines.extend(f"- {run.name}" for run in selected_runs)
    lines.append("")
    lines.append("included_metadata_files_per_run=" + ",".join(metadata_files))
    lines.append(
        "included_converted_csv_files_per_run=" + ",".join(file.replace(".parquet", ".csv") for file in parquet_files)
    )
    lines.append("context_files_if_present=" + ",".join(context_files))
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_zip(bundle_path: Path, file_paths: list[Path]) -> None:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    if bundle_path.exists():
        bundle_path.unlink()
    with ZipFile(bundle_path, mode="w", compression=ZIP_DEFLATED, compresslevel=9) as zip_file:
        for file_path in file_paths:
            zip_file.write(file_path, arcname=file_path.as_posix())


def _is_within_max_size(bundle_path: Path, max_size_mb: float) -> bool:
    max_bytes = int(max_size_mb * 1024 * 1024)
    return bundle_path.stat().st_size <= max_bytes


def _part_path(bundle_path: Path, parts_dir: Path, part_index: int) -> Path:
    return parts_dir / f"{bundle_path.stem}.part{part_index:03d}{bundle_path.suffix}"


def _cleanup_existing_parts(bundle_path: Path, parts_dir: Path) -> None:
    pattern = f"{bundle_path.stem}.part*{bundle_path.suffix}"
    for path in parts_dir.glob(pattern):
        path.unlink(missing_ok=True)
    bundle_path.unlink(missing_ok=True)


def _source_size_bytes(file_paths: list[Path]) -> int:
    return sum(path.stat().st_size for path in file_paths)


def _estimated_compressed_size_bytes(source_size_bytes: int, compression_estimate: float) -> int:
    return max(1, int(source_size_bytes * compression_estimate))


def _write_zip_parts_by_scenario(
    bundle_path: Path,
    parts_dir: Path,
    common_paths: list[Path],
    per_scenario_paths: dict[str, list[Path]],
    max_size_mb: float,
    compression_estimate: float,
) -> list[Path]:
    if max_size_mb <= 0:
        raise ValueError("--max-size-mb must be > 0.")
    if compression_estimate <= 0 or compression_estimate > 1:
        raise ValueError("--compression-estimate must be in (0, 1].")
    parts_dir.mkdir(parents=True, exist_ok=True)

    tmp_tag = f".tmp_{uuid.uuid4().hex}"

    def _staged_path(final_path: Path) -> Path:
        return final_path.with_name(f"{final_path.name}{tmp_tag}")

    scenario_names = sorted(per_scenario_paths)
    all_paths: list[Path] = [*common_paths]
    for scenario_name in scenario_names:
        all_paths.extend(per_scenario_paths[scenario_name])

    staged_single = _staged_path(bundle_path)
    _write_zip(staged_single, all_paths)
    if _is_within_max_size(staged_single, max_size_mb):
        _cleanup_existing_parts(bundle_path, parts_dir)
        staged_single.rename(bundle_path)
        return [bundle_path]
    staged_single.unlink(missing_ok=True)

    max_bytes = int(max_size_mb * 1024 * 1024)
    common_size_bytes = _source_size_bytes(common_paths)
    common_estimated_bytes = _estimated_compressed_size_bytes(common_size_bytes, compression_estimate)
    if common_estimated_bytes > max_bytes:
        raise ValueError(f"Common bundle files exceed --max-size-mb={max_size_mb}. Increase the limit.")
    scenario_source_bytes = {name: _source_size_bytes(paths) for name, paths in per_scenario_paths.items()}
    scenario_estimated_bytes = {
        name: _estimated_compressed_size_bytes(size, compression_estimate)
        for name, size in scenario_source_bytes.items()
    }

    staged_part_paths: list[Path] = []
    part_index = 1
    remaining_scenarios = list(scenario_names)

    while remaining_scenarios:
        current_scenarios: list[str] = []
        current_estimated_bytes = common_estimated_bytes
        while remaining_scenarios:
            next_scenario = remaining_scenarios[0]
            next_estimated_bytes = scenario_estimated_bytes[next_scenario]
            if current_scenarios and current_estimated_bytes + next_estimated_bytes > max_bytes:
                break
            current_scenarios.append(remaining_scenarios.pop(0))
            current_estimated_bytes += next_estimated_bytes

        while True:
            part_file_paths = [*common_paths]
            for name in current_scenarios:
                part_file_paths.extend(per_scenario_paths[name])
            staged_part = _staged_path(_part_path(bundle_path, parts_dir, part_index))
            _write_zip(staged_part, part_file_paths)
            if _is_within_max_size(staged_part, max_size_mb):
                staged_part_paths.append(staged_part)
                part_index += 1
                break
            staged_part.unlink(missing_ok=True)
            if len(current_scenarios) == 1:
                scenario_name = current_scenarios[0]
                raise ValueError(f"Scenario exceeds --max-size-mb={max_size_mb}: {scenario_name}. Increase the limit.")
            overflow_scenario = current_scenarios.pop()
            remaining_scenarios.insert(0, overflow_scenario)

    _cleanup_existing_parts(bundle_path, parts_dir)
    if len(staged_part_paths) == 1:
        staged_part_paths[0].rename(bundle_path)
        return [bundle_path]

    final_paths: list[Path] = []
    for idx, staged_path in enumerate(staged_part_paths, start=1):
        final_path = _part_path(bundle_path, parts_dir, idx)
        staged_path.rename(final_path)
        final_paths.append(final_path)
    return final_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("abm/output/runs"),
        help="Root directory containing run folders.",
    )
    parser.add_argument(
        "--selection",
        choices=("latest-batch", "scenario", "explicit", "all-scenarios"),
        default="latest-batch",
        help="Run selection strategy.",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Scenario key for --selection scenario.",
    )
    parser.add_argument(
        "--run-name",
        action="append",
        default=[],
        help="Exact run folder name (repeatable) for --selection explicit.",
    )
    parser.add_argument(
        "--batch-window-seconds",
        type=int,
        default=900,
        help="Time window used by latest-batch selection.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("abm/output/runtime_parametric_experiment_with_outcomes_metrics.zip"),
        help="Output ZIP path.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("abm/output/upload_bundle_current_experiment/README_upload_manifest.txt"),
        help="Output manifest path.",
    )
    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=20.0,
        help="Maximum allowed ZIP size in MB.",
    )
    parser.add_argument(
        "--compression-estimate",
        type=float,
        default=0.20,
        help="Estimated ZIP/source size ratio used for scenario packing.",
    )
    parser.add_argument(
        "--parts-dir",
        type=Path,
        default=Path("abm/output/runtime_parametric_experiment_with_outcomes_metrics_parts"),
        help="Directory used to write multi-part ZIP outputs.",
    )
    parser.add_argument(
        "--csv-root",
        type=Path,
        default=Path("abm/output/upload_bundle_current_experiment/csv_exports"),
        help="Directory where converted CSV files are written before zipping.",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Export only runs whose scenario starts with 'baseline_'.",
    )
    args = parser.parse_args()

    runs = _load_runs(args.runs_root)
    selected_runs = _select_runs(
        runs=runs,
        selection=args.selection,
        scenario=args.scenario,
        explicit_runs=tuple(args.run_name),
        batch_window_seconds=args.batch_window_seconds,
    )
    selected_runs = _filter_baseline_runs(selected_runs, baseline_only=bool(args.baseline_only))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    _write_manifest(
        manifest_path=args.manifest,
        selection=args.selection,
        baseline_only=bool(args.baseline_only),
        selected_runs=selected_runs,
        metadata_files=DEFAULT_METADATA_FILES,
        parquet_files=DEFAULT_PARQUET_RUN_FILES,
        context_files=DEFAULT_CONTEXT_FILES,
    )
    _convert_parquet_to_csv(selected_runs=selected_runs, csv_root=args.csv_root)
    common_paths, per_scenario_paths = _build_file_groups_by_scenario(
        selected_runs=selected_runs,
        metadata_files=DEFAULT_METADATA_FILES,
        context_files=DEFAULT_CONTEXT_FILES,
        manifest_path=args.manifest,
        csv_root=args.csv_root,
    )
    written_paths = _write_zip_parts_by_scenario(
        args.out,
        parts_dir=args.parts_dir,
        common_paths=common_paths,
        per_scenario_paths=per_scenario_paths,
        max_size_mb=args.max_size_mb,
        compression_estimate=args.compression_estimate,
    )

    for path in written_paths:
        print(f"Wrote {path}")
    print(f"Wrote {args.manifest}")
    print(f"Selected runs: {len(selected_runs)}")


if __name__ == "__main__":
    main()
