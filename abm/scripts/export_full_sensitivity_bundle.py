"""Export a compact full-sensitivity bundle excluding full run artefacts.

The bundle includes all files under ``abm/output/full_sensitivity_pipeline``
except paths that pass through excluded directory names (defaults to ``runs``).

If the ZIP would exceed ``--max-size-mb``, the script automatically writes
multi-part ZIP files.
"""

from __future__ import annotations

import argparse
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


@dataclass(frozen=True, slots=True)
class ExportInventory:
    """Collected files and basic inclusion/exclusion stats."""

    included_paths: list[Path]
    excluded_paths: list[Path]
    included_bytes: int
    excluded_bytes: int


def _collect_inventory(output_dir: Path, excluded_dir_names: tuple[str, ...]) -> ExportInventory:
    included_paths: list[Path] = []
    excluded_paths: list[Path] = []
    included_bytes = 0
    excluded_bytes = 0

    normalized_excludes = tuple(name.strip() for name in excluded_dir_names if name.strip())

    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(output_dir)
        traverses_excluded = any(part in normalized_excludes for part in rel.parts[:-1])
        size = path.stat().st_size
        if traverses_excluded:
            excluded_paths.append(path)
            excluded_bytes += size
            continue
        included_paths.append(path)
        included_bytes += size

    return ExportInventory(
        included_paths=included_paths,
        excluded_paths=excluded_paths,
        included_bytes=included_bytes,
        excluded_bytes=excluded_bytes,
    )


def _write_manifest(
    *,
    manifest_path: Path,
    output_dir: Path,
    excluded_dir_names: tuple[str, ...],
    inventory: ExportInventory,
) -> None:
    created = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    excluded_sample_limit = 200
    included_rel = [path.relative_to(output_dir).as_posix() for path in inventory.included_paths]
    excluded_rel = [path.relative_to(output_dir).as_posix() for path in inventory.excluded_paths]

    lines = [
        f"bundle_created_utc={created}",
        f"output_dir={output_dir.as_posix()}",
        f"excluded_dir_names={','.join(excluded_dir_names)}",
        f"included_file_count={len(inventory.included_paths)}",
        f"excluded_file_count={len(inventory.excluded_paths)}",
        f"included_source_size_bytes={inventory.included_bytes}",
        f"excluded_source_size_bytes={inventory.excluded_bytes}",
        "",
        "included_files:",
    ]
    lines.extend(f"- {name}" for name in included_rel)
    lines.append("")
    lines.append(f"excluded_files_sample_first_{excluded_sample_limit}:")
    lines.extend(f"- {name}" for name in excluded_rel[:excluded_sample_limit])
    remaining = max(0, len(excluded_rel) - excluded_sample_limit)
    if remaining:
        lines.append(f"... ({remaining} more excluded files omitted)")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _zip_arcname(file_path: Path) -> str:
    if not file_path.is_absolute():
        return file_path.as_posix()
    cwd = Path.cwd()
    try:
        return file_path.relative_to(cwd).as_posix()
    except ValueError:
        return file_path.name


def _write_zip(bundle_path: Path, file_paths: list[Path]) -> None:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    if bundle_path.exists():
        bundle_path.unlink()
    with ZipFile(bundle_path, mode="w", compression=ZIP_DEFLATED, compresslevel=9) as zip_file:
        for file_path in file_paths:
            zip_file.write(file_path, arcname=_zip_arcname(file_path))


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


def _write_zip_parts(
    *,
    bundle_path: Path,
    parts_dir: Path,
    common_paths: list[Path],
    payload_paths: list[Path],
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

    full_paths = [*common_paths, *payload_paths]
    staged_single = _staged_path(bundle_path)
    _write_zip(staged_single, full_paths)
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

    ordered_payload = sorted(payload_paths, key=lambda path: path.as_posix())
    payload_estimated = {
        path: _estimated_compressed_size_bytes(path.stat().st_size, compression_estimate) for path in ordered_payload
    }

    for path, estimate in payload_estimated.items():
        if common_estimated_bytes + estimate > max_bytes:
            msg = (
                "Single file exceeds --max-size-mb when combined with common files: "
                f"{path.as_posix()}. Increase the limit."
            )
            raise ValueError(msg)

    staged_part_paths: list[Path] = []
    part_index = 1
    cursor = 0

    while cursor < len(ordered_payload):
        current: list[Path] = []
        estimated_bytes = common_estimated_bytes
        while cursor < len(ordered_payload):
            candidate = ordered_payload[cursor]
            candidate_est = payload_estimated[candidate]
            if current and estimated_bytes + candidate_est > max_bytes:
                break
            current.append(candidate)
            estimated_bytes += candidate_est
            cursor += 1

        while True:
            part_file_paths = [*common_paths, *current]
            staged_part = _staged_path(_part_path(bundle_path, parts_dir, part_index))
            _write_zip(staged_part, part_file_paths)
            if _is_within_max_size(staged_part, max_size_mb):
                staged_part_paths.append(staged_part)
                part_index += 1
                break

            staged_part.unlink(missing_ok=True)
            if len(current) == 1:
                msg = (
                    "Single file still exceeds --max-size-mb after compression: "
                    f"{current[0].as_posix()}. Increase the limit."
                )
                raise ValueError(msg)

            cursor -= 1
            current.pop()

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
        "--output-dir",
        type=Path,
        default=Path("abm/output/full_sensitivity_pipeline"),
        help="Root directory containing full sensitivity outputs.",
    )
    parser.add_argument(
        "--exclude-dir-name",
        action="append",
        default=["runs"],
        help="Directory name to exclude when traversing outputs (repeatable).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("abm/output/full_sensitivity_pipeline_results_no_runs.zip"),
        help="Output ZIP path.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("abm/output/full_sensitivity_pipeline_export/README_upload_manifest.txt"),
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
        help="Estimated ZIP/source size ratio used for multipart packing.",
    )
    parser.add_argument(
        "--parts-dir",
        type=Path,
        default=Path("abm/output/full_sensitivity_pipeline_results_no_runs_parts"),
        help="Directory used to write multipart ZIP outputs.",
    )
    args = parser.parse_args()

    if not args.output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {args.output_dir}")

    excluded_dir_names = tuple(dict.fromkeys(args.exclude_dir_name))
    inventory = _collect_inventory(args.output_dir, excluded_dir_names)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    _write_manifest(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        excluded_dir_names=excluded_dir_names,
        inventory=inventory,
    )

    common_paths = [args.manifest]
    payload_paths = inventory.included_paths
    written_paths = _write_zip_parts(
        bundle_path=args.out,
        parts_dir=args.parts_dir,
        common_paths=common_paths,
        payload_paths=payload_paths,
        max_size_mb=args.max_size_mb,
        compression_estimate=args.compression_estimate,
    )

    for path in written_paths:
        print(f"Wrote {path}")
    print(f"Wrote {args.manifest}")
    print(f"Included files: {len(inventory.included_paths)}")
    print(f"Excluded files: {len(inventory.excluded_paths)}")


if __name__ == "__main__":
    main()
