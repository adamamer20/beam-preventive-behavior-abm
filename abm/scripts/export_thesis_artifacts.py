from __future__ import annotations

from pathlib import Path

import typer

from beam_abm.abm.export import export_thesis_artifacts


def main(
    repo_root: Path = typer.Option(
        Path(__file__).resolve().parents[2],
        "--repo-root",
        help="Repository root containing abm/output and thesis/artifacts.",
    ),
) -> None:
    export_thesis_artifacts(repo_root.resolve(), refresh_canonical_summary=True)
    print("Exported ABM thesis artifacts.")


if __name__ == "__main__":
    typer.run(main)
