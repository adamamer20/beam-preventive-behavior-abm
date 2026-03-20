from __future__ import annotations

from pathlib import Path

import typer

from beam_abm.evaluation.export import export_thesis_artifacts

app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def main(
    repo_root: Path = typer.Option(
        Path(__file__).resolve().parents[2],
        "--repo-root",
        help="Repository root containing evaluation/output and thesis/artifacts.",
    ),
) -> None:
    export_thesis_artifacts(repo_root.resolve())
    print("Exported evaluation thesis artifacts.")


if __name__ == "__main__":
    app()
