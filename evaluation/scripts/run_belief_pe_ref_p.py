from __future__ import annotations

import typer

from beam_abm.evaluation.workflows.belief import run_belief_pe_ref_p

app = typer.Typer(
    add_completion=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    run_belief_pe_ref_p(list(ctx.args))


if __name__ == "__main__":
    app()
