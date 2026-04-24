from __future__ import annotations

import typer

from beam_abm.llm_microvalidation.workflows.behavioural_outcomes import run_behavioural_outcomes_perturbation_response

app = typer.Typer(
    add_completion=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    run_behavioural_outcomes_perturbation_response(list(ctx.args))


if __name__ == "__main__":
    app()
