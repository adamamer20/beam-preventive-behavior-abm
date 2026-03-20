from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from beam_abm.llm_microvalidation.export import export_thesis_artifacts
from beam_abm.llm_microvalidation.workflows import behavioural_outcomes as behavioural_outcomes_workflows
from beam_abm.llm_microvalidation.workflows import psychological_profiles as psychological_profiles_workflows


def test_choice_workflow_callables_dispatch(monkeypatch) -> None:
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _stub_unperturbed(argv: list[str] | None = None) -> None:
        calls.append(("unperturbed", (argv,), {}))

    def _stub_perturbed(argv: list[str] | None = None) -> None:
        calls.append(("perturbed", (argv,), {}))

    unperturbed_mod = ModuleType("beam_abm.llm_microvalidation.behavioural_outcomes.baseline_reconstruction_workflow")
    unperturbed_mod.run_behavioural_outcomes_baseline_reconstruction = _stub_unperturbed  # type: ignore[attr-defined]
    perturbed_mod = ModuleType("beam_abm.llm_microvalidation.behavioural_outcomes.perturbation_response_workflow")
    perturbed_mod.run_behavioural_outcomes_perturbation_response = _stub_perturbed  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "beam_abm.llm_microvalidation.behavioural_outcomes.baseline_reconstruction_workflow",
        unperturbed_mod,
    )
    monkeypatch.setitem(
        sys.modules, "beam_abm.llm_microvalidation.behavioural_outcomes.perturbation_response_workflow", perturbed_mod
    )

    behavioural_outcomes_workflows.run_behavioural_outcomes_baseline_reconstruction(
        ["--output-root", "tmp/choice_unperturbed"]
    )
    behavioural_outcomes_workflows.run_behavioural_outcomes_perturbation_response(
        ["--output-root", "tmp/choice_perturbed"]
    )

    assert calls == [
        ("unperturbed", (["--output-root", "tmp/choice_unperturbed"],), {}),
        ("perturbed", (["--output-root", "tmp/choice_perturbed"],), {}),
    ]


def test_belief_workflow_callables_dispatch(monkeypatch) -> None:
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _stub_unperturbed(argv: list[str] | None = None) -> None:
        calls.append(("unperturbed", (argv,), {}))

    def _stub_perturbed(argv: list[str] | None = None) -> None:
        calls.append(("perturbed", (argv,), {}))

    unperturbed_mod = ModuleType("beam_abm.llm_microvalidation.psychological_profiles.baseline_reconstruction_workflow")
    unperturbed_mod.run_psychological_profiles_baseline_reconstruction = _stub_unperturbed  # type: ignore[attr-defined]
    perturbed_mod = ModuleType("beam_abm.llm_microvalidation.psychological_profiles.perturbation_response_workflow")
    perturbed_mod.run_psychological_profiles_perturbation_response = _stub_perturbed  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "beam_abm.llm_microvalidation.psychological_profiles.baseline_reconstruction_workflow",
        unperturbed_mod,
    )
    monkeypatch.setitem(
        sys.modules, "beam_abm.llm_microvalidation.psychological_profiles.perturbation_response_workflow", perturbed_mod
    )

    psychological_profiles_workflows.run_psychological_profiles_baseline_reconstruction(
        ["--output-root", "tmp/belief_unperturbed"]
    )
    psychological_profiles_workflows.run_psychological_profiles_perturbation_response(
        ["--output-root", "tmp/belief_perturbed"]
    )

    assert calls == [
        ("unperturbed", (["--output-root", "tmp/belief_unperturbed"],), {}),
        ("perturbed", (["--output-root", "tmp/belief_perturbed"],), {}),
    ]


def test_belief_pe_ref_p_workflow_dispatch(monkeypatch) -> None:
    calls: list[list[str] | None] = []

    def _stub(argv: list[str] | None = None) -> None:
        calls.append(argv)

    pe_ref_p_mod = ModuleType("beam_abm.llm_microvalidation.psychological_profiles.reference_effects")
    pe_ref_p_mod.run_psychological_profile_reference_effects = _stub  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "beam_abm.llm_microvalidation.psychological_profiles.reference_effects", pe_ref_p_mod
    )
    psychological_profiles_workflows.run_psychological_profile_reference_effects(
        ["--data", "in.parquet", "--spec", "spec.json", "--out", "out.parquet"]
    )
    assert calls == [["--data", "in.parquet", "--spec", "spec.json", "--out", "out.parquet"]]


def test_evaluation_export_library_smoke(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    required_paths = [
        "evaluation/output/choice_validation/unperturbed/_summary/posthoc_metrics_all.csv",
        "evaluation/output/choice_validation/unperturbed/_summary/posthoc_metrics_bootstrap.csv",
        "evaluation/output/choice_validation/unperturbed/_summary/strategy_bootstrap_cis.csv",
        "evaluation/output/choice_validation/unperturbed/_summary/cell_bootstrap_cis.csv",
        "evaluation/output/choice_validation/unperturbed/_summary/outcome_bootstrap_cis.csv",
        "evaluation/output/choice_validation/unperturbed/_summary/ref_benchmark.csv",
        "evaluation/output/choice_validation/perturbed/_summary/all_outcomes_models_strategies.csv",
        "evaluation/output/choice_validation/unperturbed_block_ablation_canonical/_summary/unperturbed_block_ablation.csv",
        "evaluation/output/belief_update_validation/summary_metrics/all_models_strategies.csv",
        "evaluation/output/belief_update_validation/summary_metrics/all_models_signals.csv",
        "empirical/output/modeling/model_plan.json",
        "config/levers.json",
        "empirical/output/modeling/model_lobo_all.csv",
    ]

    for rel in required_paths:
        path = repo_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text("{}\n", encoding="utf-8")
        else:
            path.write_text("col\n1\n", encoding="utf-8")

    (repo_root / "thesis/artifacts").mkdir(parents=True, exist_ok=True)

    export_thesis_artifacts(repo_root)

    assert (repo_root / "thesis/artifacts/evaluation/choice_validation/unperturbed/posthoc_metrics_all.csv").exists()
    assert (repo_root / "thesis/artifacts/evaluation/belief_update_validation/frontier_metrics.csv").exists()
