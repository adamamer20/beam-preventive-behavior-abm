from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from beam_abm.empirical.export import export_thesis_artifacts as export_empirical_thesis_artifacts

FORBIDDEN_RUNTIME_REF_RE = re.compile(r"(preprocess/output|empirical/output|evaluation/output|abm/output)")
ARTIFACT_REF_RE = re.compile(r"['\"]\.\./thesis/artifacts/([^'\"]+)['\"]")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _thesis_contract_files(root: Path) -> list[Path]:
    files = list((root / "thesis").glob("*.qmd"))
    files.extend((root / "thesis" / "utils").rglob("*.py"))
    return sorted(files)


def _load_evaluation_exporter(root: Path):
    script_path = root / "evaluation" / "scripts" / "export_thesis_artifacts.py"
    spec = importlib.util.spec_from_file_location("evaluation_thesis_exporter", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load evaluation thesis exporter: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_abm_exporter(root: Path):
    script_path = root / "abm" / "scripts" / "export_thesis_artifacts.py"
    spec = importlib.util.spec_from_file_location("abm_thesis_exporter", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load ABM thesis exporter: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exported_artifacts_root() -> Path:
    root = _repo_root()
    try:
        export_empirical_thesis_artifacts(root)
        evaluation_exporter = _load_evaluation_exporter(root)
        evaluation_exporter.export_thesis_artifacts(root)
        abm_exporter = _load_abm_exporter(root)
        abm_exporter.export_thesis_artifacts(root)
    except (FileNotFoundError, RuntimeError) as exc:
        pytest.skip(f"Runtime outputs unavailable for thesis artifact export smoke test: {exc}")
    return root / "thesis" / "artifacts"


def test_thesis_code_has_no_runtime_output_references() -> None:
    root = _repo_root()
    offenders: list[str] = []
    for path in _thesis_contract_files(root):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if FORBIDDEN_RUNTIME_REF_RE.search(text):
            offenders.append(str(path.relative_to(root)))
    assert offenders == []


def test_export_scripts_write_expected_contract_outputs(exported_artifacts_root: Path) -> None:
    expected_files = [
        exported_artifacts_root / "empirical" / "modeling" / "model_lobo_all.csv",
        exported_artifacts_root / "evaluation" / "choice_validation" / "unperturbed" / "posthoc_metrics_all.csv",
        exported_artifacts_root / "evaluation" / "choice_validation" / "perturbed" / "summary_metrics_cells.csv",
        exported_artifacts_root / "evaluation" / "belief_update_validation" / "frontier_metrics.csv",
        exported_artifacts_root / "abm" / "report_summary_canonical_effects.csv",
        exported_artifacts_root / "abm" / "run_context" / "series.parquet",
    ]
    missing = [str(path) for path in expected_files if not path.exists()]
    assert missing == []


def test_referenced_artifact_paths_exist_after_export(exported_artifacts_root: Path) -> None:
    root = _repo_root()
    referenced_paths: set[Path] = set()
    for path in _thesis_contract_files(root):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in ARTIFACT_REF_RE.findall(text):
            cleaned = str(match).strip()
            if not cleaned:
                continue
            referenced_paths.add(exported_artifacts_root / cleaned)

    missing = sorted(str(path) for path in referenced_paths if not path.exists())
    assert missing == []
