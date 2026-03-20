from __future__ import annotations

import re
from pathlib import Path

FORBIDDEN_IMPORT_RE = re.compile(
    r"\bbeam_abm\.(analysis|microvalidation|llm_behaviour|preprocessing|config|load_df|utils)\b"
)
FORBIDDEN_PIPELINE_PATH_RE = re.compile(r"\b(0_preprocessing|1_dataset_analysis|3_llm_microvalidation|4_abm)\b")
FORBIDDEN_SRC_SCRIPT_IMPORT_RE = re.compile(
    r"^\s*(?:from|import)\s+(?:evaluation|empirical|preprocess|abm)\.scripts(?:\.|\b)",
    re.MULTILINE,
)
FORBIDDEN_EVAL_LIBRARY_CLI_RE = re.compile(
    r"^\s*import\s+argparse\b|^\s*from\s+argparse\s+import\b|^\s*def\s+main\s*\(",
    re.MULTILINE,
)
FORBIDDEN_EVAL_LIBRARY_CLI_SHIM_RE = re.compile(
    r"^\s*from\s+beam_abm\.cli\s+import\s+argparse_shim\b",
    re.MULTILINE,
)
FORBIDDEN_EVAL_BRIDGING_RE = re.compile(
    r"\bsys\.argv\b|\bimportlib\.util\b|\binspect\b|run_step_subprocess\(|[A-Za-z_][A-Za-z0-9_]*\.main\b",
)
FORBIDDEN_EMPIRICAL_LIBRARY_CLI_RE = re.compile(
    r"^\s*import\s+argparse\b|^\s*from\s+argparse\s+import\b",
    re.MULTILINE,
)


def _iter_checked_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for base in [
        root / "src",
        root / "tests",
        root / "preprocess" / "scripts",
        root / "empirical" / "scripts",
        root / "evaluation" / "scripts",
        root / "abm" / "scripts",
        root / "scripts",
    ]:
        if not base.exists():
            continue
        files.extend(p for p in base.rglob("*") if p.is_file() and p.suffix in {".py", ".sh"})
    files.append(root / "Makefile")
    return files


def test_forbidden_legacy_imports_and_paths_absent() -> None:
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []

    for path in _iter_checked_files(root):
        if path.name == "test_package_structure_refactor.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if FORBIDDEN_IMPORT_RE.search(text) or FORBIDDEN_PIPELINE_PATH_RE.search(text):
            offenders.append(str(path.relative_to(root)))

    assert offenders == []


def test_src_library_does_not_import_external_scripts() -> None:
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []

    for path in (root / "src" / "beam_abm").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if FORBIDDEN_SRC_SCRIPT_IMPORT_RE.search(text):
            offenders.append(str(path.relative_to(root)))

    assert offenders == []


def test_evaluation_library_has_no_cli_entrypoints() -> None:
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []

    eval_root = root / "src" / "beam_abm" / "evaluation"
    for path in eval_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if FORBIDDEN_EVAL_LIBRARY_CLI_RE.search(text) or FORBIDDEN_EVAL_LIBRARY_CLI_SHIM_RE.search(text):
            offenders.append(str(path.relative_to(root)))

    assert offenders == []


def test_evaluation_library_has_no_script_style_bridging_patterns() -> None:
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []

    eval_root = root / "src" / "beam_abm" / "evaluation"
    for path in eval_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if FORBIDDEN_EVAL_BRIDGING_RE.search(text):
            offenders.append(str(path.relative_to(root)))

    assert offenders == []


def test_empirical_library_does_not_import_script_entrypoints() -> None:
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []

    empirical_root = root / "src" / "beam_abm" / "empirical"
    for path in empirical_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if FORBIDDEN_SRC_SCRIPT_IMPORT_RE.search(text):
            offenders.append(str(path.relative_to(root)))

    assert offenders == []


def test_empirical_library_has_no_argparse_usage() -> None:
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []

    empirical_root = root / "src" / "beam_abm" / "empirical"
    for path in empirical_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if FORBIDDEN_EMPIRICAL_LIBRARY_CLI_RE.search(text):
            offenders.append(str(path.relative_to(root)))

    assert offenders == []


def test_tests_do_not_import_empirical_scripts_except_cli_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []

    for path in (root / "tests").rglob("*.py"):
        if path.name == "test_empirical_cli_smoke.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"^\s*(?:from|import)\s+empirical\.scripts(?:\.|\b)", text, flags=re.MULTILINE):
            offenders.append(str(path.relative_to(root)))

    assert offenders == []


def test_tests_do_not_import_evaluation_scripts_except_cli_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []

    for path in (root / "tests").rglob("*.py"):
        if path.name == "test_evaluation_cli_smoke.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"^\s*(?:from|import)\s+evaluation\.scripts(?:\.|\b)", text, flags=re.MULTILINE):
            offenders.append(str(path.relative_to(root)))

    assert offenders == []


def test_evaluation_scripts_do_not_import_src_evaluation_modules() -> None:
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []

    scripts_root = root / "evaluation" / "scripts"
    for path in scripts_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if re.search(
            r"^\s*(?:from|import)\s+beam_abm\.evaluation\.(?!(workflows|export)(?:\.|\b))",
            text,
            flags=re.MULTILINE,
        ):
            offenders.append(str(path.relative_to(root)))

    assert offenders == []
