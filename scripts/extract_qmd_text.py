#!/usr/bin/env python3
"""Extract plain text from ordered Quarto `.qmd` sources.

Usage:
    python scripts/extract_qmd_text.py                # uses sections from thesis/_quarto.yml
    python scripts/extract_qmd_text.py thesis/05-agent-based-model.qmd
    python scripts/extract_qmd_text.py --input-dir thesis --pattern '*.qmd'

By default, the script prints the content with all ``{python}`` fenced code
blocks removed. All other content (including non-python code blocks) is
kept.
If no files are provided, book chapter and appendix sources are read from
`thesis/_quarto.yml` and concatenated in render order.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_FENCE_RE = re.compile(r"^\s*(?P<fence>`{3,}|~{3,})(?P<info>.*)$")


def _is_python_fence(info: str) -> bool:
    spec = info.strip()
    if not spec:
        return False

    if spec.startswith("{") and "}" in spec:
        inner = spec[1 : spec.index("}")].strip()
    else:
        inner = spec.split(maxsplit=1)[0].strip()

    lang = inner.split(",")[0].strip().lower()
    return lang == "python"


def _is_fence_close(line: str, fence: str) -> bool:
    marker = fence[0]
    text = line.strip()
    return text and len(text) >= len(fence) and set(text) == {marker}


def strip_python_blocks(text: str) -> str:
    """Return `text` with only Python code fences removed."""

    lines = text.splitlines(keepends=True)
    out: list[str] = []

    in_code_block = False
    skip_block = False
    fence: str | None = None

    for line in lines:
        match = _FENCE_RE.match(line)

        if match and not in_code_block:
            fence = match.group("fence")
            info = match.group("info")
            skip_block = _is_python_fence(info)
            in_code_block = True
            if not skip_block:
                out.append(line)
            continue

        if in_code_block and fence is not None and _is_fence_close(line, fence):
            if not skip_block:
                out.append(line)
            in_code_block = False
            skip_block = False
            fence = None
            continue

        if not in_code_block or not skip_block:
            out.append(line)

    return "".join(out)


def collect_sources(args: argparse.Namespace) -> list[Path]:
    if args.inputs:
        missing = [path for path in args.inputs if not path.exists()]
        if missing:
            missing_names = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"Missing input file(s): {missing_names}")

        return sorted(args.inputs)

    if args.quarto_yml is not None:
        return _load_sources_from_quarto(args.quarto_yml)

    source_root = Path(args.input_dir)
    if not source_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {source_root}")

    return sorted(source_root.rglob(args.pattern))


def _strip_comments(line: str) -> str:
    stripped = line.split("#", 1)[0]
    return stripped.rstrip("\n")


def _load_sources_from_quarto(yaml_path: str) -> list[Path]:
    quarto_root = Path(yaml_path).resolve().parent
    with Path(yaml_path).resolve().open(encoding="utf-8") as fh:
        lines = fh.readlines()

    in_book = False
    section: str | None = None
    sources: list[Path] = []

    for raw_line in lines:
        line = _strip_comments(raw_line)
        if not line.strip():
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = line.strip()

        if re.match(r"^book:\s*$", stripped):
            in_book = True
            section = None
            continue

        if in_book:
            if indent == 0 and re.match(r"^[^\s].*:\s*$", stripped):
                in_book = False
                section = None
                continue

            if indent <= 2 and stripped.startswith("chapters:"):
                section = "chapters"
                continue

            if indent <= 2 and stripped.startswith("appendices:"):
                section = "appendices"
                continue

            match = re.match(r"^\s*-\s*([^\s#]+\.qmd)\s*$", line)
            if match and section in {"chapters", "appendices"}:
                rel_path = match.group(1)
                source_path = (quarto_root / rel_path).resolve()
                if not source_path.exists():
                    print(f"Skipping missing file from quarto config: {source_path}", file=sys.stderr)
                    continue
                sources.append(source_path)

    if not sources:
        raise FileNotFoundError(f"No qmd files found in quarto config: {yaml_path}")

    return sources


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract all text from a .qmd file while removing python blocks.",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Input .qmd files. If omitted, use --quarto-yml order (or --input-dir pattern fallback).",
    )
    parser.add_argument(
        "--input-dir",
        default="thesis",
        help="Directory to scan when no input files are provided.",
    )
    parser.add_argument(
        "--pattern",
        default="*.qmd",
        help="Glob pattern to match when scanning --input-dir.",
    )
    parser.add_argument(
        "--quarto-yml",
        type=str,
        default="thesis/_quarto.yml",
        help="Quarto YAML config used to infer default chapter/appended order.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("thesis/full-text.qmd"),
        help="Output file path. Use '-' to print to stdout.",
    )
    return parser.parse_args(args=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        sources = collect_sources(args)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if not sources:
        print("No .qmd files found for processing", file=sys.stderr)
        return 2

    extracted = []
    for source in sources:
        extracted.append(strip_python_blocks(source.read_text(encoding="utf-8")))

    merged = "\n\n\n".join(extracted)

    if str(args.output) == "-":
        print(merged, end="")
        return 0

    args.output.write_text(merged, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
