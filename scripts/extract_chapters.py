#!/usr/bin/env python3
"""Small throwaway script to extract chapter content from Quarto HTML files.

Usage:
    python scripts/extract_chapters.py \
        --input-dir thesis/final_output \
        --output thesis/final_output/all-chapters-content.html

Behavior:
- For each .html file in the input directory (sorted), the script extracts the
  inner contents of the <main id="quarto-document-content"> element.
- It removes reference blocks (<div id="refs">...) and footnotes
  (<section class="footnotes">) to keep only the "real" chapter content.
- Writes a simple combined HTML file that includes each chapter's title and
  content inside a wrapper <section class="extracted-chapter">.

This is intentionally small and dependency-light: it prefers BeautifulSoup4
(if available) and falls back to a conservative regex-based extractor.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bs4 import BeautifulSoup


def extract_with_bs4(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main", {"id": "quarto-document-content"})
    if main is None:
        raise RuntimeError('<main id="quarto-document-content"> not found')

    # remove refs and footnotes if present
    refs = main.find(id="refs")
    if refs:
        refs.decompose()
    footnotes = main.find("section", {"class": "footnotes"})
    if footnotes:
        footnotes.decompose()

    # Get a chapter title if present (h1.title > span.chapter-title)
    title_el = main.find("h1")
    title_text = title_el.get_text(strip=True) if title_el else ""

    # return inner HTML of main
    content_html = "".join(str(c) for c in main.contents)
    return title_text, content_html


def extract_file(path: Path) -> tuple[str, str]:
    html = path.read_text(encoding="utf-8")
    return extract_with_bs4(html)


def build_output(chapters: list[tuple[str, str]]) -> str:
    head = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>All chapters - extracted content</title>
<style>
body{font-family:system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; max-width:900px; margin:2rem auto; padding:0 1rem;}
section.extracted-chapter{margin-bottom:3rem; border-bottom:1px solid #eee; padding-bottom:2rem}
h1.chapter-title{font-size:1.2rem; margin:0 0 0.5rem}
</style>
</head>
<body>
<h1>Combined chapters (extracted)</h1>
"""
    parts = [head]
    for title, content in chapters:
        title_html = f'<h1 class="chapter-title">{title}</h1>\n' if title else ""
        parts.append(f'<section class="extracted-chapter">{title_html}{content}</section>')

    parts.append("</body>\n</html>")
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="thesis/final_output", help="Folder with chapter HTML files")
    parser.add_argument("--output", default="thesis/final_output/all-chapters-content.html", help="Output file")
    parser.add_argument("--pattern", default="*.html", help="Glob pattern for files")
    args = parser.parse_args(argv)

    folder = Path(args.input_dir)
    if not folder.exists():
        print("Input folder does not exist:", folder, file=sys.stderr)
        return 2

    files = sorted(folder.glob(args.pattern))
    chapters: list[tuple[str, str]] = []
    for f in files:
        # skip index.html and other non-chapter-ish files if you like
        if f.name in ("index.html",):
            continue
        try:
            title, content = extract_file(f)
            pages = (title, content)
            chapters.append(pages)
            print(f"Extracted {f.name}: title='{title}'")
        except Exception as e:
            print(f"Skipping {f.name}: {e}", file=sys.stderr)

    out_html = build_output(chapters)
    out_path = Path(args.output)
    out_path.write_text(out_html, encoding="utf-8")
    print(f"Wrote combined file to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
