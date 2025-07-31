#!/usr/bin/env python3
"""
batch_pdf_to_markdown.py
------------------------
Convert *every* PDF found in PDF_Bucket/ to Markdown by re-using
single_pdf_to_markdown.convert_via_cli().

Usage
-----
$ python Pipeline/batch_pdf_to_markdown.py          # process all
$ python Pipeline/batch_pdf_to_markdown.py W1982    # filter → *W1982*.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from single_pdf_to_markdown import convert_via_cli

ROOT = Path(__file__).resolve().parent
PDF_DIR = ROOT / "PDF_Bucket"


# ───────────────────────────── helpers ─────────────────────────────────────
def _discover_pdfs(pattern: str | None = None) -> List[Path]:
    if not PDF_DIR.exists():
        sys.exit(f"❌ PDF directory not found at {PDF_DIR}")

    glob_pat = f"*{pattern}*.pdf" if pattern else "*.pdf"
    return sorted(PDF_DIR.glob(glob_pat))


# ───────────────────────────── main ────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run Marker on every PDF inside PDF_Bucket.\n"
            "Optionally supply a substring to process only matching filenames."
        )
    )
    ap.add_argument(
        "filter",
        nargs="?",
        help="Substring to filter PDF names (e.g. W1982)",
    )
    args = ap.parse_args()

    pdf_files = _discover_pdfs(args.filter)
    if not pdf_files:
        print("Nothing to do – no matching PDFs found.")
        return

    for pdf in pdf_files:
        convert_via_cli(pdf)


if __name__ == "__main__":
    main()
