#!/usr/bin/env python3
# filepath: /Users/samuelwaugh/Desktop/ArtContext/Pipeline/single_pdf_to_markdown.py
"""
Convert a single PDF to Markdown (plus extracted images) using Marker CLI.

Usage
-----
source .venv/bin/activate          # if not already active
python Pipeline/single_pdf_to_markdown.py PDF_Bucket/W1549104703.pdf
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# ─────────────────────────── logging ─────────────────────────────────────
LOG_DIR = Path(__file__).resolve().parent / "logs" / "PDF-To-Markdown-Logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _get_logger(stem: str) -> logging.Logger:
    logger = logging.getLogger(stem)
    if logger.handlers:  # already configured in same run
        return logger
    fh = logging.FileHandler(LOG_DIR / f"{stem}_to_markdown_log.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def convert_via_cli(pdf_path: Path, timeout_sec: int = 3600) -> None:
    """Run Marker CLI on *pdf_path* and write output to Marker_Output/<stem>/."""
    logger = _get_logger(pdf_path.stem)

    if not pdf_path.exists():
        logger.error("File not found: %s", pdf_path)
        sys.exit(1)

    # Output directory: Pipeline/Marker_Output/<PDF-stem>/
    # Give Marker the parent folder only; it will create <PDF-stem>/ itself
    out_dir = Path(__file__).resolve().parent / "Marker_Output"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "marker_single",
        str(pdf_path),
        "--output_format",
        "markdown",
        "--output_dir",
        str(out_dir),
    ]
    logger.info("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            check=True,
            timeout=timeout_sec,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.stdout:
            logger.info("Marker output:\n%s", result.stdout)
        md_file = out_dir / pdf_path.stem / f"{pdf_path.stem}.md"
        logger.info("Done. Markdown: %s", md_file.relative_to(Path.cwd()))
    except subprocess.TimeoutExpired:
        logger.warning("Timeout (%ss) for %s", timeout_sec, pdf_path)
    except subprocess.CalledProcessError as err:
        logger.error("Marker CLI failed: %s", err)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Convert one PDF to Markdown with Marker.\n"
            "Pass either a bare WorkID (e.g. W1549104703) or an explicit path."
        )
    )
    ap.add_argument("id_or_path", help="WorkID or path to the PDF")
    args = ap.parse_args()

    inp = args.id_or_path
    if inp.lower().endswith(".pdf") or "/" in inp:
        pdf_path = Path(inp).expanduser().resolve()
    else:
        # bare ID ⇒ assume Pipeline/PDF_Bucket/<ID>.pdf
        root = Path(__file__).resolve().parent
        pdf_path = root / "PDF_Bucket" / f"{inp}.pdf"

    convert_via_cli(pdf_path)


if __name__ == "__main__":
    main()
