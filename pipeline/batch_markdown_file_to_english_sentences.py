#!/usr/bin/env python3
"""
batch_markdown_file_to_english_sentences.py
------------------------------------------
Walk through Pipeline/Marker_Output/** and run
markdown_file_to_english_sentences.process_markdown() on every Markdown
file found.

Usage
-----
$ python Pipeline/batch_markdown_file_to_english_sentences.py          # all
$ python Pipeline/batch_markdown_file_to_english_sentences.py W1982    # filter
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from markdown_file_to_english_sentences import process_markdown

ROOT = Path(__file__).resolve().parent  # Pipeline/
MARKER_DIR = ROOT / "Marker_Output"
LOG_DIR = ROOT / "logs/Markdown-To-Sentences-Logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ───────────────────────────── logging setup ──────────────────────────────
def _setup_logger() -> logging.Logger:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"batch_markdown_to_sentences_{timestamp}.log"

    logger = logging.getLogger("batch_markdown")
    logger.setLevel(logging.INFO)

    # File handler only (no console output)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    logger.addHandler(fh)
    logger.propagate = False

    return logger


# ───────────────────────────── helpers ─────────────────────────────────────
def _discover_markdowns(substring: str | None = None) -> List[Path]:
    """
    Return a sorted list of *.md files within Marker_Output/.  Optionally
    filter by *substring* (case-insensitive) on filename.
    """
    if not MARKER_DIR.exists():
        sys.exit(f"❌ Marker output directory not found at {MARKER_DIR}")

    pattern = f"*{substring.lower()}*.md" if substring else "*.md"
    return sorted(MARKER_DIR.rglob(pattern))


# ───────────────────────────── main ────────────────────────────────────────
def main() -> None:
    logger = _setup_logger()

    ap = argparse.ArgumentParser(
        description=(
            "Convert every Markdown in Marker_Output to JSON sentence objects.\n"
            "Optionally provide a substring to process only matching WorkIDs."
        )
    )
    ap.add_argument(
        "filter",
        nargs="?",
        help="Substring to filter Markdown filenames (e.g. W1982)",
    )
    args = ap.parse_args()

    md_files = _discover_markdowns(args.filter)
    if not md_files:
        logger.info("Nothing to process – no matching Markdown files found.")
        return

    logger.info(f"Processing {len(md_files)} Markdown files")

    for i, md in enumerate(md_files, 1):
        logger.info(f"[{i}/{len(md_files)}] Processing: {md.name}")
        try:
            # Capture all stdout/stderr from process_markdown
            with contextlib.redirect_stdout(
                io.StringIO()
            ) as stdout_capture, contextlib.redirect_stderr(
                io.StringIO()
            ) as stderr_capture:
                process_markdown(md)

            # Log any captured output
            if stdout_output := stdout_capture.getvalue().strip():
                logger.info(f"  Output: {stdout_output}")
            if stderr_output := stderr_capture.getvalue().strip():
                logger.warning(f"  Errors: {stderr_output}")

            logger.info("  → Completed successfully")
        except Exception as exc:
            logger.error(f"  → Failed: {exc}")

    logger.info("Batch processing completed")


if __name__ == "__main__":
    main()
