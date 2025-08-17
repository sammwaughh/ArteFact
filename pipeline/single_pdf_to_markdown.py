#!/usr/bin/env python3
"""
Convert a single PDF to text + images using a Bede-compatible CPU-only backend.
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

from pdf_to_markdown_pymupdf import convert_pdf_to_markdown

LOG_DIR = Path(__file__).resolve().parent / "logs" / "PDF-To-Markdown-Logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def _get_logger(stem: str) -> logging.Logger:
    logger = logging.getLogger(stem)
    if logger.handlers:
        return logger
    fh = logging.FileHandler(LOG_DIR / f"{stem}_to_markdown_log.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.propagate = False
    return logger

def convert_via_cli(pdf_path: Path, timeout_sec: int = 3600, output_dir: Path = None) -> None:
    # Retain signature used by batch code; ignore timeout in CPU backend
    logger = _get_logger(pdf_path.stem)
    if not pdf_path.exists():
        logger.error("File not found: %s", pdf_path)
        sys.exit(1)

    out_root = output_dir if output_dir is not None else Path(__file__).resolve().parent
    logger.info("Converting via PyMuPDF backend: %s", pdf_path)
    md_path = convert_pdf_to_markdown(pdf_path, out_root)
    logger.info("Done. Markdown: %s", md_path)

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert one PDF to Markdown with a Bede-compatible backend."
    )
    ap.add_argument("id_or_path", help="WorkID or path to the PDF")
    ap.add_argument("--output-dir", type=Path, help="Output directory root for Marker_Output")
    args = ap.parse_args()

    inp = args.id_or_path
    if inp.lower().endswith(".pdf") or "/" in inp:
        pdf_path = Path(inp).expanduser().resolve()
    else:
        raise ValueError(f"Cannot resolve bare ID '{inp}' without full path in sharded mode")

    convert_via_cli(pdf_path, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
