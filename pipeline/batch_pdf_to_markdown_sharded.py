#!/usr/bin/env python3
"""
batch_pdf_to_markdown_sharded.py
--------------------------------
Convert PDFs from all 32 shards to Markdown by re-using
single_pdf_to_markdown.convert_via_cli().

This script is adapted to work with the sharded PDF download structure
where PDFs are distributed across 32 shard directories.

Usage
-----
$ python batch_pdf_to_markdown_sharded.py          # process all shards
$ python batch_pdf_to_markdown_sharded.py 5        # process only shard 5
$ python batch_pdf_to_markdown_sharded.py W1982    # filter → *W1982*.pdf across all shards
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from single_pdf_to_markdown import convert_via_cli
from sharding_config import SHARDS, RUN_ROOT, SHARDS_DIR

# ─────────────────────────── logging ─────────────────────────────────────
LOG_DIR = Path(__file__).resolve().parent / "logs" / "PDF-To-Markdown-Logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def _get_logger() -> logging.Logger:
    logger = logging.getLogger("batch_pdf_to_markdown_sharded")
    if logger.handlers:  # already configured in same run
        return logger
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    
    # File handler
    fh = logging.FileHandler(LOG_DIR / "batch_pdf_to_markdown_sharded.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    
    logger.setLevel(logging.INFO)
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False
    return logger

# ───────────────────────────── helpers ─────────────────────────────────────
def _discover_pdfs_in_shard(shard_idx: int, pattern: Optional[str] = None) -> List[Path]:
    """Discover PDFs in a specific shard's PDF_Bucket directory."""
    pdf_dir = SHARDS_DIR / f"shard_{shard_idx:02d}" / "PDF_Bucket"
    
    if not pdf_dir.exists():
        return []
    
    glob_pat = f"*{pattern}*.pdf" if pattern else "*.pdf"
    return sorted(pdf_dir.glob(glob_pat))

def _discover_pdfs_all_shards(pattern: Optional[str] = None, specific_shard: Optional[int] = None) -> List[Path]:
    """Discover PDFs across all shards or a specific shard."""
    all_pdfs = []
    
    if specific_shard is not None:
        # Process only the specified shard
        if 0 <= specific_shard < SHARDS:
            all_pdfs.extend(_discover_pdfs_in_shard(specific_shard, pattern))
        else:
            raise ValueError(f"Invalid shard index: {specific_shard}. Must be 0-{SHARDS-1}")
    else:
        # Process all shards
        for shard_idx in range(SHARDS):
            pdfs = _discover_pdfs_in_shard(shard_idx, pattern)
            all_pdfs.extend(pdfs)
    
    return all_pdfs

def _process_pdf(pdf_path: Path, logger: logging.Logger, output_dir: Path) -> bool:
    """Process a single PDF file."""
    try:
        logger.info(f"Processing: {pdf_path}")
        convert_via_cli(pdf_path, output_dir=output_dir)
        logger.info(f"✅ Completed: {pdf_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to process {pdf_path}: {e}")
        return False

# ───────────────────────────── main ────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run Marker on PDFs from sharded PDF_Bucket directories.\n"
            "Optionally supply a shard number to process only that shard,\n"
            "or a substring to filter PDF names (e.g. W1982)."
        )
    )
    ap.add_argument(
        "filter",
        nargs="?",
        help="Either a shard number (0-31) or substring to filter PDF names (e.g. W1982)",
    )
    ap.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of parallel workers (default: 8)"
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for Marker_Output (defaults to RUN_ROOT)"
    )
    args = ap.parse_args()

    logger = _get_logger()
    
    # Validate sharding configuration
    try:
        from sharding_config import validate_config
        validate_config()
    except Exception as e:
        logger.error(f"Sharding configuration error: {e}")
        sys.exit(1)

    # Set output directory
    if args.output_dir is None:
        output_dir = RUN_ROOT
    else:
        output_dir = args.output_dir
    
    logger.info(f"Output directory: {output_dir}")

    # Parse filter argument
    specific_shard = None
    pattern = None
    
    if args.filter:
        try:
            # Try to parse as shard number
            shard_num = int(args.filter)
            if 0 <= shard_num < SHARDS:
                specific_shard = shard_num
                logger.info(f"Processing only shard {shard_num}")
            else:
                raise ValueError(f"Shard number must be 0-{SHARDS-1}")
        except ValueError:
            # Not a number, treat as pattern
            pattern = args.filter
            logger.info(f"Filtering PDFs by pattern: {pattern}")

    # Discover PDFs
    pdf_files = _discover_pdfs_all_shards(pattern, specific_shard)
    
    if not pdf_files:
        logger.warning("Nothing to do – no matching PDFs found.")
        return

    logger.info(f"Found {len(pdf_files)} PDFs to process")
    
    # Process PDFs with parallel execution
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_pdf = {
            executor.submit(_process_pdf, pdf, logger, output_dir): pdf 
            for pdf in pdf_files
        }
        
        # Process completed tasks
        for future in as_completed(future_to_pdf):
            pdf = future_to_pdf[future]
            try:
                if future.result():
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Unexpected error processing {pdf}: {e}")
                failed += 1
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"PDF Processing Complete:")
    logger.info(f"  Total PDFs: {len(pdf_files)}")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Success rate: {successful/len(pdf_files)*100:.1f}%")
    logger.info(f"  Output: {output_dir}/Marker_Output/")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
