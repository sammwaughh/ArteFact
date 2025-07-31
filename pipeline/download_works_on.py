#!/usr/bin/env python3
"""
download_works_on.py  –  batch-download every OpenAlex work for one artist

CLI
---
$ python Pipeline/download_works_on.py Titian
$ python Pipeline/download_works_on.py "Leonardo da Vinci"
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

# Local helper – PDF fetcher only
from download_single_work import download_pdf

# ─────────────────────────── paths & logging ───────────────────────────────

ROOT = Path(__file__).resolve().parent
JSON_DIR = ROOT / "Artist-JSONs"
LOG_DIR = ROOT / "logs/Download-PDF-Logs"
LOG_DIR.mkdir(exist_ok=True)


# one file per artist: logs/<artist>_download_log.log
def _setup_logger(slug: str) -> logging.Logger:
    logger = logging.getLogger(slug)
    if logger.handlers:  # already configured in same run
        return logger

    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_DIR / f"{slug}_download_log.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    logger.addHandler(fh)
    logger.propagate = False  # avoid duplicate lines on root
    return logger


# ───────────────────────────── core routine ────────────────────────────────


def _load_works(json_path: Path) -> List[Dict]:
    try:
        return json.loads(json_path.read_text())
    except Exception as exc:
        raise RuntimeError(f"Cannot read {json_path}: {exc}") from exc


def process_artist(artist: str) -> None:
    slug = artist.lower().replace(" ", "_")
    json_fp = JSON_DIR / f"{slug}.json"
    logger = _setup_logger(slug)
    logger.info("Reading %s", json_fp)

    works = _load_works(json_fp)
    if not works:
        logger.warning("No works inside %s – nothing to do.", json_fp)
        return

    ok, fail = 0, 0
    for w in works:
        work_id = w.get("id", "")
        if not work_id:
            continue
        logger.info("▶  %s  –  %s", work_id, w.get("title", "")[:80])

        if download_pdf(work_id):
            ok += 1
        else:
            fail += 1
            logger.error("✗  Download failed for %s", work_id)

    summary = f"Done.  Success: {ok}   Fail: {fail}   Total: {ok + fail}"
    logger.info(summary)
    print(summary)  # also echo once to terminal


# ────────────────────────────── entry-point ────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python download_works_on.py <Artist Name>")
    process_artist(sys.argv[1].strip())
