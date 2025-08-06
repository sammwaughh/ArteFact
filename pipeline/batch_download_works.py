#!/usr/bin/env python3

### Fill this in with new HPC code

"""
batch_download_works.py
-----------------------
Iterate over every <artist>.json file in Artist-JSONs/ and call
download_works_on.process_artist() for the corresponding painter.

Example
-------
$ python Pipeline/batch_download_works.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List
import pandas as pd
# NEW – parallel execution
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Re-use the single-artist routine & its paths
from download_works_on import JSON_DIR, process_artist


ROOT = Path(__file__).resolve().parent
PAINTERS_FILE = ROOT / "painters.xlsx"
CHECKPOINT_FILE = ROOT / "download_checkpoint.txt"

# ---------- painter-row range (1-based, inclusive) -------------------------
RANGE_START: int = 1        # first row to include (1 = first painter)
RANGE_END:   int = 5323     # last  row to include (inclusive)
# --------------------------------------------------------------------------


def _load_painters() -> List[str]:                     # NEW
    """Return painter names from painters.xlsx in sheet order."""
    if not PAINTERS_FILE.exists():
        sys.exit(f"❌ {PAINTERS_FILE} not found")
    df = pd.read_excel(PAINTERS_FILE, usecols=[0])
    names = df.iloc[:, 0].dropna().astype(str).str.strip()
    return [n for n in names if n]


def _load_checkpoint() -> set[str]:                    # NEW
    try:
        return {ln.strip() for ln in CHECKPOINT_FILE.read_text().splitlines()}
    except FileNotFoundError:
        return set()


def _append_checkpoint(artist: str) -> None:           # NEW
    with CHECKPOINT_FILE.open("a") as fh:
        fh.write(artist + "\n")


# ───────────────────────────── main ────────────────────────────────────────
def main() -> None:
    if not JSON_DIR.exists():
        sys.exit(f"❌ Artist-JSONs directory not found at {JSON_DIR}")

    painters   = _load_painters()

    # Apply 1-based inclusive slice
    slice_start = max(RANGE_START - 1, 0)      # convert to 0-based
    slice_end   = RANGE_END                    # Python slice end is exclusive
    painters    = painters[slice_start:slice_end]

    if not painters:
        sys.exit("❌ Selected painter range is empty – adjust RANGE_START / RANGE_END")

    completed  = _load_checkpoint()
    remaining  = [p for p in painters if p not in completed]

    if not remaining:
        sys.exit("✅ All painters in the selected range already processed.")

    # ─── run several artists in parallel ────────────────────────────────
    MAX_WORKERS = min(os.cpu_count() or 4, 16)   # avoid oversubscription

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_name = {pool.submit(process_artist, n): n for n in remaining}

        for fut in as_completed(future_to_name):
            name = future_to_name[fut]
            try:
                fut.result()           # re-raise any exception from worker
                _append_checkpoint(name)
            except Exception as exc:
                print(f"⚠️  {name}: {exc}")


# ───────────────────────────── entry point ─────────────────────────────────
if __name__ == "__main__":
    main()
