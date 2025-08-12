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
import json            # NEW – read blacklist
from pathlib import Path
from typing import List
import pandas as pd
# NEW – parallel execution
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Re-use the single-artist routine & its paths
from download_works_on import JSON_DIR, process_artist
# Add import for sharding functionality
from sharding import SHARDS_DIR


ROOT = Path(__file__).resolve().parent
PAINTERS_FILE = ROOT / "painters.xlsx"
CHECKPOINT_FILE = ROOT / "download_checkpoint.txt"
# blacklist file (optional)
BLACKLIST_FILE  = ROOT / "blacklist.json"

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


# ───────────────────────── blacklist ──────────────────────────
def _load_blacklist() -> set[str]:
    """Return a set of painters to skip; empty if file absent/malformed."""
    try:
        return set(json.loads(BLACKLIST_FILE.read_text()))
    except FileNotFoundError:
        return set()
    except Exception:
        print("⚠️  Could not parse blacklist.json – ignoring")
        return set()


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

    # PATCH: Ensure shards directory exists and is properly configured
    SHARDS_DIR.mkdir(exist_ok=True)
    print(f"📁 Shards directory: {SHARDS_DIR}")
    print(f"🔢 Number of shards: {len([_ for _ in SHARDS_DIR.glob('shard_*')])}")

    painters   = _load_painters()

    # Apply 1-based inclusive slice
    slice_start = max(RANGE_START - 1, 0)      # convert to 0-based
    slice_end   = RANGE_END                    # Python slice end is exclusive
    painters    = painters[slice_start:slice_end]

    if not painters:
        sys.exit("❌ Selected painter range is empty – adjust RANGE_START / RANGE_END")

    completed  = _load_checkpoint()
    blacklist  = _load_blacklist()
    remaining  = [p for p in painters
                  if p not in completed and p not in blacklist]

    if not remaining:
        sys.exit("✅ All painters in the selected range already processed.")

    # ─── run several artists in parallel ────────────────────────────────
    MAX_WORKERS = 32

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_name = {pool.submit(process_artist, n): n for n in remaining}

        for fut in as_completed(future_to_name):
            name = future_to_name[fut]
            try:
                fut.result()           # re-raise any exception from worker
                _append_checkpoint(name)
            except Exception as exc:
                print(f"⚠️  {name}: {exc}")

    # PATCH: Print summary of sharded results
    print("\n📊 Sharding Summary:")
    total_works = 0
    for i in range(32):
        shard_file = SHARDS_DIR / f"shard_{i:02d}" / f"works_shard_{i:02d}.json"
        if shard_file.exists():
            try:
                shard_data = json.loads(shard_file.read_text())
                count = len(shard_data)
                total_works += count
                print(f"  Shard {i:02d}: {count:6d} works")
            except Exception:
                print(f"  Shard {i:02d}: ERROR reading file")
        else:
            print(f"  Shard {i:02d}: No file")
    
    print(f"  Total: {total_works} works across all shards")
    print(f"\n💡 Run merge_works_and_artists.py to consolidate metadata")


# ───────────────────────────── entry point ─────────────────────────────────
if __name__ == "__main__":
    main()
