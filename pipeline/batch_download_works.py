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

# Re-use the single-artist routine & its paths
from download_works_on import JSON_DIR, process_artist


# ───────────────────────────── helpers ─────────────────────────────────────
def _slug_to_name(slug: str) -> str:
    """
    Convert “arthur_hughes”  →  “Arthur Hughes”
    """
    return " ".join(part.capitalize() for part in slug.split("_") if part)


def _discover_artists(dir_: Path) -> List[str]:
    """
    Return all artists detected via *.json files in *dir_* as pretty names.
    """
    slugs = sorted(p.stem for p in dir_.glob("*.json"))
    return [_slug_to_name(s) for s in slugs if s]


# ───────────────────────────── main ────────────────────────────────────────
def main() -> None:
    if not JSON_DIR.exists():
        sys.exit(f"❌ Artist-JSONs directory not found at {JSON_DIR}")

    artists = _discover_artists(JSON_DIR)
    if not artists:
        sys.exit("❌ No artist JSON files to process.")

    for name in artists:
        print(f"\n=== Downloading works for {name} ===")
        try:
            process_artist(name)
        except Exception as exc:
            print(f"⚠️  {name}: {exc}")


# ───────────────────────────── entry point ─────────────────────────────────
if __name__ == "__main__":
    main()
