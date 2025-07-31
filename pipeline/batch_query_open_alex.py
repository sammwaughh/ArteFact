#!/usr/bin/env python3
"""
batch_query_open_alex.py
------------------------
Read every painter name from painters.xlsx (first column) and query
OpenAlex for their works using query_open_alex_with.py functionality.

Each painter gets their own JSON file in Artist-JSONs/<painter>.json.

Usage
-----
$ python Pipeline/batch_query_open_alex.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pandas as pd

# Re-use the single-painter routine
from query_open_alex_with import _write_json, query_open_alex_with

ROOT = Path(__file__).resolve().parent
PAINTERS_FILE = ROOT / "painters.xlsx"


# ───────────────────────────── helpers ─────────────────────────────────────
def _load_painters() -> List[str]:
    """Return list of painter names from first column of painters.xlsx."""
    if not PAINTERS_FILE.exists():
        sys.exit(f"❌ {PAINTERS_FILE} not found")

    df = pd.read_excel(PAINTERS_FILE, usecols=[0])  # first column only
    names = df.iloc[:, 0].dropna().astype(str).str.strip()
    return [name for name in names if name]


# ───────────────────────────── main ────────────────────────────────────────
def main() -> None:
    painters = _load_painters()
    if not painters:
        sys.exit("❌ No painter names found in painters.xlsx")

    print(f"Found {len(painters)} painters to query")

    for i, painter in enumerate(painters, 1):
        print(f"\n[{i:2d}/{len(painters)}] Querying: {painter}")
        try:
            works = query_open_alex_with(painter)
            path = _write_json(painter, works)
            print(f"   → {len(works)} works saved to {path.name}")
        except Exception as exc:
            print(f"   ⚠️  Error: {exc}")


if __name__ == "__main__":
    main()
