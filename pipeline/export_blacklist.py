#!/usr/bin/env python3
"""
export_blacklist.py
-------------------
Read painter_counts_ordered.xlsx, find rows where the “Blacklist” column
is truthy (1 / True / non-zero), and write the painter names to
blacklist.json.
"""

from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

ROOT        = Path(__file__).resolve().parent
IN_XL       = ROOT / "painter_counts_ordered.xlsx"
OUT_JSON    = ROOT / "blacklist.json"


def main() -> None:
    if not IN_XL.exists():
        raise SystemExit(f"❌ {IN_XL} not found")

    df = pd.read_excel(IN_XL)

    if "Blacklist" not in df.columns:
        raise SystemExit("❌ Column “Blacklist” missing in the spreadsheet")

    names = (
        df.loc[df["Blacklist"].astype(bool), "Painter"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    OUT_JSON.write_text(json.dumps(names, indent=2, ensure_ascii=False))
    print(f"✓ Saved {len(names)} painters → {OUT_JSON}")


if __name__ == "__main__":
    main()