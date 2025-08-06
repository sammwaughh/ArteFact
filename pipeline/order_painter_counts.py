#!/usr/bin/env python3
"""
order_painter_counts.py
-----------------------
Load *painter_counts.xlsx*, drop rows where Works == 0, sort descending by
Works, and save to *painter_counts_ordered.xlsx*.
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd

ROOT        = Path(__file__).resolve().parent
IN_XL       = ROOT / "painter_counts.xlsx"
OUT_XL      = ROOT / "painter_counts_ordered.xlsx"


def main() -> None:
    if not IN_XL.exists():
        raise SystemExit(f"❌ {IN_XL} not found")

    df = (
        pd.read_excel(IN_XL)
        .query("Works > 0")
        .sort_values("Works", ascending=False, ignore_index=True)
    )

    if df.empty:
        print("No painters with >0 works found.")
        return

    df.to_excel(OUT_XL, index=False)
    print(f"✓ Saved {len(df)} painters → {OUT_XL}")


if __name__ == "__main__":
    main()