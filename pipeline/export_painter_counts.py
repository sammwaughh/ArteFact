#!/usr/bin/env python3
"""
export_painter_counts.py
------------------------
Read painters.xlsx, count how many JSON work-records exist for every painter
and write the result to *painter_counts.xlsx* (two columns: Painter, Works).
painters.xlsx itself is never modified.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd

ROOT        = Path(__file__).resolve().parent
JSON_DIR    = ROOT / "Artist-JSONs"
PAINTERS_XL = ROOT / "painters.xlsx"
OUT_XL      = ROOT / "painter_counts.xlsx"


def _load_ordered_painters() -> List[str]:
    df = pd.read_excel(PAINTERS_XL, usecols=[0])
    names = df.iloc[:, 0].dropna().astype(str).str.strip()
    return [n for n in names if n]


def _count_works(painter: str) -> int:
    slug = painter.lower().replace(" ", "_")
    fp = JSON_DIR / f"{slug}.json"
    if not fp.is_file():
        return 0
    try:
        data = json.loads(fp.read_text())
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0


def main() -> None:
    painters = _load_ordered_painters()
    rows: List[Tuple[str, int]] = [(name, _count_works(name)) for name in painters]

    df_out = pd.DataFrame(rows, columns=["Painter", "Works"])
    df_out.to_excel(OUT_XL, index=False)
    print(f"✓ Wrote counts for {len(df_out)} painters → {OUT_XL}")


if __name__ == "__main__":
    main()