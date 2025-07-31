#!/usr/bin/env python3
"""
clean_painters.py
-----------------
Read painters.xlsx, filter out URLs and non-painter entries, and save
only clean painter names to cleaned.xlsx (single column).

Usage
-----
$ python clean_painters.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "painters.xlsx"
OUTPUT_FILE = ROOT / "cleaned.xlsx"


def is_painter_name(text: str) -> bool:
    """Return True if text looks like a painter name, not a URL or junk."""
    text = text.strip()

    # Skip empty or very short strings
    if len(text) < 2:
        return False

    # Skip anything containing URL indicators
    url_patterns = [
        r"https?://",  # http:// or https://
        r"www\.",  # www.
        r"\.com",  # .com
        r"\.org",  # .org
        r"\.net",  # .net
        r"\.edu",  # .edu
    ]

    for pattern in url_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False

    # Skip strings that are mostly numbers or special characters
    if re.match(r"^[\d\s\-\.]+$", text):
        return False

    # Skip common non-name patterns
    junk_patterns = [
        r"^(page|chapter|section)\s*\d+",
        r"^\d+\.",  # numbered lists
        r"^(see|ref|reference)",  # references
        r"^(table|figure|fig)\s*\d+",  # figure/table references
    ]

    for pattern in junk_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False

    return True


def clean_painters() -> pd.DataFrame:
    """Load, filter, and return cleaned painter names in original order."""
    if not INPUT_FILE.exists():
        sys.exit(f"❌ {INPUT_FILE} not found")

    # Read first column only, preserve order
    df = pd.read_excel(INPUT_FILE, usecols=[0])
    first_col = df.iloc[:, 0].dropna().astype(str).str.strip()

    # Filter while preserving order
    clean_names = [name for name in first_col if is_painter_name(name)]

    return pd.DataFrame({"Painter": clean_names})


def main() -> None:
    print(f"Reading {INPUT_FILE.name}...")
    cleaned_df = clean_painters()

    if cleaned_df.empty:
        sys.exit("❌ No valid painter names found after cleaning")

    cleaned_df.to_excel(OUTPUT_FILE, index=False)
    print(f"✅ Saved {len(cleaned_df)} clean painter names to {OUTPUT_FILE.name}")

    # Show first few for verification
    print("\nFirst 10 entries:")
    for name in cleaned_df["Painter"].head(10):
        print(f"  • {name}")


if __name__ == "__main__":
    main()
