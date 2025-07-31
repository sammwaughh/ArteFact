#!/usr/bin/env python3
"""
check_topic.py – quick helper to resolve OpenAlex topic / concept IDs
to their English display names.

Usage
-----
$ python check_topic.py T13922 C52119013

For each supplied ID the script prints:
ID → "Display Name"
"""
from __future__ import annotations

import sys

import requests

CONTACT_EMAIL = "samjmwaugh@gmail.com"
BASE = "https://api.openalex.org"

# NEW ────────────────────────── predefined topic IDs ──────────────────────────
TOPIC_IDS_STR: str = (
    "C52119013|"  # Art History
    "T13922|"  # Historical Art and Culture Studies
    "T12632|"  # Visual Culture and Art Theory
    "T12650|"  # Aesthetic Perception and Analysis
    "C204034006|"  # Art Criticism
    "C501303744|"  # Iconography
    "C554736915|"  # Ancient Art
    "C138634970|"  # Medieval Art
    "T12076|"  # Renaissance and Early Modern Studies
    "C189135316|"  # Modern Art
    "C85363599|"  # Contemporary Art
    "C32685002|"  # Romanticism
    "C12183850|"  # Indian / Asian Art
    "C2993994385|"  # Islamic Art
    "C64626740"  # African Art
)
PRESET_IDS: list[str] = [tid for tid in TOPIC_IDS_STR.split("|") if tid]


def fetch_display_name(oa_id: str) -> str | None:
    """
    Return the English display name for *oa_id* (“C…” or “T…”).
    None is returned for HTTP errors or malformed responses.
    """
    if not oa_id or oa_id[0] not in "CT":
        return None  # clearly not a valid OpenAlex concept/topic ID

    endpoint = "/concepts/" if oa_id.startswith("C") else "/topics/"
    url = f"{BASE}{endpoint}{oa_id}"
    try:
        resp = requests.get(url, params={"mailto": CONTACT_EMAIL}, timeout=20)
        resp.raise_for_status()
        return resp.json().get("display_name")
    except Exception:
        return None


def main(argv: list[str]) -> None:
    # If no IDs given on the CLI, fall back to the predefined list above.
    ids = argv or PRESET_IDS

    for tid in ids:
        name = fetch_display_name(tid)
        if name:
            print(f"{tid} → {name!r}")
        else:
            print(f"{tid} → ERROR: not found or invalid ID")


if __name__ == "__main__":
    main(sys.argv[1:])
