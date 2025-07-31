# -*- coding: utf-8 -*-
"""
openalex_query.py – minimal helper to fetch English, open-access works
matching a painter’s name and a fixed set of art-history topic IDs.

Usage
-----
>>> from openalex_query import query_open_alex_with
>>> works = query_open_alex_with("Leonardo da Vinci")
>>> print(len(works), "works returned")
"""

from __future__ import annotations

import json
import re  # NEW
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ───────────────────────── constants & config ──────────────────────────────

CONTACT_EMAIL: str = "samjmwaugh@gmail.com"

TOPIC_IDS: str = (
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

# API tuning
DEFAULT_PER_PAGE: int = 200
DEFAULT_SLEEP_SEC: float = 0.15
MAX_RETRIES: int = 5
BACKOFF_FACTOR: float = 0.5
SESSION_TIMEOUT_SEC: int = 30

JSON_DIR: Path = Path("Artist-JSONs")  # NEW – output folder

# ─────────────────────────── HTTP session helper ────────────────────────────


def _make_session() -> requests.Session:
    """Return a retry-enabled requests session."""
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        raise_on_status=False,
    )
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": f"ArtContext/0.1 (mailto:{CONTACT_EMAIL})"})
    return s


SESSION: requests.Session = _make_session()

# ────────────────────────────── small utils ─────────────────────────────────


def _safe_json_parse(value):
    """Return parsed JSON or None on failure."""
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:  # pragma: no cover
        return None


def _concept_ids(concepts_field) -> List[str]:
    """Extract bare concept IDs from the concepts array."""
    ids: List[str] = []
    for c in _safe_json_parse(concepts_field) or []:
        if isinstance(c, dict) and "id" in c:
            ids.append(c["id"].rsplit("/", 1)[-1])
    return ids


def _extract_pdf_links(work: Dict) -> Tuple[str, str]:
    """
    Return (best, backup) URLs from the various OA location blobs.
    “Best” = first direct .pdf link if present, else first OA url.
    """
    candidates: List[str] = []

    def _push(src):
        if isinstance(src, dict):
            candidates.extend(
                [
                    src.get("pdf_url"),
                    src.get("landing_page_url"),
                    src.get("url"),  # primary_location sometimes
                ]
            )

    _push(_safe_json_parse(work.get("best_oa_location")))
    _push(_safe_json_parse(work.get("primary_location")))
    for loc in _safe_json_parse(work.get("locations")) or []:
        _push(loc)

    candidates = [c for c in dict.fromkeys(candidates) if c]  # dedupe / drop None
    best = next((c for c in candidates if ".pdf" in c.lower()), "") or (
        candidates[0] if candidates else ""
    )
    backup = next((c for c in candidates if c != best), "")
    return best, backup


def _sanitize(name: str) -> str:  # NEW
    """Return *name* as a safe path component."""
    # 1. remove forbidden characters, 2. collapse whitespace to “_”
    cleaned = re.sub(r'[\\/*?:"<>|]', "", name.lower())
    cleaned = re.sub(r"\s+", "_", cleaned).strip("_")
    return cleaned or "untitled"


def _write_json(painter: str, works: List[Dict]) -> Path:  # NEW
    """Dump *works* to Artist-JSONs/<painter>.json and return the path."""
    JSON_DIR.mkdir(exist_ok=True)
    dest = JSON_DIR / f"{_sanitize(painter)}.json"
    dest.write_text(json.dumps(works, indent=2, ensure_ascii=False))
    return dest


# ─────────────────────────── main public helper ─────────────────────────────


def query_open_alex_with(
    painter: str,
    *,
    per_page: int = DEFAULT_PER_PAGE,
    sleep_sec: float = DEFAULT_SLEEP_SEC,
) -> List[Dict]:
    """
    Return metadata for all English, open-access OpenAlex works
    mentioning *painter* and tagged with ≥1 art-history topic in TOPIC_IDS.

    Each dict contains:
        id, title, relevance_score, doi, topic_ids, best_pdf, backup_pdf
    (plus any other raw OpenAlex fields you might need later).
    """
    base_url = "https://api.openalex.org/works"
    oa_filter = f"language:en,is_oa:true,topics.id:{TOPIC_IDS}"
    cursor: str | None = "*"
    works: List[Dict] = []
    page = 0

    while cursor:
        page += 1
        params: Dict[str, str | int] = {
            "filter": oa_filter,
            "search": painter,
            "per_page": per_page,
            "cursor": cursor,
            "select": (
                "id,display_name,relevance_score,doi,concepts,"
                "primary_location,open_access,locations,best_oa_location"
            ),
            "mailto": CONTACT_EMAIL,
        }

        # Handle 429 manually because we may need to honour Retry-After
        while True:
            resp = SESSION.get(base_url, params=params, timeout=SESSION_TIMEOUT_SEC)
            if resp.status_code != 429:
                break
            time.sleep(int(resp.headers.get("Retry-After", "60")))

        if resp.status_code != 200:  # abort on fatal error
            raise RuntimeError(f"OpenAlex HTTP {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        for w in data.get("results", []):
            if w.get("relevance_score", 0) <= 1:
                continue  # early filter

            best, backup = _extract_pdf_links(w)
            works.append(
                {
                    "id": w.get("id"),
                    "title": w.get("display_name"),
                    "relevance_score": w.get("relevance_score"),
                    "doi": w.get("doi"),
                    "topic_ids": _concept_ids(w.get("concepts")),
                    "best_pdf": best,
                    "backup_pdf": backup,
                    # keep entire raw work for later if you wish
                    "raw": w,
                }
            )

        cursor = data.get("meta", {}).get("next_cursor")
        time.sleep(sleep_sec)

    return works


# ───────────────────────────── CLI test hook ────────────────────────────────
if __name__ == "__main__":
    import sys

    painter_arg = sys.argv[1] if len(sys.argv) > 1 else "Leonardo da Vinci"
    print(f"Querying OpenAlex for: {painter_arg!r}")
    out = query_open_alex_with(painter_arg)
    print(f"{len(out)} works found (relevance > 1)")
    path = _write_json(painter_arg, out)  # CHANGED
    print(f"Results saved to {path}")
