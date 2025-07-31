#!/usr/bin/env python3
# filepath: /Users/samuelwaugh/Desktop/ArtContext/Pipeline/download_single_work.py
"""
download_work_pdf.py ─ fetch a single PDF for an OpenAlex work ID.

CLI
---
$ python download_work_pdf.py W1549104703          # bare ID
$ python download_work_pdf.py https://openalex.org/W1549104703

The script looks through every JSON file in Artist-JSONs/, finds the work
dictionary whose “id” matches, pulls the best/backup PDF URL and downloads
to PDF_Bucket/<ID>.pdf.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Optional deep PDF validation
try:
    from PyPDF2 import PdfReader  # type: ignore
except ImportError:
    PdfReader = None  # fallback – basic header check only

# ─────────────────────────── folders & constants ────────────────────────────
# Always resolve relative to this file, not the CWD.
ROOT_DIR = Path(__file__).resolve().parent

JSON_DIR = ROOT_DIR / "Artist-JSONs"
OUT_DIR = ROOT_DIR / "PDF_Bucket"
OUT_DIR.mkdir(exist_ok=True)

CONTACT_EMAIL = "samjmwaugh@gmail.com"
TIMEOUT_SEC = 30
BACKOFF_FACTOR = 0.5
MAX_RETRIES = 5

# artists → [work-ids …] mapping ---------------------------
ARTISTS_FILE = ROOT_DIR / "artists.json"

# ───────────────────────────── PDF sanity check ────────────────────────────


def _pdf_is_valid(path: Path) -> bool:
    """
    Return True iff *path* is a readable PDF.
    • Always check for %PDF- header.
    • When PyPDF2 is available, try to parse the file.
    """
    try:
        with path.open("rb") as fh:
            if fh.read(5) != b"%PDF-":
                return False
            if PdfReader:
                fh.seek(0)
                PdfReader(fh, strict=False)  # raises on corruption
    except Exception:
        return False
    return True


def _update_artists_json(artist: str, work_id: str) -> None:
    """
    Ensure *work_id* is present in the list for *artist* inside artists.json.
    """
    try:
        data = json.loads(ARTISTS_FILE.read_text())
        if not isinstance(data, dict):
            data = {}
    except FileNotFoundError:
        data = {}
    except Exception:
        data = {}

    lst = data.get(artist, [])
    if work_id not in lst:
        lst.append(work_id)
        data[artist] = lst
        ARTISTS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# ───────────────────────────── HTTP session ─────────────────────────────────


def _make_session() -> requests.Session:
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


SESSION = _make_session()


# ───────────────────────────── helpers (copied) ─────────────────────────────


def _safe_json_parse(value):
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return None


def _get_candidate_links(work: Dict) -> List[str]:
    """Collect distinct OA / PDF links from OpenAlex location blobs."""
    cand: List[str] = []

    def _push(src):
        if isinstance(src, dict):
            cand.extend(
                [src.get("pdf_url"), src.get("landing_page_url"), src.get("url")]
            )

    _push(_safe_json_parse(work.get("best_oa_location")))
    _push(_safe_json_parse(work.get("primary_location")))
    for loc in _safe_json_parse(work.get("locations")) or []:
        _push(loc)

    # deduplicate + drop blanks
    return [c for c in dict.fromkeys(cand) if c]


def _best_and_backup(work: Dict) -> Tuple[str, str]:
    """Return (best, backup) download URLs."""
    # First honour pre-extracted fields if present
    best = (work.get("best_pdf") or "").strip()
    backup = (work.get("backup_pdf") or "").strip()

    if best:  # already good
        if not backup:
            # pick another candidate not equal to best
            for c in _get_candidate_links(work.get("raw") or work):
                if c != best:
                    backup = c
                    break
        return best, backup

    cand = _get_candidate_links(work.get("raw") or work)
    best = next((c for c in cand if ".pdf" in c.lower()), "") or (
        cand[0] if cand else ""
    )
    backup = next((c for c in cand if c != best), "")
    return best, backup


def _slug(openalex_id: str) -> str:
    """Return ‘W…’ part."""
    return openalex_id.rstrip("/").rsplit("/", 1)[-1]


# ───────────────────────────── main routine ─────────────────────────────────


def download_pdf(openalex_id: str) -> Path | None:
    target_slug = _slug(openalex_id)
    # Search every JSON file
    for fp in JSON_DIR.glob("*.json"):
        try:
            works = json.loads(fp.read_text())
        except Exception:
            continue
        artist_slug = fp.stem.lower()  # e.g. "titian"
        logger = logging.getLogger(artist_slug)

        for w in works:
            if _slug(str(w.get("id", ""))) == target_slug:
                best, backup = _best_and_backup(w)
                for url in (best, backup):
                    if not url:
                        continue
                    try:
                        r = SESSION.get(url, timeout=TIMEOUT_SEC, allow_redirects=True)
                        if r.status_code == 429:
                            time.sleep(int(r.headers.get("Retry-After", "60")))
                            r = SESSION.get(url, timeout=TIMEOUT_SEC)
                        if r.status_code == 200 and r.content:
                            dest = OUT_DIR / f"{target_slug}.pdf"
                            dest.write_bytes(r.content)

                            # validate PDF before proceeding
                            if not _pdf_is_valid(dest):
                                logger.warning(
                                    "Corrupt PDF detected → %s (deleted)", dest
                                )
                                dest.unlink(missing_ok=True)
                                continue  # try backup/next URL

                            logger.info("Saved PDF → %s", dest)

                            # ────── update works.json  (kept in Pipeline/) ──────
                            works_file = ROOT_DIR / "works.json"
                            try:
                                db = json.loads(works_file.read_text())
                                if not isinstance(db, dict):  # safeguard
                                    db = {}
                            except FileNotFoundError:
                                db = {}
                            except Exception:
                                db = {}

                            work_key = target_slug  # e.g. "W1549104703"
                            # keep existing sentence/image info if present
                            prev = db.get(work_key, {})
                            entry = {
                                "Artist": fp.stem.lower(),
                                "Link": url,
                                "Number of Sentences": prev.get(
                                    "Number of Sentences", 0
                                ),
                                "DOI": str(w.get("doi", "")).strip(),
                                "ImageIDs": prev.get("ImageIDs", []),
                                "TopicIDs": [str(t) for t in w.get("topic_ids", [])],
                                "Relevance": float(
                                    w.get("relevance_score", w.get("relevance", 0.0))
                                ),
                            }
                            db[work_key] = entry  # (re)write unconditionally

                            works_file.write_text(
                                json.dumps(db, indent=2, ensure_ascii=False)
                            )
                            logger.info("↳ metadata (re)written to %s", works_file)

                            # --------------- update artists.json --------------
                            _update_artists_json(fp.stem.lower(), work_key)
                            return dest
                        else:
                            logger.warning("%s → HTTP %s", url, r.status_code)
                    except Exception as exc:
                        logger.error("Error on %s: %s", url, exc)
                logger.warning("No working link found.")
                return None
    logging.getLogger("download_single_work").error(
        "No work with ID %s found in %s", openalex_id, JSON_DIR
    )
    return None


def main(argv: List[str]) -> None:
    if not argv:
        print("Usage: python download_work_pdf.py <OpenAlex-ID-or-URL>")
        sys.exit(1)
    download_pdf(argv[0])


if __name__ == "__main__":
    main(sys.argv[1:])
