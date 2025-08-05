#!/usr/bin/env python3
"""
Harvest painting metadata from Wikidata and save to Parquet, using
endpoint‑friendly settings that comply with WDQS rate limits.

Same logic as the Excel version, but:
✓ Final output is 'paintings.parquet'
✓ A checkpoint file is overwritten every few pages      (defaults to 5)
✓ SIGTERM/SIGINT handlers write a last‑minute checkpoint
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util import Retry

# ─────────────────────────────────── constants ──────────────────────────────────
LIMIT: int = 32_000
INITIAL_CHUNK_SIZE: int = 400
MIN_CHUNK_SIZE: int = 100
CHUNK_GROW_BACK_FACTOR: float = 1.2

SITELINKS_THRESHOLD: int = 1
SUBCHUNK_SIZE: int = 200
WORKERS: int = 3

# contact / etiquette
CONTACT_EMAIL: str = "samjmwaugh@gmail.com"
ORG_URL: str = "https://github.com/sammwaughh/ArtContext"
PAUSE_BETWEEN_HTTP: float = 0.3
PAUSE_BETWEEN_PAGES: float = 1.5
MAX_ATTEMPTS: int = 6
REQUEST_TIMEOUT: Tuple[int, int] = (10, 75)

# output & checkpointing
OUT_FILE: Path = Path("paintings.parquet")
CHECKPOINT_FILE: Path = Path("checkpoint.parquet")
CHECKPOINT_EVERY_PAGES: int = 5  # overwrite checkpoint this often

# ─────────────────────────────────── logging ────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/wikidata_harvest.log",
    level=logging.INFO,
    filemode="w",
    format="%(asctime)s  %(levelname)s  %(message)s",
)
print("Harvesting painting metadata…")

# ──────────────────────────── helper: chunk an iterable ─────────────────────────
def chunked(it: Iterable, size: int):
    it = iter(it)
    while chunk := tuple(islice(it, size)):
        yield chunk

# ─────────────────────── requests session with retry policy ─────────────────────
def make_session() -> requests.Session:
    retry_cfg = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"POST"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_cfg)
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": f"ArtContextHarvester/0.2 ({ORG_URL}; mailto:{CONTACT_EMAIL})",
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
        }
    )
    s.mount("https://", adapter)
    return s

SESSION = make_session()
TRANSIENT = {502, 503, 504}

# ──────────────────────── robust SPARQL POST with retries ───────────────────────
def query_wd(query: str) -> dict:
    url = "https://query.wikidata.org/sparql"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            logging.info("HTTP POST to WDQS (attempt %d/%d)", attempt, MAX_ATTEMPTS)
            r = SESSION.post(url, data={"query": query}, timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:
                retry_for = int(r.headers.get("Retry-After", "60"))
                logging.warning("429 Too Many Requests – waiting %s s", retry_for)
                time.sleep(retry_for)
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as exc:
            code = getattr(exc.response, "status_code", None)
            if code in TRANSIENT and attempt < MAX_ATTEMPTS:
                wait = 2**attempt
                logging.warning("%s from WDQS – retry in %s s", code, wait)
                time.sleep(wait)
                continue
            logging.error("Fatal HTTP error: %s", exc)
            raise
        except requests.exceptions.RequestException as exc:
            if attempt < MAX_ATTEMPTS:
                wait = 2**attempt
                logging.warning("Network error: %s – retry in %s s", exc, wait)
                time.sleep(wait)
                continue
            raise
        finally:
            time.sleep(PAUSE_BETWEEN_HTTP)

# ────────────────────────────── basic‑page query ───────────────────────────────
def get_basic_records(offset: int, page_size: int, threshold: int):
    q = f"""
        SELECT ?painting ?paintingLabel ?creator ?creatorLabel
               ?inception ?wikipedia_url ?linkCount
               ?materialLabel ?height ?width ?locationLabel ?collectionLabel
        WHERE {{
          ?painting wdt:P31 wd:Q3305213 ;
                    wikibase:sitelinks ?linkCount .
          FILTER(?linkCount >= {threshold})
          OPTIONAL {{ ?painting wdt:P170 ?creator }}
          OPTIONAL {{ ?painting wdt:P571 ?inception }}
          OPTIONAL {{ ?painting wdt:P186 ?material }}
          OPTIONAL {{ ?painting wdt:P2048 ?height }}
          OPTIONAL {{ ?painting wdt:P2049 ?width }}
          OPTIONAL {{ ?painting wdt:P276 ?location }}
          OPTIONAL {{ ?painting wdt:P195 ?collection }}
          OPTIONAL {{
            ?paintingArticle schema:about ?painting ;
                             schema:inLanguage "en" ;
                             schema:isPartOf <https://en.wikipedia.org/> .
            BIND(?paintingArticle AS ?wikipedia_url)
          }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT {page_size} OFFSET {offset}
        """
    try:
        data = query_wd(q)
    except (requests.exceptions.ReadTimeout, requests.exceptions.HTTPError, json.JSONDecodeError) as exc:
        if page_size > MIN_CHUNK_SIZE:
            new_size = max(page_size // 2, MIN_CHUNK_SIZE)
            logging.warning("Query failed (%s) – retrying with size %d", exc, new_size)
            return get_basic_records(offset, new_size, threshold)
        raise

    bindings = data.get("results", {}).get("bindings", [])
    records, pids = {}, []
    for b in bindings:
        pid = b["painting"]["value"]
        records[pid] = {
            "Painting ID": pid,
            "Title": b.get("paintingLabel", {}).get("value", ""),
            "Creator ID": b.get("creator", {}).get("value", ""),
            "Creator": b.get("creatorLabel", {}).get("value", ""),
            "Inception": b.get("inception", {}).get("value", ""),
            "Wikipedia URL": b.get("wikipedia_url", {}).get("value", ""),
            "Link Count": int(b["linkCount"]["value"]),
            "Material": b.get("materialLabel", {}).get("value", ""),
            "Height": b.get("height", {}).get("value", ""),
            "Width": b.get("width", {}).get("value", ""),
            "Location": b.get("locationLabel", {}).get("value", ""),
            "Collection": b.get("collectionLabel", {}).get("value", ""),
        }
        pids.append(pid)
    return records, pids

# ───────────────────────── aggregate fields in sub‑chunks ──────────────────────
def get_agg_fields(painting_ids: List[str]) -> Dict[str, dict]:
    agg: Dict[str, dict] = {}
    def one_chunk(ids: Tuple[str, ...]) -> Dict[str, dict]:
        vals = " ".join(f"<{x}>" for x in ids)
        q = f"""
            SELECT ?painting
                   (GROUP_CONCAT(DISTINCT ?depictsLabel;  separator=", ") AS ?depicts)
                   (GROUP_CONCAT(DISTINCT ?movementLabel; separator=", ") AS ?movs)
                   (GROUP_CONCAT(DISTINCT ?movement;      separator=", ") AS ?movIDs)
            WHERE {{
              VALUES ?painting {{ {vals} }}
              OPTIONAL {{
                  ?painting wdt:P180 ?depicts .
                  ?depicts  rdfs:label ?depictsLabel .
                  FILTER(LANG(?depictsLabel) = "en")
              }}
              OPTIONAL {{
                  ?painting wdt:P135 ?movement .
                  ?movement rdfs:label ?movementLabel .
                  FILTER(LANG(?movementLabel) = "en")
              }}
            }}
            GROUP BY ?painting
            """
        try:
            d = query_wd(q)
        except requests.exceptions.HTTPError as exc:
            if exc.response and exc.response.status_code in TRANSIENT and len(ids) > 1:
                mid = len(ids) // 2
                return {**one_chunk(ids[:mid]), **one_chunk(ids[mid:])}
            raise
        res = {}
        for b in d.get("results", {}).get("bindings", []):
            pid = b["painting"]["value"]
            res[pid] = {
                "Depicts": b.get("depicts", {}).get("value", ""),
                "Movements": b.get("movs", {}).get("value", ""),
                "Movement IDs": b.get("movIDs", {}).get("value", ""),
            }
        return res

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(one_chunk, tuple(ch)) for ch in chunked(painting_ids, SUBCHUNK_SIZE)]
        for f in as_completed(futs):
            agg.update(f.result())
    return agg

# ────────────────────────────── checkpoint helpers ─────────────────────────────
def write_checkpoint(rows: List[dict]) -> None:
    """Overwrite checkpoint.parquet with current accumulated rows."""
    if not rows:
        return
    pd.DataFrame(rows).to_parquet(CHECKPOINT_FILE, index=False)
    logging.info("Checkpoint written: %s (%d rows)", CHECKPOINT_FILE, len(rows))

def handle_sigterm(signum, frame):
    logging.warning("SIGTERM received – saving checkpoint and exiting")
    write_checkpoint(all_rows)
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

# ─────────────────────────────────── main loop ─────────────────────────────────
all_rows: List[dict] = []
offset, page_size = 0, INITIAL_CHUNK_SIZE
page_count = 0
bar = tqdm(total=LIMIT, unit=" rows")

try:
    while len(all_rows) < LIMIT:
        basic, ids = get_basic_records(offset, page_size, SITELINKS_THRESHOLD)
        if not basic:
            break

        agg = get_agg_fields(ids)
        merged = [
            {**basic[pid], **agg.get(pid, {"Depicts": "", "Movements": "", "Movement IDs": ""})}
            for pid in basic
        ]

        existing = {r["Painting ID"] for r in all_rows}
        new = [r for r in merged if r["Painting ID"] not in existing]
        all_rows.extend(new)
        bar.update(len(new))

        page_count += 1
        if page_count % CHECKPOINT_EVERY_PAGES == 0:
            write_checkpoint(all_rows)

        offset += len(basic)
        if page_size < INITIAL_CHUNK_SIZE and len(new) == len(basic):
            page_size = min(int(page_size * CHUNK_GROW_BACK_FACTOR), INITIAL_CHUNK_SIZE)

        time.sleep(PAUSE_BETWEEN_PAGES)

except KeyboardInterrupt:
    logging.warning("Interrupted – writing final checkpoint")
    write_checkpoint(all_rows)
    sys.exit(0)
finally:
    bar.close()

logging.info("Total unique paintings: %d", len(all_rows))

# ───────────────────────────── dataframe & Parquet ────────────────────────────
df = pd.DataFrame(all_rows)
df["File Name"] = df["Painting ID"].str.extract(r"([^/]+)$", expand=False).fillna("") + "_0.png"
df["Year"] = pd.to_numeric(df["Inception"].str[:4], errors="coerce")

df["Dimensions"] = ""
if {"Height", "Width"}.issubset(df.columns):
    mask = (
        df["Height"].notna()
        & df["Width"].notna()
        & (df["Height"] != "")
        & (df["Width"] != "")
    )
    if mask.any():
        df.loc[mask, "Dimensions"] = (
            df.loc[mask, "Height"].astype(str)
            + " × "
            + df.loc[mask, "Width"].astype(str)
            + " cm"
        )

desired_cols = [
    "Title", "File Name", "Creator", "Year", "Material", "Dimensions",
    "Location", "Collection", "Movements", "Depicts", "Wikipedia URL",
    "Link Count", "Painting ID", "Creator ID", "Movement IDs",
]
for col in desired_cols:
    if col not in df.columns:
        df[col] = ""
df = df[desired_cols]
df["Link Count"] = pd.to_numeric(df["Link Count"], errors="coerce")
df.sort_values("Link Count", ascending=False, inplace=True)

df.to_parquet(OUT_FILE, index=False)
logging.info("Saved final Parquet: %s", OUT_FILE)
print("Done – saved", OUT_FILE)
