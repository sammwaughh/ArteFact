#!/usr/bin/env python3
"""
markdown_file_to_english_sentences.py
-------------------------------------
Extract English sentences from a Markdown file produced by Marker and
store them in sentences.json.  For every sentence we store:

• SentenceID          – e.g.  W1549104703_s0007
• Work                – the parent work ID (W-number)
• English Original    – the sentence text
• CLIP Embedding      – placeholder (null)
• PaintingCLIP Embedding – placeholder (null)

After writing the sentences the script also updates works.json so the
corresponding Work entry lists all newly created SentenceIDs.

Usage
-----
$ python Pipeline/markdown_file_to_english_sentences.py \
      Marker_Output/W1549104703/W1549104703.md
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import nltk

# ── ensure required NLTK data are present ──────────────────────────────
for res in ("punkt", "punkt_tab"):
    try:
        # punkt ⇒ …/tokenizers/punkt/english.pickle
        # punkt_tab ⇒ …/tokenizers/punkt_tab/english/…
        lookup_path = f"tokenizers/{res}/english"
        nltk.data.find(lookup_path)
    except LookupError:
        nltk.download(res, quiet=True)


# ───────────────────────────── helpers ──────────────────────────────────────
def extract_sentences_from_markdown(file_path: str) -> List[str]:
    """
    Lightweight Markdown → sentence extractor.
    Copied from 10_cache_painting_clip.py (so that file can be removed).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove fenced code blocks and inline code
    content = re.sub(r"```[\s\S]*?```", "", content)
    content = re.sub(r"`[^`]*`", "", content)

    # Replace Markdown links with just the link text
    content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content)

    # Remove image links
    content = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", content)

    # Strip remaining markdown markup
    content = re.sub(r"[#>*_~\-]", "", content)

    sentences = nltk.sent_tokenize(content)
    return [s.strip() for s in sentences if len(s.split()) > 3]


# ───────────────────────────── I/O utils ────────────────────────────────────
ROOT = Path(__file__).resolve().parent  # Pipeline/
SENTENCES_FILE = ROOT / "sentences.json"
WORKS_FILE = ROOT / "works.json"


def _load_json(path: Path, default):
    try:
        data = json.loads(path.read_text())
        if isinstance(data, type(default)):
            return data
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return default


def _save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# ───────────────────────────── core routine ────────────────────────────────
def process_markdown(md_path: Path) -> None:
    if not md_path.exists():
        sys.exit(f"❌ Markdown not found: {md_path}")

    work_id = md_path.stem  # Marker writes <WorkID>/<WorkID>.md
    sentences = extract_sentences_from_markdown(str(md_path))
    if not sentences:
        print("No sentences extracted.")
        return

    sentences_db: Dict[str, Dict] = _load_json(SENTENCES_FILE, {})
    works_db: Dict[str, Dict] = _load_json(WORKS_FILE, {})

    new_ids: List[str] = []
    rewritten_ids: List[str] = []
    for idx, sentence in enumerate(sentences, start=1):
        sent_id = f"{work_id}_s{idx:04d}"
        entry = {
            "English Original": sentence,
            "Has CLIP Embedding": False,
            "Has PaintingCLIP Embedding": False,
        }
        if sent_id in sentences_db:
            rewritten_ids.append(sent_id)  # reset & overwrite
        else:
            new_ids.append(sent_id)
        sentences_db[sent_id] = entry  # (re)write unconditionally

    if not new_ids and not rewritten_ids:
        print("No sentences extracted.")
        return

    # update works.json
    work_entry = works_db.setdefault(
        work_id,
        {
            "Artist": "",
            "Link": "",
            "Number of Sentences": 0,
            "DOI": "",
            "ImageIDs": [],
            "TopicIDs": [],
        },
    )
    # Replace / set total sentence count
    work_entry["Number of Sentences"] = sum(
        sid.startswith(work_id + "_s") for sid in sentences_db
    )

    # write back
    _save_json(SENTENCES_FILE, sentences_db)
    _save_json(WORKS_FILE, works_db)

    print(
        f"✅ Added {len(new_ids)} new, rewrote {len(rewritten_ids)} "
        f"sentences for {work_id}"
    )
    print(f"↳ updated {SENTENCES_FILE.name} and {WORKS_FILE.name}")


# ────────────────────────────── entry point ────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(
            "Usage: python markdown_file_to_english_sentences.py "
            "<file.md | WorkID (e.g. W3110840203)>"
        )

    arg = sys.argv[1]
    if arg.lower().endswith(".md"):
        md_path = Path(arg).expanduser().resolve()
    else:
        work_id = arg.strip()
        md_path = (ROOT / "Marker_Output" / work_id / f"{work_id}.md").resolve()

    process_markdown(md_path)
