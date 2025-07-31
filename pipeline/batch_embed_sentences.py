#!/usr/bin/env python3


## NEEDS TO BE EDITED FOR HPC

"""
batch_embed_sentences.py
------------------------
Generate missing CLIP and PaintingCLIP embeddings for **every** sentence
listed in sentences.json.

The script re-uses helpers from embed_sentence_with_clip.py so we avoid
duplicate code and model downloads.

CLI
$ python Pipeline/batch_embed_sentences.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from tqdm import tqdm

# ── import helpers from the single-sentence script ───────────────────────
from embed_sentence_with_clip import (
    CLIP_EMB_DIR,
    DEFAULT_MODEL,
    ROOT,
    embed_sentence,
    load_model_and_processor,
    load_sentences,
    save_sentences,
)

# LoRA adapter folder (must exist inside Pipeline/)
PAINTING_ADAPTER = "PaintingCLIP"
PAINTING_EMB_DIR = ROOT / f"{PAINTING_ADAPTER}_Embeddings"
PAINTING_EMB_DIR.mkdir(exist_ok=True)

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)


def ensure_embedding(
    entry: dict,
    sent_id: str,
    model_name: str,
    out_dir: Path,
    json_key: str,
    file_suffix: str,
    model_cache: dict,
) -> None:
    """
    Compute & save the embedding if it is missing, then update *entry*.
    *model_cache* keeps already-loaded model/processor pairs.
    """
    # Skip when JSON already points to an existing file
    if entry.get(json_key) is True:
        fp = out_dir / f"{sent_id}_{file_suffix}.pt"
        if fp.exists():
            return

    # lazy-load model once
    if model_name not in model_cache:
        model_cache[model_name] = load_model_and_processor(model_name)
    model, processor = model_cache[model_name]

    emb = embed_sentence(entry["English Original"], model, processor).cpu()
    file_name = f"{sent_id}_{file_suffix}.pt"
    torch.save(emb, out_dir / file_name)

    entry[json_key] = True


def main() -> None:
    db = load_sentences()
    if not db:
        sys.exit("No sentences in sentences.json")

    model_cache: dict[str, tuple] = {}

    for sent_id, entry in tqdm(db.items(), desc="Embedding sentences"):
        text = entry.get("English Original")
        if not isinstance(text, str) or not text.strip():
            continue  # malformed

        # vanilla CLIP
        ensure_embedding(
            entry,
            sent_id,
            DEFAULT_MODEL,
            CLIP_EMB_DIR,
            "Has CLIP Embedding",
            "clip",
            model_cache,
        )

        # PaintingCLIP
        ensure_embedding(
            entry,
            sent_id,
            PAINTING_ADAPTER,
            PAINTING_EMB_DIR,
            "Has PaintingCLIP Embedding",
            "painting_clip",
            model_cache,
        )

    save_sentences(db)
    print("✅ All embeddings generated & sentences.json updated")


if __name__ == "__main__":
    main()
