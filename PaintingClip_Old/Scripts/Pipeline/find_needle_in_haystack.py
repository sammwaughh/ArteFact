#!/usr/bin/env python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import torch
import numpy as np
import pandas as pd
import unicodedata
from sentence_transformers import SentenceTransformer

# ---------------------------
# Configuration defaults
# ---------------------------
DEFAULT_INPUT_EXCEL   = "paintings_metadata.xlsx"
DEFAULT_OUTPUT_EXCEL  = "paintings_with_labels.xlsx"
PAINTERS_EXCEL        = "painters.xlsx"
CACHE_DIR             = "cache"
MODEL_NAME            = "paraphrase-MiniLM-L6-v2"
COMBINE_THRESHOLD     = 0.5

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def normalize_name(s: str) -> str:
    s = s.strip().casefold()
    for dash in ["–", "—", "-", "−", "\u2011"]:
        s = s.replace(dash, "-")
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def build_query(title, creator, year, depicts):
    if year.isdigit():
        article = "an" if year.startswith(("8", "11", "18")) else "a"
        q = f"{title} is {article} {year} painting"
    else:
        q = f"{title} is a painting"
    if creator:
        q += f" by {creator}"
    if depicts:
        q += f" depicting {depicts}"
    return q + "."


def select_label(sentences, embeddings, query_emb):
    sims = embeddings @ query_emb
    best_idx = int(torch.argmax(sims))
    best = sentences[best_idx]
    # try combining with prev
    if best_idx > 0 and best.lower().startswith(("it ", "this ", "these ")):
        prev = sentences[best_idx - 1]
        combo = prev.rstrip(".!?") + ". " + best
        if len(combo.split()) <= 50 and float(sims[best_idx - 1]) > 0.3:
            return combo
    # try next
    if best_idx < len(sentences) - 1:
        nxt = sentences[best_idx + 1]
        combo = best.rstrip(".!?") + ". " + nxt
        if len(combo.split()) <= 50 and float(sims[best_idx + 1]) > COMBINE_THRESHOLD:
            return combo
    return best


def main():
    parser = argparse.ArgumentParser(
        description="Generate ~50-token SBERT labels for paintings."
    )
    parser.add_argument("-i", "--input", default=DEFAULT_INPUT_EXCEL)
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_EXCEL)
    parser.add_argument("-s", "--start", type=int, default=0)
    parser.add_argument("-e", "--end", type=int, default=None)
    args = parser.parse_args()

    # 1) painter to query lookup
    painters_df = pd.read_excel(PAINTERS_EXCEL)
    painter_norm_to_raw_qs = {}
    painter_norm_to_norm_qs = {}
    for _, r in painters_df.iterrows():
        art = str(r["Artist"])
        qs_raw = str(r["Query String"]).strip()
        art_norm = normalize_name(art)
        qs_norm = normalize_name(qs_raw)
        painter_norm_to_raw_qs[art_norm] = qs_raw
        painter_norm_to_norm_qs[art_norm] = qs_norm

    # 2) scan cache dir
    file_map = {}
    norm_file_map = {}
    for fn in os.listdir(CACHE_DIR):
        if not fn.lower().endswith(".pt"):
            continue
        raw_key = fn.rsplit("_cache.pt", 1)[0].strip()
        file_map[raw_key] = fn
        norm_file_map[normalize_name(raw_key)] = raw_key

    # 3) load & slice metadata
    df_full = pd.read_excel(args.input)
    if args.end is not None:
        df = df_full.iloc[args.start : args.end + 1].copy()
    else:
        df = df_full.iloc[args.start :].copy()

    # 4) prepare SBERT
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    embeddings_cache = {}
    labels = []

    for _, row in df.iterrows():
        creator = str(row.get("Creator", "")).strip()

        # --- skip URIs or empty ---
        if not creator or creator.lower().startswith("http"):
            print(f"Warning: skipping URI/empty creator '{creator}'")
            labels.append("")
            continue

        norm_creator = normalize_name(creator)
        raw_cache_key = None
        if norm_creator in painter_norm_to_norm_qs:
            qs_norm = painter_norm_to_norm_qs[norm_creator]
            raw_cache_key = norm_file_map.get(qs_norm) or painter_norm_to_raw_qs[norm_creator]
        if raw_cache_key is None and norm_creator in norm_file_map:
            raw_cache_key = norm_creator
        if raw_cache_key is None and "brueghel" in norm_creator:
            alt = norm_creator.replace("brueghel", "bruegel")
            raw_cache_key = norm_file_map.get(alt)
        if raw_cache_key is None:
            hits = [v for k, v in norm_file_map.items() if k.startswith(norm_creator)]
            if len(hits) == 1:
                raw_cache_key = hits[0]
        if raw_cache_key is None:
            hits = [v for k, v in norm_file_map.items() if norm_creator in k]
            if len(hits) == 1:
                raw_cache_key = hits[0]
        if raw_cache_key is None:
            print(f"Warning: no cache match for '{creator}'. Skipping.")
            labels.append("")
            continue
        if raw_cache_key not in embeddings_cache:
            fn = file_map.get(raw_cache_key)
            if fn is None:
                print(f"Warning: cache file missing for key '{raw_cache_key}'. Skipping.")
                labels.append("")
                continue

            data = torch.load(os.path.join(CACHE_DIR, fn), map_location="cpu")
            sents = data["candidate_sentences"]
            embs  = data["candidate_embeddings"]
            if isinstance(embs, np.ndarray):
                embs = torch.tensor(embs, dtype=torch.float)
            embs = torch.nn.functional.normalize(embs, p=2, dim=1).to(DEVICE)
            embeddings_cache[raw_cache_key] = (sents, embs)

        sents, embs = embeddings_cache[raw_cache_key]

        # encode & select
        title, year, depicts = [str(row.get(f, "")).strip()
                                 for f in ("Title", "Year", "Depicts")]
        query = build_query(title, creator, year, depicts)
        q_emb = model.encode(query, convert_to_tensor=True)
        q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=0).to(DEVICE)
        labels.append(select_label(sents, embs, q_emb))

    # 5) merge back & save
    print("Saving merged output…")
    df_full["TextualLabel"] = ""
    df_full.loc[df.index, "TextualLabel"] = labels
    df_full.to_excel(args.output, index=False)
    span = f"{args.start}" + (f"–{args.end}" if args.end is not None else "")
    print(f"Processed rows {span}; wrote {len(labels)} labels into '{args.output}'.\n")


if __name__ == "__main__":
    main()
