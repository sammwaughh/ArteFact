#!/usr/bin/env python3

from __future__ import annotations
import pathlib
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from peft import PeftModel
from tqdm import tqdm
from openpyxl import load_workbook
from PIL import Image

# ── new imports ─────────────────────────────────────────
import re
try:                                 
    from openpyxl.utils.cell import ILLEGAL_CHARACTERS_RE
except ImportError:                  
    ILLEGAL_CHARACTERS_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")
# ────────────────────────────────────────────────────────


# ─────────────── paths ───────────────
ROOT          = pathlib.Path(__file__).resolve().parent.parent
CACHE_DIR     = ROOT / "Dataset" / "cache_clip_embeddings"
PAINTERS_XLS  = ROOT / "Dataset" / "painters.xlsx"
LABELS_XLS    = ROOT / "Dataset" / "paintings_with_labels.xlsx"
FINETUNE_XLS  = ROOT / "FineTune" / "fine_tune_dataset.xlsx"
IMAGES_DIR    = ROOT / "Dataset" / "Images"

LORA_DIR      = ROOT / "Results" / "clip_finetuned_lora_best"

OUT_VANILLA   = ROOT / "Results" / "vanilla_clip.xlsx"
OUT_MINT      = ROOT / "Results" / "mint_clip.xlsx"

TOP_K = 10
# ──────────────────────────────────────


def load_text_cache(path: pathlib.Path) -> Tuple[torch.Tensor, List[str]]:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    return blob["candidate_embeddings"], blob["candidate_sentences"]


@torch.no_grad()
def topk_for_image(
    model: CLIPModel,
    proc: CLIPProcessor,
    img_path: pathlib.Path,
    txt_embs: torch.Tensor,
    k: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    img = Image.open(img_path).convert("RGB")
    inputs = proc(images=img, return_tensors="pt").to(device)
    img_vec = F.normalize(model.get_image_features(**inputs).squeeze(0), dim=-1)
    txt_vec = F.normalize(txt_embs.to(device), dim=-1)
    sims    = (txt_vec @ img_vec).cpu()
    top     = torch.topk(sims, k=k, largest=True, sorted=True)
    return top.indices.numpy(), top.values.numpy()


# ─── 1. metadata & caches ───
print("Loading painter metadata …")
painters_df = pd.read_excel(PAINTERS_XLS)
stem_to_qs: Dict[str, str] = {
    artist: qs.lower()
    for artist, qs in zip(painters_df["Artist"], painters_df["Query String"])
}

print("Loading cached sentence embeddings …")
caches: Dict[str, Tuple[torch.Tensor, List[str]]] = {}
for pt in CACHE_DIR.glob("*_clip_embedding_cache.pt"):
    artist = pt.stem.replace("_clip_embedding_cache", "")
    qs = stem_to_qs.get(artist)
    if qs:
        caches[qs] = load_text_cache(pt)
if not caches:
    raise RuntimeError("No cache files matched any query string.")

labels_df = pd.read_excel(LABELS_XLS)
ft_df     = pd.read_excel(FINETUNE_XLS)
title_col = "Title" if "Title" in ft_df.columns else "Name"

merged = ft_df.merge(
    labels_df[["File Name", "Creator"]],
    on="File Name", how="left", validate="many_to_one"
).rename(columns={"File Name": "File_Name"})

def match_qs(creator):
    if not isinstance(creator, str):
        return None
    low = creator.lower()
    for qs in caches:
        if qs in low:
            return qs
    return None

merged["cache_key"] = merged["Creator"].apply(match_qs)
paintings = merged.dropna(subset=["cache_key"])
if paintings.empty:
    raise RuntimeError("No paintings match the cached artists.")


# ─── 2. models ───
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
MODEL_ID  = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(MODEL_ID, use_fast=True)

print("Loading Vanilla-CLIP …")
vanilla = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()

print("Loading Mint-CLIP (LoRA) …")
base = CLIPModel.from_pretrained(MODEL_ID)
mint = PeftModel.from_pretrained(base, LORA_DIR).to(device).eval()


# ─── 3. harvesting ───
def clean(value):
    """Strip characters Excel refuses."""
    return ILLEGAL_CHARACTERS_RE.sub("", value) if isinstance(value, str) else value

def harvest(model: CLIPModel, out_path: pathlib.Path) -> None:
    rows: List[dict] = []
    processed = skipped = 0

    bar = tqdm(
        paintings.itertuples(index=False),
        total=len(paintings),
        desc=f"Top-{TOP_K} → {out_path.stem}",
    )

    for tup in bar:
        title, fn, creator, qs = (
            getattr(tup, title_col),
            tup.File_Name,
            tup.Creator,
            tup.cache_key,
        )
        img_p = IMAGES_DIR / fn
        if not img_p.exists():
            skipped += 1
            continue

        txt_embs, sentences = caches[qs]
        idx, sims = topk_for_image(model, processor, img_p, txt_embs, TOP_K, device)

        for rank, (i, score) in enumerate(zip(idx, sims), start=1):
            rows.append({
                "File_Name":    fn,
                "Title":        clean(title),
                "Creator":      clean(creator),
                "SentenceRank": rank,
                "Sentence":     clean(sentences[i]),
                "Score":        float(score),
                "Label":        "",
            })
        processed += 1

    df = pd.DataFrame(rows)
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].map(clean)

    df.to_excel(out_path, index=False, engine="openpyxl")

    wb = load_workbook(out_path)
    ws = wb.active
    for col in ws.columns:
        width = max(len(str(c.value)) for c in col) + 2
        ws.column_dimensions[col[0].column_letter].width = width
    wb.save(out_path)

    print(f"{out_path.name}: processed {processed} paintings, skipped {skipped} (image missing).")


# ─── 4. run ───
if __name__ == "__main__":
    harvest(vanilla, OUT_VANILLA)
    harvest(mint, OUT_MINT)
