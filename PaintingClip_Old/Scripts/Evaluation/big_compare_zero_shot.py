#!/usr/bin/env python3
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from peft import PeftModel
from tqdm import tqdm
from openpyxl import load_workbook
from PIL import Image

# ──────────── PATH SETUP ────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

CACHE_DIR = PROJECT_ROOT / "Dataset" / "cache_clip_embeddings"
PAINTERS_PATH = PROJECT_ROOT / "Dataset" / "painters.xlsx"
LABELS_PATH = PROJECT_ROOT / "Dataset" / "paintings_with_labels.xlsx"
FINETUNE_PATH = PROJECT_ROOT / "FineTune" / "fine_tune_dataset.xlsx"
IMAGES_DIR = PROJECT_ROOT / "Dataset" / "Images"

LORA_DIR = SCRIPT_DIR / "clip_finetuned_lora_best"
OUTPUT_PATH = SCRIPT_DIR / "zero_shot_results.xlsx"
# ─────────────────────────────────────


def load_text_cache(cache_path: Path):
    data = torch.load(cache_path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise ValueError(f"{cache_path!r} did not load as a dict")
    for key in ("candidate_embeddings", "candidate_sentences"):
        if key not in data:
            raise ValueError(f"{cache_path!r} missing key '{key}'")
    return data["candidate_embeddings"], data["candidate_sentences"]


def evaluate_zero_shot(model, processor, img_path, text_embs, sentences, device):
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        img_feats = model.get_image_features(**inputs).squeeze(0)
    img_feats = F.normalize(img_feats, dim=-1)
    txt_feats = F.normalize(text_embs.to(device), dim=-1)
    sims = (txt_feats @ img_feats).cpu()
    best = sims.argmax().item()
    return sentences[best], sims[best].item()


def main():
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load painters → query‐string map
    painters_df = pd.read_excel(PAINTERS_PATH)
    stem_to_qs = {
        artist: qs.lower()
        for artist, qs in zip(painters_df["Artist"], painters_df["Query String"])
    }

    # Load all caches
    caches = {}
    for p in sorted(CACHE_DIR.glob("*_clip_embedding_cache.pt")):
        raw = p.stem.removesuffix("_clip_embedding_cache")
        qs = stem_to_qs.get(raw)
        if qs:
            caches[qs] = load_text_cache(p)
    if not caches:
        raise RuntimeError("No cache files found")

    # Read labels & fine‐tune sheets
    labels_df = pd.read_excel(LABELS_PATH)
    ft_df = pd.read_excel(FINETUNE_PATH)

    title_col = "Title" if "Title" in ft_df.columns else "Name"
    merged = ft_df.merge(
        labels_df[["File Name", "Creator"]],
        on="File Name",
        how="left",
        validate="many_to_one",
    )

    def find_qs(creator):
        if not isinstance(creator, str):
            return None
        c = creator.lower()
        for qs in caches:
            if qs in c:
                return qs
        return None

    merged["cache_key"] = merged["Creator"].apply(find_qs)
    subset = merged.dropna(subset=["cache_key"])
    if subset.empty:
        raise RuntimeError("No fine-tune entries match any cached artist")

    # Prepare CLIP + LoRA
    MODEL_ID = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(MODEL_ID, use_fast=True)
    vanilla = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
    base_clip = CLIPModel.from_pretrained(MODEL_ID)
    mint = PeftModel.from_pretrained(base_clip, LORA_DIR).to(device).eval()

    total = len(subset)
    processed = 0
    ignored = 0
    records = []

    # Zero-shot eval with running counters
    pbar = tqdm(
        subset.itertuples(index=False, name=None), total=total, desc="Zero-shot eval"
    )
    for title, fn, txt_lbl, creator, qs in pbar:
        img_p = IMAGES_DIR / fn
        if not img_p.exists():
            ignored += 1
        else:
            text_embs, sents = caches[qs]
            v_sent, v_score = evaluate_zero_shot(
                vanilla, processor, img_p, text_embs, sents, device
            )
            m_sent, m_score = evaluate_zero_shot(
                mint, processor, img_p, text_embs, sents, device
            )

            records.append(
                {
                    title_col: title,
                    "File Name": fn,
                    "Creator": creator,
                    "Vanilla Sentence": v_sent,
                    "Vanilla Score": v_score,
                    "Mint Sentence": m_sent,
                    "Mint Score": m_score,
                }
            )
            processed += 1

        left = total - processed - ignored
        pbar.set_postfix({"processed": processed, "ignored": ignored, "left": left})

    # Save & auto‐format
    out_df = pd.DataFrame(records)
    out_df.to_excel(OUTPUT_PATH, index=False)

    wb = load_workbook(OUTPUT_PATH)
    ws = wb.active
    for col in ws.columns:
        width = max(len(str(cell.value)) for cell in col) + 2
        ws.column_dimensions[col[0].column_letter].width = width
    wb.save(OUTPUT_PATH)

    print(
        f"\n Done! Processed {processed}, ignored {ignored}, results in {OUTPUT_PATH.resolve()}"
    )


if __name__ == "__main__":
    main()
