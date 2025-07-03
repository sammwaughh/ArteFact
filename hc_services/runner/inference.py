"""
Wraps PaintingCLIP + LoRA (stubbed for now).

`run_inference` MUST stay pure: take a local image path, return
JSON‑serialisable Python objects.  That makes it easy to unit‑test.
"""

from pathlib import Path
from typing import List, Dict, Any  # add Any
from functools import lru_cache

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from peft import PeftModel


# ─── paths ──────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]          # artefact-context/
LORA_DIR  = ROOT / "PaintingClip" / "clip_finetuned_lora_best"
EMB_PATH  = ROOT / "Sentence Embeddings" / "Rembrandt van Rijn_clip_embedding_cache.pt"
MODEL_ID  = "openai/clip-vit-base-patch32"
TOP_K     = 5                                            # results returned
# ────────────────────────────────────────────────────────


def _load_text_cache(pt: Path):
    blob = torch.load(pt, map_location="cpu", weights_only=True)
    return blob["candidate_embeddings"], blob["candidate_sentences"]


@lru_cache(maxsize=1)
def _lazy_init():
    """
    Load processor, CLIP + LoRA weights, and Rembrandt sentence embeddings
    exactly once (cached across calls).  All heavy I/O stays outside
    run_inference() to keep it fast and pure.
    """
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    proc  = CLIPProcessor.from_pretrained(MODEL_ID, use_fast=True)
    base  = CLIPModel.from_pretrained(MODEL_ID)
    model = PeftModel.from_pretrained(base, LORA_DIR).to(device).eval()

    txt_embs, sentences = _load_text_cache(EMB_PATH)
    return proc, model, txt_embs, sentences, device


# ----------------------------------------------------------------------
def run_inference(image_path: str) -> List[Dict[str, Any]]:
    """
    Given a local image, return TOP_K Rembrandt sentences + similarity scores.
    """
    proc, model, txt_embs, sentences, device = _lazy_init()

    img = Image.open(image_path).convert("RGB")
    inputs = proc(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        img_vec = F.normalize(model.get_image_features(**inputs).squeeze(0), dim=-1)

    txt_vec = F.normalize(txt_embs.to(device), dim=-1)
    sims    = (txt_vec @ img_vec).cpu()                     # (N,)
    top     = torch.topk(sims, k=min(TOP_K, sims.numel()))

    results: List[Dict[str, Any]] = []
    for rank, (idx, score) in enumerate(zip(top.indices.tolist(),
                                            top.values.tolist()), start=1):
        results.append({
            "label":    sentences[idx],
            "score":    float(score),
            "evidence": {"rank": rank},                     # placeholder
        })
    return results


# Helper(s) for later ---------------------------------------------------
def _ensure_asset(s3_key: str, dest: Path) -> Path:
    """
    Downloads `s3_key` from MODEL_BUCKET to `dest` if not already present.
    To be filled in once we wire S3 + EFS.
    """
    # Placeholder – keeps import errors away during phase A dev.
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.write_text("TODO – downloaded model asset")
    return dest
