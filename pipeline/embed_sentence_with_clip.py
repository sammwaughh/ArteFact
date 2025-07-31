#!/usr/bin/env python3
"""
embed_sentence_with_clip.py
----------------------
Create a CLIP embedding for one sentence stored in sentences.json.

Usage
-----
$ python Pipeline/embed_sentence_clip.py W3110840203_s0001
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import CLIPModel, CLIPProcessor

# Optional – only needed when the user chooses a LoRA adapter such as PaintingCLIP
try:
    from peft import PeftModel
except ImportError:  # PEFT not installed → LoRA adapters will not work
    PeftModel = None

# ─────────────────────────── configuration ──────────────────────────────
ROOT = Path(__file__).resolve().parent  # Pipeline/
SENTENCES_FILE = ROOT / "sentences.json"
# Default directory for vanilla CLIP embeddings
CLIP_EMB_DIR = ROOT / "CLIP_Embeddings"
CLIP_EMB_DIR.mkdir(exist_ok=True)

DEFAULT_MODEL = "openai/clip-vit-base-patch32"  # fallback when no --model given
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)


# ───────────────────────────── helpers ───────────────────────────────────
def load_sentences() -> dict:
    try:
        return json.loads(SENTENCES_FILE.read_text())
    except FileNotFoundError:
        sys.exit(f"❌ {SENTENCES_FILE} not found.")
    except Exception as exc:
        sys.exit(f"❌ Cannot load {SENTENCES_FILE}: {exc}")


def save_sentences(db: dict) -> None:
    SENTENCES_FILE.write_text(json.dumps(db, indent=2, ensure_ascii=False))


def embed_sentence(text: str, model, processor) -> torch.Tensor:
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    ).to(DEVICE)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    return features / features.norm(dim=-1, keepdim=True)


# ───────────────────────────── main ──────────────────────────────────────
def main(sent_id: str, model_name: str) -> None:
    db = load_sentences()
    entry = db.get(sent_id)
    if entry is None:
        sys.exit(f"❌ SentenceID {sent_id} not found in sentences.json")

    text = entry.get("English Original", "").strip()
    if not text:
        sys.exit(f"❌ No text for {sent_id}")

    print(f"Embedding sentence ({len(text)} chars) with model {model_name} …")

    model, processor = load_model_and_processor(model_name)

    emb = embed_sentence(text, model, processor).cpu()

    # ── decide output folder / json‐key depending on the model ──────────
    if model_name == DEFAULT_MODEL:
        out_dir = CLIP_EMB_DIR
        json_key = "Has CLIP Embedding"
        file_name = f"{sent_id}_clip.pt"
    else:
        adapter_name = Path(model_name).name  # e.g. PaintingCLIP / MyLora
        out_dir = ROOT / f"{adapter_name}_Embeddings"
        out_dir.mkdir(exist_ok=True)
        json_key = f"Has {adapter_name} Embedding"
        suffix = adapter_name.lower().replace("clip", "_clip")
        file_name = f"{sent_id}_{suffix}.pt"

    emb_path = out_dir / file_name
    torch.save(emb, emb_path)
    print(f"✅ Saved embedding → {emb_path.relative_to(ROOT)}")

    # update JSON
    entry[json_key] = True
    save_sentences(db)
    print("✅ sentences.json updated")


def load_model_and_processor(model_name: str):
    """
    1. If *model_name* resolves to a directory inside Pipeline/ that contains
       `adapter_config.json`, treat it as a LoRA adapter and load it with PEFT.
    2. Otherwise assume *model_name* is a full model checkpoint on HF Hub
       or a local directory and load it directly via Transformers.
    """
    candidate_dir = ROOT / model_name
    is_adapter = (
        candidate_dir.is_dir() and (candidate_dir / "adapter_config.json").exists()
    )

    if is_adapter:
        if PeftModel is None:
            sys.exit("❌ PEFT not installed – unable to load LoRA adapter")
        base = CLIPModel.from_pretrained(DEFAULT_MODEL).to(DEVICE)
        model = PeftModel.from_pretrained(base, str(candidate_dir)).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(DEFAULT_MODEL)
    else:
        model = CLIPModel.from_pretrained(model_name).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(model_name)

    model.eval()
    return model, processor


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Embed one sentence with CLIP / LoRA")
    ap.add_argument("sentence_id", help="SentenceID, e.g. W3110840203_s0001")
    ap.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Model name or directory. "
            "Give a local folder with adapter_config.json to use a LoRA adapter "
            "(e.g. PaintingCLIP). Default: openai/clip-vit-base-patch32"
        ),
    )
    args = ap.parse_args()

    main(args.sentence_id.strip(), args.model)
