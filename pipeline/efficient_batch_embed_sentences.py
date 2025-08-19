#!/usr/bin/env python3
"""
efficient_batch_embed_sentences.py
---------------------------------
Efficient batch embedding generation for ArteFact-HF pipeline on Bede.
Generates CLIP and PaintingCLIP embeddings for all sentences in sentences.json
and saves to consolidated safetensors format for optimal performance.

This script runs after batch_markdown_file_to_english_sentences.py and
replaces the individual file approach with a much more efficient batch approach.

Usage:
    python pipeline/efficient_batch_embed_sentences.py
"""

from __future__ import annotations

import json
import sys
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from peft import PeftModel
from safetensors.torch import save_file
from tqdm import tqdm

# ───────────────────────────── configuration ──────────────────────────────
CODE_ROOT = Path(__file__).resolve().parent  # Pipeline/
RUN_ROOT = Path(os.getenv("RUN_ROOT", str(CODE_ROOT)))
SENTENCES_FILE = RUN_ROOT / "sentences.json"
EMBEDDINGS_DIR = RUN_ROOT / "Embeddings"
EMBEDDINGS_DIR.mkdir(exist_ok=True)

# Model configurations
DEFAULT_MODEL = "openai/clip-vit-base-patch32"
PAINTING_ADAPTER = "PaintingCLIP"
PAINTING_ADAPTER_DIR = RUN_ROOT / PAINTING_ADAPTER

# Performance settings
BATCH_SIZE = 64  # Adjust based on GPU memory
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

# ───────────────────────────── helpers ───────────────────────────────────
def load_sentences() -> Dict[str, Dict[str, Any]]:
    """Load sentences from sentences.json."""
    try:
        with open(SENTENCES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        sys.exit(f"❌ {SENTENCES_FILE} not found.")
    except Exception as exc:
        sys.exit(f"❌ Cannot load {SENTENCES_FILE}: {exc}")

def load_model_and_processor(model_name: str) -> Tuple[CLIPModel, CLIPProcessor]:
    """
    Load CLIP model with optional LoRA adapter.
    Reuses logic from embed_sentence_with_clip.py
    """
    candidate_dir = RUN_ROOT / model_name
    is_adapter = (
        candidate_dir.is_dir() and (candidate_dir / "adapter_config.json").exists()
    )

    if is_adapter:
        if PeftModel is None:
            sys.exit("❌ PEFT not installed – unable to load LoRA adapter")
        print(f"🔧 Loading LoRA adapter from {candidate_dir}")
        base = CLIPModel.from_pretrained(DEFAULT_MODEL).to(DEVICE)
        model = PeftModel.from_pretrained(base, str(candidate_dir)).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(DEFAULT_MODEL)
    else:
        print(f"🔧 Loading model: {model_name}")
        model = CLIPModel.from_pretrained(model_name).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(model_name)

    model.eval()
    return model, processor

def extract_sentences_data(sentences_db: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """Extract sentence IDs and texts from the database."""
    sentence_ids = []
    sentence_texts = []
    
    for sentence_id, entry in sentences_db.items():
        text = entry.get("English Original", "").strip()
        if isinstance(text, str) and text:  # Skip empty or malformed entries
            sentence_ids.append(sentence_id)
            sentence_texts.append(text)
    
    return sentence_ids, sentence_texts

def generate_embeddings_batch(
    texts: List[str], 
    processor: CLIPProcessor, 
    model: CLIPModel, 
    batch_size: int = 64
) -> torch.Tensor:
    """Generate embeddings for a batch of texts efficiently."""
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i + batch_size]
        
        # Process batch
        inputs = processor(
            text=batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=77
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            # Normalize embeddings
            text_embeddings = F.normalize(text_features, dim=-1)
            all_embeddings.append(text_embeddings.cpu())
    
    # Concatenate all batches
    return torch.cat(all_embeddings, dim=0)

def save_embeddings_consolidated(
    embeddings: torch.Tensor, 
    sentence_ids: List[str], 
    model_type: str
) -> None:
    """Save embeddings in consolidated safetensors format."""
    
    # Save safetensors file
    st_file = EMBEDDINGS_DIR / f"{model_type}_embeddings.safetensors"
    save_file({"embeddings": embeddings}, str(st_file))
    
    # Save sentence IDs
    ids_file = EMBEDDINGS_DIR / f"{model_type}_embeddings_sentence_ids.json"
    with open(ids_file, 'w', encoding='utf-8') as f:
        json.dump(sentence_ids, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved {len(sentence_ids)} {model_type} embeddings to {st_file}")
    print(f"✅ Embedding shape: {embeddings.shape}")
    print(f"✅ File size: {st_file.stat().st_size / (1024**3):.2f} GB")

def update_sentences_json_with_embeddings(
    sentences_db: Dict[str, Dict[str, Any]], 
    model_type: str
) -> None:
    """Update sentences.json to mark all sentences as having embeddings."""
    json_key = f"Has {model_type} Embedding"
    
    for entry in sentences_db.values():
        entry[json_key] = True
    
    # Save updated database
    with open(SENTENCES_FILE, 'w', encoding='utf-8') as f:
        json.dump(sentences_db, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Updated sentences.json with {model_type} embedding status")

def copy_to_backend_format() -> None:
    """Copy embeddings to backend-expected format."""
    backend_dir = CODE_ROOT.parent / "data" / "embeddings"
    backend_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy safetensors files
    for st_file in EMBEDDINGS_DIR.glob("*_embeddings.safetensors"):
        backend_file = backend_dir / st_file.name
        shutil.copy2(st_file, backend_file)
        print(f"📋 Copied {st_file.name} to {backend_file}")
    
    # Copy sentence ID files
    for ids_file in EMBEDDINGS_DIR.glob("*_embeddings_sentence_ids.json"):
        backend_file = backend_dir / ids_file.name
        shutil.copy2(ids_file, backend_file)
        print(f"📋 Copied {ids_file.name} to {backend_file}")
    
    print(f"✅ All embeddings copied to {backend_dir}")

def generate_model_embeddings(
    sentences_db: Dict[str, Dict[str, Any]], 
    model_name: str, 
    model_type: str
) -> None:
    """Generate embeddings for a specific model."""
    print(f"\n🔧 Generating {model_type.upper()} embeddings...")
    
    # Extract sentence data
    sentence_ids, sentence_texts = extract_sentences_data(sentences_db)
    print(f"📖 Processing {len(sentence_ids)} sentences")
    
    if not sentence_ids:
        print(f"⚠️  No valid sentences found for {model_type}")
        return
    
    # Load model
    model, processor = load_model_and_processor(model_name)
    
    # Generate embeddings
    embeddings = generate_embeddings_batch(sentence_texts, processor, model, BATCH_SIZE)
    
    # Save in consolidated format
    save_embeddings_consolidated(embeddings, sentence_ids, model_type)
    
    # Update sentences.json
    update_sentences_json_with_embeddings(sentences_db, model_type)
    
    # Clean up model to free memory
    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ───────────────────────────── main ──────────────────────────────────────
def main() -> None:
    """Main function to generate embeddings for both CLIP and PaintingCLIP."""
    print(f"�� Starting efficient batch embedding generation on {DEVICE}")
    print(f"�� Working directory: {CODE_ROOT}")
    print(f"📁 Run directory: {RUN_ROOT}")
    print(f"📁 Embeddings directory: {EMBEDDINGS_DIR}")
    
    # Load sentences database
    print("\n📖 Loading sentences database...")
    sentences_db = load_sentences()
    
    if not sentences_db:
        sys.exit("❌ No sentences found in sentences.json")
    
    print(f"📖 Loaded {len(sentences_db)} sentence entries")
    
    # Generate CLIP embeddings
    generate_model_embeddings(sentences_db, DEFAULT_MODEL, "CLIP")
    
    # Generate PaintingCLIP embeddings
    if PAINTING_ADAPTER_DIR.exists():
        generate_model_embeddings(sentences_db, PAINTING_ADAPTER, "PaintingCLIP")
    else:
        print(f"⚠️  PaintingCLIP adapter not found at {PAINTING_ADAPTER_DIR}")
        print("   Skipping PaintingCLIP embedding generation")
    
    # Copy to backend format
    print("\n�� Copying embeddings to backend format...")
    copy_to_backend_format()
    
    print("\n✅ All embeddings generated successfully!")
    print(f"📁 Embeddings saved to: {EMBEDDINGS_DIR}")
    print(f"📁 Backend copies: {CODE_ROOT.parent}/data/embeddings/")
    print("\n📋 Summary:")
    print("   - CLIP embeddings: CLIP_embeddings.safetensors")
    print("   - PaintingCLIP embeddings: PaintingCLIP_embeddings.safetensors")
    print("   - Sentence IDs: *_embeddings_sentence_ids.json")
    print("   - Updated sentences.json with embedding status")

if __name__ == "__main__":
    main()