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
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import gc

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

# Performance settings - CORRECTED for GH200
BATCH_SIZE = 1024  # Single consistent batch size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True  # Enable Automatic Mixed Precision
NUM_WORKERS = min(36, mp.cpu_count())  # Reduced to 36 for safety

# Memory optimization
TORCH_CUDA_EMPTY_CACHE_FREQ = 10  # Clear cache every 10 batches

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

def extract_sentences_data_parallel(sentences_db: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """Extract sentence IDs and texts using parallel processing with error handling."""
    try:
        sentence_ids = []
        sentence_texts = []
        
        # Use parallel processing for large datasets
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Split work into chunks
            items = list(sentences_db.items())
            chunk_size = max(1, len(items) // NUM_WORKERS)
            
            def process_chunk(chunk):
                ids, texts = [], []
                for sentence_id, entry in chunk:
                    text = entry.get("English Original", "").strip()
                    if isinstance(text, str) and text:
                        ids.append(sentence_id)
                        texts.append(text)
                return ids, texts
            
            # Process chunks in parallel
            futures = []
            for i in range(0, len(items), chunk_size):
                chunk = items[i:i + chunk_size]
                futures.append(executor.submit(process_chunk, chunk))
            
            # Collect results
            for future in futures:
                chunk_ids, chunk_texts = future.result()
                sentence_ids.extend(chunk_ids)
                sentence_texts.extend(chunk_texts)
        
        return sentence_ids, sentence_texts
    
    except Exception as e:
        print(f"⚠️  Parallel processing failed: {e}")
        print("🔄 Falling back to sequential processing...")
        return extract_sentences_data(sentences_db)

def generate_embeddings_batch(
    texts: List[str], 
    processor: CLIPProcessor, 
    model: CLIPModel, 
    batch_size: int = 256  # Updated default
) -> torch.Tensor:
    """Generate embeddings for a batch of texts efficiently with AMP."""
    all_embeddings = []
    
    # Enable mixed precision for faster computation
    if USE_AMP and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
    
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
            if USE_AMP and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    text_features = model.get_text_features(**inputs)
                    text_embeddings = F.normalize(text_features, dim=-1)
            else:
                text_features = model.get_text_features(**inputs)
                text_embeddings = F.normalize(text_features, dim=-1)
            
            all_embeddings.append(text_embeddings.cpu())
    
    # Concatenate all batches
    return torch.cat(all_embeddings, dim=0)

def generate_embeddings_batch_optimized(
    texts: List[str], 
    processor: CLIPProcessor, 
    model: CLIPModel, 
    batch_size: int = 1024
) -> torch.Tensor:
    """Generate embeddings with maximum GPU utilization."""
    all_embeddings = []
    
    # Pre-allocate tensors for better memory management
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    print(f"🎯 Processing {len(texts)} sentences in {total_batches} batches of {batch_size}")
    print(f"🎮 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
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
        inputs = {k: v.to(DEVICE, non_blocking=True) for k, v in inputs.items()}
        
        with torch.no_grad():
            if USE_AMP and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    text_features = model.get_text_features(**inputs)
                    text_embeddings = F.normalize(text_features, dim=-1)
            else:
                text_features = model.get_text_features(**inputs)
                text_embeddings = F.normalize(text_features, dim=-1)
            
            # Move to CPU and clear GPU memory immediately
            cpu_embeddings = text_embeddings.cpu()
            all_embeddings.append(cpu_embeddings)
            
            # Clear GPU memory periodically
            if (i // batch_size + 1) % TORCH_CUDA_EMPTY_CACHE_FREQ == 0:
                del text_features, text_embeddings
                torch.cuda.empty_cache()
                gc.collect()
    
    # Concatenate all batches
    print("🔗 Concatenating embeddings...")
    result = torch.cat(all_embeddings, dim=0)
    
    # Final cleanup
    del all_embeddings
    torch.cuda.empty_cache()
    gc.collect()
    
    return result

def save_embeddings_consolidated(
    embeddings: torch.Tensor, 
    sentence_ids: List[str], 
    model_type: str
) -> None:
    """Save embeddings in consolidated safetensors format."""
    
    # Convert model_type to lowercase for backend compatibility
    model_type_lower = model_type.lower()
    
    # Save safetensors file with lowercase naming
    st_file = EMBEDDINGS_DIR / f"{model_type_lower}_embeddings.safetensors"
    save_file({"embeddings": embeddings}, str(st_file))
    
    # Save sentence IDs with lowercase naming
    ids_file = EMBEDDINGS_DIR / f"{model_type_lower}_embeddings_sentence_ids.json"
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
    """Copy embeddings to backend-expected format with correct naming."""
    
    # Backend embeddings directory
    backend_emb_dir = CODE_ROOT.parent / "data" / "embeddings"
    backend_emb_dir.mkdir(parents=True, exist_ok=True)
    
    # Backend json_info directory
    backend_json_dir = CODE_ROOT.parent / "data" / "json_info"
    backend_json_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy embeddings (already lowercase from save function)
    for st_file in EMBEDDINGS_DIR.glob("*_embeddings.safetensors"):
        backend_file = backend_emb_dir / st_file.name
        shutil.copy2(st_file, backend_file)
        print(f"📋 Copied {st_file.name} to {backend_file}")
    
    # Copy sentence ID files (already lowercase)
    for ids_file in EMBEDDINGS_DIR.glob("*_embeddings_sentence_ids.json"):
        backend_file = backend_emb_dir / ids_file.name
        shutil.copy2(ids_file, backend_file)
        print(f"📋 Copied {ids_file.name} to {backend_file}")
    
    # Copy updated metadata files
    sentences_src = RUN_ROOT / "sentences.json"
    works_src = RUN_ROOT / "works.json"
    
    if sentences_src.exists():
        shutil.copy2(sentences_src, backend_json_dir / "sentences.json")
        print(f"📋 Copied sentences.json to {backend_json_dir}")
    
    if works_src.exists():
        shutil.copy2(works_src, backend_json_dir / "works.json")
        print(f"📋 Copied works.json to {backend_json_dir}")
    
    print(f"✅ All files copied to backend format:")
    print(f"   Embeddings: {backend_emb_dir}")
    print(f"   Metadata: {backend_json_dir}")

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

def generate_model_embeddings_optimized(
    sentences_db: Dict[str, Dict[str, Any]], 
    model_name: str, 
    model_type: str
) -> None:
    """Generate embeddings with maximum optimization and error handling."""
    print(f"\n🔧 Generating {model_type.upper()} embeddings (OPTIMIZED)...")
    
    try:
        # Use parallel extraction for large datasets
        if len(sentences_db) > 1000000:  # 1M+ sentences
            print("📖 Using parallel processing for large dataset...")
            sentence_ids, sentence_texts = extract_sentences_data_parallel(sentences_db)
        else:
            sentence_ids, sentence_texts = extract_sentences_data(sentences_db)
        
        print(f"📖 Processing {len(sentence_ids)} sentences")
        
        if not sentence_ids:
            print(f"⚠️  No valid sentences found for {model_type}")
            return
        
        # Load model
        model, processor = load_model_and_processor(model_name)
        
        # Generate embeddings with maximum optimization
        embeddings = generate_embeddings_batch_optimized(sentence_texts, processor, model, BATCH_SIZE)
        
        # Save in consolidated format
        save_embeddings_consolidated(embeddings, sentence_ids, model_type)
        
        # Update sentences.json
        update_sentences_json_with_embeddings(sentences_db, model_type)
        
        # Clean up model to free memory
        del model, processor, embeddings
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"❌ Error generating {model_type} embeddings: {e}")
        print("🔄 Attempting to continue with next model...")
        return

# ───────────────────────────── main ──────────────────────────────────────
def main() -> None:
    """Main function with maximum optimization."""
    print(f"🚀 Starting MAXIMUM OPTIMIZED embedding generation on {DEVICE}")
    print(f"📁 Working directory: {CODE_ROOT}")
    print(f"📁 Run directory: {RUN_ROOT}")
    print(f"📁 Embeddings directory: {EMBEDDINGS_DIR}")
    
    # GPU info
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🎮 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"🎮 CUDA Version: {torch.version.cuda}")
        print(f"⚡ Mixed Precision: {'Enabled' if USE_AMP else 'Disabled'}")
    
    # CPU info
    print(f"🖥️  CPU Cores: {mp.cpu_count()}")
    print(f"🖥️  Batch Size: {BATCH_SIZE}")
    print(f"🖥️  Workers: {NUM_WORKERS}")
    
    # Load sentences database
    print("\n📖 Loading sentences database...")
    sentences_db = load_sentences()
    
    if not sentences_db:
        sys.exit("❌ No sentences found in sentences.json")
    
    print(f"📖 Loaded {len(sentences_db)} sentence entries")
    
    # Generate CLIP embeddings
    generate_model_embeddings_optimized(sentences_db, DEFAULT_MODEL, "clip")
    
    # Generate PaintingCLIP embeddings
    if PAINTING_ADAPTER_DIR.exists():
        generate_model_embeddings_optimized(sentences_db, PAINTING_ADAPTER, "paintingclip")
    else:
        print(f"⚠️  PaintingCLIP adapter not found at {PAINTING_ADAPTER_DIR}")
        print("   Skipping PaintingCLIP embedding generation")
    
    # Copy to backend format
    print("\n🔄 Copying embeddings to backend format...")
    copy_to_backend_format()
    
    print("\n✅ All embeddings generated successfully!")
    print(f"📁 Embeddings saved to: {EMBEDDINGS_DIR}")
    print(f"📁 Backend copies: {CODE_ROOT.parent}/data/embeddings/")
    print(f"📁 Backend metadata: {CODE_ROOT.parent}/data/json_info/")

if __name__ == "__main__":
    main()