"""
PaintingCLIP inference pipeline for art-historical text retrieval.

This module provides a pure functional interface for comparing artwork images
against a corpus of pre-computed sentence embeddings using CLIP models with
optional LoRA fine-tuning.

The pipeline:
1. Loads an image and computes its embedding using CLIP/PaintingCLIP
2. Compares against pre-computed sentence embeddings via cosine similarity
3. Returns the top-K most similar sentences with their metadata
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Literal, Optional
from functools import lru_cache
import time
import base64
import io

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from peft import PeftModel
# on-demand Grad-ECLIP & region-aware ranking
from .heatmap import generate_heatmap


# ─── Configuration ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]  # artefact-context/

# Model selection - change this to switch between models
MODEL_TYPE: Literal["clip", "paintingclip"] = "paintingclip"

# Model paths and settings
MODEL_CONFIG = {
    "clip": {
        "model_id": "openai/clip-vit-base-patch32",
        "embeddings_dir": ROOT / "CLIP_Embeddings",
        "use_lora": False,
        "lora_dir": None,
    },
    "paintingclip": {
        "model_id": "openai/clip-vit-base-patch32",
        "embeddings_dir": ROOT / "PaintingCLIP_Embeddings",
        "use_lora": True,
        "lora_dir": ROOT / "PaintingCLIP",
    }
}

# Data paths
SENTENCES_JSON = ROOT / "hc_services" / "runner" / "data" / "sentences.json"

# Inference settings
TOP_K = 10  # Number of results to return
# ─────────────────────────────────────────────────────────────────────────────

def _load_embeddings(embeddings_dir: Path) -> Tuple[torch.Tensor, List[str]]:
    """
    Load pre-computed sentence embeddings from individual .pt files.
    
    Each embedding file follows the naming convention:
    - CLIP: {sentence_id}_clip.pt (e.g., W1982215463_s0001_clip.pt)
    - PaintingCLIP: {sentence_id}_painting_clip.pt (e.g., W1982215463_s0001_painting_clip.pt)
    
    Args:
        embeddings_dir: Directory containing individual embedding files
        
    Returns:
        embeddings: Stacked tensor of shape (N, embedding_dim)
        sentence_ids: List of sentence IDs corresponding to each embedding
        
    Raises:
        ValueError: If no embedding files are found in the directory
    """
    embeddings = []
    sentence_ids = []
    
    # Glob all .pt files and sort for consistent ordering
    pt_files = sorted(embeddings_dir.glob("*.pt"))
    
    if not pt_files:
        raise ValueError(
            f"No embedding files (*.pt) found in {embeddings_dir}. "
            f"Please ensure embeddings are generated and stored correctly."
        )
    
    for pt_file in pt_files:
        # Extract sentence ID by removing the appropriate suffix based on model type
        stem = pt_file.stem
        
        # Remove the suffix based on which embeddings we're loading
        if "_painting_clip" in stem:
            # PaintingCLIP embeddings: remove "_painting_clip"
            sentence_id = stem.replace("_painting_clip", "")
        elif "_clip" in stem:
            # Regular CLIP embeddings: remove "_clip"
            sentence_id = stem.replace("_clip", "")
        else:
            # Fallback: use the stem as-is
            sentence_id = stem
        
        # Load the embedding tensor
        embedding = torch.load(pt_file, map_location="cpu", weights_only=True)
        
        # Handle various storage formats (dict vs direct tensor)
        if isinstance(embedding, dict):
            # Try common dictionary keys
            for key in ["embedding", "embeddings", "features"]:
                if key in embedding:
                    embedding = embedding[key]
                    break
        
        # Ensure 1D tensor shape
        if embedding.ndim > 1:
            embedding = embedding.squeeze()
        
        # Validate embedding dimension
        if embedding.ndim != 1:
            raise ValueError(
                f"Invalid embedding shape {embedding.shape} in {pt_file}. "
                f"Expected 1D tensor."
            )
        
        embeddings.append(embedding)
        sentence_ids.append(sentence_id)
    
    # Stack all embeddings into a single tensor
    embeddings_tensor = torch.stack(embeddings, dim=0)
    
    return embeddings_tensor, sentence_ids


def _load_sentences_metadata(sentences_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load sentence metadata from sentences.json.
    
    Args:
        sentences_path: Path to sentences.json file
        
    Returns:
        Dictionary mapping sentence IDs to their metadata
    """
    with open(sentences_path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _initialize_pipeline():
    """
    Initialize the inference pipeline components (cached).
    
    This function loads all heavy resources once and caches them:
    - CLIP model (with optional LoRA adapter)
    - Pre-computed sentence embeddings
    - Sentence metadata
    
    Returns:
        Tuple of (processor, model, embeddings, sentence_ids, sentences_data, device)
    """
    # Select configuration based on MODEL_TYPE
    config = MODEL_CONFIG[MODEL_TYPE]
    
    # Determine compute device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Load CLIP processor and base model
    processor = CLIPProcessor.from_pretrained(config["model_id"], use_fast=False)
    base_model = CLIPModel.from_pretrained(config["model_id"])
    
    # Apply LoRA adapter if configured
    if config["use_lora"] and config["lora_dir"]:
        model = PeftModel.from_pretrained(base_model, str(config["lora_dir"]))
    else:
        model = base_model
    
    # Move model to device and set to evaluation mode
    model = model.to(device).eval()
    
    # Load pre-computed embeddings
    embeddings, sentence_ids = _load_embeddings(config["embeddings_dir"])
    
    # Load sentence metadata
    sentences_data = _load_sentences_metadata(SENTENCES_JSON)
    
    return processor, model, embeddings, sentence_ids, sentences_data, device


# ========================================================================== #
#  Optional saliency overlay                                                 #
# ========================================================================== #
def compute_heatmap(
    image_path: str,
    sentence: str,
    *,
    layer_idx: int = -1,
) -> str:
    """
    Generate a Grad-ECLIP heat-map for (image, sentence).

    Parameters
    ----------
    image_path : str
        Path to the input image (same one sent to run_inference).
    sentence : str
        Caption text to explain (usually one of the sentences returned by
        run_inference).
    layer_idx : int, optional
        Vision transformer block to analyse (default last).

    Returns
    -------
    data_url : str
        PNG overlay encoded as ``data:image/png;base64,...`` suitable for the
        front-end.
    """
    # Re-use cached objects
    processor, model, _, _, _, device = _initialize_pipeline()

    pil_img = Image.open(image_path).convert("RGB")

    overlay = generate_heatmap(
        image=pil_img,
        sentence=sentence,
        model=model,
        processor=processor,
        device=device,
        layer_idx=layer_idx,
    )

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

# ========================================================================== #
#  Main retrieval routine                                                    #
# ========================================================================== #
def run_inference(
    image_path: str,
    *,
    cell: Optional[Tuple[int, int]] = None,
    grid_size: Tuple[int, int] = (8, 8),
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    """
    Perform semantic similarity search.
    
    Parameters
    ----------
    image_path : str
        Local path of the RGB image.
    cell : (int, int) | None
        If supplied (row, col) → return region-aware ranking using
        `patch_inference.rank_sentences_for_cell`.  If *None* (default)
        compute whole-painting similarity (legacy behaviour).
    grid_size : (int, int), default (8, 8)
        UI grid resolution for region mode.
    top_k : int, default 10
        Number of sentences to return.
    
    Returns:
        List of dictionaries, each containing:
        - sentence_id: Unique identifier (e.g., "W1982215463_s0001")
        - score: Cosine similarity score [0, 1]
        - sentence: Full sentence metadata including "English Original"
        - rank: Result ranking (1-based)
        
    Example:
        >>> results = run_inference("painting.jpg")
        >>> print(results[0]["sentence"]["English Original"])
        "The artist's use of light creates dramatic contrast..."
    """
    # ---- Region-aware pathway --------------------------------------------
    if cell is not None:
        # Lazy-import to avoid circular dependency at module load time
        from .patch_inference import rank_sentences_for_cell

        row, col = cell
        return rank_sentences_for_cell(
            image_path=image_path,
            cell_row=row,
            cell_col=col,
            grid_size=grid_size,
            top_k=top_k,
        )

    # ---- Whole-painting pathway (original implementation) ----------------
    # DEBUG: Start timing the entire inference
    start_time = time.time()
    
    # Load cached pipeline components
    init_start = time.time()
    processor, model, embeddings, sentence_ids, sentences_data, device = _initialize_pipeline()
    init_end = time.time()
    # debug logs removed
    
    # Load and preprocess the image
    img_load_start = time.time()
    image = Image.open(image_path).convert("RGB")
    
    preprocess_start = time.time()
    inputs = processor(images=image, return_tensors="pt").to(device)
    preprocess_end = time.time()
    # debug log removed
    
    # Compute image embedding
    embed_start = time.time()
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        print(f"[DEBUG] Raw image features shape: {image_features.shape}")
        image_embedding = F.normalize(image_features.squeeze(0), dim=-1)
    embed_end = time.time()
    # debug logs removed
    
    # Normalize sentence embeddings and compute similarities
    sim_start = time.time()
    sentence_embeddings = F.normalize(embeddings.to(device), dim=-1)
    
    similarities = torch.matmul(sentence_embeddings, image_embedding).cpu()
    sim_end = time.time()
    # debug logs removed
    
    # Get top-K results
    topk_start = time.time()
    k = min(TOP_K, len(similarities))
    top_scores, top_indices = torch.topk(similarities, k=k)
    topk_end = time.time()
    # debug logs removed
    
    # Build results with full sentence metadata
    build_start = time.time()
    results = []
    for rank, (idx, score) in enumerate(zip(top_indices.tolist(), top_scores.tolist()), start=1):
        sentence_id = sentence_ids[idx]
        
        # Get sentence metadata
        sentence_data = sentences_data.get(sentence_id, {
            "English Original": f"[Sentence data not found for {sentence_id}]",
            "Has PaintingCLIP Embedding": True
        }).copy()                       # <- make editable copy

        # ------------------------------------------------------------------
        # Ensure the UI-required “Work” field exists.
        work_id = sentence_id.split("_")[0]   # e.g. W1234567890
        sentence_data.setdefault("Work", work_id)
        # ------------------------------------------------------------------

        # DEBUG: Print first 50 chars of the sentence
        sentence_text = sentence_data.get("English Original", "N/A")
         
        results.append({
            "sentence_id": sentence_id,
            "score": float(score),
            "english_original": sentence_data.get("English Original", "N/A"),  # <- ADD
            "work": work_id,  # <- ADD (already computed above)
            "rank": rank
        })
    
    build_end = time.time()
    # debug log removed

    # DEBUG: Total time
    total_time = time.time() - start_time
    # debug logs removed
     
    return results


# ─── Utilities ───────────────────────────────────────────────────────────────
def get_available_models() -> List[str]:
    """Return list of available model types."""
    return list(MODEL_CONFIG.keys())


def set_model_type(model_type: str) -> None:
    """
    Change the active model type.
    
    Args:
        model_type: Either "clip" or "paintingclip"
        
    Raises:
        ValueError: If model_type is not recognized
    """
    global MODEL_TYPE
    if model_type not in MODEL_CONFIG:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available options: {', '.join(MODEL_CONFIG.keys())}"
        )
    MODEL_TYPE = model_type
    # Clear the cache to force reinitialization
    _initialize_pipeline.cache_clear()