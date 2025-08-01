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
import cv2

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from peft import PeftModel

# on-demand Grad-ECLIP & region-aware ranking
from .heatmap import generate_heatmap
from .filtering import get_filtered_sentence_ids


# ─── Configuration ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]  # artefact-context/

# Model selection - change this to switch between models
MODEL_TYPE: Literal["clip", "paintingclip"] = "paintingclip"

# Model paths and settings
MODEL_CONFIG = {
    "clip": {
        "model_id": "openai/clip-vit-base-patch32",
        "embeddings_dir": ROOT / "data" / "embeddings" / "CLIP_Embeddings",
        "use_lora": False,
        "lora_dir": None,
    },
    "paintingclip": {
        "model_id": "openai/clip-vit-base-patch32",
        "embeddings_dir": ROOT / "data" / "embeddings" / "PaintingCLIP_Embeddings",
        "use_lora": True,
        "lora_dir": ROOT / "data" / "models" / "PaintingCLIP",
    },
}

# Data paths
SENTENCES_JSON = ROOT / "data" / "json_info" / "sentences.json"

# Inference settings
TOP_K = 25  # Number of results to return
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

    # Check for meta tensors and handle them properly
    has_meta_tensors = any(p.device.type == "meta" for p in model.parameters())

    if has_meta_tensors:
        # Meta tensors mean the model needs to be materialized
        print("[inference] meta tensors detected – materializing model on CPU")
        device = torch.device("cpu")
        
        # Materialize the model by moving it to CPU
        # This converts meta tensors to actual tensors with allocated memory
        model = model.to(device)
        
        # Ensure all parameters are properly initialized
        for param in model.parameters():
            if param.device.type == "meta":
                # This shouldn't happen after .to(device), but as a safety check
                param.data = param.data.to(device)
    else:
        # Normal case: move model to selected device
        if device.type != "cpu":
            model = model.to(device)

    model = model.eval()

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
    alpha: float = 0.45,  # Add this
    colormap: int = cv2.COLORMAP_JET,  # Add this
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
    alpha : float, optional
        Heatmap overlay opacity (default: 0.45)
    colormap : int, optional
        OpenCV colormap for visualization (default: COLORMAP_JET)

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
        alpha=alpha,  # Pass through
        colormap=colormap,  # Pass through
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
    grid_size: Tuple[int, int] = (7, 7),
    top_k: int = TOP_K,
    filter_topics: List[str] = None,
    filter_creators: List[str] = None,
    model_type: str = None,
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
    grid_size : (int, int), default (7, 7)
        UI grid resolution for region mode.
    top_k : int, default 25
        Number of sentences to return.
    filter_topics : List[str], optional
        List of topic codes to filter results by
    filter_creators : List[str], optional
        List of creator names to filter results by
    model_type : str, optional
        Model type to use ("clip" or "paintingclip")

    Returns:
        List of dictionaries with filtered results
    """
    # Set model type if specified
    if model_type:
        set_model_type(model_type.lower())

    # ---- Region-aware pathway --------------------------------------------
    if cell is not None:
        from .patch_inference import rank_sentences_for_cell

        row, col = cell
        results = rank_sentences_for_cell(
            image_path=image_path,
            cell_row=row,
            cell_col=col,
            grid_size=grid_size,
            top_k=top_k * 3,  # Get more results to filter from
        )

        # Apply filtering
        if filter_topics or filter_creators:
            from .filtering import apply_filters_to_results

            results = apply_filters_to_results(results, filter_topics, filter_creators)
            results = results[:top_k]  # Trim to requested top_k

        return results

    # ---- Whole-painting pathway (original implementation) ----------------
    start_time = time.time()

    # Load cached pipeline components
    processor, model, embeddings, sentence_ids, sentences_data, device = (
        _initialize_pipeline()
    )

    # Get valid sentence IDs based on filters
    if filter_topics or filter_creators:
        valid_sentence_ids = get_filtered_sentence_ids(filter_topics, filter_creators)

        # Create mask for valid sentences
        valid_indices = [
            i for i, sid in enumerate(sentence_ids) if sid in valid_sentence_ids
        ]

        if not valid_indices:
            # No sentences match the filters
            return []

        # Filter embeddings and sentence_ids
        filtered_embeddings = embeddings[valid_indices]
        filtered_sentence_ids = [sentence_ids[i] for i in valid_indices]
    else:
        # No filtering, use all
        filtered_embeddings = embeddings
        filtered_sentence_ids = sentence_ids

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    # Ensure inputs are on the correct device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Compute image embedding
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_embedding = F.normalize(image_features.squeeze(0), dim=-1)

    # Normalize sentence embeddings and compute similarities
    sentence_embeddings = F.normalize(filtered_embeddings.to(device), dim=-1)
    similarities = torch.matmul(sentence_embeddings, image_embedding).cpu()

    # Get top-K results
    k = min(top_k, len(similarities))
    top_scores, top_indices = torch.topk(similarities, k=k)

    # Build results with full sentence metadata
    results = []
    for rank, (idx, score) in enumerate(
        zip(top_indices.tolist(), top_scores.tolist()), start=1
    ):
        sentence_id = filtered_sentence_ids[idx]

        # Get sentence metadata
        sentence_data = sentences_data.get(
            sentence_id,
            {
                "English Original": f"[Sentence data not found for {sentence_id}]",
                "Has PaintingCLIP Embedding": True,
            },
        ).copy()

        work_id = sentence_id.split("_")[0]
        sentence_data.setdefault("Work", work_id)

        results.append(
            {
                "sentence_id": sentence_id,
                "score": float(score),
                "english_original": sentence_data.get("English Original", "N/A"),
                "work": work_id,
                "rank": rank,
            }
        )

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
