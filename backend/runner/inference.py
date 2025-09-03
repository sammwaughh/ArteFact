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

import base64
import io
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import cv2
import torch
import torch.nn.functional as F
from peft import PeftModel
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset

from .filtering import get_filtered_sentence_ids
# on-demand Grad-ECLIP & region-aware ranking
from .heatmap import generate_heatmap
from .config import (
    JSON_INFO_DIR,
    EMBEDDINGS_DIR,
    JSON_DATASETS,
    EMBEDDINGS_DATASETS,
    PAINTINGCLIP_MODEL_DIR,
    ARTEFACT_EMBEDDINGS_DATASET,
    sentences,  # Add this
    CLIP_EMBEDDINGS_ST,  # Add these for backward compatibility
    PAINTINGCLIP_EMBEDDINGS_ST,
    CLIP_SENTENCE_IDS,
    PAINTINGCLIP_SENTENCE_IDS,
    CLIP_EMBEDDINGS_DIR,
    PAINTINGCLIP_EMBEDDINGS_DIR
)

# ─── Configuration ───────────────────────────────────────────────────────────
MODEL_TYPE: Literal["clip", "paintingclip"] = "paintingclip"

# Model selection - change this to switch between models
MODEL_CONFIG = {
    "clip": {
        "model_id": "openai/clip-vit-base-patch32",
        "use_lora": False,
        "lora_dir": None,
    },
    "paintingclip": {
        "model_id": "openai/clip-vit-base-patch32",
        "use_lora": True,
        "lora_dir": PAINTINGCLIP_MODEL_DIR,  # This should now point to the correct path
    },
}

# Inference settings
TOP_K = 25  # Number of results to return
# ─────────────────────────────────────────────────────────────────────────────

def load_embeddings_from_hf():
    """Load embeddings from HF dataset using safetensors files"""
    try:
        print(f" Loading embeddings from {ARTEFACT_EMBEDDINGS_DATASET}...")
        
        if not EMBEDDINGS_DATASETS:
            print("❌ No embeddings datasets loaded")
            return None
            
        # Check if we're using direct download
        if EMBEDDINGS_DATASETS.get('use_direct_download', False):
            print("✅ Using direct file download for embeddings")
            
            # Download the safetensors files
            from huggingface_hub import hf_hub_download
            import safetensors
            
            # Download CLIP embeddings
            print("🔍 Downloading CLIP embeddings...")
            clip_embeddings_path = hf_hub_download(
                repo_id=ARTEFACT_EMBEDDINGS_DATASET,
                filename="clip_embeddings.safetensors",
                repo_type="dataset"
            )
            
            clip_ids_path = hf_hub_download(
                repo_id=ARTEFACT_EMBEDDINGS_DATASET,
                filename="clip_embeddings_sentence_ids.json",
                repo_type="dataset"
            )
            
            # Download PaintingCLIP embeddings
            print("🔍 Downloading PaintingCLIP embeddings...")
            paintingclip_embeddings_path = hf_hub_download(
                repo_id=ARTEFACT_EMBEDDINGS_DATASET,
                filename="paintingclip_embeddings.safetensors",
                repo_type="dataset"
            )
            
            paintingclip_ids_path = hf_hub_download(
                repo_id=ARTEFACT_EMBEDDINGS_DATASET,
                filename="paintingclip_embeddings_sentence_ids.json",
                repo_type="dataset"
            )
            
            # Load the embeddings
            print("🔍 Loading CLIP embeddings...")
            clip_embeddings = safetensors.torch.load_file(clip_embeddings_path)['embeddings']
            
            print("🔍 Loading PaintingCLIP embeddings...")
            paintingclip_embeddings = safetensors.torch.load_file(paintingclip_embeddings_path)['embeddings']
            
            # Load the sentence IDs
            with open(clip_ids_path, 'r') as f:
                clip_sentence_ids = json.load(f)
                
            with open(paintingclip_ids_path, 'r') as f:
                paintingclip_sentence_ids = json.load(f)
            
            print(f"✅ Loaded CLIP embeddings: {clip_embeddings.shape}")
            print(f"✅ Loaded PaintingCLIP embeddings: {paintingclip_embeddings.shape}")
            
            return {
                "clip": (clip_embeddings, clip_sentence_ids),
                "paintingclip": (paintingclip_embeddings, paintingclip_sentence_ids)
            }
        else:
            # Fallback to old method if not using direct download
            print("⚠️  Using fallback embedding loading method")
            return None
            
    except Exception as e:
        print(f"❌ Failed to load embeddings from HF: {e}")
        return None

def _load_sentences_metadata() -> Dict[str, Dict[str, Any]]:
    """
    Get sentence metadata from global config (loaded from HF datasets).
    """
    if not sentences:
        print("⚠️  No sentence metadata available - check if HF datasets loaded successfully")
        return {}
    return sentences

@lru_cache(maxsize=1)
def _initialize_pipeline():
    """
    Initialize the inference pipeline components (cached).

    This function loads all heavy resources once and caches them:
    - CLIP model (with optional LoRA adapter)
    - Pre-computed sentence embeddings from HF
    - Sentence metadata from HF

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

    # Apply LoRA adapter if configured and available
    if config["use_lora"] and config["lora_dir"]:
        lora_path = Path(config["lora_dir"])
        adapter_config_path = lora_path / "adapter_config.json"
        
        if adapter_config_path.exists():
            print(f"✅ Loading LoRA adapter from {lora_path}")
            model = PeftModel.from_pretrained(base_model, str(lora_path))
        else:
            print(f"⚠️  LoRA adapter not found at {lora_path}")
            print(f"⚠️  Missing file: {adapter_config_path}")
            print(f"⚠️  Falling back to base CLIP model without LoRA adapter")
            model = base_model
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

    # Load pre-computed embeddings from HF
    try:
        embeddings_data = load_embeddings_from_hf()
        if embeddings_data is None:
            raise ValueError(f"Failed to load embeddings from HF dataset: {ARTEFACT_EMBEDDINGS_DATASET}")
        
        # Check if we're using streaming (old approach)
        if embeddings_data.get("streaming", False):
            print("✅ Using streaming embeddings - will load on-demand")
            return processor, model, "STREAMING", "STREAMING", "STREAMING", device
        else:
            # New code path for direct file download
            if MODEL_TYPE == "clip":
                embeddings, sentence_ids = embeddings_data["clip"]
            else:
                embeddings, sentence_ids = embeddings_data["paintingclip"]

            if embeddings is None or sentence_ids is None:
                raise ValueError(f"Failed to load embeddings for model type: {MODEL_TYPE}")
            
            print(f"🔍 Loaded {len(sentence_ids)} embeddings with shape {embeddings.shape}")
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        raise

    # Get sentence metadata from global config
    sentences_data = _load_sentences_metadata()
    print(f"🔍 Loaded {len(sentences_data)} sentence metadata entries")
    if sentences_data:
        sample_key = next(iter(sentences_data.keys()))
        print(f"🔍 Sample sentence data structure: {sentences_data[sample_key]}")

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
    print(f"🔍 run_inference called with:")
    print(f"🔍   image_path: {image_path}")
    print(f"🔍   cell: {cell}")
    print(f"🔍   filter_topics: {filter_topics}")
    print(f"🔍   filter_creators: {filter_creators}")
    print(f"🔍   model_type: {model_type}")
    
    try:
        # Set model type if specified
        if model_type:
            print(f"🔍 Setting model type to: {model_type}")
            set_model_type(model_type.lower())

        # ---- Region-aware pathway --------------------------------------------
        if cell is not None:
            print(f"🔍 Using region-aware pathway for cell {cell}")
            from .patch_inference import rank_sentences_for_cell

            row, col = cell
            results = rank_sentences_for_cell(
                image_path=image_path,
                cell_row=row,
                cell_col=col,
                grid_size=grid_size,
                top_k=top_k * 3,
            )

            # Apply filtering
            if filter_topics or filter_creators:
                from .filtering import apply_filters_to_results
                results = apply_filters_to_results(results, filter_topics, filter_creators)
                results = results[:top_k]

            return results

        # ---- Whole-painting pathway (original implementation) ----------------
        print(f"🔍 Using whole-painting pathway")
        
        # Load cached pipeline components
        print(f"🔍 Loading pipeline components...")
        processor, model, embeddings, sentence_ids, sentences_data, device = (
            _initialize_pipeline()
        )
        print(f"✅ Pipeline components loaded successfully")

        # Check if we're in streaming mode
        if embeddings == "STREAMING":
            print("✅ Streaming mode detected - using streaming embeddings")
            return run_inference_streaming(
                image_path=image_path,
                filter_topics=filter_topics,
                filter_creators=filter_creators,
                model_type=model_type,
                top_k=top_k,
                processor=processor,
                model=model,
                device=device
            )

        # Non-streaming mode - continue with existing logic
        # Get valid sentence IDs based on filters
        if filter_topics or filter_creators:
            print(f"🔍 Applying filters...")
            valid_sentence_ids = get_filtered_sentence_ids(filter_topics, filter_creators)
            print(f"✅ Filtered to {len(valid_sentence_ids)} valid sentences")

            # Create mask for valid sentences
            valid_indices = [
                i for i, sid in enumerate(sentence_ids) if sid in valid_sentence_ids
            ]

            if not valid_indices:
                print(f"⚠️  No sentences match the filters")
                return []

            # Filter embeddings and sentence_ids
            filtered_embeddings = embeddings[valid_indices]
            filtered_sentence_ids = [sentence_ids[i] for i in valid_indices]
        else:
            print(f"🔍 No filtering applied")
            filtered_embeddings = embeddings
            filtered_sentence_ids = sentence_ids

        # Load and preprocess the image
        print(f"🔍 Loading and preprocessing image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        print(f"✅ Image loaded successfully, size: {image.size}")

        # Compute image embedding
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

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
        for rank, (idx, score) in enumerate(zip(top_indices.tolist(), top_scores.tolist()), start=1):
            sentence_id = filtered_sentence_ids[idx]
            sentence_data = sentences_data.get(
                sentence_id,
                {"English Original": f"[Sentence data not found for {sentence_id}]", "Has PaintingCLIP Embedding": True},
            ).copy()
            work_id = sentence_id.split("_")[0]
            sentence_data.setdefault("Work", work_id)
            results.append({
                "id": sentence_id,
                "score": float(score),
                "english_original": sentence_data.get("English Original", "N/A"),
                "work": work_id,
                "rank": rank,
            })

        print(f"🔍 run_inference returning {len(results)} results")
        if results:
            print(f"🔍 First result: {results[0]}")
        return results
        
    except Exception as e:
        print(f"❌ Error in run_inference: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        print(f"❌ Full traceback:")
        traceback.print_exc()
        raise


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


def load_consolidated_embeddings(embedding_file: Path, metadata_file: Path):
    """Load embeddings from consolidated file with metadata"""
    print(f"Loading consolidated embeddings from {embedding_file}")
    
    # Load consolidated data with weights_only=False for compatibility
    # This is safe since we're loading our own pre-computed embeddings
    try:
        consolidated_data = torch.load(embedding_file, map_location='cpu', weights_only=False)
        print(f"✅ Successfully loaded consolidated embeddings")
    except Exception as e:
        print(f"❌ Failed to load with weights_only=False: {e}")
        # Fallback: try with weights_only=True (may fail if file has non-tensor data)
        try:
            print(f"🔍 Trying fallback with weights_only=True...")
            consolidated_data = torch.load(embedding_file, map_location='cpu', weights_only=True)
            print(f"✅ Successfully loaded with weights_only=True")
        except Exception as e2:
            print(f"❌ Both loading methods failed:")
            print(f"   weights_only=False: {e}")
            print(f"   weights_only=True: {e2}")
            raise RuntimeError(f"Cannot load embedding file with either method: {e2}")
    
    embeddings = consolidated_data['embeddings']
    
    # Load metadata for file mapping
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Create filename to index mapping
    filename_to_index = {item['filename']: item['index'] for item in metadata['file_mapping']}
    
    print(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    
    return embeddings, filename_to_index

def load_consolidated_embeddings_st(embedding_st_file: Path, ids_json_file: Path):
	print(f"Loading safetensors embeddings from {embedding_st_file}")
	if not embedding_st_file.exists():
		raise FileNotFoundError(f"Missing {embedding_st_file}")
	if not ids_json_file.exists():
		raise FileNotFoundError(f"Missing {ids_json_file}")

	data = st_load_file(str(embedding_st_file))
	if "embeddings" not in data:
		raise KeyError(f"'embeddings' tensor missing in {embedding_st_file}")
	embeddings = data["embeddings"].to(dtype=torch.float32, device="cpu").contiguous()

	with open(ids_json_file, "r", encoding="utf-8") as f:
		sentence_ids = json.load(f)
	if not isinstance(sentence_ids, list):
		raise ValueError(f"IDs file malformed: {ids_json_file}")

	print(f"Loaded {len(sentence_ids)} embeddings with dim {embeddings.shape[1]}")
	return embeddings, sentence_ids

# Update your embedding loading logic
def load_embeddings_for_model(model_type: str):
	"""Load embeddings for the specified model type with safetensors-first strategy."""
	if model_type == "clip":
		st_file = CLIP_EMBEDDINGS_ST
		ids_file = CLIP_SENTENCE_IDS
		# Legacy PT locations for fallback (if repo still has them)
		pt_file = EMBEDDINGS_DIR / "clip_embeddings_consolidated.pt"
		meta_file = EMBEDDINGS_DIR / "clip_embeddings_metadata.json"
		indiv_dir = CLIP_EMBEDDINGS_DIR
	else:
		st_file = PAINTINGCLIP_EMBEDDINGS_ST
		ids_file = PAINTINGCLIP_SENTENCE_IDS
		pt_file = EMBEDDINGS_DIR / "paintingclip_embeddings_consolidated.pt"
		meta_file = EMBEDDINGS_DIR / "paintingclip_embeddings_metadata.json"
		indiv_dir = PAINTINGCLIP_EMBEDDINGS_DIR

	# 1) safetensors
	if st_file.exists() and ids_file.exists():
		try:
			return load_consolidated_embeddings_st(st_file, ids_file)
		except Exception as e:
			print(f"⚠️  Safetensors load failed: {e}")

	# 2) legacy PT (if present)
	if pt_file.exists() and meta_file.exists():
		try:
			return load_consolidated_embeddings(pt_file, meta_file)
		except Exception as e:
			print(f"⚠️  Legacy PT load failed: {e}")

	# 3) final fallback: refuse to scan 10k files here (HF Spaces file count limits)
	print("❌ No valid consolidated embeddings found. Make sure you committed:")
	print(f"   - {st_file.name}")
	print(f"   - {ids_file.name}")
	return None, None

# Add this function for backward compatibility
def st_load_file(file_path: Path) -> Any:
    """Load a file using safetensors or other methods"""
    try:
        if file_path.suffix == '.safetensors':
            import safetensors
            return safetensors.safe_open(str(file_path), framework="pt")
        else:
            import torch
            return torch.load(str(file_path))
    except ImportError:
        print(f"⚠️  Required library not available for loading {file_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

def load_embedding_for_sentence(sentence_id: str, model_type: str = "clip") -> Optional[torch.Tensor]:
    """Load a single embedding for a specific sentence using streaming"""
    try:
        if not EMBEDDINGS_DATASETS or not EMBEDDINGS_DATASETS.get('use_streaming', False):
            print("❌ Streaming embeddings not available")
            return None
            
        dataset = EMBEDDINGS_DATASETS['streaming_dataset']
        
        # Search for the sentence in the streaming dataset
        for item in dataset:
            if item.get('sentence_id') == sentence_id:
                # Extract the appropriate embedding based on model type
                if model_type == "clip" and 'clip_embedding' in item:
                    return torch.tensor(item['clip_embedding'])
                elif model_type == "paintingclip" and 'paintingclip_embedding' in item:
                    return torch.tensor(item['paintingclip_embedding'])
                else:
                    print(f"⚠️  Embedding not found for {model_type} in sentence {sentence_id}")
                    return None
        
        print(f"⚠️  Sentence {sentence_id} not found in streaming dataset")
        return None
        
    except Exception as e:
        print(f"❌ Error loading streaming embedding for {sentence_id}: {e}")
        return None

def get_top_k_embeddings(query_embedding: torch.Tensor, k: int = 10, model_type: str = "clip") -> List[Tuple[str, float]]:
    """Get top-k most similar embeddings using streaming"""
    try:
        if not EMBEDDINGS_DATASETS or not EMBEDDINGS_DATASETS.get('use_streaming', False):
            print("❌ Streaming embeddings not available")
            return []
            
        dataset = EMBEDDINGS_DATASETS['streaming_dataset']
        similarities = []
        
        # Process embeddings in batches to avoid memory issues
        batch_size = 1000
        batch = []
        
        for item in dataset:
            batch.append(item)
            
            if len(batch) >= batch_size:
                # Process batch
                batch_similarities = process_embedding_batch(batch, query_embedding, model_type)
                similarities.extend(batch_similarities)
                batch = []
                
                # Keep only top-k so far
                similarities.sort(key=lambda x: x[1], reverse=True)
                similarities = similarities[:k]
        
        # Process remaining items
        if batch:
            batch_similarities = process_embedding_batch(batch, query_embedding, model_type)
            similarities.extend(batch_similarities)
            similarities.sort(key=lambda x: x[1], reverse=True)
            similarities = similarities[:k]
        
        return similarities
        
    except Exception as e:
        print(f"❌ Error getting top-k embeddings: {e}")
        return []

def process_embedding_batch(batch: List[Dict], query_embedding: torch.Tensor, model_type: str) -> List[Tuple[str, float]]:
    """Process a batch of embeddings to find similarities"""
    similarities = []
    
    for item in batch:
        try:
            sentence_id = item.get('sentence_id', '')
            
            # Get the appropriate embedding
            if model_type == "clip" and 'clip_embedding' in item:
                embedding = torch.tensor(item['clip_embedding'])
            elif model_type == "paintingclip" and 'paintingclip_embedding' in item:
                embedding = torch.tensor(item['paintingclip_embedding'])
            else:
                continue
            
            # Calculate similarity
            similarity = F.cosine_similarity(query_embedding.unsqueeze(0), embedding.unsqueeze(0), dim=1)
            similarities.append((sentence_id, similarity.item()))
            
        except Exception as e:
            print(f"⚠️  Error processing item in batch: {e}")
            continue
    
    return similarities

def run_inference_streaming(
    image_path: str,
    filter_topics: List[str] = None,
    filter_creators: List[str] = None,
    model_type: str = "CLIP",
    top_k: int = 10,
    processor=None,
    model=None,
    device=None
) -> List[Dict[str, Any]]:
    """Run inference using streaming embeddings"""
    try:
        print(f"🔍 Running streaming inference for {image_path}")
        start_time = time.time()
        
        # Load and preprocess the image
        print(f"🔍 Loading and preprocessing image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        print(f"✅ Image loaded successfully, size: {image.size}")

        # Compute image embedding
        print(f"🔍 Computing image embedding...")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_embedding = F.normalize(image_features.squeeze(0), dim=-1)
        print(f"✅ Image embedding computed successfully")

        # Get streaming dataset
        if not EMBEDDINGS_DATASETS or not EMBEDDINGS_DATASETS.get('use_streaming', False):
            raise ValueError("Streaming embeddings not available")
            
        dataset = EMBEDDINGS_DATASETS['streaming_dataset']
        
        # Process embeddings in streaming mode
        results = []
        batch_size = 1000
        batch = []
        total_processed = 0
        batch_count = 0
        
        print(f"🔍 Starting streaming processing of 3.1M+ sentence embeddings...")
        print(f"🔍 Batch size: {batch_size}")
        print(f"🔍 Target top-k: {top_k}")
        
        # Estimate total items for progress tracking
        try:
            # Try to get dataset size if available
            if hasattr(dataset, '__len__'):
                total_items = len(dataset)
                print(f"🔍 Total embeddings to process: {total_items:,}")
            else:
                total_items = None
                print(f"🔍 Dataset size unknown (streaming mode)")
        except:
            total_items = None
        
        for item in dataset:
            batch.append(item)
            total_processed += 1
            
            if len(batch) >= batch_size:
                batch_count += 1
                batch_start_time = time.time()
                
                # Process batch
                print(f"🔍 Processing batch {batch_count} ({total_processed:,} items processed)...")
                batch_results = process_embedding_batch_streaming(
                    batch, image_embedding, model_type, device
                )
                results.extend(batch_results)
                batch = []
                
                # Keep only top-k so far
                results.sort(key=lambda x: x["score"], reverse=True)
                results = results[:top_k]
                
                batch_time = time.time() - batch_start_time
                elapsed_time = time.time() - start_time
                
                # Progress reporting
                if total_items:
                    progress_pct = (total_processed / total_items) * 100
                    print(f"🔍 Batch {batch_count} completed in {batch_time:.2f}s")
                    print(f"🔍 Progress: {total_processed:,}/{total_items:,} ({progress_pct:.1f}%)")
                    print(f"🔍 Elapsed time: {elapsed_time:.1f}s")
                    print(f"🔍 Current top score: {results[0]['score']:.4f}" if results else "🔍 Current top score: N/A")
                    print(f"🔍 Estimated time remaining: {((elapsed_time / total_processed) * (total_items - total_processed)):.1f}s")
                else:
                    print(f"🔍 Batch {batch_count} completed in {batch_time:.2f}s")
                    print(f"🔍 Total processed: {total_processed:,}")
                    print(f"🔍 Elapsed time: {elapsed_time:.1f}s")
                    print(f"🔍 Current top score: {results[0]['score']:.4f}" if results else "🔍 Current top score: N/A")
                
                print(f"🔍 Current top result: {results[0]['english_original'][:100]}..." if results else "🔍 No results yet")
                print("─" * 80)
        
        # Process remaining items
        if batch:
            print(f"🔍 Processing final batch of {len(batch)} items...")
            batch_results = process_embedding_batch_streaming(
                batch, image_embedding, model_type, device
            )
            results.extend(batch_results)
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:top_k]
        
        total_time = time.time() - start_time
        print(f"✅ Streaming inference completed!")
        print(f"🔍 Total time: {total_time:.2f}s")
        print(f"🔍 Total embeddings processed: {total_processed:,}")
        print(f"🔍 Final results: {len(results)} items")
        if results:
            print(f"🔍 Top result score: {results[0]['score']:.4f}")
            print(f"🔍 Top result: {results[0]['english_original'][:100]}...")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in streaming inference: {e}")
        raise

def process_embedding_batch_streaming(
    batch: List[Dict], 
    image_embedding: torch.Tensor, 
    model_type: str, 
    device: torch.device
) -> List[Dict[str, Any]]:
    """Process a batch of streaming embeddings"""
    results = []
    processed_count = 0
    error_count = 0
    
    print(f"🔍 Processing batch of {len(batch)} items...")
    
    # Debug: show first few items to understand the data structure
    for i, item in enumerate(batch[:3]):
        print(f" Item {i}: keys = {list(item.keys())}")
        print(f" Item {i}: full item = {item}")
    
    for item in batch:
        try:
            sentence_id = item.get('sentence_id', '')
            
            # Get the appropriate embedding
            if model_type == "CLIP" and 'clip_embedding' in item:
                embedding = torch.tensor(item['clip_embedding'])
            elif model_type == "PaintingCLIP" and 'paintingclip_embedding' in item:
                embedding = torch.tensor(item['paintingclip_embedding'])
            else:
                if processed_count < 3:  # Only show first few errors
                    print(f"⚠️  No embedding found for {model_type} in item: {list(item.keys())}")
                continue
            
            # Calculate similarity
            embedding = embedding.to(device)
            similarity = F.cosine_similarity(
                image_embedding.unsqueeze(0), 
                embedding.unsqueeze(0), 
                dim=1
            ).item()
            
            # Get sentence metadata
            sentences_data = _load_sentences_metadata()
            sentence_data = sentences_data.get(sentence_id, {})
            work_id = sentence_id.split("_")[0]
            
            results.append({
                "id": sentence_id,
                "score": similarity,
                "english_original": sentence_data.get("English Original", "N/A"),
                "work": work_id,
                "rank": len(results) + 1,
            })
            processed_count += 1
            
        except Exception as e:
            error_count += 1
            if error_count < 3:  # Only show first few errors
                print(f"⚠️  Error processing item in streaming batch: {e}")
            continue
    
    print(f"🔍 Batch processing complete: {processed_count} successful, {error_count} errors")
    return results
