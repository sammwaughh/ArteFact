"""
patch_inference.py
──────────────────
Fast patch‑text similarity ranking on top of the existing PaintingCLIP
inference pipeline.

Public API
----------
rank_sentences_for_cell(...)
list_grid_scores(...)          # optional diagnostic helper
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

# Local import: reuse the heavyweight initialiser & sentence metadata
from .inference import _initialize_pipeline   # same package, no circular import


# ════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ════════════════════════════════════════════════════════════════════════════
def _infer_patch_hw(num_patches: int) -> Tuple[int, int]:
    """
    Infer the ViT patch grid (H, W) from the flat token count.
    Works for square layouts only (ViT‑B/32 → 7×7; ViT‑B/16 → 14×14).
    """
    root = int(math.sqrt(num_patches))
    if root * root == num_patches:
        return root, root
    raise ValueError(f"Unexpected non‑square patch layout: {num_patches}")


@lru_cache(maxsize=8)              # cache a few recent paintings × grid sizes
def _prepare_image(
    image_path: str | Path,
    grid_size: Tuple[int, int] = (8, 8),
) -> torch.Tensor:
    """
    Convert *one* painting into a bank of L2‑normalised grid‑cell vectors.

    Returns
    -------
    torch.Tensor
        Shape (cells, D) on the same device as sentence embeddings.
    """
    processor, model, *_ , device = _initialize_pipeline()

    # 1. Forward pass through the vision tower (one‑time per painting ↔ cache)
    image = Image.open(image_path).convert("RGB")
    pixel = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        vision_out = model.vision_model(
            pixel_values=pixel["pixel_values"],
            output_hidden_states=True,
        )
        # Exclude CLS (token‑0), keep patch tokens
        patch_tokens = vision_out.last_hidden_state[:, 1:, :]               # (1, N, 768)
        patch_tokens = model.vision_model.post_layernorm(patch_tokens)      # LayerNorm
        patch_feats = model.visual_projection(patch_tokens)                 # (1, N, 512)
        patch_feats = F.normalize(patch_feats.squeeze(0), dim=-1)           # (N, 512)

    # 2. Reshape → (D, H, W) to pool channel‑wise
    num_patches, dim = patch_feats.shape
    H, W = _infer_patch_hw(num_patches)
    patch_grid = (
        patch_feats.view(H, W, dim)
        .permute(2, 0, 1)           # (D, H, W)
        .unsqueeze(0)               # (1, D, H, W) for pooling
    )

    # 3. Adaptive average‑pool down to UI grid resolution
    grid_h, grid_w = grid_size
    cell_grid = F.adaptive_avg_pool2d(patch_grid, output_size=(grid_h, grid_w)).squeeze(0)

    # 4. Flatten → (cells, D) & L2‑normalise
    cell_vecs = cell_grid.permute(1, 2, 0).reshape(-1, dim)                # (g², 512)
    return F.normalize(cell_vecs, dim=-1).to(device)


# ════════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════════
def rank_sentences_for_cell(
    image_path: str | Path,
    cell_row: int,
    cell_col: int,
    grid_size: Tuple[int, int] = (8, 8),
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Retrieve the *top‑k* sentences whose text embeddings align most strongly
    with a specific grid cell of the painting.

    Parameters
    ----------
    image_path : str | Path
        Path (local or mounted) to the RGB image file.
    cell_row, cell_col : int
        Zero‑indexed row/column of the clicked grid cell.
    grid_size : (int, int), default (8, 8)
        Resolution of the UI grid.
    top_k : int, default 10
        How many sentences to return.

    Returns
    -------
    List[dict]
        Each item is the same schema as `run_inference`, facilitating front‑end reuse:
        { "sentence_id", "score", "english_original", "work", "rank" }
    """
    # Shared resources
    _proc, _model, sent_mat, sentence_ids, sent_meta, device = _initialize_pipeline()
    sent_mat = F.normalize(sent_mat.to(device), dim=-1)                    # (S, 512)

    # Validate cell indices
    grid_h, grid_w = grid_size
    if not (0 <= cell_row < grid_h and 0 <= cell_col < grid_w):
        raise ValueError(f"Cell ({cell_row}, {cell_col}) outside grid {grid_size}")

    # Cell feature vector
    cell_idx = cell_row * grid_w + cell_col
    cell_vecs = _prepare_image(image_path, grid_size)                      # (g², 512)
    cell_vec = cell_vecs[cell_idx]                                         # (512,)

    # Cosine similarity and ranking
    scores = torch.matmul(sent_mat, cell_vec)                              # (S,)
    k = min(top_k, scores.size(0))
    top_scores, top_idx = torch.topk(scores, k)

    # Assemble output
    out: List[Dict[str, Any]] = []
    for rank, (idx, sc) in enumerate(zip(top_idx.tolist(), top_scores.tolist()), 1):
        sid = sentence_ids[idx]
        meta = sent_meta.get(
            sid,
            {"English Original": f"[Sentence data not found for {sid}]"},
        )
        work_id = sid.split("_")[0]
        out.append(
            {
                "sentence_id": sid,
                "score": float(sc),
                "english_original": meta.get("English Original", ""),
                "work": work_id,
                "rank": rank,
            }
        )
    return out


# Optional helper for debugging / heat‑map pre‑computation
def list_grid_scores(
    image_path: str | Path,
    grid_size: Tuple[int, int] = (8, 8),
) -> torch.Tensor:
    """
    Return the full similarity matrix of shape (sentences, cells).
    Primarily for diagnostics or off‑line analysis.
    """
    _p, _m, sent_mat, *_ , device = _initialize_pipeline()
    sent_mat = F.normalize(sent_mat.to(device), dim=-1)
    cell_vecs = _prepare_image(image_path, grid_size)
    return sent_mat @ cell_vecs.T                                          # (S, g²)
