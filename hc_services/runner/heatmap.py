"""
heatmap.py
----------

Grad-ECLIP visual explanations for CLIP/PaintingCLIP models.
Generates heatmap overlays showing which image regions contribute to image-text similarity.

Based on "Gradient-based Visual Explanation for Transformer-based CLIP" 
by Zhao et al. (ICML 2024)

Public entry point:
------------------
generate_heatmap(
    image,                 # str | PIL.Image.Image
    sentence,              # caption text
    model,                 # CLIPModel or PEFT-wrapped model
    processor,             # CLIPProcessor
    device,                # torch.device
    *,
    layer_idx: int = -1,   # which visual transformer block to explain
    alpha: float = 0.45,   # overlay opacity
    colormap: int = cv2.COLORMAP_JET,
    resize: Optional[Tuple[int, int]] = None,
) -> PIL.Image.Image       # RGB overlay for display
"""

from __future__ import annotations

import contextlib
from typing import Dict, Tuple, Union, Optional, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from peft import PeftModel


# ============================================================================ #
# Core Grad-ECLIP Implementation                                               #
# ============================================================================ #

class _GradECLIPHooks:
    """
    Context manager for forward/backward hooks to capture Grad-ECLIP components.
    """
    
    def __init__(self, model: CLIPModel, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self.captures: Dict[str, Any] = {}
        self.handles = []
        
    def __enter__(self):
        # Get target layer
        vision_layers = self.model.vision_model.encoder.layers
        if self.layer_idx < 0:
            self.layer_idx = len(vision_layers) + self.layer_idx
        self.target_layer = vision_layers[self.layer_idx]
        
        # Register hooks
        self._register_forward_hook()
        self._register_backward_hook()
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up hooks
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        
    def _register_forward_hook(self):
        """Register forward hook to capture Q, K, V and attention weights."""
        def forward_hook(module, input, output):
            if len(input) > 0:
                hidden_states = input[0]
                
                # Get attention inputs
                x = hidden_states
                if hasattr(module.self_attn, 'layer_norm'):
                    x = module.self_attn.layer_norm(x)
                
                # Compute Q, K, V
                if hasattr(module.self_attn, 'q_proj'):
                    batch_size, seq_len, hidden_dim = x.shape
                    
                    Q = module.self_attn.q_proj(x)
                    K = module.self_attn.k_proj(x)
                    V = module.self_attn.v_proj(x)
                    
                    # Store raw projections
                    self.captures['V'] = V
                    self.captures['hidden_states_pre'] = hidden_states
                    
                    # Compute attention for head-averaged weights
                    head_dim = hidden_dim // module.self_attn.num_heads
                    num_heads = module.self_attn.num_heads
                    
                    # Reshape for multi-head attention
                    Q_heads = Q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                    K_heads = K.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                    
                    # Compute attention weights
                    scale = head_dim ** -0.5
                    attn_weights = torch.matmul(Q_heads, K_heads.transpose(-2, -1)) * scale
                    attn_weights = torch.softmax(attn_weights, dim=-1)
                    
                    # Store for later use
                    self.captures['Q'] = Q_heads
                    self.captures['K'] = K_heads
                    self.captures['attn_weights'] = attn_weights.mean(dim=1)  # Average over heads
        
        handle = self.target_layer.register_forward_hook(forward_hook)
        self.handles.append(handle)
        
    def _register_backward_hook(self):
        """Register backward hook to capture gradients."""
        def backward_hook(module, grad_input, grad_output):
            if len(grad_output) > 0:
                self.captures['grad_attn'] = grad_output[0]
        
        handle = self.target_layer.register_full_backward_hook(backward_hook)
        self.handles.append(handle)
        
    def get_captures(self) -> Dict[str, torch.Tensor]:
        """Return captured tensors."""
        return self.captures


def _compute_gradeclip_importance(
    captures: Dict[str, torch.Tensor],
    use_k_similarity: bool = True,
    device: torch.device = None
) -> torch.Tensor:
    """
    Compute Grad-ECLIP importance scores from captured tensors.
    
    Args:
        captures: Dictionary with captured tensors from hooks
        use_k_similarity: Whether to use Q-K similarity weighting
        device: Computation device
        
    Returns:
        Importance scores for each patch (excluding CLS token)
    """
    # Extract captured tensors
    V = captures.get('V')
    grad_attn = captures.get('grad_attn')
    attn_weights = captures.get('attn_weights')
    
    if V is None or grad_attn is None:
        raise ValueError("Missing required captures for Grad-ECLIP computation")
    
    # 1. Channel importance: gradients at CLS token
    grad_cls = grad_attn[0, 0, :]  # Shape: (hidden_dim,)
    
    # 2. Extract patch values (exclude CLS token)
    V_patches = V[0, 1:, :]  # Shape: (num_patches, hidden_dim)
    num_patches = V_patches.shape[0]
    
    # 3. Get spatial attention weights
    if attn_weights is not None:
        # Use captured attention from CLS to patches
        cls_attn = attn_weights[0, 0, 1:num_patches+1]
    else:
        # Fallback: uniform weights
        cls_attn = torch.ones(num_patches, device=device or V.device) / num_patches
    
    # 4. Optional: Apply Q-K similarity normalization
    if use_k_similarity and 'Q' in captures and 'K' in captures:
        Q = captures['Q']
        K = captures['K']
        
        # Get CLS token query (average over heads)
        q_cls = Q[:, :, 0:1, :].mean(dim=1)  # Shape: (1, 1, head_dim)
        k_patches = K[:, :, 1:, :].mean(dim=1)  # Shape: (1, num_patches, head_dim)
        
        # Normalize and compute cosine similarity
        q_cls = F.normalize(q_cls, dim=-1)
        k_patches = F.normalize(k_patches, dim=-1)
        
        k_similarity = torch.matmul(q_cls, k_patches.transpose(-2, -1)).squeeze()
        # Normalize to [0, 1]
        k_similarity = (k_similarity - k_similarity.min()) / (k_similarity.max() - k_similarity.min() + 1e-8)
        
        # Apply K-similarity weighting
        cls_attn = cls_attn * k_similarity[:num_patches]
    
    # 5. Compute importance: ReLU(Σ_c grad_c * v_i,c * attn_i)
    importance = (grad_cls * V_patches).sum(dim=-1)  # Channel-wise importance
    importance = importance * cls_attn  # Spatial weighting
    importance = torch.relu(importance)  # ReLU activation
    
    return importance


# ============================================================================ #
# Public API                                                                   #
# ============================================================================ #

def generate_heatmap(
    image: Union[str, Image.Image],
    sentence: str,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    *,
    layer_idx: int = -1,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
    resize: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    """
    Generate Grad-ECLIP heatmap overlay for image-text pair.
    
    Parameters
    ----------
    image : str or PIL.Image
        Input image path or PIL Image object
    sentence : str
        Text description to explain
    model : CLIPModel
        Pre-loaded CLIP model (possibly with LoRA adapter)
    processor : CLIPProcessor
        CLIP processor for preprocessing
    device : torch.device
        Computation device
    layer_idx : int, optional
        Which vision transformer layer to analyze (default: -1 for last layer)
    alpha : float, optional
        Heatmap overlay opacity (default: 0.45)
    colormap : int, optional
        OpenCV colormap for visualization (default: COLORMAP_JET)
    resize : tuple, optional
        Target (width, height) for output image
        
    Returns
    -------
    PIL.Image
        RGB image with heatmap overlay
    """
    # Load image if path provided
    if isinstance(image, str):
        pil_image = Image.open(image).convert('RGB')
    else:
        pil_image = image.convert('RGB')
    
    # Store original size
    orig_size = pil_image.size  # (width, height)
    
    # Apply resize if requested
    if resize:
        display_image = pil_image.resize(resize, Image.Resampling.BICUBIC)
    else:
        display_image = pil_image
    
    # Prepare inputs
    inputs = processor(
        images=pil_image,
        text=sentence,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Temporarily enable gradients
    model_requires_grad = [p.requires_grad for p in model.parameters()]
    for param in model.parameters():
        param.requires_grad = True
    
    try:
        # Forward and backward pass with hooks
        with torch.set_grad_enabled(True):
            with _GradECLIPHooks(model, layer_idx) as hooks:
                # Forward pass
                outputs = model(**inputs, output_attentions=False)
                
                # Get normalized embeddings
                image_embeds = F.normalize(outputs.image_embeds, dim=-1)
                text_embeds = F.normalize(outputs.text_embeds, dim=-1)
                
                # Compute similarity
                similarity = (image_embeds @ text_embeds.T).squeeze()
                
                # Backward pass
                model.zero_grad()
                similarity.backward(retain_graph=False)
                
                # Get captured tensors
                captures = hooks.get_captures()
        
        # Compute Grad-ECLIP importance
        importance = _compute_gradeclip_importance(
            captures,
            use_k_similarity=True,
            device=device
        )
        
        # Reshape to 2D grid
        num_patches = importance.shape[0]
        grid_size = int(np.sqrt(num_patches))
        importance_map = importance.reshape(grid_size, grid_size)
        
        # Convert to numpy and normalize
        saliency_map = importance_map.detach().cpu().numpy()
        saliency_map = saliency_map - saliency_map.min()
        saliency_map = saliency_map / (saliency_map.max() + 1e-8)
        
        # Resize saliency map to match display image
        saliency_resized = cv2.resize(
            saliency_map,
            display_image.size,  # (width, height)
            interpolation=cv2.INTER_CUBIC
        )
        
        # Apply colormap
        heatmap_uint8 = (saliency_resized * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        
        # Blend with original image
        img_array = np.array(display_image).astype(np.float32)
        overlay = (1 - alpha) * img_array + alpha * heatmap_rgb
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        return Image.fromarray(overlay, mode="RGB")
        
    finally:
        # Restore original gradient settings
        for param, requires_grad in zip(model.parameters(), model_requires_grad):
            param.requires_grad = requires_grad


__all__ = ["generate_heatmap"]