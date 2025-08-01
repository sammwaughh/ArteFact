#!/usr/bin/env python3
"""
grad_eclip.py
=============

Clean implementation of Grad-ECLIP for CLIP models.
Generates visual explanations showing which image regions contribute
to image-text similarity.

Usage:
    python grad_eclip.py

Based on: "Gradient-based Visual Explanation for Transformer-based CLIP"
by Zhao et al. (ICML 2024)
"""

import json  # Add this for pretty printing
import warnings
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Suppress the scaled_dot_product_attention warning
warnings.filterwarnings("ignore", message=".*scaled_dot_product_attention.*")


# ============================================================================ #
# CONFIGURATION                                                                #
# ============================================================================ #
IMAGE_PATH = "Test-Images/kiss_of_judas.jpg"
# SENTENCE = "Judas is kissing Jesus in the centre of the painting"
SENTENCE = "Fiery red torches are held in the sky"
OUTPUT_PATH = "test-outputs/grad_eclip_output6.png"

# Model configuration
MODEL_NAME = "openai/clip-vit-base-patch32"
LAYER_INDEX = -1  # Which layer to analyze (-1 = last)
USE_K_SIMILARITY = True  # Whether to use Q-K similarity weighting
ALPHA = 0.45  # Heatmap opacity
COLORMAP = cv2.COLORMAP_JET

DEBUG = True  # Set to False to disable debug output


# ============================================================================ #
# Core Grad-ECLIP Implementation                                               #
# ============================================================================ #


class GradECLIP:
    """
    Grad-ECLIP: Gradient-based Explainability for CLIP

    Implements the method from "Gradient-based Visual Explanation for
    Transformer-based CLIP" (Zhao et al., ICML 2024)
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """Initialize CLIP model and processor."""
        self.device = self._get_device()
        print(f"Loading model: {model_name}")

        # Load with use_fast=False to avoid compatibility issues
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        print(f"Model loaded on: {self.device}")

    def _get_device(self) -> torch.device:
        """Get the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def debug_print(self, message: str, data=None, level: int = 0):
        """Helper function for debug printing with indentation."""
        if DEBUG:
            indent = "  " * level
            print(f"{indent}[DEBUG] {message}")
            if data is not None:
                if isinstance(data, torch.Tensor):
                    print(
                        f"{indent}  Shape: {data.shape}, Device: {data.device}, Dtype: {data.dtype}"
                    )
                    # Only compute statistics for floating point tensors
                    if data.dtype in [torch.float16, torch.float32, torch.float64]:
                        print(
                            f"{indent}  Range: [{data.min().item():.4f}, {data.max().item():.4f}]"
                        )
                        print(
                            f"{indent}  Mean: {data.mean().item():.4f}, Std: {data.std().item():.4f}"
                        )
                    else:
                        # For integer tensors, just show range
                        print(
                            f"{indent}  Range: [{data.min().item()}, {data.max().item()}]"
                        )
                elif isinstance(data, dict):
                    for k, v in data.items():
                        print(f"{indent}  {k}: {v}")
                else:
                    print(f"{indent}  {data}")

    def _forward_with_hooks(
        self, inputs: Dict[str, torch.Tensor], layer_idx: int = -1
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Enhanced forward pass with better attention capture."""
        self.debug_print("Starting _forward_with_hooks", level=1)
        self.debug_print(f"Target layer index: {layer_idx}", level=1)

        captures = {}
        handles = []

        # Get the target vision layer
        vision_layers = self.model.vision_model.encoder.layers
        if layer_idx < 0:
            layer_idx = len(vision_layers) + layer_idx
        target_layer = vision_layers[layer_idx]
        self.debug_print(
            f"Using layer {layer_idx} of {len(vision_layers)} total layers", level=1
        )

        # Hook for the encoder layer to manually compute attention
        def forward_hook_layer(module, input, output):
            self.debug_print("Layer forward hook triggered", level=2)

            if len(input) > 0:
                hidden_states = input[0]
                self.debug_print("Input hidden states", hidden_states, level=3)

                # Get normalized hidden states for attention
                x = (
                    module.self_attn.layer_norm(hidden_states)
                    if hasattr(module.self_attn, "layer_norm")
                    else hidden_states
                )

                # Compute Q, K, V using the attention module's projections
                if hasattr(module.self_attn, "q_proj"):
                    batch_size, seq_len, hidden_dim = x.shape

                    Q = module.self_attn.q_proj(x)
                    K = module.self_attn.k_proj(x)
                    V = module.self_attn.v_proj(x)

                    self.debug_print("Q projection", Q, level=3)
                    self.debug_print("K projection", K, level=3)
                    self.debug_print("V projection", V, level=3)

                    # Store V in its original shape
                    captures["V"] = V
                    captures["hidden_states_pre"] = hidden_states

                    # Manually compute attention
                    head_dim = hidden_dim // module.self_attn.num_heads
                    num_heads = module.self_attn.num_heads

                    # Reshape for attention computation
                    Q = Q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                    K = K.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                    V_heads = V.view(
                        batch_size, seq_len, num_heads, head_dim
                    ).transpose(1, 2)

                    # Compute attention weights
                    scale = head_dim**-0.5
                    attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
                    attn_weights = torch.softmax(attn_weights, dim=-1)

                    # Apply attention to values
                    attn_output = torch.matmul(attn_weights, V_heads)
                    attn_output = (
                        attn_output.transpose(1, 2)
                        .contiguous()
                        .view(batch_size, seq_len, hidden_dim)
                    )

                    # Store attention-related tensors
                    captures["Q"] = Q
                    captures["K"] = K
                    captures["attn_weights"] = attn_weights.mean(
                        dim=1
                    )  # Average over heads
                    captures["attn_output"] = attn_output

                    self.debug_print(
                        f"Attention weights shape: {attn_weights.shape}", level=3
                    )
                    self.debug_print(
                        f"Attention output shape: {attn_output.shape}", level=3
                    )

            # Store the layer output
            hidden_states_out = output[0]
            captures["hidden_states"] = hidden_states_out
            self.debug_print(
                "Output hidden states captured", hidden_states_out, level=3
            )

        # Backward hook for gradients on attention output
        def backward_hook_attn(module, grad_input, grad_output):
            self.debug_print("Backward hook triggered", level=2)

            if "attn_output" in captures:
                # We need to capture gradients w.r.t attention output
                # Since we computed it manually, we'll use the layer output gradient
                if len(grad_output) > 0:
                    grad_hidden = grad_output[0]
                    captures["grad_attn"] = grad_hidden
                    self.debug_print("Gradient captured", grad_hidden, level=3)

        # Register hooks
        h1 = target_layer.register_forward_hook(forward_hook_layer)
        h2 = target_layer.register_full_backward_hook(backward_hook_attn)
        handles = [h1, h2]
        self.debug_print("Hooks registered", level=1)

        try:
            # Forward pass - need to track attention output for gradients
            captures["attn_output_ref"] = []

            def track_attn_output(module, input, output):
                if hasattr(output, "requires_grad"):
                    output.retain_grad()
                captures["attn_output_ref"].append(output)
                return output

            # Temporarily add hook to track attention output
            if hasattr(target_layer.self_attn, "register_forward_hook"):
                h3 = target_layer.self_attn.register_forward_hook(track_attn_output)
                handles.append(h3)

            outputs = self.model(**inputs, output_attentions=False)
            self.debug_print("Forward pass complete", level=1)
            self.debug_print(f"Captures keys: {list(captures.keys())}", level=1)
            return outputs, captures
        finally:
            # Clean up
            for h in handles:
                h.remove()
            self.debug_print("Hooks removed", level=1)

    def compute_gradeclip_map(
        self,
        image_path: str,
        text: str,
        layer_idx: int = -1,
        use_k_similarity: bool = True,
    ) -> np.ndarray:
        """
        Compute Grad-ECLIP saliency map.

        Args:
            image_path: Path to input image
            text: Text description
            layer_idx: Which layer to analyze
            use_k_similarity: Whether to use Q-K similarity weighting

        Returns:
            Saliency map as 2D numpy array
        """
        self.debug_print(f"=== Starting compute_gradeclip_map ===")
        self.debug_print(f"Image: {image_path}")
        self.debug_print(f"Text: '{text}'")
        self.debug_print(
            f"Layer index: {layer_idx}, Use K-similarity: {use_k_similarity}"
        )

        # Prepare inputs
        image = Image.open(image_path).convert("RGB")
        self.debug_print(f"Image loaded: {image.size}")

        inputs = self.processor(
            images=image, text=text, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        self.debug_print(f"Inputs prepared: {list(inputs.keys())}")

        # Enable gradients temporarily
        for param in self.model.parameters():
            param.requires_grad = True
        self.debug_print("Gradients enabled")

        try:
            # Forward pass with hooks
            with torch.set_grad_enabled(True):
                outputs, captures = self._forward_with_hooks(inputs, layer_idx)

                # Get normalized embeddings
                image_embeds = F.normalize(outputs.image_embeds, dim=-1)
                text_embeds = F.normalize(outputs.text_embeds, dim=-1)
                self.debug_print("Image embeddings", image_embeds)
                self.debug_print("Text embeddings", text_embeds)

                # Compute similarity
                similarity = (image_embeds @ text_embeds.T).squeeze()
                self.debug_print(f"Similarity score: {similarity.item():.4f}")

                # Backward pass
                self.model.zero_grad()

                # Complete backward pass to get gradients
                similarity.backward(retain_graph=False)
                self.debug_print("Backward pass complete")

                # Extract captured tensors
                V = captures.get("V", None)
                grad_attn = captures.get("grad_attn", None)
                attn_weights = captures.get("attn_weights", None)

                if V is None or grad_attn is None:
                    self.debug_print("WARNING: Missing required captures!")
                    self.debug_print(f"Available captures: {list(captures.keys())}")
                    return np.zeros((7, 7))  # Return empty map

                self.debug_print("V tensor", V)
                self.debug_print("grad_attn tensor", grad_attn)

                # Get attention weights for spatial importance
                if attn_weights is not None:
                    cls_attn = attn_weights[0, 0, :]  # CLS to all tokens
                    self.debug_print("Using captured attention weights", cls_attn)
                else:
                    # Fallback: use cosine similarity
                    if "hidden_states_pre" in captures:
                        hidden = captures["hidden_states_pre"]
                        cls_hidden = hidden[0, 0:1, :]  # CLS token
                        patch_hidden = hidden[0, :, :]  # All tokens

                        # Compute cosine similarity
                        cls_norm = F.normalize(cls_hidden, dim=-1)
                        patch_norm = F.normalize(patch_hidden, dim=-1)
                        cls_attn = (cls_norm @ patch_norm.T).squeeze(0)
                        cls_attn = torch.softmax(
                            cls_attn * 5.0, dim=-1
                        )  # Temperature scaling
                        self.debug_print(
                            "Using cosine similarity attention (fallback)", cls_attn
                        )
                    else:
                        # Create non-uniform weights
                        num_tokens = V.shape[1] if V is not None else 50
                        cls_attn = torch.ones(num_tokens, device=self.device)
                        cls_attn[0] = 5.0  # Higher weight for CLS
                        cls_attn = cls_attn / cls_attn.sum()
                        self.debug_print("Using default attention weights", cls_attn)

                # Compute Grad-ECLIP components
                # 1. Channel importance: gradients at CLS token
                grad_cls = grad_attn[0, 0, :]  # Gradient at CLS position
                self.debug_print("Gradient at CLS (grad_cls)", grad_cls)

                # 2. Extract patch values and attention
                V_patches = V[0, 1:, :]  # Exclude CLS token
                num_patches = V_patches.shape[0]
                attn_patches = cls_attn[
                    1 : num_patches + 1
                ]  # Attention from CLS to patches

                self.debug_print(f"Number of patches: {num_patches}")
                self.debug_print("V_patches", V_patches)
                self.debug_print("Attention to patches", attn_patches)

                # 3. Optional: K-similarity normalization
                if use_k_similarity and "Q" in captures and "K" in captures:
                    Q = captures["Q"]
                    K = captures["K"]
                    self.debug_print("Computing Q-K similarity")

                    # Get CLS token query
                    q_cls = Q[:, :, 0:1, :].mean(dim=1)  # Average over heads
                    k_patches = K[:, :, 1:, :].mean(dim=1)

                    # Normalize and compute cosine similarity
                    q_cls = F.normalize(q_cls, dim=-1)
                    k_patches = F.normalize(k_patches, dim=-1)

                    k_similarity = torch.matmul(
                        q_cls, k_patches.transpose(-2, -1)
                    ).squeeze()
                    k_similarity = (k_similarity - k_similarity.min()) / (
                        k_similarity.max() - k_similarity.min() + 1e-8
                    )
                    self.debug_print("K-similarity", k_similarity)

                    # Apply K-similarity weighting to attention
                    attn_patches = attn_patches * k_similarity[:num_patches]
                    self.debug_print("Attention after K-sim weighting", attn_patches)

                # 4. Compute importance: ReLU(Σ_c grad_c * v_i,c * attn_i)
                # This is the key Grad-ECLIP formula
                importance = (grad_cls * V_patches).sum(
                    dim=-1
                )  # Element-wise multiply and sum over channels
                importance = importance * attn_patches  # Weight by attention
                importance = torch.relu(importance)  # ReLU activation

                self.debug_print(
                    "Importance (before ReLU)", (grad_cls * V_patches).sum(dim=-1)
                )
                self.debug_print("Importance (after attention weighting)", importance)

                # 5. Reshape to 2D grid
                grid_size = int(np.sqrt(num_patches))
                importance_map = importance.reshape(grid_size, grid_size)
                self.debug_print(f"Final importance map shape: {importance_map.shape}")
                self.debug_print("Importance map", importance_map)

                return importance_map.detach().cpu().numpy()

        finally:
            # Restore gradient settings
            for param in self.model.parameters():
                param.requires_grad = False
            self.debug_print("Gradients disabled")
            self.debug_print("=== compute_gradeclip_map complete ===\n")

    def create_visualization(
        self,
        image_path: str,
        saliency_map: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> Image.Image:
        """
        Create heatmap overlay visualization.

        Args:
            image_path: Path to original image
            saliency_map: 2D saliency map
            alpha: Overlay transparency
            colormap: OpenCV colormap

        Returns:
            PIL Image with heatmap overlay
        """
        # Load original image
        original = Image.open(image_path).convert("RGB")
        orig_size = original.size  # (width, height)

        # Normalize saliency map
        saliency_norm = saliency_map - saliency_map.min()
        saliency_norm = saliency_norm / (saliency_norm.max() + 1e-8)

        # Resize to match image
        saliency_resized = cv2.resize(
            saliency_norm, orig_size, interpolation=cv2.INTER_CUBIC
        )

        # Apply colormap
        heatmap_uint8 = (saliency_resized * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Blend with original
        orig_array = np.array(original).astype(np.float32)
        overlay = (1 - alpha) * orig_array + alpha * heatmap_rgb
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        return Image.fromarray(overlay)


# ============================================================================ #
# Main execution                                                               #
# ============================================================================ #


def main():
    """Run Grad-ECLIP visualization."""
    print("=" * 60)
    print("Grad-ECLIP: Visual Explanations for CLIP")
    print("=" * 60)
    print(f"Image: {IMAGE_PATH}")
    print(f"Text: '{SENTENCE}'")
    print(f"Output: {OUTPUT_PATH}")
    print()

    # Initialize
    gradeclip = GradECLIP(MODEL_NAME)

    # Compute saliency map
    print("Computing Grad-ECLIP saliency map...")
    saliency = gradeclip.compute_gradeclip_map(
        IMAGE_PATH, SENTENCE, layer_idx=LAYER_INDEX, use_k_similarity=USE_K_SIMILARITY
    )

    # Create visualization
    print("Creating visualization...")
    visualization = gradeclip.create_visualization(
        IMAGE_PATH, saliency, alpha=ALPHA, colormap=COLORMAP
    )

    # Save output
    visualization.save(OUTPUT_PATH)
    print(f"✓ Saved to: {OUTPUT_PATH}")

    # Print statistics
    print(f"\nSaliency map statistics:")
    print(f"  Shape: {saliency.shape}")

    # Raw values with scientific notation if needed
    min_val, max_val = saliency.min(), saliency.max()
    if max_val < 0.01:
        print(f"  Raw range: [{min_val:.2e}, {max_val:.2e}]")
    else:
        print(f"  Raw range: [{min_val:.4f}, {max_val:.4f}]")

    # Normalized statistics (what's actually visualized)
    saliency_norm = saliency - min_val
    if max_val > min_val:
        saliency_norm = saliency_norm / (max_val - min_val)
    print(f"  Normalized range: [0.0000, 1.0000]")

    # Distribution statistics
    print(f"  Mean: {saliency.mean():.2e}")
    print(f"  Std: {saliency.std():.2e}")

    # Additional helpful statistics
    nonzero_count = np.count_nonzero(saliency)
    total_count = saliency.size
    print(
        f"  Non-zero pixels: {nonzero_count}/{total_count} ({100*nonzero_count/total_count:.1f}%)"
    )

    # Peak location
    peak_idx = np.unravel_index(np.argmax(saliency), saliency.shape)
    print(f"  Peak location: {peak_idx}")


if __name__ == "__main__":
    main()
