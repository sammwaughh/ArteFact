"""
heatmap.py
-----------

Create Grad‑ECLIP saliency overlays for CLIP / PaintingCLIP.

Public entry‑point
------------------
generate_heatmap(
    image,                 # str | PIL.Image.Image
    sentence,              # caption text
    model,                 # CLIPModel or PEFT‑wrapped model
    processor,             # CLIPProcessor
    device,                # torch.device
    *,
    layer_idx: int = -1,   # which visual transformer block to explain
    alpha: float = 0.45,   # overlay opacity
    colormap: int =  cv2.COLORMAP_JET,
) -> PIL.Image.Image       # RGB overlay the UI can send to the browser

The implementation is a faithful re‑creation of Grad‑ECLIP’s
“channel‑× spatial” weighting ⁠— see Section 3.2, Eq 19 of the
paper :contentReference[oaicite:2]{index=2}.
"""

from __future__ import annotations

import contextlib
import types
from typing import Callable, Tuple, Union, Optional, Iterator

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers.models.clip.modeling_clip import CLIPModel, CLIPVisionEmbeddings
from transformers import CLIPProcessor

# -----------------------------------------------------------------------------#
# Internal helpers                                                             #
# -----------------------------------------------------------------------------#

_AttnTensors = Tuple[torch.Tensor, torch.Tensor]  # (value (B,N,C), raw_qk (B,N,N))


def _select_block(
    model: CLIPModel,
    layer_idx: int,
) -> torch.nn.Module:
    """
    Return the `layer_idx`‑th transformer block inside the visual tower.
    Negative indices follow Python slice semantics.
    """
    blocks = model.vision_model.encoder.layers
    if layer_idx < 0:
        layer_idx += len(blocks)
    if layer_idx < 0 or layer_idx >= len(blocks):
        raise IndexError(f"layer_idx {layer_idx} out of range [0,{len(blocks)-1}]")
    return blocks[layer_idx]


@contextlib.contextmanager
def _grad_eclip_hooks(
    model: CLIPModel,
    layer_idx: int,
) -> Iterator[Callable[[], _AttnTensors]]:
    """
    Forward‑/backward‑hook manager that captures:
        • V  (values after projection)             – forward
        • raw QKᵀ/√d (before softmax)             – forward
        • ∂s/∂V   (channel gradients)              – backward
    Returns a getter that, after backward(), yields needed tensors.
    """
    block = _select_block(model, layer_idx)
    holder: dict[str, torch.Tensor] = {}

    def fwd_hook(module, inputs, output):
        # module: CLIPEncoderLayer
        # output[0] == hidden_states, output[1] == attn_weights (if asked)
        # We ensure `output_attentions=True` in forward() call later.
        hidden, attn_weights = output
        # save V (value vectors)   shape: (B, N, C)
        holder["v"] = module.self_attn.v_proj(
            hidden
        ).detach()  # store detached copy to avoid interfering with autograd
        # save raw qk pre‑softmax   shape: (B, N, N)
        holder["raw_qk"] = attn_weights.detach()

    def bwd_hook(module, grad_input, grad_output):
        # grad_output[0] corresponds to grad wrt hidden_states
        holder["v_grad"] = module.self_attn.v_proj(
            grad_output[0]
        ).detach()

    h1 = block.register_forward_hook(fwd_hook, with_kwargs=False)
    h2 = block.register_full_backward_hook(bwd_hook)

    try:
        yield lambda: (
            holder["v"],
            holder["raw_qk"],
            holder["v_grad"],
        )
    finally:
        h1.remove()
        h2.remove()
        holder.clear()


# -----------------------------------------------------------------------------#
# Public API                                                                   #
# -----------------------------------------------------------------------------#


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
    Compute a Grad‑ECLIP heat‑map for (image, sentence) and blend it
    over the original picture.

    Parameters
    ----------
    image:
        Path or already‑loaded PIL image.
    sentence:
        Caption query.
    model / processor / device:
        Re‑use the cached objects from `inference._initialize_pipeline`.
    layer_idx:
        Index of vision transformer block to explain (default: last).
    alpha:
        Heat‑map opacity in overlay.
    colormap:
        OpenCV colormap code.
    resize:
        Optional target size (W,H) for output overlay; if None, keep original.

    Returns
    -------
    overlay : PIL.Image.Image
        RGB image with heat‑map blended for immediate saving / streaming.
    """
    # ------------------------------------------------------------#
    # 1. Pre‑process inputs                                       #
    # ------------------------------------------------------------#
    if isinstance(image, str):
        pil = Image.open(image).convert("RGB")
    else:
        pil = image.copy()

    if resize:
        pil = pil.resize(resize, Image.BICUBIC)

    # Call order “images first, then text” avoids secondary kwargs filtering bug
    inputs = processor(images=pil, text=[sentence], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ------------------------------------------------------------#
    # 2. Forward + backward with hooks                            #
    # ------------------------------------------------------------#
    # temporarily switch parameters to require_grad=True
    prev = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(True)

    model.eval()
    with torch.set_grad_enabled(True), _grad_eclip_hooks(model, layer_idx) as get_tensors:
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
        image_emb = F.normalize(outputs.image_embeds, dim=-1)  # (1,512)
        text_emb = F.normalize(outputs.text_embeds, dim=-1)    # (1,512)

        # similarity scalar
        sim = (image_emb @ text_emb.T).squeeze()
        # backward to populate gradients captured by hook
        model.zero_grad(set_to_none=True)
        sim.backward(retain_graph=False)

        v, raw_qk, v_grad = get_tensors()  # shapes: (1,N,C), (1,N,N), (1,N,C)

    # restore original requires_grad flags
    for flag, param in zip(prev, model.parameters()):
        param.requires_grad_(flag)

    # ------------------------------------------------------------#
    # 3. Channel & spatial importance (Eq 19)                     #
    # ------------------------------------------------------------#
    # channel weights wc – gradients for CLS row
    # CLS token is position 0
    wc = v_grad[0, 0]  # (C,)

    # loosened spatial weights λᵢ – 0‑1 norm of raw_qk CLS row
    cls_row = raw_qk[0, 0]  # (N,)
    lam = (cls_row - cls_row.min()) / (cls_row.max() - cls_row.min() + 1e-8)

    # feature map (exclude CLS) → (N-1,C)
    v_patches = v[0, 1:]                 # (M,C)

    # Ensure the λ vector has the same length as the patch sequence.
    # Some CLIP checkpoints include an extra token that appears in raw_qk
    # but not in the value tensor, which creates a 50 vs 49 size mismatch.
    # `v_patches` has shape (M,C).  Keep only the first M spatial weights.
    num_patch_tokens = v_patches.size(0)
    lam_patches = lam.narrow(0, 1, num_patch_tokens)   # (M,)

    # importance per patch: ReLU( Σ_c wc * λ_i * v_i,c )
    imp = torch.relu((v_patches * wc).sum(dim=-1) * lam_patches)  # (N-1,)

    # ------------------------------------------------------------#
    # 4. Reshape to patch grid & upscale                          #
    # ------------------------------------------------------------#
    num_patches = imp.shape[0]
    side = int(np.sqrt(num_patches))
    heat = imp.view(side, side).cpu().numpy()
    heat = cv2.resize(heat, pil.size, interpolation=cv2.INTER_CUBIC)
    heat = heat - heat.min()
    heat = heat / (heat.max() + 1e-8)
    heat_uint8 = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat_uint8, colormap)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------#
    # 5. Blend & return                                           #
    # ------------------------------------------------------------#
    orig = np.asarray(pil).astype(np.float32)
    overlay = (1 - alpha) * orig + alpha * heat_color
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay, mode="RGB")


__all__ = ["generate_heatmap"]
