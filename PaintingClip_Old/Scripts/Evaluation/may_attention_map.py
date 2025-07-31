import time, textwrap, torch, pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import zoom, gaussian_filter
from transformers import CLIPProcessor, CLIPModel

# ── 1. MODEL / INPUT (unchanged) ────────────────────────────────────────────────
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

image_path = "bacchus-and-ariadne.jpg"
text_input = (
    "Bacchus' chariot is normally drawn by tigers or panthers, but Alfonso d'Este "
    "is known to have had a menagerie at the palace in which he kept a cheetah or "
    "a cheetah-like member of the cat family."
)
threshold = 0.7
image = Image.open(image_path)

inputs = processor(text=[text_input], images=image, return_tensors="pt", padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

gradients = None


def save_grads(_m, _gi, go):  # hook for token gradients
    global gradients
    gradients = go[0]


hook = model.text_model.embeddings.register_full_backward_hook(save_grads)
inputs["pixel_values"].requires_grad_(True)

logits = model(**inputs).logits_per_image
logits[0, 0].backward()
hook.remove()

token_grads = gradients.norm(dim=-1).cpu().numpy().flatten()
tokens = processor.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# aggregate sub-words
agg = {}
word = ""
for tok, g in zip(tokens, token_grads):
    if tok in ("<|startoftext|>", "<|endoftext|>"):
        continue
    if tok.endswith("</w>"):
        tok = tok.replace("</w>", "")
        word += tok
        agg[word] = agg.get(word, 0) + g
        word = ""
    else:
        word += tok
if word:
    agg[word] = agg.get(word, 0)

# rank by saliency
ranking = sorted(agg.items(), key=lambda x: x[1], reverse=True)
df = pd.DataFrame(ranking, columns=["word", "saliency"])
df.to_excel("attention-rank.xlsx", index=False)

# ── 3. IMAGE HEAT MAP ──────────────────
grads_img = inputs["pixel_values"].grad.mean(dim=1).squeeze().cpu().numpy()
grads_img = np.log1p(grads_img)
grads_img -= grads_img.min()
grads_img /= grads_img.max()
grads_img[grads_img < threshold] = 0

N = grads_img.size
gsz = int(np.sqrt(N))
if gsz**2 != N:
    raise ValueError("Gradient grid is not square.")
grads_img = gaussian_filter(zoom(grads_img.reshape(gsz, gsz), zoom=32), sigma=2)
grads_img -= grads_img.min()
grads_img /= grads_img.max()

image_res = image.resize(grads_img.shape[::-1])

# ── 4. OUTPUT ───────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 8))
plt.imshow(image_res, alpha=0.6)
plt.imshow(grads_img, cmap="viridis", alpha=0.4)
plt.axis("off")
plt.title("Attention heat-map overlay")

base = image_path.rsplit("/", 1)[-1].split(".")[0]
stamp = time.strftime("%H-%M-%S")
outimg = f"{base}_{stamp}_overlay.png"
plt.savefig(outimg, bbox_inches="tight")
plt.close()

print("\nSentence:")
print(textwrap.fill(text_input, width=80))
print(f"\nHeat-map saved to  {outimg}")
print("Word-saliency ranking saved to  attention-rank.xlsx")
