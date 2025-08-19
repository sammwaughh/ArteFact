#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "embeddings"
CLIP_DIR = DATA_DIR / "CLIP_Embeddings"
PAINTINGCLIP_DIR = DATA_DIR / "PaintingCLIP_Embeddings"

def load_one(pt_path: Path) -> torch.Tensor:
	"""Load a single .pt embedding, handling dict-or-tensor variants."""
	obj = torch.load(pt_path, map_location="cpu", weights_only=True)
	if isinstance(obj, torch.Tensor):
		return obj
	if isinstance(obj, dict):
		for k in ("embedding", "embeddings", "features"):
			if k in obj:
				t = obj[k]
				if isinstance(t, torch.Tensor):
					return t
	raise ValueError(f"Unsupported .pt content in {pt_path}")

def derive_id_from_filename(stem: str) -> str:
	"""
	- CLIP:       Wxxxx_sYYYY_clip → Wxxxx_sYYYY
	- PaintingCLIP: Wxxxx_sYYYY_painting_clip → Wxxxx_sYYYY
	"""
	if stem.endswith("_painting_clip"):
		return stem[: -len("_painting_clip")]
	if stem.endswith("_clip"):
		return stem[: -len("_clip")]
	return stem  # fallback

def consolidate_dir(indir: Path) -> Tuple[torch.Tensor, List[str]]:
	pt_files = sorted(indir.glob("*.pt"))
	if not pt_files:
		raise RuntimeError(f"No .pt files found under {indir}")

	embs: List[torch.Tensor] = []
	ids: List[str] = []

	for i, p in enumerate(pt_files, 1):
		e = load_one(p).float()
		if e.ndim > 1:
			e = e.squeeze()
		if e.ndim != 1:
			raise ValueError(f"Embedding is not 1D in {p}: shape={tuple(e.shape)}")
		embs.append(e)
		ids.append(derive_id_from_filename(p.stem))
		if i % 1000 == 0:
			print(f"... processed {i} files from {indir}")

	# Stack to [N, D]
	embeddings = torch.stack(embs, dim=0).contiguous()
	return embeddings, ids

def save_as_safetensors(embeddings: torch.Tensor, ids: List[str], out_prefix: Path) -> None:
	out_st = out_prefix.with_suffix(".safetensors")
	out_json = out_prefix.with_name(out_prefix.name + "_sentence_ids.json")
	save_file({"embeddings": embeddings}, str(out_st))
	with open(out_json, "w", encoding="utf-8") as f:
		json.dump(ids, f, ensure_ascii=False, indent=2)
	print(f"Saved embeddings: {out_st} [{tuple(embeddings.shape)}]")
	print(f"Saved sentence IDs: {out_json} [{len(ids)} ids]")

def main():
	print("Consolidating CLIP...")
	clip_emb, clip_ids = consolidate_dir(CLIP_DIR)
	save_as_safetensors(clip_emb, clip_ids, DATA_DIR / "clip_embeddings")

	print("Consolidating PaintingCLIP...")
	pclip_emb, pclip_ids = consolidate_dir(PAINTINGCLIP_DIR)
	save_as_safetensors(pclip_emb, pclip_ids, DATA_DIR / "paintingclip_embeddings")

if __name__ == "__main__":
	main()