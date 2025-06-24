"""
Wraps PaintingCLIP + LoRA (stubbed for now).

`run_inference` MUST stay pure: take a local image path, return
JSON‑serialisable Python objects.  That makes it easy to unit‑test.
"""

from pathlib import Path
from typing import List, Dict, Any  # add Any


# ----------------------------------------------------------------------
def run_inference(image_path: str) -> List[Dict[str, Any]]:
    """
    Parameters
    ----------
    image_path : str
        Absolute or relative path to the JPEG/PNG file on disk.

    Returns
    -------
    list[dict]
        [{ "label": str, "score": float, "evidence": dict }, …]
    """
    # TODO: load PaintingCLIP once, run encode + FAISS search, etc.
    # Temporary stub: return three labels so we can confirm the UI
    # renders multiple results correctly.  Replace with real model
    # outputs in a later step.
    return [
        {
            "label": "leaf motif",
            "score": 0.92,
            "evidence": {"note": "dummy evidence – will be CLIP-based"},
        },
        {
            "label": "gilded frame",
            "score": 0.85,
            "evidence": {"note": "dummy evidence – will be CLIP-based"},
        },
        {
            "label": "Renaissance style",
            "score": 0.77,
            "evidence": {"note": "dummy evidence – will be CLIP-based"},
        },
    ]


# Helper(s) for later ---------------------------------------------------
def _ensure_asset(s3_key: str, dest: Path) -> Path:
    """
    Downloads `s3_key` from MODEL_BUCKET to `dest` if not already present.
    To be filled in once we wire S3 + EFS.
    """
    # Placeholder – keeps import errors away during phase A dev.
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.write_text("TODO – downloaded model asset")
    return dest
