"""
Wraps PaintingCLIP + LoRA (stubbed for now).

`run_inference` MUST stay pure: take a local image path, return
JSON‑serialisable Python objects.  That makes it easy to unit‑test.
"""

from pathlib import Path
from typing import List, Dict

# ----------------------------------------------------------------------
def run_inference(image_path: str) -> List[Dict]:
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
    # TODO: load PaintingCLIP once, run encode + FAISS search, etc.
    # For skeleton we emit a single stub label.
    return [
        {
            "label": "stub‑label",
            "score": 0.99,
            "evidence": {"note": "replace with real provenance"},
        }
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
