#!/usr/bin/env python3
# test_heatmap.py
"""
Smoke-test for the Grad-ECLIP pipeline.

Usage
-----
python test_heatmap.py  --run-id  d72545f1...  \
                        --sentence "The men held torches with large flames above their heads" \
                        --out overlay.png

Options
-------
--run-id        ID previously returned by /presign (maps to artifacts/<id>.jpg)
--image-path    Direct path to an RGB image (skip --run-id)
--sentence      Caption text to explain (required)
--layer-idx     Vision transformer block (default -1 = last)
--out           Output PNG file (default overlay.png)
"""

import argparse
import base64
import os
from pathlib import Path

from hc_services.runner.inference import compute_heatmap

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-id", help="Existing run id → artifacts/<id>.jpg")
    g.add_argument("--image-path", help="Path to any local image")

    parser.add_argument("--sentence", required=True, help="Caption text")
    parser.add_argument("--layer-idx", type=int, default=-1, help="VT block")
    parser.add_argument("--out", default="overlay.png", help="Output PNG")

    args = parser.parse_args()

    if args.run_id:
        img_path = ARTIFACTS_DIR / f"{args.run_id}.jpg"
    else:
        img_path = Path(args.image_path).expanduser()

    if not img_path.exists():
        raise SystemExit(f"[ERROR] image not found: {img_path}")

    print(f"[INFO] Computing heat-map for\n  image   = {img_path}\n  sentence= {args.sentence}")

    data_url = compute_heatmap(str(img_path), args.sentence, layer_idx=args.layer_idx)
    _, b64 = data_url.split(",", 1)

    out_file = Path(args.out)
    out_file.write_bytes(base64.b64decode(b64))
    print(f"[OK] Saved overlay to {out_file.absolute()}")


if __name__ == "__main__":
    main()