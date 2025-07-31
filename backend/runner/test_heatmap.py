#!/usr/bin/env python3
"""
test_heatmap.py
===============

Test script for the Grad-ECLIP heatmap generation pipeline.

This script allows you to test the heatmap generation functionality either
with images from the artifacts directory (using run IDs) or with any local image.

Usage Examples
--------------
# Using a run ID from artifacts directory:
python test_heatmap.py --run-id d72545f1-xxxx-xxxx-xxxx \
                       --sentence "The men held torches with large flames above their heads" \
                       --out torch_heatmap.png

# Using a direct image path:
python test_heatmap.py --image-path ~/Pictures/painting.jpg \
                       --sentence "A beautiful sunset over the mountains" \
                       --layer-idx -2 \
                       --out sunset_heatmap.png

# Minimal usage with defaults:
python test_heatmap.py --image-path test.jpg --sentence "A cat sitting on a mat"

Arguments
---------
--run-id : str
    Run ID from a previous /presign call (maps to artifacts/<id>.jpg)
    
--image-path : str
    Direct path to an RGB image file
    
--sentence : str
    Text description to generate explanation for (required)
    
--layer-idx : int
    Which vision transformer layer to analyze (default: -1 for last layer)
    
--out : str
    Output PNG filename (default: overlay.png)
    
--alpha : float
    Heatmap overlay opacity, between 0 and 1 (default: 0.45)
    
--colormap : str
    Color scheme for heatmap: 'jet', 'hot', 'viridis', 'plasma' (default: 'jet')
"""

import argparse
import base64
import sys
from pathlib import Path
from typing import Optional

import cv2

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from hc_services.runner.inference import compute_heatmap


# Colormap name to OpenCV constant mapping
COLORMAP_OPTIONS = {
    "jet": cv2.COLORMAP_JET,
    "hot": cv2.COLORMAP_HOT,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "plasma": cv2.COLORMAP_PLASMA,
    "cool": cv2.COLORMAP_COOL,
    "rainbow": cv2.COLORMAP_RAINBOW,
}


def resolve_image_path(run_id: Optional[str], image_path: Optional[str]) -> Path:
    """
    Resolve the image path from either a run ID or direct path.

    Args:
        run_id: Optional run ID that maps to artifacts/<id>.jpg
        image_path: Optional direct path to image

    Returns:
        Resolved Path object

    Raises:
        FileNotFoundError: If the resolved path doesn't exist
    """
    if run_id:
        # Get artifacts directory relative to this script
        artifacts_dir = Path(__file__).resolve().parent / "artifacts"
        resolved_path = artifacts_dir / f"{run_id}.jpg"
    else:
        resolved_path = Path(image_path).expanduser().resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(f"Image not found: {resolved_path}")

    if not resolved_path.is_file():
        raise ValueError(f"Path is not a file: {resolved_path}")

    return resolved_path


def save_heatmap_from_data_url(data_url: str, output_path: Path) -> None:
    """
    Extract and save PNG image from base64 data URL.

    Args:
        data_url: Data URL containing base64-encoded PNG
        output_path: Where to save the decoded image
    """
    if not data_url.startswith("data:image/png;base64,"):
        raise ValueError("Expected PNG data URL")

    # Extract base64 portion
    _, base64_data = data_url.split(",", 1)

    # Decode and save
    image_bytes = base64.b64decode(base64_data)
    output_path.write_bytes(image_bytes)


def main() -> None:
    """Main entry point for the test script."""
    parser = argparse.ArgumentParser(
        description="Test Grad-ECLIP heatmap generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Image source (mutually exclusive)
    image_group = parser.add_mutually_exclusive_group(required=True)
    image_group.add_argument(
        "--run-id",
        help="Run ID from previous /presign call (looks in artifacts/<id>.jpg)",
    )
    image_group.add_argument("--image-path", help="Direct path to an RGB image file")

    # Required arguments
    parser.add_argument("--sentence", required=True, help="Text description to explain")

    # Optional arguments
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=-1,
        help="Vision transformer layer index (default: -1 for last layer)",
    )
    parser.add_argument(
        "--out",
        default="overlay.png",
        help="Output PNG filename (default: overlay.png)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Heatmap overlay opacity 0-1 (default: 0.45)",
    )
    parser.add_argument(
        "--colormap",
        choices=list(COLORMAP_OPTIONS.keys()),
        default="jet",
        help="Color scheme for heatmap (default: jet)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print additional debug information"
    )

    args = parser.parse_args()

    # Validate arguments
    if not 0 <= args.alpha <= 1:
        parser.error("--alpha must be between 0 and 1")

    try:
        # Resolve image path
        image_path = resolve_image_path(args.run_id, args.image_path)

        if args.verbose:
            print(f"[DEBUG] Resolved image path: {image_path}")
            print(f"[DEBUG] Layer index: {args.layer_idx}")
            print(f"[DEBUG] Alpha: {args.alpha}")
            print(f"[DEBUG] Colormap: {args.colormap}")

        # Print info
        print(f"[INFO] Computing heatmap for:")
        print(f"       Image:    {image_path}")
        print(f"       Sentence: '{args.sentence}'")
        print(f"       Output:   {args.out}")

        # Generate heatmap
        print("[INFO] Generating heatmap...")
        data_url = compute_heatmap(
            str(image_path),
            args.sentence,
            layer_idx=args.layer_idx,
            alpha=args.alpha,
            colormap=COLORMAP_OPTIONS[args.colormap],
        )

        # Save output
        output_path = Path(args.out)
        save_heatmap_from_data_url(data_url, output_path)

        print(f"[SUCCESS] Saved heatmap overlay to: {output_path.absolute()}")

        # Print file info if verbose
        if args.verbose:
            file_size = output_path.stat().st_size
            print(f"[DEBUG] Output file size: {file_size:,} bytes")

    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] Invalid input: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
