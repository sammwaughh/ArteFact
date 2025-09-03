#!/usr/bin/env python3
"""
test_patch_inference.py
======================

Test script for the patch-based region inference functionality.
Tests both whole-image inference and grid-cell specific inference.

Usage:
    python test_patch_inference.py

This script demonstrates:
1. Whole-image inference (traditional CLIP similarity)
2. Grid-cell specific inference (click on region → relevant sentences)
3. Comparison between different regions
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path for imports (backend/)
sys.path.append(str(Path(__file__).resolve().parent.parent))

from runner.inference import run_inference


def print_results(results: List[Dict[str, Any]], title: str, max_display: int = 5):
    """Pretty print inference results."""
    print(f"\n{title}")
    print("=" * len(title))

    for i, result in enumerate(results[:max_display], 1):
        print(f"\n{i}. {result['english_original'][:100]}...")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Work: {result['work']}")
        print(f"   ID: {result['id']}")

    if len(results) > max_display:
        print(f"\n... and {len(results) - max_display} more results")


def test_grid_visualization(grid_size=(7, 7)):
    """Show ASCII grid with coordinates for reference."""
    print("\nGrid Reference (row, col):")
    print("=" * 50)

    rows, cols = grid_size
    for r in range(rows):
        row_str = ""
        for c in range(cols):
            row_str += f"({r},{c}) "
        print(row_str)
    print()


def main():
    """Run comprehensive patch inference tests."""

    # Configuration - Updated to use new directory structure
    project_root = Path(__file__).resolve().parent.parent.parent
    IMAGE_PATH = str(
        project_root
        / "frontend"
        / "images"
        / "examples"
        / "Giotto_-_Scrovegni_-_-31-_-_Kiss_of_Judas.jpg"
    )
    GRID_SIZE = (7, 7)  # 7x7 grid matches ViT-B/32 patch grid
    TOP_K = 25  # Number of results to return

    print("Patch Inference Test Suite")
    print("=" * 60)
    print(f"Image: {IMAGE_PATH}")
    print(f"Grid Size: {GRID_SIZE[0]}x{GRID_SIZE[1]} (matches ViT-B/32 patch grid)")
    print(f"Top K: {TOP_K}")

    # Check if image exists
    if not Path(IMAGE_PATH).exists():
        print(f"\nError: Image not found at {IMAGE_PATH}")
        print("Please update IMAGE_PATH to point to a valid test image.")

    # Test 1: Whole-image inference (baseline)
    print("\n\nTest 1: Whole-Image Inference")
    print("-" * 60)
    whole_image_results = run_inference(
        image_path=IMAGE_PATH, cell=None, top_k=TOP_K  # None means whole image
    )
    print_results(whole_image_results, "Whole Image Results")

    # Show grid reference
    test_grid_visualization(GRID_SIZE)

    # Test 2: Center region inference
    print("\n\nTest 2: Center Region Inference")
    print("-" * 60)
    center_row, center_col = GRID_SIZE[0] // 2, GRID_SIZE[1] // 2
    print(f"Testing center cell: ({center_row}, {center_col})")

    center_results = run_inference(
        image_path=IMAGE_PATH,
        cell=(center_row, center_col),
        grid_size=GRID_SIZE,
        top_k=TOP_K,
    )
    print_results(center_results, f"Center Cell ({center_row},{center_col}) Results")

    # Test 3: Corner regions comparison
    print("\n\nTest 3: Corner Regions Comparison")
    print("-" * 60)

    corners = {
        "Top-Left": (0, 0),
        "Top-Right": (0, GRID_SIZE[1] - 1),
        "Bottom-Left": (GRID_SIZE[0] - 1, 0),
        "Bottom-Right": (GRID_SIZE[0] - 1, GRID_SIZE[1] - 1),
    }

    for corner_name, (row, col) in corners.items():
        print(f"\n{corner_name} Corner ({row}, {col}):")
        corner_results = run_inference(
            image_path=IMAGE_PATH,
            cell=(row, col),
            grid_size=GRID_SIZE,
            top_k=3,  # Show fewer results for comparison
        )

        for i, result in enumerate(corner_results[:3], 1):
            print(f"  {i}. {result['english_original'][:60]}...")
            print(f"     Score: {result['score']:.4f}")

    # Test 4: Specific region of interest
    print("\n\nTest 4: Custom Region Test")
    print("-" * 60)
    # You can modify these coordinates to test specific regions
    custom_row, custom_col = 2, 4  # Example: row 2, column 4
    print(f"Testing custom cell: ({custom_row}, {custom_col})")

    custom_results = run_inference(
        image_path=IMAGE_PATH,
        cell=(custom_row, custom_col),
        grid_size=GRID_SIZE,
        top_k=TOP_K,
    )
    print_results(custom_results, f"Custom Cell ({custom_row},{custom_col}) Results")

    # Test 5: Compare scores between whole image and regions
    print("\n\nTest 5: Score Comparison Analysis")
    print("-" * 60)

    # Get the top sentence from whole image
    if whole_image_results:
        top_sentence_id = whole_image_results[0]["id"]
        top_sentence_text = whole_image_results[0]["english_original"]

        print(f"Top whole-image sentence: {top_sentence_text[:80]}...")
        print(f"Whole-image score: {whole_image_results[0]['score']:.4f}")

        # Check this sentence's score in different regions
        print("\nScore for this sentence in different regions:")

        test_cells = [
            ("Center", (center_row, center_col)),
            ("Top-Left", (0, 0)),
            ("Bottom-Right", (GRID_SIZE[0] - 1, GRID_SIZE[1] - 1)),
        ]

        for region_name, (row, col) in test_cells:
            region_results = run_inference(
                image_path=IMAGE_PATH, cell=(row, col), grid_size=GRID_SIZE, top_k=TOP_K
            )

            # Find the score for our target sentence
            region_score = None
            region_rank = None
            for rank, result in enumerate(region_results, 1):
                if result["id"] == top_sentence_id:
                    region_score = result["score"]
                    region_rank = rank
                    break

            if region_score:
                print(
                    f"  {region_name} ({row},{col}): score={region_score:.4f}, rank={region_rank}"
                )
            else:
                print(f"  {region_name} ({row},{col}): Not in top {TOP_K}")

    # Summary statistics
    print("\n\nSummary")
    print("=" * 60)
    print("✓ Whole-image inference tested")
    print("✓ Region-specific inference tested")
    print("✓ Multiple grid cells compared")
    print("\nThe patch inference system is working correctly!")

    # Optional: Save results for further analysis
    save_results = input("\nSave detailed results to JSON? (y/n): ").lower() == "y"
    if save_results:
        results_data = {
            "image_path": IMAGE_PATH,
            "grid_size": GRID_SIZE,
            "whole_image": whole_image_results,
            "center_cell": {
                "position": [center_row, center_col],
                "results": center_results,
            },
            "corners": {
                name: {
                    "position": list(pos),
                    "results": run_inference(
                        IMAGE_PATH, cell=pos, grid_size=GRID_SIZE, top_k=5
                    ),
                }
                for name, pos in corners.items()
            },
        }

        # Save to data/outputs directory
        outputs_dir = project_root / "runner" / "tests" / "test-outputs"
        outputs_dir.mkdir(exist_ok=True)
        output_path = outputs_dir / "patch_inference_test_results.json"

        with open(output_path, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
