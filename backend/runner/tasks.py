"""
Background task processing (no Celery, no AWS).
"""

import json
import os
import threading
import time
from datetime import datetime, timezone

from .inference import run_inference
from .config import OUTPUTS_DIR

# In-memory runs store and lock for thread-safe updates
runs: dict[str, dict] = {}
runs_lock = threading.Lock()

# Optional dev flags for testing
FORCE_ERROR = os.getenv("FORCE_ERROR") == "1"
SLEEP_SECS = int(os.getenv("SLEEP_SECS", "0"))

# Get the base directory for file storage (project root)
# BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# OUTPUTS_DIR = os.path.join(BASE_DIR, "data", "outputs")

# OUTPUTS_DIR is now imported from config


def run_task(
    run_id: str,
    image_path: str,
    topics: list = None,
    creators: list = None,
    model: str = "paintingclip",
) -> None:
    """
    Process a single run: load image from disk, run ML inference, save output, update status.
    """
    print(f"🚀 Starting task for run {run_id}")
    print(f"🚀 Image path: {image_path}")
    print(f"🚀 Topics: {topics}, Creators: {creators}, Model: {model}")
    
    # Enhanced logging: Check environment and paths
    print(f"🔍 Environment check:")
    print(f"   STUB_MODE: {os.getenv('STUB_MODE', 'not set')}")
    print(f"   Current working directory: {os.getcwd()}")
    print(f"   Image file exists: {os.path.exists(image_path)}")
    if os.path.exists(image_path):
        print(f"   Image file size: {os.path.getsize(image_path)} bytes")
    
    # Clear any cached images from patch inference
    try:
        from .patch_inference import _prepare_image
        _prepare_image.cache_clear()
        print(f"✅ Cleared patch inference cache")
    except ImportError as e:
        print(f"⚠️  patch_inference import failed: {e}")

    # Mark as processing
    with runs_lock:
        if run_id not in runs:
            print(f"❌ Run {run_id} not found in runs store")
            return
        runs[run_id]["status"] = "processing"
        runs[run_id]["startedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        runs[run_id]["updatedAt"] = runs[run_id]["startedAt"]
        print(f"✅ Run {run_id} marked as processing")

    try:
        # 1. Check if the image file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if SLEEP_SECS:
            time.sleep(SLEEP_SECS)

        print(f"🔍 About to call run_inference...")
        
        # 2. Run the ML inference with filtering
        labels = run_inference(
            image_path, filter_topics=topics, filter_creators=creators, model_type=model
        )
        
        print(f"✅ run_inference completed successfully")
        print(f"✅ Labels type: {type(labels)}")
        print(f"✅ Labels length: {len(labels) if isinstance(labels, list) else 'not a list'}")

        # If FORCE_ERROR is enabled (for testing), raise an error to simulate a failure
        if FORCE_ERROR:
            raise RuntimeError("Forced error for testing")

        # 3. Save the labels to a JSON file in the outputs folder
        print(f"🔍 Saving results to outputs directory...")
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        output_filename = f"{run_id}.json"
        output_path = os.path.join(OUTPUTS_DIR, output_filename)
        output_key = f"outputs/{output_filename}"

        with open(output_path, "w") as f:
            json.dump(labels, f)

        # Verify the file was created
        if not os.path.exists(output_path):
            raise RuntimeError(f"Failed to create output file: {output_path}")

        # 4. Mark the run as done and store the output path
        with runs_lock:
            runs[run_id]["status"] = "done"
            runs[run_id]["outputKey"] = output_key
            runs[run_id]["finishedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            runs[run_id]["updatedAt"] = runs[run_id]["finishedAt"]
            runs[run_id].pop("errorMessage", None)
            print(f"✅ Task completed successfully for run {run_id}")
            print(f"✅ Output saved to: {output_path}")
            print(f"✅ Output key: {output_key}")

    except Exception as exc:
        # Enhanced error logging
        print(f"❌ Error in run {run_id}: {exc}")
        print(f"❌ Error type: {type(exc).__name__}")
        import traceback
        print(f"❌ Full traceback:")
        traceback.print_exc()

        with runs_lock:
            if run_id in runs:
                runs[run_id]["status"] = "error"
                runs[run_id]["errorMessage"] = str(exc)[:500]
                runs[run_id]["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                print(f"❌ Run {run_id} marked as error: {runs[run_id]['errorMessage']}")
