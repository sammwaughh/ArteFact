"""
Background task processing (no Celery, no AWS).
"""

import json
import os
import time
import threading
from datetime import datetime, timezone

from .inference import run_inference

# In-memory runs store and lock for thread-safe updates
runs: dict[str, dict] = {}
runs_lock = threading.Lock()

# Optional dev flags for testing
FORCE_ERROR = os.getenv("FORCE_ERROR") == "1"
SLEEP_SECS = int(os.getenv("SLEEP_SECS", "0"))

# Get the base directory for file storage (project root)
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")


def run_task(run_id: str, image_path: str, topics: list = None, creators: list = None, model: str = "paintingclip") -> None:
    """
    Process a single run: load image from disk, run ML inference, save output, update status.
    
    Args:
        run_id: The unique run identifier
        image_path: Full path to the image file
        topics: List of topic codes to filter by (optional)
        creators: List of creator names to filter by (optional)
        model: Model type to use ("clip" or "paintingclip")
    """
    # Mark as processing (with a check to ensure the run exists)
    with runs_lock:
        if run_id not in runs:
            return
        runs[run_id]["status"] = "processing"
        runs[run_id]["startedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        runs[run_id]["updatedAt"] = runs[run_id]["startedAt"]

    try:
        # 1. Check if the image file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if SLEEP_SECS:
            time.sleep(SLEEP_SECS)          # simulate slow inference if desired

        # 2. Run the ML inference with filtering
        labels = run_inference(
            image_path,
            filter_topics=topics,
            filter_creators=creators,
            model_type=model
        )

        # If FORCE_ERROR is enabled (for testing), raise an error to simulate a failure
        if FORCE_ERROR:
            raise RuntimeError("Forced error for testing")

        # 3. Save the labels to a JSON file in the outputs folder
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        output_filename = f"{run_id}.json"
        output_path = os.path.join(OUTPUTS_DIR, output_filename)
        output_key = f"outputs/{output_filename}"  # This is what the API expects
        
        with open(output_path, "w") as f:
            json.dump(labels, f)

        # Verify the file was created
        if not os.path.exists(output_path):
            raise RuntimeError(f"Failed to create output file: {output_path}")
        
        # 4. Mark the run as done and store the output path
        with runs_lock:
            runs[run_id]["status"] = "done"
            runs[run_id]["outputKey"] = output_key  # Store the relative path for the API
            runs[run_id]["finishedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            runs[run_id]["updatedAt"] = runs[run_id]["finishedAt"]
            # Clear any previous error message if present
            runs[run_id].pop("errorMessage", None)

    except Exception as exc:
        # On any error, mark the run as failed and record the error message
        print(f"Error in run {run_id}: {exc}")  # This should already be there
        import traceback
        traceback.print_exc()  # Add full traceback
        
        with runs_lock:
            if run_id in runs:  # Be defensive here too
                runs[run_id]["status"] = "error"
                runs[run_id]["errorMessage"] = str(exc)[:500]  # truncate to 500 chars
                runs[run_id]["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
