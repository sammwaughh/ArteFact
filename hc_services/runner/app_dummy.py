"""
Flask API gateway (local-only version).

Routes
------
POST /presign
POST /upload/<runId>
POST /runs
GET  /runs/<runId>
GET  /artifacts/<filename>
GET  /outputs/<filename>
"""

import os
import uuid
import json
import random
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Import the tasks module (contains run_task, runs dict, etc.)
# from . import tasks

# --------------------------------------------------------------------------- #
#  Flask application & thread pool setup                                      #
# --------------------------------------------------------------------------- #
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # allow SPA on :5173

# Thread pool to handle background inference tasks
executor = ThreadPoolExecutor(max_workers=4)

# Get the base directory for file storage (project root)
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Load data/sentences.json into a variable called sentences
sentences = {}
with open(
    os.path.join(BASE_DIR, "hc_services", "runner", "data", "sentences.json"), "r"
) as f:
    sentences = json.load(f)
    # Print the first sentence for debugging
    print(f"First sentence: {list(sentences.keys())[0]}")

works = {}
with open(
    os.path.join(BASE_DIR, "hc_services", "runner", "data", "works.json"), "r"
) as f:
    works = json.load(f)


# --------------------------------------------------------------------------- #
#  Routes                                                                     #
# --------------------------------------------------------------------------- #
@app.route("/health")
def health() -> str:
    return "ok"


@app.route("/presign", methods=["POST"])
def presign_upload():
    """
    Body:     { "fileName": "myfile.jpg" }
    Response: {
        "runId": "...",
        "s3Key": "artifacts/<id>.jpg",
        "upload": { "url": "...", "fields": { } }
    }
    """

    return jsonify(
        {
            "runId": "001",
            "s3Key": "key",
            "upload": {
                "url": "/images/upload/001.jpg",  # Local URL for upload
                "fields": {},  # no fields needed for local upload
            },
        }
    )


@app.route("/runs", methods=["POST"])
def create_run():
    """
    Body: { "runId": "...", "s3Key": "artifacts/...jpg" }
    - Save initial run status in memory
    - Launch background thread for processing
    """
    payload = request.get_json(force=True)
    run_id = payload["runId"]
    s3_key = payload["s3Key"]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # # Store initial run info in the in-memory dictionary
    # with tasks.runs_lock:
    #     tasks.runs[run_id] = {
    #         "runId": run_id,
    #         "status": "queued",
    #         "s3Key": s3_key,
    #         "createdAt": now,
    #         "updatedAt": now
    #     }

    # # Submit the background inference task to the thread pool
    # # Pass the absolute path to the image
    # image_path = os.path.join(BASE_DIR, s3_key)
    # executor.submit(tasks.run_task, run_id, image_path)

    return "{}", 202  # Accepted


@app.route("/runs/<run_id>", methods=["GET"])
def get_run(run_id: str):
    """
    Return the status of the run (from in-memory store).
    """

    run = {}
    run["status"] = "processing"
    run["startedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run["updatedAt"] = run["startedAt"]
    # If a random number is less than 0.1 then set status to "completed"
    if uuid.uuid4().int % 2 == 0:
        run["status"] = "done"

    return jsonify(run)


@app.route("/artifacts/<path:filename>", methods=["GET"])
def get_artifact_file(filename: str):
    """Serve an uploaded image from the artifacts directory."""
    return send_from_directory(ARTIFACTS_DIR, filename)


@app.route("/outputs/<path:filename>", methods=["GET"])
def get_output_file(filename: str):
    """Serve a JSON output file from the outputs directory."""

    # For now, we return a static set of labels for demonstration purposes
    # In a real implementation, you would fetch the actual labels based on the filename or run_id
    # For example, you might query a database or perform some computation to generate these labels
    # Here we just return the static labels defined above
    labels = [
        {
            "sentence": sentences[sentence_key],
            "score": random.uniform(0.1, 0.9),
            "evidence": {"rank": i + 1},
        }
        for i, sentence_key in enumerate(random.sample(list(sentences), 20))
    ]

    return labels


@app.route("/work/<id>", methods=["GET"])
def get_work(id: str):
    # Get the work by id and return it as JSON
    work = works.get(id, {})
    return work


# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # invoked via  python -m …
    app.run(host="0.0.0.0", port=8000, debug=True)
