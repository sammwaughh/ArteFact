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
from . import tasks

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


# --------------------------------------------------------------------------- #
#  Global Data                                      #
# --------------------------------------------------------------------------- #
# Load data/sentences.json into a variable called sentences
sentences = {}
with open(os.path.join(BASE_DIR,"hc_services", "runner" , "data", "sentences.json"), "r") as f:
    sentences = json.load(f)  
    # Print the first sentence for debugging
    print(f"First sentence: {list(sentences.keys())[0]}")

works = {}
with open(os.path.join(BASE_DIR, "hc_services", "runner" , "data", "works.json"), "r") as f:
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
    data = request.get_json(force=True)
    run_id = uuid.uuid4().hex
    # We'll use a local file path as the "s3Key"
    s3_key = f"artifacts/{run_id}.jpg"
    # Local upload endpoint URL (where the front-end will POST the file)
    upload_url = request.host_url + f"upload/{run_id}"
    # Return the run info and upload instructions (fields remain empty for compatibility)
    return jsonify({
        "runId": run_id,
        "s3Key": s3_key,
        "upload": {
            "url": upload_url,
            "fields": {}  # no fields needed for local upload
        }
    })


@app.route("/upload/<run_id>", methods=["POST"])
def upload_file(run_id: str):
    """
    Receives the image file upload for the given runId and saves it to disk.
    """
    if 'file' not in request.files:
        return jsonify({"error": "no-file"}), 400
    file = request.files['file']
    # Ensure artifacts directory exists
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    # Save the file as artifacts/<runId>.jpg
    file_path = os.path.join(ARTIFACTS_DIR, f"{run_id}.jpg")
    file.save(file_path)
    # Check file exists otherwise 400
    if not os.path.exists(file_path):
        return jsonify({"error": "file-not-saved"}), 500
    # Respond with 204 No Content (success, no response body)
    return "{}", 204


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
    
    # Store initial run info in the in-memory dictionary
    with tasks.runs_lock:
        tasks.runs[run_id] = {
            "runId": run_id,
            "status": "queued",
            "s3Key": s3_key,
            "createdAt": now,
            "updatedAt": now
        }
    
    # Submit the background inference task to the thread pool
    # Pass the absolute path to the image
    image_path = os.path.join(BASE_DIR, s3_key)
    executor.submit(tasks.run_task, run_id, image_path)
    
    return "", 202  # Accepted


@app.route("/runs/<run_id>", methods=["GET"])
def get_run(run_id: str):
    """
    Return the status of the run (from in-memory store).
    """
    run = tasks.runs.get(run_id)
    if run is None:
        return jsonify({"error": "not-found"}), 404
    return jsonify(run)


@app.route("/artifacts/<path:filename>", methods=["GET"])
def get_artifact_file(filename: str):
    """Serve an uploaded image from the artifacts directory."""
    return send_from_directory(ARTIFACTS_DIR, filename)


@app.route("/outputs/<path:filename>", methods=["GET"])
def get_output_file(filename: str):
    """Serve a JSON output file from the outputs directory."""
    return send_from_directory(OUTPUTS_DIR, filename)



@app.route("/work/<id>", methods=["GET"])
def get_work(id: str):
    # Get the work by id and return it as JSON
    work = works.get(id, {})
    return work

# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # invoked via  python -m …
    app.run(host="0.0.0.0", port=8000, debug=True)
