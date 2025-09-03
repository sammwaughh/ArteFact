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
GET  /work/<id>
GET  /topics
GET  /creators
GET  /cell-sim
POST /heatmap
"""

import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from mimetypes import guess_type
from pathlib import Path
from threading import RLock

from flask import Flask, jsonify, request, send_from_directory, render_template_string
from flask_cors import CORS

# --------------------------------------------------------------------------- #
#  Phase 1: Stub mode for Hugging Face Spaces                                #
# --------------------------------------------------------------------------- #
STUB_MODE = os.getenv("STUB_MODE", "1") == "1"  # set to "0" later for real ML

# Global variables for tasks module
tasks = None
inference = None

# Add this near the top, after the STUB_MODE check
if not STUB_MODE:
    try:
        # Test basic ML imports
        import torch
        import transformers
        import peft
        import cv2
        print(f"✅ ML imports successful: torch {torch.__version__}")
        print(f"✅ ML imports successful: transformers {transformers.__version__}")
        print(f"✅ ML imports successful: peft {peft.__version__}")
        print(f"✅ ML imports successful: opencv {cv2.__version__}")
        
        # Import ML modules
        from . import inference as inference_module, tasks as tasks_module
        from .inference import compute_heatmap
        tasks = tasks_module
        inference = inference_module
        RUNS = tasks.runs
        RUNS_LOCK = tasks.runs_lock
    except Exception as e:
        print(f"❌ ML import failed: {e}")
        # Fall back to stub mode
        STUB_MODE = True

# Use tasks.runs if available; otherwise a local in-memory store
if STUB_MODE or tasks is None:
    # Stub mode: lightweight imports only
    RUNS: dict[str, dict] = {}
    RUNS_LOCK = RLock()

# --------------------------------------------------------------------------- #
#  Flask application & thread pool setup                                      #
# --------------------------------------------------------------------------- #
app = Flask(__name__)
# Configure CORS for HF Spaces
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://huggingface.co",
            "https://*.hf.space",
            "http://localhost:5173",  # Local development
            "http://localhost:3000"   # Local development
        ]
    }
})

# Configure port for HF Spaces
if os.getenv("PORT"):
    app.config['SERVER_NAME'] = f"0.0.0.0:{os.getenv('PORT')}"

# Thread pool to handle background inference tasks
# Reduce workers for HF Spaces memory constraints
max_workers = int(os.getenv("MAX_WORKERS", "2"))  # Default to 2 for HF Spaces
executor = ThreadPoolExecutor(max_workers=max_workers)

# Use the Space data volume, not the repo folder
from .config import (
    ARTIFACTS_DIR,
    OUTPUTS_DIR,
    JSON_INFO_DIR,
    MARKER_DIR,
    JSON_DATASETS,
    EMBEDDINGS_DATASETS
)

# Import data from config (loaded from HF datasets)
from .config import sentences, works, creators, topics, topic_names

# --------------------------------------------------------------------------- #
#  Global Data (loaded from HF datasets via config)                            #
# --------------------------------------------------------------------------- #
# Data is now loaded from Hugging Face datasets in config.py
# No need to load from local files anymore

# Debug logging for data loading
print(f"📊 Data loaded from HF datasets:")
print(f"📊   Sentences: {len(sentences)} entries")
print(f"📊   Works: {len(works)} entries")
print(f"📊   Topics: {len(topics)} entries")
print(f"📊   Creators: {len(creators)} entries")
print(f"📊   Topic names: {len(topic_names)} entries")


# --------------------------------------------------------------------------- #
#  Routes                                                                     #
# --------------------------------------------------------------------------- #
@app.route("/health")
def health() -> str:
    return "ok"


@app.route("/")
def index():
    """Serve the main frontend page."""
    # Read the HTML file and serve it
    html_path = Path(__file__).parent.parent.parent / "frontend" / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    else:
        return "Frontend not found", 404

# Serve static frontend files
@app.route("/css/<path:filename>")
def serve_css(filename):
    """Serve CSS files."""
    css_dir = Path(__file__).parent.parent.parent / "frontend" / "css"
    return send_from_directory(css_dir, filename)

# Serve static frontend files with proper error handling
@app.route("/js/<path:filename>")
def serve_js(filename):
    try:
        js_dir = Path(__file__).parent.parent.parent / "frontend" / "js"
        if not js_dir.exists():
            return "JavaScript directory not found", 404
        return send_from_directory(js_dir, filename)
    except Exception as e:
        print(f"Error serving JS file {filename}: {e}")
        return "Internal server error", 500

# Route for work_id images (only matches actual work_ids)
@app.route("/images/W<work_id>", methods=["GET"])
def list_work_images(work_id: str):
    """
    Return absolute URLs for all JPEG / PNG images that belong to <work_id>.
    Only matches work_ids that start with 'W' followed by numbers.
    """
    # Validate that work_id is numeric
    if not work_id.isdigit():
        return "Invalid work_id format", 400
    
    full_work_id = f"W{work_id}"
    print(f"🔍 list_work_images called with work_id: {full_work_id}")
    img_dir = MARKER_DIR / full_work_id
    if not img_dir.is_dir():
        print(f"❌ Work directory not found: {img_dir}")
        return jsonify([])

    files = sorted(
        f for f in img_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    host = request.host_url.rstrip("/")
    urls = [f"{host}/marker/{full_work_id}/{f.name}" for f in files]
    return jsonify(urls)

# Route for frontend images (catches everything else)
@app.route("/images/<path:filename>")
def serve_images(filename):
    """Serve image files."""
    print(f"🔍 serve_images called with filename: {filename}")
    images_dir = Path(__file__).parent.parent.parent / "frontend" / "images"
    print(f"🔍 Images directory: {images_dir}")
    print(f"🔍 Images directory exists: {images_dir.exists()}")
    print(f"🔍 Looking for file: {images_dir / filename}")
    print(f"🔍 File exists: {(images_dir / filename).exists()}")
    
    if not images_dir.exists():
        return "Images directory not found", 404
    
    if not (images_dir / filename).exists():
        return f"Image file {filename} not found", 404
    
    mime, _ = guess_type(filename)
    print(f"🔍 MIME type: {mime}")
    return send_from_directory(images_dir, filename, mimetype=mime)


@app.route("/presign", methods=["POST"])
def presign_upload():
    run_id = uuid.uuid4().hex
    image_key = f"artifacts/{run_id}.jpg"
    
    # Use HF Spaces environment variables
    if os.getenv("SPACE_URL"):
        base_url = os.getenv("SPACE_URL")
    elif os.getenv("SPACE_HOST"):
        base_url = f"https://{os.getenv('SPACE_HOST')}"
    else:
        # Fallback for local development
        base_url = request.host_url.rstrip("/")
    
    upload_url = f"{base_url}/upload/{run_id}"
    
    return jsonify({
        "runId": run_id,
        "imageKey": image_key,
        "upload": {"url": upload_url, "fields": {}},
    })


@app.route("/upload/<run_id>", methods=["POST"])
def upload_file(run_id: str):
    """
    Receives the image file upload for the given runId and saves it to disk.
    """
    try:
        print(f"📤 Upload request for run {run_id}")
        
        if "file" not in request.files:
            print(f"❌ No file in request for run {run_id}")
            return jsonify({"error": "no-file"}), 400
        
        file = request.files["file"]
        print(f"📤 File received: {file.filename}, size: {file.content_length if hasattr(file, 'content_length') else 'unknown'}")
        
        # Ensure artifacts directory exists
        try:
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            print(f"📤 Artifacts directory: {ARTIFACTS_DIR} (exists: {ARTIFACTS_DIR.exists()})")
        except Exception as e:
            print(f"❌ Failed to create artifacts directory: {e}")
            return jsonify({"error": f"directory-creation-failed: {str(e)}"}), 500
        
        # Save the file as artifacts/<runId>.jpg
        file_path = ARTIFACTS_DIR / f"{run_id}.jpg"
        print(f"📤 Saving file to {file_path}")
        
        try:
            file.save(str(file_path))
        except Exception as e:
            print(f"❌ Failed to save file: {e}")
            return jsonify({"error": f"file-save-failed: {str(e)}"}), 500
        
        # Check file exists otherwise 500
        if not file_path.exists():
            print(f"❌ File not saved for run {run_id}")
            return jsonify({"error": "file-not-saved"}), 500
        
        file_size = file_path.stat().st_size
        print(f"✅ File saved successfully for run {run_id}, size: {file_size} bytes")
        
        # Respond with 204 No Content (success, no response body)
        return "", 204
        
    except Exception as e:
        print(f"❌ Unexpected error in upload_file: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"unexpected-error: {str(e)}"}), 500


@app.route("/runs", methods=["POST"])
def create_run():
    """
    Body: { 
    "runId": "...", 
    "imageKey": "artifacts/...jpg", 
    "topics": [...], 
    "creators": [...], 
    "model": "..." }
    - Save initial run status in memory
    - Launch background thread for processing
    """
    payload = request.get_json(force=True)
    print(f"🔍 create_run called with payload: {payload}")
    
    run_id = payload["runId"]
    image_key = payload["imageKey"]
    topics = payload.get("topics", [])
    creators = payload.get("creators", [])
    model = payload.get("model", "paintingclip").lower()  # Convert to lowercase
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    
    print(f"🔍 Parsed: run_id={run_id}, image_key={image_key}, topics={topics}, creators={creators}, model={model}")

    # Store initial run info in the in-memory dictionary
    with RUNS_LOCK:
        RUNS[run_id] = {
            "runId": run_id,
            "status": "queued",
            "imageKey": image_key,
            "topics": topics,
            "creators": creators,
            "model": model,
            "createdAt": now,
            "updatedAt": now,
        }

    if STUB_MODE:
        print(f"🔍 Stub mode: generating fake results for {run_id}")
        # Write a tiny fake result so the UI flows
        results = {
            "runId": run_id,
            "model": model,
            "top_k": 25,
            "sentences": [
                {
                    "id": f"W123_s{i:04d}", 
                    "text": f"Stub sentence {i}.", 
                    "english_original": f"Stub sentence {i}.",  # Add this field
                    "work": f"W123",  # Add this field
                    "score": 0.9 - i*0.01
                }
                for i in range(1, 6)
            ],
        }
        out_path = OUTPUTS_DIR / f"{run_id}.json"
        print(f"🔍 Stub mode: writing results to {out_path}")
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        
        with RUNS_LOCK:
            RUNS[run_id].update({
                "status": "done",  # ← Change from "completed" to "done"
                "outputKey": f"outputs/{out_path.name}",
                "finishedAt": now,
                "updatedAt": now
            })
        print(f"🔍 Stub mode: returning results directly for {run_id}")
        return jsonify(results), 200
    else:
        # Submit the background inference task to the thread pool
        image_path = ARTIFACTS_DIR / f"{run_id}.jpg"
        print(f"🔍 Real ML mode: submitting task for {run_id} with image {image_path}")
        print(f"🔍 Topics: {topics}, Creators: {creators}, Model: {model}")
        executor.submit(tasks.run_task, run_id, str(image_path), topics, creators, model)
        return jsonify({"status": "accepted"}), 202


@app.route("/runs/<run_id>", methods=["GET"])
def get_run(run_id: str):
    """
    Return the status of the run (from in-memory store).
    """
    run = RUNS.get(run_id)
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

    # If the filename doesn't end with .json, add it
    if not filename.endswith(".json"):
        filename = filename + ".json"

    # Check if file exists
    file_path = OUTPUTS_DIR / filename
    if not file_path.exists():
        return jsonify({"error": "file-not-found"}), 404

    return send_from_directory(OUTPUTS_DIR, filename)


@app.route("/work/<id>", methods=["GET"])
def get_work(id: str):
    """
    Return metadata for a work plus (optionally) the paragraph that contains
    a given sentence.

    Query params
    ------------
    sentence : original-English sentence text (URL-encoded)
    """
    work = works.get(id)
    if work is None:
        return jsonify({}), 404

    # ---------------- context lookup ----------------
    sentence = request.args.get("sentence", "").strip()
    context = ""
    if sentence:
        md_path = MARKER_DIR / id / f"{id}.md"
        if md_path.is_file():
            content = md_path.read_text(encoding="utf-8", errors="ignore")
            import re
            from difflib import SequenceMatcher

            def normalise(txt: str) -> str:
                """lower-case, remove punctuation, collapse whitespace"""
                txt = re.sub(r"[^\w\s]", " ", txt.lower())
                return re.sub(r"\s+", " ", txt).strip()

            target_norm = normalise(sentence)
            best_para = ""
            best_ratio = 0.0

            # split on blank lines → paragraphs
            for para in (p.strip() for p in content.split("\n\n") if p.strip()):
                para_norm = normalise(para)

                # 1) quick exact-substring on normalised text
                if target_norm in para_norm:
                    context = para
                    break

                # 2) otherwise keep best fuzzy match
                ratio = SequenceMatcher(None, target_norm, para_norm).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_para = para

            # accept fuzzy hit if fairly close
            if not context and best_ratio >= 0.55:
                context = best_para

    payload = {**work, "context": context}
    return jsonify(payload)


@app.route("/topics", methods=["GET"])
def get_topics():
    if STUB_MODE:
        return jsonify({
            "C52119013": "Art History",
            "T13922": "Historical Art and Culture Studies",
            "T12632": "Visual Culture and Art Theory"
        })
    return jsonify(topic_names)


@app.route("/creators", methods=["GET"])
def get_creators():
    if STUB_MODE:
        return jsonify({
            "arthur_hughes": ["W4206160935", "W2029124454"],
            "francesco_hayez": ["W1982215463", "W4388661114"],
            "george_stubbs": ["W2020798572", "W2021094421"]
        })
    return jsonify(creators)


@app.route("/models", methods=["GET"])
def get_models():
    """
    Return the list of models.
    """
    return jsonify(["CLIP", "PaintingCLIP"])


@app.route("/cell-sim", methods=["GET"])
def cell_similarity():
    if STUB_MODE:
        # Return stub results that match the expected frontend structure
        return jsonify({
            "sentences": [
                {
                    "sentence_id": f"W123_s{i:04d}",
                    "english_original": f"Stub cell sentence {i} for testing.",
                    "work": "W123",
                    "score": 0.9 - i*0.01,
                    "rank": i
                }
                for i in range(1, 6)
            ]
        })
    
    try:
        run_id = request.args["runId"]
        row = int(request.args["row"])
        col = int(request.args["col"])
        k = int(request.args.get("k", 25))

        # Get the run info to retrieve filtering parameters
        run_info = RUNS.get(run_id, {})
        topics = run_info.get("topics", [])
        creators = run_info.get("creators", [])
        model = run_info.get("model", "paintingclip").lower()  # Convert to lowercase

        img_path = ARTIFACTS_DIR / f"{run_id}.jpg"
        if not img_path.exists():
            return jsonify({"error": "Image not found"}), 404
            
        results = inference.run_inference(
            str(img_path),
            cell=(row, col),
            top_k=k,
            filter_topics=topics,
            filter_creators=creators,
            model_type=model,
        )
        return jsonify(results)
    except Exception as e:
        print(f"❌ Error in cell_similarity: {e}")
        return jsonify({"error": str(e)}), 500


# --------------------------------------------------------------------------- #
#  Accurate Grad-ECLIP heat-map                                              #
# --------------------------------------------------------------------------- #
@app.route("/heatmap", methods=["POST"])
def heatmap():
    """
    Body:
        {
            "runId":   "...",
            "sentence": "Full English Original text …",
            "layerIdx": -1          # optional, defaults to last block
        }

    Response:
        { "dataUrl": "data:image/png;base64,..." }
    """
    if STUB_MODE:
        # Return a stub heatmap for Phase 1
        return jsonify({"dataUrl": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="})
    
    payload = request.get_json(force=True)
    run_id = payload["runId"]
    sentence = payload["sentence"]
    layer = int(payload.get("layerIdx", -1))

    # Truncate sentence if it's too long for CLIP (max 77 tokens)
    MAX_SENTENCE_LENGTH = 300
    if len(sentence) > MAX_SENTENCE_LENGTH:
        sentence = sentence[: MAX_SENTENCE_LENGTH - 3] + "..."

    # Path of the already-uploaded artefact
    img_path = ARTIFACTS_DIR / f"{run_id}.jpg"
    if not img_path.exists():
        return jsonify({"error": "image-not-found"}), 404

    try:
        data_url = compute_heatmap(str(img_path), sentence, layer_idx=layer)
        return jsonify({"dataUrl": data_url})
    except Exception as exc:
        print(f"Heatmap generation error: {exc}")
        return jsonify({"error": str(exc)}), 500


# --------------------------------------------------------------------------- #
#  NEW:  marker-output image helpers                                          #
# --------------------------------------------------------------------------- #
@app.route("/marker/<work_id>/<path:filename>", methods=["GET"])
def serve_marker_image(work_id: str, filename: str):
    """
    Static file server for data/marker_output/<work_id>/<filename>
    """
    img_dir = MARKER_DIR / work_id
    img_path = img_dir / filename
    mime, _ = guess_type(filename)
    if not img_path.exists():
        return jsonify({"error": "not-found"}), 404
    return send_from_directory(img_dir, filename, mimetype=mime)


# --------------------------------------------------------------------------- #
#  Error Handlers                                                             #
# --------------------------------------------------------------------------- #
@app.errorhandler(413)  # Payload too large
def too_large(e):
    return jsonify({"error": "File too large for HF Spaces"}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # invoked via  python -m …
    # Use PORT environment variable for Hugging Face Spaces
    port = int(os.getenv("PORT", 7860))  # Default to 7860 for HF Spaces
    app.run(host="0.0.0.0", port=port, debug=False)
