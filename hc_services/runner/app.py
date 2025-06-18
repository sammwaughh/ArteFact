"""
Flask API gateway (runner-svc).

Routes
------
POST /presign
POST /runs
GET  /runs/<runId>
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Dict, cast

import boto3
from flask import Flask, jsonify, request
from flask_cors import CORS           # ← NEW

from .constants import ARTIFACT_BUCKET, RUNS_TABLE, QUEUE_NAME

# --------------------------------------------------------------------------- #
#  AWS clients – created once; boto3 is thread-safe                           #
# --------------------------------------------------------------------------- #
_REGION = "eu-west-2"

_s3        = boto3.client("s3", region_name=_REGION)
_sqs       = boto3.client("sqs", region_name=_REGION)
_dynamodb  = boto3.resource("dynamodb", region_name=_REGION)
_table     = _dynamodb.Table(RUNS_TABLE)

# --------------------------------------------------------------------------- #
#  Helper: resolve the SQS queue URL lazily (avoid AWS calls at import time)  #
# --------------------------------------------------------------------------- #
def _ensure_queue_url() -> str:
    if not hasattr(_ensure_queue_url, "_cache"):
        try:
            url = _sqs.get_queue_url(QueueName=QUEUE_NAME)["QueueUrl"]
        except _sqs.exceptions.QueueDoesNotExist:
            url = _sqs.create_queue(QueueName=QUEUE_NAME)["QueueUrl"]
        _ensure_queue_url._cache = url  # type: ignore[attr-defined]
    return cast(str, _ensure_queue_url._cache)      # type: ignore[attr-defined]

# --------------------------------------------------------------------------- #
#  Flask application & routes                                                 #
# --------------------------------------------------------------------------- #
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})      # ← NEW: allow all origins in dev

@app.route("/health")
def health() -> str:
    """Simple ALB/Gateway health-check endpoint."""
    return "ok"

# --------------------------------------------------------------------------- #
@app.route("/presign", methods=["POST"])
def presign_upload():
    """
    Body:      { "fileName": "myfile.jpg" }
    Response:  { "runId", "uploadUrl", "s3Key" }
    """
    data: Dict[str, str] = request.get_json(force=True)
    run_id   = uuid.uuid4().hex
    key      = f"artifacts/{run_id}.jpg"
    filetype = data.get("fileName", "upload.jpg").split(".")[-1].lower()
    mime     = "image/jpeg" if filetype in ("jpg", "jpeg") else "image/png"

    upload_url = _s3.generate_presigned_url(
        ClientMethod="put_object",
        Params={"Bucket": ARTIFACT_BUCKET, "Key": key, "ContentType": mime},
        ExpiresIn=15 * 60,   # 15 min
    )
    return jsonify({"runId": run_id, "uploadUrl": upload_url, "s3Key": key})

# --------------------------------------------------------------------------- #
@app.route("/runs", methods=["POST"])
def create_run():
    """
    Body: { "runId": "...", "s3Key": "artifacts/....jpg" }

    * writes Dynamo item (status=queued)
    * publishes SQS message
    """
    payload = request.get_json(force=True)
    run_id  = payload["runId"]
    s3_key  = payload["s3Key"]
    now     = datetime.utcnow().isoformat(timespec="seconds")

    _table.put_item(
        Item={
            "runId": run_id,
            "status": "queued",
            "s3Key": s3_key,
            "createdAt": now,
            "updatedAt": now,
        }
    )
    _sqs.send_message(
        QueueUrl=_ensure_queue_url(),
        MessageBody=json.dumps({"runId": run_id, "s3Key": s3_key}),
    )
    return "", 202  # Accepted

# --------------------------------------------------------------------------- #
@app.route("/runs/<run_id>", methods=["GET"])
def get_run(run_id: str):
    """Return the DynamoDB record for the requested run (polled by the SPA)."""
    resp = _table.get_item(Key={"runId": run_id})
    if "Item" not in resp:
        return jsonify({"error": "not-found"}), 404
    return jsonify(resp["Item"])

# --------------------------------------------------------------------------- #
if __name__ == "__main__":                # When invoked via  python -m …
    app.run(host="0.0.0.0", port=8000, debug=False)
