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
from typing import Dict, Optional, cast

import boto3
from flask import Flask, jsonify, request

from .constants import ARTIFACT_BUCKET, RUNS_TABLE, QUEUE_NAME

# --------------------------------------------------------------------------- #
#  AWS clients – created once; boto3 is thread-safe                           #
# --------------------------------------------------------------------------- #
_REGION = "eu-west-2"

_s3 = boto3.client("s3", region_name=_REGION)
_sqs = boto3.client("sqs", region_name=_REGION)
_dynamodb = boto3.resource("dynamodb", region_name=_REGION)
_table = _dynamodb.Table(RUNS_TABLE)

# --------------------------------------------------------------------------- #
#  Helper: resolve the queue URL lazily so no AWS call occurs at import time  #
# --------------------------------------------------------------------------- #
def _ensure_queue_url() -> str:
    """
    Return the cached SQS queue URL, creating or looking it up on first call.

    This avoids hitting AWS when the module is imported (important for tests
    where Moto hasn’t patched SQS yet) and also dodges the need for credentials
    until the route is actually invoked.
    """
    if not hasattr(_ensure_queue_url, "_cache"):
        try:
            url = _sqs.get_queue_url(QueueName=QUEUE_NAME)["QueueUrl"]
        except _sqs.exceptions.QueueDoesNotExist:
            url = _sqs.create_queue(QueueName=QUEUE_NAME)["QueueUrl"]

        # attribute trick: attach cache to the function object itself
        _ensure_queue_url._cache = url  # type: ignore[attr-defined]

    return cast(str, _ensure_queue_url._cache)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
#  Flask application & routes                                                 #
# --------------------------------------------------------------------------- #
app = Flask(__name__)


@app.route("/health")
def health() -> str:
    """Simple ALB/Gateway health-check."""
    return "ok"


@app.route("/presign", methods=["POST"])
def presign_upload():
    """
    Request body:   { "fileName": "myfile.jpg" }
    Response JSON:  { "runId", "uploadUrl", "s3Key" }
    """
    data: Dict[str, str] = request.get_json(force=True)
    file_name = data.get("fileName", "upload.jpg")

    run_id = uuid.uuid4().hex
    key = f"artifacts/{run_id}.jpg"

    upload_url = _s3.generate_presigned_url(
        ClientMethod="put_object",
        Params={
            "Bucket": ARTIFACT_BUCKET,
            "Key": key,
            "ContentType": "image/jpeg",
        },
        ExpiresIn=15 * 60,  # 15 minutes
    )
    return jsonify({"runId": run_id, "uploadUrl": upload_url, "s3Key": key})


@app.route("/runs", methods=["POST"])
def create_run():
    """
    Request body: { "runId": "...", "s3Key": "artifacts/....jpg" }

    Side-effects
    ------------
    * DynamoDB item  (status = "queued")
    * SQS message    (kicks the worker)
    """
    payload: Dict[str, str] = request.get_json(force=True)
    run_id: str = payload["runId"]
    s3_key: str = payload["s3Key"]

    now = datetime.utcnow().isoformat(timespec="seconds")

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

    return "", 202  # HTTP 202 Accepted


@app.route("/runs/<run_id>", methods=["GET"])
def get_run(run_id: str):
    """Return the DynamoDB record for the requested run."""
    resp = _table.get_item(Key={"runId": run_id})
    if "Item" not in resp:
        return jsonify({"error": "not-found"}), 404
    return jsonify(resp["Item"])

# ----------------------------------------------------------------------
# Launch the API if the module is executed as a script
# (that’s what `python -m hc_services.runner.app` does in the container).
if __name__ == "__main__":
    # Listen on all interfaces so Docker can expose the port
    # Port 8000 matches the mapping in docker-compose.yml
    app.run(host="0.0.0.0", port=8000, debug=False)