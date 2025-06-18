"""
Flask API gateway (runner‑svc).

Routes:
    POST /presign
    POST /runs
    GET  /runs/<runId>
"""

import json
import uuid
from datetime import datetime
from typing import Dict

from flask import Flask, request, jsonify
import boto3

from .constants import ARTIFACT_BUCKET, RUNS_TABLE, QUEUE_NAME

# ----------------------------------------------------------------------
app = Flask(__name__)

# One‑time AWS clients (safe to share – boto3 is thread‑safe)
_s3          = boto3.client("s3", region_name="eu-west-2")  # adjust region
_dynamodb = boto3.resource("dynamodb", region_name="eu-west-2")
_table       = _dynamodb.Table(RUNS_TABLE)
_sqs         = boto3.client("sqs")
_queue_url   = _sqs.get_queue_url(QueueName=QUEUE_NAME)["QueueUrl"]

# Health‑check for ALB --------------------------------------------------
@app.route("/health")
def health() -> str:  # ALB expects 200 OK plain‑text
    return "ok"

# ----------------------------------------------------------------------
@app.route("/presign", methods=["POST"])
def presign_upload():
    """
    Body:
        { "fileName": "myfile.jpg" }

    Response:
        { "runId": "...", "uploadUrl": "https://...", "s3Key": "artifacts/....jpg" }
    """
    data: Dict = request.get_json(force=True)
    file_name = data.get("fileName", "upload.jpg")
    run_id    = uuid.uuid4().hex
    key       = f"artifacts/{run_id}.jpg"

    url = _s3.generate_presigned_url(
        ClientMethod="put_object",
        Params={
            "Bucket": ARTIFACT_BUCKET,
            "Key": key,
            "ContentType": "image/jpeg",  # browsers send correct header
        },
        ExpiresIn=900,  # 15 minutes
    )
    return jsonify({"runId": run_id, "uploadUrl": url, "s3Key": key})


# ----------------------------------------------------------------------
@app.route("/runs", methods=["POST"])
def create_run():
    """
    Body:
        { "runId": "...", "s3Key": "artifacts/....jpg" }

    Side‑effects:
        * DynamoDB put_item (status="queued")
        * SQS SendMessage to kick the worker
    """
    payload: Dict = request.get_json(force=True)
    run_id: str   = payload["runId"]
    s3_key: str   = payload["s3Key"]

    # DDB item
    _table.put_item(
        Item={
            "runId":      run_id,
            "status":     "queued",
            "s3Key":      s3_key,
            "createdAt":  datetime.utcnow().isoformat(timespec="seconds"),
            "updatedAt":  datetime.utcnow().isoformat(timespec="seconds"),
        }
    )

    # SQS message
    _sqs.send_message(
        QueueUrl=_queue_url,
        MessageBody=json.dumps({"runId": run_id, "s3Key": s3_key}),
    )

    return "", 202  # Accepted


# ----------------------------------------------------------------------
@app.route("/runs/<run_id>", methods=["GET"])
def get_run(run_id: str):
    """
    Returns the live DynamoDB record – SPA polls this.
    """
    resp = _table.get_item(Key={"runId": run_id})
    if "Item" not in resp:
        return jsonify({"error": "not-found"}), 404
    return jsonify(resp["Item"])
