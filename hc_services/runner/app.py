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
import os
import uuid
from datetime import datetime
from typing import Dict, cast

import boto3
from botocore.exceptions import ClientError  # ← NEW
from flask import Flask, jsonify, request
from flask_cors import CORS

from .constants import ARTIFACT_BUCKET, RUNS_TABLE, QUEUE_NAME
from .tasks import run_task  # ← add

# --------------------------------------------------------------------------- #
#  AWS / LocalStack configuration                                             #
# --------------------------------------------------------------------------- #
_REGION = "eu-west-2"
AWS_ENDPOINT = os.getenv("AWS_ENDPOINT_URL")  # e.g. http://localstack:4566

# --------------------------------------------------------------------------- #
#  Clients – created once; boto3 is thread-safe                               #
# --------------------------------------------------------------------------- #
_s3 = boto3.client("s3", region_name=_REGION, endpoint_url=AWS_ENDPOINT)
_sqs = boto3.client("sqs", region_name=_REGION, endpoint_url=AWS_ENDPOINT)
_dynamodb = boto3.resource("dynamodb", region_name=_REGION, endpoint_url=AWS_ENDPOINT)
_table = _dynamodb.Table(RUNS_TABLE)


# --------------------------------------------------------------------------- #
#  Dev-only bootstrap: ensure bucket exists & permissive CORS on LocalStack   #
# --------------------------------------------------------------------------- #
def _ensure_bucket() -> None:
    try:
        _s3.head_bucket(Bucket=ARTIFACT_BUCKET)
    except ClientError as err:
        code = err.response.get("Error", {}).get("Code", "")
        if code not in ("NoSuchBucket", "404"):
            raise  # genuine AWS error
        # --- create bucket (add LocationConstraint outside us-east-1) -------
        create_kwargs = {"Bucket": ARTIFACT_BUCKET}
        if _REGION != "us-east-1":  # ← NEW
            create_kwargs["CreateBucketConfiguration"] = {"LocationConstraint": _REGION}
        _s3.create_bucket(**create_kwargs)  # ← pass the dict
        # --- open CORS so the SPA can PUT directly --------------------------
        _s3.put_bucket_cors(
            Bucket=ARTIFACT_BUCKET,
            CORSConfiguration={
                "CORSRules": [
                    {
                        "AllowedOrigins": ["*"],
                        "AllowedMethods": ["PUT", "GET", "HEAD"],
                        "AllowedHeaders": ["*"],
                        "MaxAgeSeconds": 3600,
                    }
                ]
            },
        )


def _ensure_table() -> None:
    """Create DynamoDB table `hc-runs` in LocalStack if missing."""
    client = _dynamodb.meta.client
    try:
        client.describe_table(TableName=RUNS_TABLE)
    except client.exceptions.ResourceNotFoundException:
        client.create_table(
            TableName=RUNS_TABLE,
            KeySchema=[{"AttributeName": "runId", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "runId", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        client.get_waiter("table_exists").wait(TableName=RUNS_TABLE)


if AWS_ENDPOINT:  # only when using LocalStack
    _ensure_bucket()
    _ensure_table()


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
    return cast(str, _ensure_queue_url._cache)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
#  Flask application & routes                                                 #
# --------------------------------------------------------------------------- #
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # allow SPA on :5173


@app.route("/health")
def health() -> str:
    return "ok"


# --------------------------------------------------------------------------- #
@app.route("/presign", methods=["POST"])
def presign_upload():
    """
    Body:     { "fileName": "myfile.jpg" }
    Response: { "runId", "uploadUrl", "s3Key" }
    """
    data: Dict[str, str] = request.get_json(force=True)
    run_id = uuid.uuid4().hex
    key = f"artifacts/{run_id}.jpg"
    filetype = data.get("fileName", "upload.jpg").split(".")[-1].lower()
    mime = "image/jpeg" if filetype in ("jpg", "jpeg") else "image/png"

    upload_url = _s3.generate_presigned_url(
        ClientMethod="put_object",
        Params={"Bucket": ARTIFACT_BUCKET, "Key": key, "ContentType": mime},
        ExpiresIn=15 * 60,
    )

    # --- DEV-ONLY tweak: make the URL accessible from the host browser ----------
    if AWS_ENDPOINT:  # we’re talking to LocalStack
        public_host = os.getenv("AWS_PUBLIC_ENDPOINT", "http://localhost:4566")
        # internal presign points to  http://localstack:4566/...
        upload_url = upload_url.replace("http://localstack:4566", public_host)

    return jsonify({"runId": run_id, "uploadUrl": upload_url, "s3Key": key})


# --------------------------------------------------------------------------- #
@app.route("/runs", methods=["POST"])
def create_run():
    """
    Body: { "runId": "...", "s3Key": "artifacts/....jpg" }

    * writes Dynamo item (status = queued)
    * publishes SQS message
    """
    payload = request.get_json(force=True)
    run_id = payload["runId"]
    s3_key = payload["s3Key"]
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
    # hand the work off to Celery (goes via SQS broker)
    run_task.apply_async(args=[run_id, s3_key], queue=QUEUE_NAME)

    return "", 202  # Accepted


# --------------------------------------------------------------------------- #
@app.route("/runs/<run_id>", methods=["GET"])
def get_run(run_id: str):
    resp = _table.get_item(Key={"runId": run_id})
    if "Item" not in resp:
        return jsonify({"error": "not-found"}), 404
    return jsonify(resp["Item"])


# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # invoked via  python -m …
    app.run(host="0.0.0.0", port=8000, debug=False)
