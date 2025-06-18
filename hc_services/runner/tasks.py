"""
Celery configuration + asynchronous job definition (worker‑svc).
"""

import json
import os
from datetime import datetime
from tempfile import NamedTemporaryFile

import boto3
from celery import Celery

from .constants import (
    ARTIFACT_BUCKET,
    RUNS_TABLE,
    QUEUE_NAME,
)
from .inference import run_inference

# ----------------------------------------------------------------------
BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost")  # dev default

celery = Celery(__name__, broker=BROKER_URL)

# AWS clients (inside worker container)
_s3        = boto3.client("s3",  region_name="eu-west-2")
_dynamodb = boto3.resource("dynamodb", region_name="eu-west-2")
_table    = _dynamodb.Table(RUNS_TABLE)


# ----------------------------------------------------------------------
@celery.task(name="run_task")  # explicit name so we can change module paths later
def run_task(run_id: str, s3_key: str) -> None:
    """
    Receive message → run inference → upload labels → update DDB.
    """
    # 1. Download image from S3
    with NamedTemporaryFile(suffix=".jpg") as tmp:
        _s3.download_file(ARTIFACT_BUCKET, s3_key, tmp.name)

        # 2. Run ML (stubbed)
        labels = run_inference(tmp.name)

    # 3. Upload outputs JSON
    out_key = f"outputs/{run_id}.json"
    with NamedTemporaryFile(suffix=".json", mode="w+") as tmp_json:
        json.dump(labels, tmp_json)
        tmp_json.flush()
        _s3.upload_file(tmp_json.name, ARTIFACT_BUCKET, out_key)

    # 4. Update DynamoDB
    _table.update_item(
        Key={"runId": run_id},
        UpdateExpression="SET #s = :s, outputKey = :k, updatedAt = :u",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={
            ":s": "done",
            ":k": out_key,
            ":u": datetime.utcnow().isoformat(timespec="seconds"),
        },
    )
