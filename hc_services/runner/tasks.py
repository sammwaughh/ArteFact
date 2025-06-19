"""
Celery configuration + asynchronous job definition (worker‑svc).
"""

import json
import os
import time                                  # ← add
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

# NEW – default queue name so worker declares/consumes it
celery.conf.task_default_queue = QUEUE_NAME

# AWS clients (inside worker container)
_s3        = boto3.client("s3",  region_name="eu-west-2")
_dynamodb = boto3.resource("dynamodb", region_name="eu-west-2")
_table    = _dynamodb.Table(RUNS_TABLE)

FORCE_ERROR = os.getenv("FORCE_ERROR") == "1"

SLEEP_SECS = int(os.getenv("SLEEP_SECS", "0"))   # ← add

# ----------------------------------------------------------------------
@celery.task(name="run_task", autoretry_for=(Exception,), retry_backoff=True,
             retry_kwargs={"max_retries": 2})
def run_task(run_id: str, s3_key: str) -> None:
    """
    Receive message → run inference → upload labels → update DDB.
    """
    # --- mark as processing ------------------------------------------
    _table.update_item(
        Key={"runId": run_id},
        UpdateExpression="SET #s = :s, startedAt = :t",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={
            ":s": "processing",
            ":t": datetime.utcnow().isoformat(timespec="seconds"),
        },
    )

    try:
        # 1. download image
        with NamedTemporaryFile(suffix=".jpg") as tmp:
            _s3.download_file(ARTIFACT_BUCKET, s3_key, tmp.name)

        if SLEEP_SECS:                         # simulate slow inference
            time.sleep(SLEEP_SECS)

        labels = run_inference(tmp.name)

        # DEV-only toggle (kept for future manual tests)
        # if FORCE_ERROR:
        #     raise RuntimeError("forced-error test")

        # 2. upload outputs
        out_key = f"outputs/{run_id}.json"
        with NamedTemporaryFile(suffix=".json", mode="w+") as jf:
            json.dump(labels, jf); jf.flush()
            _s3.upload_file(jf.name, ARTIFACT_BUCKET, out_key)

        # 3. mark done
        _table.update_item(
            Key={"runId": run_id},
            UpdateExpression="""
              SET #s = :s, outputKey = :k,
                  finishedAt = :t, updatedAt = :t
            """,
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={
                ":s": "done",
                ":k": out_key,
                ":t": datetime.utcnow().isoformat(timespec="seconds"),
            },
        )

    except Exception as exc:
        _table.update_item(
            Key={"runId": run_id},
            UpdateExpression="SET #s = :s, errorMessage = :e, updatedAt = :t",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={
                ":s": "error",
                ":e": str(exc)[:500],                       # keep concise
                ":t": datetime.utcnow().isoformat(timespec="seconds"),
            },
        )
        raise   # lets Celery retry up to max_retries
