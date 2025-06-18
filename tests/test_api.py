"""
Tiny smoke test: starts Flask app w/ test_client + Moto mocks S3 & DynamoDB.

The test region **must** match the service’s hard-coded `_REGION = "eu-west-2"`
(otherwise Moto will create resources in the wrong partition and the service
will fail with ResourceNotFound).
"""

from __future__ import annotations

import os
import json

# --------------------------------------------------------------------------- #
#  Configure dummy credentials & default region for boto3                     #
# --------------------------------------------------------------------------- #
_REGION = "eu-west-2"  # keep in sync with hc_services.runner.app

os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("AWS_DEFAULT_REGION", _REGION)

# --------------------------------------------------------------------------- #
from moto import mock_s3, mock_dynamodb  # noqa: E402  (import after env vars)
import boto3
from hc_services.runner.app import app
from hc_services.runner.constants import ARTIFACT_BUCKET, RUNS_TABLE

# --------------------------------------------------------------------------- #
@mock_s3
@mock_dynamodb
def test_presign_and_create_run() -> None:
    """Happy-path: presign upload → create run → 202 Accepted."""
    # -- Moto infra ---------------------------------------------------------
    s3 = boto3.client("s3", region_name=_REGION)
    s3.create_bucket(
    Bucket=ARTIFACT_BUCKET,
    CreateBucketConfiguration={"LocationConstraint": _REGION},  # NEW ✔
    )

    ddb = boto3.client("dynamodb", region_name=_REGION)
    ddb.create_table(
        TableName=RUNS_TABLE,
        KeySchema=[{"AttributeName": "runId", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "runId", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )

    # -- Flask client -------------------------------------------------------
    client = app.test_client()

    presign_resp = client.post("/presign", json={"fileName": "x.jpg"})
    assert presign_resp.status_code == 200
    data = presign_resp.get_json()
    assert "runId" in data

    run_id, s3_key = data["runId"], data["s3Key"]

    create_resp = client.post("/runs", json={"runId": run_id, "s3Key": s3_key})
    assert create_resp.status_code == 202
