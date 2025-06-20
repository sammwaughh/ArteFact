"""
Tiny smoke test: starts Flask app w/ test_client + Moto mocks
S3 - DynamoDB - SQS.

⚠️ The test region **must** match the constant in the runner service
(`_REGION = "eu-west-2"`). Moto resources created in a different region
won’t be visible to the code under test.
"""

from __future__ import annotations
import os

# --------------------------------------------------------------------------- #
#  Third-party                                                               #
# --------------------------------------------------------------------------- #
from moto import mock_aws
import boto3

#  Local imports (after env vars are set)                                     #
from hc_services.runner.app import app
from hc_services.runner.constants import (
    ARTIFACT_BUCKET,
    RUNS_TABLE,
    QUEUE_NAME,
)

# --------------------------------------------------------------------------- #
#  Dummy credentials + default region so SigV4 & boto3 don’t error            #
# --------------------------------------------------------------------------- #
_REGION = "eu-west-2"  # keep in sync with hc_services.runner.app

os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("AWS_DEFAULT_REGION", _REGION)


# --------------------------------------------------------------------------- #
@mock_aws
def test_presign_and_create_run() -> None:
    """Happy-path: /presign then /runs returns 202 Accepted."""
    # -- Moto infra ---------------------------------------------------------
    s3 = boto3.client("s3", region_name=_REGION)
    s3.create_bucket(
        Bucket=ARTIFACT_BUCKET,
        CreateBucketConfiguration={"LocationConstraint": _REGION},
    )

    ddb = boto3.client("dynamodb", region_name=_REGION)
    ddb.create_table(
        TableName=RUNS_TABLE,
        KeySchema=[{"AttributeName": "runId", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "runId", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )

    sqs = boto3.client("sqs", region_name=_REGION)
    sqs.create_queue(QueueName=QUEUE_NAME)  # required by /runs endpoint

    # -- Flask client -------------------------------------------------------
    client = app.test_client()

    presign_resp = client.post("/presign", json={"fileName": "x.jpg"})
    assert presign_resp.status_code == 200
    data = presign_resp.get_json()
    assert "runId" in data

    run_id, s3_key = data["runId"], data["s3Key"]
    create_resp = client.post("/runs", json={"runId": run_id, "s3Key": s3_key})
    assert create_resp.status_code == 202
