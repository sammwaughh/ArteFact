"""
Tiny smoke test: starts Flask app w/ test_client + Moto mocks S3 & DDB.
"""

from moto import mock_s3, mock_dynamodb
import boto3
import json

from services.runner.app import app
from services.runner.constants import ARTIFACT_BUCKET, RUNS_TABLE

@mock_s3
@mock_dynamodb
def test_presign_and_create_run():
    # --- Moto infra -------------------------------------------------------
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=ARTIFACT_BUCKET)

    ddb = boto3.client("dynamodb", region_name="us-east-1")
    ddb.create_table(
        TableName=RUNS_TABLE,
        KeySchema=[{"AttributeName": "runId", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "runId", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )

    # --- Flask test_client ------------------------------------------------
    client = app.test_client()

    resp = client.post("/presign", json={"fileName": "x.jpg"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert "runId" in body

    run_id, s3_key = body["runId"], body["s3Key"]
    resp2 = client.post("/runs", json={"runId": run_id, "s3Key": s3_key})
    assert resp2.status_code == 202
