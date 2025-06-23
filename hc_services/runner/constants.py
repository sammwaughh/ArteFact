"""
Centralised names/URLs so we never scatter literal strings.
All values can be overridden by env‑vars for prod vs. local.
"""

import os

ARTIFACT_BUCKET: str = os.getenv("ARTIFACT_BUCKET", "artefact-context-artifacts-eu2")
MODEL_BUCKET: str = os.getenv("MODEL_BUCKET", "hc-model-assets")
RUNS_TABLE: str = os.getenv("RUNS_TABLE", "hc-runs")
QUEUE_NAME: str = os.getenv("QUEUE_NAME", "hc-artifacts-queue")
CLOUDFRONT_URL: str = os.getenv(
    "CLOUDFRONT_URL", "https://dxxxxxxxx.cloudfront.net"
)  # TODO replace
