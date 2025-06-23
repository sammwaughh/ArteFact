# ArteFact v2

<img src="viewer/public/images/logo-16-9.JPEG" alt="ArteFact Logo" width="400">

Modern web application for analyzing artwork using machine-learning models.  
Upload paintings and get academic annotations, contextual labels, and confidence scores in real-time.

🌐 **Live Demo**: [artefactcontext.com](https://artefactcontext.com)

---

## Components at a Glance

| Layer | Component | What it runs | Purpose |
|-------|-----------|--------------|---------|
| **Front-end** | React/TypeScript SPA | Browser | Upload UI, live-status polling, IIIF-style viewer |
| **Runner Service** | Flask API (`runner-svc`) | Fargate task (ECS) | Presigned upload URLs, job creation, status endpoint |
| **Worker Service** | Celery worker (`worker-svc`) | Fargate task (ECS) | ML inference + post-processing |
| **Storage** | S3 (`artefact-context-artifacts-eu2`) | AWS | Images **and** JSON label outputs |
| **Queue** | SQS (`hc-artifacts-queue`) | AWS | Decouples API from heavy ML work |
| **Metadata** | DynamoDB (`hc-runs`) | AWS | Run status & bookkeeping |

---

## How the Pieces Talk 🚦

```text
Browser ──(1) POST /presign────────► Runner API ──▶ S3   (presigned PUT URL)
Browser ──(2) PUT image to S3
Browser ──(3) POST /runs──────────► Runner API ──▶ SQS   (job message)
Worker ◄────────────────────────────── SQS
Worker ──▶ run_inference ──▶ S3 (outputs/<id>.json)
Worker ──▶ DynamoDB status = done
Browser ──(4) poll /runs/<id>────────► Runner API (reads DDB)
Browser ──(5) GET outputs/<id>.json ─► CloudFront → S3   (labels)
SPA overlays red boxes ⇢ 🎉
