# ArteFact v2 &nbsp;&nbsp;<img src="viewer/public/images/logo-16-9.JPEG" alt="ArteFact Logo" width="320">

Modern web application for analysing artwork with machine-learning models.  
Upload paintings, then receive academic annotations, contextual labels and confidence scores – all in real-time (seconds, not minutes).

🌐 **Live demo:** <https://artefactcontext.com>

---

## Components at a Glance

| Layer          | Component / Resource                  | Runtime / Service           | Role in the system |
|----------------|---------------------------------------|-----------------------------|--------------------|
| **Front-end**  | React / Vite SPA                      | Browser (+ CloudFront)      | Upload UI, polling, deep-zoom viewer |
| **Runner**     | Flask API `runner-svc`                | ECS Fargate task            | Presigned uploads, create runs, status endpoint |
| **Worker**     | Celery worker `worker-svc`            | ECS Fargate task            | ML inference · label-file creation |
| **Storage**    | `artefact-context-artifacts-eu2` (S3) | Amazon S3 (+ CF origin)     | Holds both *uploaded images* and *JSON outputs* |
| **Queue**      | `hc-artifacts-queue` (SQS)            | Amazon SQS                  | Decouples “quick API” from “slow AI” |
| **Metadata**   | `hc-runs` (DynamoDB)                  | Amazon DynamoDB             | Persistent run status & timing metrics |
| **Delivery**   | `E2UQ55MZYMCNFO` (CloudFront dist.)   | Amazon CloudFront           | Serves SPA & artefacts privately (via OAI) |
| **Ingress**    | Public ALB + ACM cert                 | Load Balancing              | TLS termination for `api.artefactcontext.com` |

---

## 🛠 How the app works – request-by-request

<details>
<summary>Click to expand the sequence diagram ▶</summary>

```text
User selects image
└─► (1) POST /presign …………………………… Runner API
        ↳ returns presigned  <S3 POST URL + fields>

Browser performs multipart/form-data POST directly to S3 (2)
        S3 stores artifacts/<runId>.jpg   (private)

Browser POSTs /runs  (3) ………………… Runner API
        • putItem   hc-runs            status=queued
        • publish   SQS message        {runId, s3Key}

Celery worker (4) pulls SQS
        • GET image from S3
        • run_inference()
        • PUT outputs/<runId>.json     to S3
        • updateItem hc-runs           status=done

Browser polls /runs/<id> every 3 s
        As soon as status=done:
        GET /outputs/<id>.json  (5) … CloudFront → S3
        GET /artifacts/<id>.jpg (6) … CloudFront → S3
        React renders image + overlays  ✅
```
</details>

### Detailed walk-through

| #                     | Actor / Service          | Action & headers                                                                                                                  | Result |
|-----------------------|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------|--------|
| **1 Presign**         | **Browser → Runner API** | `POST /presign {fileName}`                                                                                                        | Flask issues a *presigned POST* for the artifacts bucket, plus a fresh `runId`. |
|                       | **API → Browser**        | JSON with `upload.url`, `upload.fields`, `runId`, `s3Key`.                                                                        | Browser can now upload without exposing AWS keys. |
| **2 Upload**          | **Browser → S3**         | HTML `<form>` POST to `https://artefact-context-artifacts-eu2.s3…` with policy + sig.                                            | S3 stores the image, returns **204** and CORS headers. |
| **3 Create run**      | **Browser → Runner API** | `POST /runs {runId, s3Key}`                                                                                                       | API writes a **queued** item in DynamoDB and publishes a Celery task to **hc-artifacts-queue** (SQS). |
| **4 Process**         | **Worker Fargate task**  | Long-polls SQS, receives task, downloads image, runs the ML model, uploads `outputs/<id>.json`, updates DynamoDB (`status=done`). | Heavy lifting kept off the API path. |
| **5 Poll**            | **Browser → Runner API** | Repeats `GET /runs/<id>` until `status===done`.                                                                                   | API simply reads DynamoDB; negligible latency. |
| **6 Fetch artefacts** | **Browser → CloudFront** | `GET /outputs/<id>.json` then `GET /artifacts/<id>.jpg`.                                                                          | CloudFront presents its **Origin Access Identity (OAI)** to the bucket; S3 serves the objects even though **Block Public Access is ON**. First request is a *Miss*, subsequent hits are cached at the edge. |
| **7 Render**          | **React SPA**            | Receives `{labels, imageUrl}` from hook state.                                                                                    | Shows the painting and (if enabled) bounding-box overlays. |

**Round-trip time:** ≈ 3 – 5 s for a 400 KB JPEG and a small label file.

---

## Local development

```bash
# 1. Run the stack (LocalStack + API + worker)
docker compose up --build --detach

# 2. Front-end (Vite dev-server hot-reloads on :5173)
cd viewer
npm ci && npm run dev
```

The compose file points the Python code at `AWS_ENDPOINT_URL=http://localstack:4566`; Moto-style mocks for S3, SQS and DynamoDB are auto-bootstrapped on first run.

---

## Deployment workflow (CI/CD)

1. **push → main** triggers **`.github/workflows/deploy.yml`**  
   • Builds the SPA, uploads to *viewer-spa-prod* bucket.  
   • Invalidates CloudFront (`aws cloudfront create-invalidation`).  
   • Builds & pushes `viewer-backend` Docker image to ECR.  
   • `aws ecs update-service --force-new-deployment` for both tasks.

2. **cdk synth / deploy** in `infra/cdk` manages:  
   VPC, ALB, ECS services, IAM, SQS, DynamoDB and bucket policy (including the OAI grant).

Secrets required (`Settings → Secrets & variables`):

| Name                                          | Used for                                |
|-----------------------------------------------|-----------------------------------------|
| `CF_DIST_ID`                                  | CloudFront invalidation                 |
| `CF_URL`                                      | Passed to Vite as `VITE_CLOUDFRONT_URL` |
| `SPA_BUCKET`                                  | SPA origin bucket                       |
| `ECR_REPO`                                    | Docker push target                      |
| `CLUSTER`, `RUNNER_SERVICE`, `WORKER_SERVICE` | `aws ecs update-service`                |

---
