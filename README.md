# ArteFact v2 Test Reboot MacBook

<img src="viewer/public/images/logo-16-9.JPEG" alt="ArteFact Logo" width="400">

Modern web application for analyzing artwork using machine learning models. Upload paintings and get academic annotations, contextual labels, and confidence scores in real-time.

🌐 **Live Demo**: [artefactcontext.com](https://artefactcontext.com)

### Components

- **Front-end**: React/TypeScript SPA with real-time status updates
- **Runner Service**: Flask API handling presigned URLs and job coordination
- **Worker Service**: Celery-based ML inference worker consuming SQS messages
- **Storage**: S3 for images, DynamoDB for metadata, SQS for job queue

## Local Development

### Prerequisites

- Docker Desktop
- Node.js 20+
- Python 3.11+
- AWS CDK v2

### Quick Start

1. Clone and install dependencies:

```bash
git clone <repo-url>
cd artefact-context

# Front-end
cd viewer
npm install

# Back-end
pip install -e ".[dev]"
```

2. Start the local stack:

```bash
docker compose up -d
```

This launches:

- LocalStack (S3, SQS, DynamoDB emulation)
- Flask runner service
- Celery worker service

3. Start the development SPA:

```bash
cd viewer
npm run dev
```

Visit http://localhost:5173

### Testing

```bash
# Front-end tests
cd viewer
npm test

# Back-end tests (with Moto AWS mocking)
pytest -q

# Manual long-task test
SLEEP_SECS=90 docker compose run --rm -e SLEEP_SECS=90 worker
```

### Infrastructure

AWS resources are defined in TypeScript using CDK:

```bash
cd infra/cdk
npm install
npx cdk synth        # Generate CloudFormation
npx cdk diff         # Review changes
npx cdk deploy       # Deploy to AWS
```

Key resources:

- S3 bucket (`hc-artifacts`) - Image storage
- SQS queue (`hc-artifacts-queue`) - Task queue
- DynamoDB table (`hc-runs`) - Run metadata
- ECS services - Runner and worker containers

## CI/CD

GitHub Actions workflow:

1. Builds and tests front-end
2. Runs Python tests with LocalStack
3. Deploys infrastructure via CDK
4. Pushes container images to ECR
5. Updates ECS services

## Architecture Details

### Request Flow

1. SPA requests presigned S3 URL from runner
2. SPA uploads image directly to S3
3. Runner queues task in SQS
4. Worker consumes message, processes image
5. Worker updates DynamoDB with results
6. SPA polls runner until complete

### Error Handling

- Graceful failure display in UI
- Task retries via Celery
- DynamoDB status tracking
- 5-minute SQS visibility timeout
