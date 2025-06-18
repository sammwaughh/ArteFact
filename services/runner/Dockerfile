# ---------- Base image -------------------------------------------------
FROM python:3.11-slim AS base

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ---------- Dependencies ----------------------------------------------
COPY services/runner/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Application code ------------------------------------------
COPY services/runner /app

# We parametrize the command at ECS‑task‑definition level.
# Runner‑svc:   ["python", "app.py"]
# Worker‑svc:   ["celery", "-A", "tasks", "worker", "--loglevel=info"]
CMD ["python", "app.py"]
