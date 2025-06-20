# ────────────────────────────────────────────────────────────────────────
#  hc_services/runner/Dockerfile
#  (image is shared by both the Flask “runner” and the Celery “worker”)
# ────────────────────────────────────────────────────────────────────────

# ---------- Base image -------------------------------------------------
FROM python:3.11-slim AS runtime

# Faster installs on slim images
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Make Python happier in containers
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ---------- Python dependencies ---------------------------------------
# Copy only the requirements file first so Docker can cache the layer.
COPY hc_services/runner/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Application code ------------------------------------------
# Copy the service source tree *after* the deps layer for better caching.
COPY hc_services/runner /app

# ---------- Entrypoint -------------------------------------------------
# We now run the Flask API as a *module* so Python finds it inside the
# package we renamed to `hc_services`.  The same image is reused for the
# Celery worker:
#
#   • Runner API  → default CMD below
#   • Worker      → override with:
#       ["celery", "-A", "hc_services.runner.tasks", "worker", "--loglevel=info"]
#
CMD ["python", "-m", "hc_services.runner.app"]
