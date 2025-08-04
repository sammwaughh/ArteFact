#!/usr/bin/env bash
set -euo pipefail

# ── 1. Activate the venv ────────────────────────────────────────────────────
source .venv/bin/activate

# ── 2. Array to remember background PIDs ────────────────────────────────────
pids=()

# ── 3. Define a graceful clean-up function ──────────────────────────────────
cleanup() {
  echo                      # newline for readability
  echo "⏹  Stopping…"

  # Ask each child to exit nicely (CTRL-C equivalent)
  kill -INT "${pids[@]}" 2>/dev/null || true

  # Give them a second to run atexit handlers, flush logs, etc.
  sleep 1

  # Anything still alive?  Send SIGTERM, then SIGKILL as a last resort.
  kill -TERM "${pids[@]}" 2>/dev/null || true
  sleep 1
  kill -KILL "${pids[@]}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT   # run on Ctrl-C or normal exit

# ── 4. Start backend and frontend in parallel ──────────────────────────────
(
  cd backend
  ./run_backend.sh           # Flask on :8000
) &
pids+=("$!")                 # remember its PID

(
  cd frontend
  ./run_frontend.sh          # http.server on :8080
) &
pids+=("$!")                 # remember its PID

# ── 5. Wait for both children so the script stays in the foreground ─────────
wait "${pids[@]}"
