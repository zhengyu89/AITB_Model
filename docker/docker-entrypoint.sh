#!/bin/bash
set -euo pipefail

APP_ENV=${APP_ENV:-production}
APP_PORT=${APP_PORT:-8000}
UVICORN_WORKERS=${UVICORN_WORKERS:-1}
QDRANT_URL=${QDRANT_URL:-http://localhost:6333}
QDRANT_COLLECTION=${QDRANT_COLLECTION:-malaysia_landmarks}
ATTRACTION_CHECKPOINT=my_landmark_attraction.pth
FOOD_CHECKPOINT=my_landmark_food.pth

# Smoke tests: Check critical environment variables before starting
echo "=== Running startup smoke tests ==="

# Check Qdrant URL (required)
if [ -z "${QDRANT_URL:-}" ]; then
  echo "❌ ERROR: QDRANT_URL is not set"
  exit 1
fi
echo "✓ QDRANT_URL is set: ${QDRANT_URL}"

# Check Qdrant collection (required)
if [ -z "${QDRANT_COLLECTION:-}" ]; then
  echo "❌ ERROR: QDRANT_COLLECTION is not set"
  exit 1
else
  echo "✓ QDRANT_COLLECTION is set: ${QDRANT_COLLECTION}"
fi

if [ -f "${ATTRACTION_CHECKPOINT}" ] || [ -f "${FOOD_CHECKPOINT}" ]; then
  echo "✓ At least one checkpoint file is available"
else
  echo "❌ ERROR: Missing checkpoints. Expected ${ATTRACTION_CHECKPOINT} and/or ${FOOD_CHECKPOINT}"
  exit 1
fi

if [ -n "${API_KEY:-}" ]; then
  echo "✓ API_KEY protection is enabled"
else
  echo "⚠️  WARNING: API_KEY is not set; /api/v1 routes will be public"
fi

echo "=== Smoke tests completed ==="
echo ""

if [ "$APP_ENV" = "local" ]; then
  echo "Running in LOCAL mode with auto-reload"
  exec uvicorn app.main:app --host 0.0.0.0 --port "$APP_PORT" --reload
else
  echo "Running in PRODUCTION mode"
  exec uvicorn app.main:app --host 0.0.0.0 --port "$APP_PORT" --workers "$UVICORN_WORKERS"
fi
