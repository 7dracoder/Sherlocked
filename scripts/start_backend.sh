#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  set -a
  source .env
  set +a
fi

PORT="${SHERLOCKED_PORT:-8787}"
HOST="${SHERLOCKED_HOST:-0.0.0.0}"

exec "$ROOT_DIR/.venv/bin/uvicorn" backend.server:app --host "$HOST" --port "$PORT"
