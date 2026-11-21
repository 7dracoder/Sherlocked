#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=== [1/4] Running Pre-Application Research Harness & Weight Calibration ==="
"$ROOT_DIR/.venv/bin/python" research/run_experiments.py

echo "=== [2/4] Running Backend & Multimodal Acceptance Pytest Suite ==="
"$ROOT_DIR/.venv/bin/pytest" -v

echo "=== [3/4] Starting Live Sherlocked Backend Server on :8787 ==="
"$ROOT_DIR/.venv/bin/uvicorn" backend.server:app --host 127.0.0.1 --port 8787 >/tmp/sherlocked_server.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

for _ in {1..30}; do
  if curl -fsS http://127.0.0.1:8787/health >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done
curl -fsS http://127.0.0.1:8787/health | python3 -m json.tool

echo "=== [4/4] Running Xcode iOS Unit + UI Acceptance Tests ==="
cd "$ROOT_DIR/ios"
xcodegen generate
xcodebuild test \
  -project Sherlocked.xcodeproj \
  -scheme Sherlocked \
  -destination 'platform=iOS Simulator,name=iPhone 18 Pro' \
  -derivedDataPath DerivedData

echo "=== All automated verifications passed! ==="
