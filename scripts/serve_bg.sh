#!/usr/bin/env bash
# Start the demo server in the background (used by screenshot and smoke-test scripts).
# Usage: scripts/serve_bg.sh [port] [speed] [replay]
set -euo pipefail
PORT="${1:-8077}"; SPEED="${2:-6000}"; REPLAY="${3:-svb_2023-03}"
cd "$(dirname "$0")/.."
(fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true)
PYTHONPATH=src TRIDENT_SPEED="$SPEED" TRIDENT_REPLAY="$REPLAY" nohup python3 -m trident demo --port "$PORT" > /tmp/trident_srv.log 2>&1 &
for i in $(seq 1 30); do
  if curl -sf "http://localhost:${PORT}/healthz" >/dev/null; then echo "server up on :${PORT}"; exit 0; fi
  sleep 1
done
echo "server failed to start"; tail -20 /tmp/trident_srv.log; exit 1
