#!/usr/bin/env bash
set -euo pipefail

if [ ! -f model.bin ]; then
  echo "model.bin not found — training model (this may take a while)..."
  python train.py
else
  echo "Found model.bin — skipping training."
fi

echo "Starting FastAPI (uvicorn)..."
exec uvicorn predict:app --host 0.0.0.0 --port 9696
