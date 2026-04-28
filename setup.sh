#!/bin/bash
# One-click install script for ToThinkVision
set -e

echo "=== ToThinkVision Setup ==="
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "Creating directories..."
mkdir -p uploads outputs models

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the server:"
echo "  cd /Users/a1-6/Documents/kaggle/ToThinkVision"
echo "  uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "Then visit: http://localhost:8000"
echo ""
echo "NOTE: First run will download model weights automatically."
echo "For production, pre-download models by setting MODEL_CACHE_DIR env var."
