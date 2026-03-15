#!/bin/bash
# ============================================================
# F1 Australian GP 2026 Predictor - Setup Script
# ============================================================
# Run this after cloning or downloading the project.
# Usage: bash setup.sh

set -e

echo "Setting up F1 Predictor project..."

# Python environment
echo "[1/4] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "[2/4] Installing Python dependencies..."
pip install -r requirements.txt

echo "[3/4] Running model..."
python src/predict.py

echo "[4/4] Running tests..."
python tests/test_model.py

echo ""
echo "Setup complete."
echo ""
echo "To start the React dashboard:"
echo "  cd dashboard"
echo "  npm install"
echo "  npm run dev"
echo ""
echo "To re-run the model:"
echo "  source venv/bin/activate"
echo "  python src/predict.py"
