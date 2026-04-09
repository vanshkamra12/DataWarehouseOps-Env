#!/bin/bash
set -e

# 1. Start the FastAPI server in the background
# We run it on port 7860 as expected by HF Spaces
echo "--- Starting DataWarehouseOps-Env Server ---"
uvicorn server.app:app --host 0.0.0.0 --port 7860 &

# 2. Wait for the server to be healthy
echo "--- Waiting for server readiness ---"
until curl -s http://localhost:7860/health > /dev/null; do
  sleep 1
done

# 3. Run the inference baseline script
# This script prints [START], [STEP], [END] to stdout
# The platform's evaluation pipeline reads this stdout for scoring and LLM validation.
echo "--- Running LLM Baseline Inference ---"
python inference.py

# 4. Keep the container alive after inference finishes if needed, 
# although usually the platform kills it after parsing [END]
wait
