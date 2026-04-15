#!/bin/bash

# Install/Update vLLM to ensure dependencies are correct
# echo "Installing/Updating vLLM..."
# pip install --upgrade vllm

# Model name provided by user
MODEL_NAME="Qwen/Qwen3-VL-2B-Instruct"

echo "Starting vLLM server for model: $MODEL_NAME"
echo "Server will be available at http://localhost:8000/v1"

# Launch vLLM server
# --trust-remote-code is often needed for newer/custom models
# --gpu-memory-utilization can be adjusted if OOM occurs
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --trust-remote-code \
    --port 8000 \
    --dtype auto \
    --gpu-memory-utilization 0.6 \
    --max-model-len 16384 \
    --api-key "EMPTY"

