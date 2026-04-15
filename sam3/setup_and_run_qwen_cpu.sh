#!/bin/bash

# Ensure we have the server dependencies
pip install "llama-cpp-python[server]"

MODEL_REPO="bartowski/Qwen2-VL-2B-Instruct-GGUF"
MODEL_FILE="Qwen2-VL-2B-Instruct-Q4_K_M.gguf"
MMPROJ_FILE="mmproj-Qwen2-VL-2B-Instruct-f16.gguf"

mkdir -p models

echo "Downloading model files from $MODEL_REPO..."
huggingface-cli download $MODEL_REPO $MODEL_FILE --local-dir ./models --local-dir-use-symlinks False
huggingface-cli download $MODEL_REPO $MMPROJ_FILE --local-dir ./models --local-dir-use-symlinks False

echo "Starting llama-cpp-python server..."
# --clip_model_path is the argument for mmproj in llama-cpp-python server
# --chat_format qwen2-vl is supported in newer versions

python -m llama_cpp.server \
    --model ./models/$MODEL_FILE \
    --clip_model_path ./models/$MMPROJ_FILE \
    --n_ctx 4096 \
    --host 0.0.0.0 \
    --port 8000 \
    --chat_format qwen2-vl 
