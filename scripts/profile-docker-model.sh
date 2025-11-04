#!/bin/bash
#
# Convenience wrapper to profile a model running in a Docker container
# Automatically detects vLLM vs HuggingFace Transformers
#

set -e

CONTAINER_NAME=$1
shift

if [ -z "$CONTAINER_NAME" ]; then
    echo "Usage: $0 <container-name> [profile options]"
    echo ""
    echo "Examples:"
    echo "  $0 vllm-inference"
    echo "  $0 vllm-inference --max-tokens 200 --batch-size 2"
    echo "  $0 llama2-7b --model /path/to/model"
    echo ""
    echo "Available options:"
    echo "  --model NAME/PATH     Model name or path"
    echo "  --prompt TEXT         Prompt text (default: 'Hello, how are you?')"
    echo "  --max-tokens N        Max tokens to generate (default: 100)"
    echo "  --batch-size N        Batch size (default: 1)"
    echo "  --dtype TYPE          Data type: auto|float16|bfloat16|float32 (default: auto)"
    echo "  --model-params N      Number of parameters (for better estimates)"
    exit 1
fi

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Container '$CONTAINER_NAME' is not running"
    echo ""
    echo "Running containers:"
    docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
    exit 1
fi

echo "=== Profiling Model in Container: $CONTAINER_NAME ==="
echo ""

# Create results directory
RESULTS_DIR="./results/memory-profiles"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${RESULTS_DIR}/${CONTAINER_NAME}_${TIMESTAMP}.json"

# Detect if container is running vLLM
echo "[1/5] Detecting framework..."
IS_VLLM=$(docker exec "$CONTAINER_NAME" bash -c 'ps aux | grep -i "vllm" | grep -v grep' 2>/dev/null || echo "")

if [ -n "$IS_VLLM" ]; then
    echo "  Detected: vLLM API server"
    FRAMEWORK="vllm"
    
    # Detect API port (default 8000)
    API_PORT=$(docker port "$CONTAINER_NAME" 2>/dev/null | grep "8000/tcp" | head -1 | cut -d':' -f2 | xargs || echo "8000")
    API_URL="http://localhost:${API_PORT}"
    
    echo "  API URL: $API_URL"
    
    # Copy vLLM profiler script
    echo "[2/5] Copying vLLM profiler script to container..."
    docker cp scripts/profile-vllm-model.py "${CONTAINER_NAME}:/tmp/"
    
    # Install requests if needed
    echo "[3/5] Checking dependencies..."
    docker exec "$CONTAINER_NAME" bash -c 'python3 -c "import requests" 2>/dev/null || pip install -q requests' > /dev/null 2>&1
    
    # Extract model name from arguments or auto-detect
    MODEL_NAME=""
    for arg in "$@"; do
        if [[ "$arg" == --model=* ]]; then
            MODEL_NAME="${arg#*=}"
        elif [[ "$prev_arg" == "--model" ]]; then
            MODEL_NAME="$arg"
        fi
        prev_arg="$arg"
    done
    
    if [ -z "$MODEL_NAME" ]; then
        echo "[4/5] Auto-detecting model name..."
        MODEL_NAME=$(docker exec "$CONTAINER_NAME" curl -s "http://localhost:8000/v1/models" 2>/dev/null | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['data'][0]['id'] if 'data' in data and len(data['data']) > 0 else '')" || echo "")
        if [ -n "$MODEL_NAME" ]; then
            echo "  Detected model: $MODEL_NAME"
        fi
    else
        echo "[4/5] Using provided model: $MODEL_NAME"
    fi
    
    # Run vLLM profiling
    echo "[5/5] Running memory profiler..."
    echo ""
    
    docker exec -it "$CONTAINER_NAME" python3 /tmp/profile-vllm-model.py \
        --api-url "$API_URL" \
        ${MODEL_NAME:+--model "$MODEL_NAME"} \
        --output /tmp/memory-profile.json \
        "$@" || {
        echo ""
        echo "Error: Profiling failed. Common issues:"
        echo "  - vLLM API not accessible at $API_URL"
        echo "  - Model not loaded or incorrect model name"
        echo ""
        echo "Try checking vLLM logs:"
        echo "  docker logs $CONTAINER_NAME | tail -20"
        exit 1
    }
    
else
    echo "  Detected: HuggingFace Transformers"
    FRAMEWORK="transformers"
    
    # Copy standard profiler script
    echo "[2/5] Copying profiler script to container..."
    docker cp scripts/profile-model-memory.py "${CONTAINER_NAME}:/tmp/"
    
    # Detect model path in container (common locations)
    echo "[3/5] Detecting model path..."
    MODEL_PATH=$(docker exec "$CONTAINER_NAME" bash -c '
        for path in /model /models /workspace/model /app/model $MODEL_PATH $HF_MODEL; do
            if [ -d "$path" ] && [ -f "$path/config.json" ]; then
                echo "$path"
                exit 0
            fi
        done
        # If not found, try to find via Python
        python3 -c "import os; print(os.environ.get(\"MODEL_PATH\", os.environ.get(\"HF_MODEL\", \"/model\")))" 2>/dev/null || echo "/model"
    ' | head -1)
    
    echo "  Detected model path: $MODEL_PATH"
    
    # Check if model path was explicitly provided in args
    CUSTOM_MODEL=""
    for arg in "$@"; do
        if [[ "$arg" == --model=* ]]; then
            CUSTOM_MODEL="${arg#*=}"
        fi
    done
    
    if [ -n "$CUSTOM_MODEL" ]; then
        MODEL_PATH="$CUSTOM_MODEL"
        echo "  Using custom model path: $MODEL_PATH"
    fi
    
    # Run profiling
    echo "[4/5] Running memory profiler..."
    echo ""
    
    docker exec -it "$CONTAINER_NAME" python3 /tmp/profile-model-memory.py \
        --model "$MODEL_PATH" \
        --output /tmp/memory-profile.json \
        "$@" || {
        echo ""
        echo "Error: Profiling failed. Common issues:"
        echo "  - Model path incorrect (detected: $MODEL_PATH)"
        echo "  - PyTorch not installed in container"
        echo "  - Insufficient GPU memory"
        echo ""
        echo "Try specifying model path explicitly:"
        echo "  $0 $CONTAINER_NAME --model /path/to/model"
        exit 1
    }
fi

# Copy results out
echo ""
echo "[Done] Copying results..."
docker cp "${CONTAINER_NAME}:/tmp/memory-profile.json" "$OUTPUT_FILE"

echo ""
echo "=== Profiling Complete ==="
echo ""
echo "Results saved to: $OUTPUT_FILE"
echo ""

# Display summary
if command -v jq &> /dev/null; then
    echo "Memory Breakdown:"
    jq '.memory_breakdown' "$OUTPUT_FILE"

    echo ""
    echo "Model Info:"
    jq '.model_info | {model_name, num_parameters, total_sequence_length}' "$OUTPUT_FILE"

    if jq -e '.gpu_info.num_gpus > 1' "$OUTPUT_FILE" &> /dev/null; then
        echo ""
        echo "Multi-GPU Info:"
        jq '.gpu_info | {gpu_type, num_gpus, total_memory_gb}' "$OUTPUT_FILE"
    fi
else
    echo "Install 'jq' for formatted output:"
    echo "  sudo apt install jq"
    echo ""
    echo "Or view raw JSON:"
    echo "  cat $OUTPUT_FILE"
fi

echo ""
echo "Next steps:"
echo "  1. Open the calculator in your browser"
echo "  2. Input the same model parameters"
echo "  3. Compare memory_breakdown values"
echo "  4. Adjust calculator formulas if needed"
