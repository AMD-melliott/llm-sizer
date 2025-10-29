#!/bin/bash
#
# Convenience wrapper to profile a model running in a Docker container
#
# Usage:
#   ./scripts/profile-docker-model.sh <container-name> [options]
#
# Examples:
#   ./scripts/profile-docker-model.sh llama2-7b
#   ./scripts/profile-docker-model.sh llama2-7b --max-tokens 200 --batch-size 2
#

set -e

CONTAINER_NAME=$1
shift

if [ -z "$CONTAINER_NAME" ]; then
    echo "Usage: $0 <container-name> [profile options]"
    echo ""
    echo "Examples:"
    echo "  $0 llama2-7b"
    echo "  $0 llama2-7b --max-tokens 200 --batch-size 2"
    echo ""
    echo "Available options:"
    echo "  --model PATH          Model path (default: auto-detect)"
    echo "  --prompt TEXT         Prompt text (default: 'Hello, how are you?')"
    echo "  --max-tokens N        Max tokens to generate (default: 100)"
    echo "  --batch-size N        Batch size (default: 1)"
    echo "  --dtype TYPE          Data type: auto|float16|bfloat16|float32 (default: auto)"
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

# Copy profiler script to container
echo "[1/4] Copying profiler script to container..."
docker cp scripts/profile-model-memory.py "${CONTAINER_NAME}:/tmp/"

# Detect model path in container (common locations)
echo "[2/4] Detecting model path..."
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
echo "[3/4] Running memory profiler..."
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

# Copy results out
echo ""
echo "[4/4] Copying results..."
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
    jq '.model_info | {model_name, num_parameters, dtype, sequence_length: .total_sequence_length}' "$OUTPUT_FILE"

    if jq -e '.gpu_info.num_gpus > 1' "$OUTPUT_FILE" &> /dev/null; then
        echo ""
        echo "Multi-GPU Info:"
        jq '.gpu_info' "$OUTPUT_FILE"
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
