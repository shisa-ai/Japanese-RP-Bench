#!/bin/bash

# Default values
MODEL="${MODEL:-}"  # Use empty string if MODEL is not set
OPENAI_URL="${OPENAI_URL:-}"  # API URL for the model
JUDGE_URL="${JUDGE_URL:-}"    # API URL for the judge

# Validate required arguments
if [ -z "$MODEL" ] || [ -z "$OPENAI_URL" ] || [ -z "$JUDGE_URL" ]; then
    echo "Error: Required environment variables are missing"
    echo "Usage: MODEL=<model_name> OPENAI_URL=<api_url> JUDGE_URL=<judge_url> ./$0"
    echo "Example:"
    echo "  MODEL=mistral OPENAI_URL=http://localhost:8000/v1 JUDGE_URL=http://localhost:8001/v1 ./$0"
    exit 1
fi

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "Starting eval script"

# Set environment variables for API endpoints
export OPENAI_COMPATIBLE_API_KEY="cat"
export OPENAI_COMPATIBLE_API_URL="$OPENAI_URL"
export JUDGE_OPENAI_COMPATIBLE_API_KEY="cat"
export JUDGE_OPENAI_COMPATIBLE_API_URL="$JUDGE_URL"

# Initialize and activate conda/mamba
source /fsx/ubuntu/miniforge3/etc/profile.d/conda.sh
source /fsx/ubuntu/miniforge3/etc/profile.d/mamba.sh
mamba activate Japanese-RP-Bench

# Create temporary config with model name substituted
envsubst < ./configs/simple_config.yaml > ./configs/temp_config.yaml

# Run the benchmark
japanese-rp-bench --config ./configs/temp_config.yaml

# Clean up
rm ./configs/temp_config.yaml