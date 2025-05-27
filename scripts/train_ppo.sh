#!/bin/bash
# PPO Training Script for ADAPT
# Usage: bash scripts/train_ppo.sh [model_name] [split] [additional_args...]

set -e

# Default values for debugging
MODEL_NAME=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
SPLIT=${2:-0}
OUTPUT_DIR="ppo_models/split${SPLIT}"
WANDB_PROJECT="adapt-ppo-debug"

# Shift to get additional arguments
shift 2 2>/dev/null || true

echo "🚀 Starting PPO training for ADAPT"
echo "Model: $MODEL_NAME"
echo "Split: $SPLIT"
echo "Output: $OUTPUT_DIR"
echo "Additional args: $@"

# Check if we're in Colab environment
if [ -n "$COLAB_GPU" ] || [ -n "$COLAB_TPU_ADDR" ]; then
    echo "🔧 Detected Colab environment - using optimized settings"
    USE_4BIT="--use_4bit"
    BATCH_SIZE=4
    MINI_BATCH_SIZE=1
    MAX_STEPS=50
    MAX_EPISODE_LENGTH=15
else
    echo "🖥️  Running locally - using standard settings"
    USE_4BIT=""
    BATCH_SIZE=8
    MINI_BATCH_SIZE=2
    MAX_STEPS=100
    MAX_EPISODE_LENGTH=20
fi

# Create necessary directories
mkdir -p $OUTPUT_DIR
mkdir -p logs

# Log system info
echo "📊 System Information:"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo 'No GPU detected')"
echo "CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'CUDA not found')"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'PyTorch not found')"

# Run PPO training with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/ppo_training_${TIMESTAMP}.log"

echo "📝 Logging to: $LOG_FILE"

python scripts/train_ppo.py \
    --model_name "$MODEL_NAME" \
    --split $SPLIT \
    --output_dir "$OUTPUT_DIR" \
    --config_path "cfg/split${SPLIT}_run_config_train.json" \
    --batch_size $BATCH_SIZE \
    --mini_batch_size $MINI_BATCH_SIZE \
    --max_steps $MAX_STEPS \
    --max_episode_length $MAX_EPISODE_LENGTH \
    --learning_rate 1.41e-5 \
    --checkpoint_frequency 25 \
    --log_frequency 5 \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "debug_split${SPLIT}_${TIMESTAMP}" \
    $USE_4BIT \
    "$@" 2>&1 | tee "$LOG_FILE"

echo "✅ PPO training completed!"
echo "📁 Model saved to: $OUTPUT_DIR"
echo "📄 Log file: $LOG_FILE"