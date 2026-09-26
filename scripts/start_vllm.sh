#!/bin/bash
# Start vLLM servers for VEGA-KG
# GPU 0: Text LLM (Qwen3-8B) for assertion extraction + reasoning
# GPU 1: VLM (Qwen3-VL-8B-Instruct) for visual assertion extraction + reading

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$PROJECT_DIR/venv/bin/activate" ] && source "$PROJECT_DIR/venv/bin/activate"

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "=== Starting vLLM servers ==="

# Server 1: Text LLM (port 8000)
echo "Starting Text LLM on GPU 0 (port 8000)..."
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-8B \
    --port 8000 \
    --gpu-memory-utilization 0.80 \
    --max-model-len 8192 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    > "$LOG_DIR/vllm_text_llm.log" 2>&1 &
TEXT_PID=$!
echo "Text LLM PID: $TEXT_PID"

# Server 2: VLM (port 8001)
echo "Starting VLM on GPU 1 (port 8001)..."
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --port 8001 \
    --gpu-memory-utilization 0.55 \
    --max-model-len 8192 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --limit-mm-per-prompt '{"image": 4}' \
    > "$LOG_DIR/vllm_vlm.log" 2>&1 &
VLM_PID=$!
echo "VLM PID: $VLM_PID"

echo ""
echo "Servers starting... Check logs:"
echo "  Text LLM: tail -f $LOG_DIR/vllm_text_llm.log"
echo "  VLM:      tail -f $LOG_DIR/vllm_vlm.log"
echo ""
echo "Wait ~2-3 minutes for models to load."
echo "Test with: curl http://localhost:8000/v1/models"
echo "           curl http://localhost:8001/v1/models"
echo ""
echo "To stop: kill $TEXT_PID $VLM_PID"

# Save PIDs for cleanup
echo "$TEXT_PID" > "$LOG_DIR/text_llm.pid"
echo "$VLM_PID" > "$LOG_DIR/vlm.pid"
