#!/bin/bash
# Stop vLLM servers for VEGA-KG

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"

echo "=== Stopping vLLM servers ==="

# Try PID files first
for name in text_llm vlm; do
    pid_file="$LOG_DIR/${name}.pid"
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping $name (PID $pid)..."
            kill "$pid"
        fi
        rm -f "$pid_file"
    fi
done

# Also kill any remaining vllm processes owned by current user
remaining=$(pgrep -u "$(whoami)" -f "vllm serve" 2>/dev/null)
if [ -n "$remaining" ]; then
    echo "Stopping remaining vLLM processes: $remaining"
    kill $remaining 2>/dev/null
fi

sleep 2

# Verify
if pgrep -u "$(whoami)" -f "vllm serve" > /dev/null 2>&1; then
    echo "WARNING: Some vLLM processes still running, sending SIGKILL..."
    pkill -9 -u "$(whoami)" -f "vllm serve" 2>/dev/null
fi

echo "vLLM servers stopped."
