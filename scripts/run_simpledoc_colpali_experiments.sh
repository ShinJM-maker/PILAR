#!/bin/bash
# Run SimpleDoc and ColPali-only experiments
# 2 backends × 2 datasets × 4 agents = 16 experiments
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
[ -f venv/bin/activate ] && source venv/bin/activate

DATASETS=("m3docvqa" "frames")
AGENTS=("naive_rag" "react" "planrag" "autogen")
BACKENDS=("colpali_page" "simpledoc")
LIMIT=100

echo "=== Starting SimpleDoc + ColPali experiments ==="
echo "Date: $(date)"
echo "Limit: $LIMIT samples per experiment"
echo ""

for dataset in "${DATASETS[@]}"; do
    for backend in "${BACKENDS[@]}"; do
        for agent in "${AGENTS[@]}"; do
            echo "--- Running: ${agent} x ${backend} x ${dataset} ---"
            PYTHONPATH=. python run_experiment.py \
                --dataset "$dataset" \
                --agent "$agent" \
                --backend "$backend" \
                --limit "$LIMIT" \
                --concurrent 32 \
                --no-shutdown \
                2>&1 | tail -5
            echo ""
        done
    done
done

echo "=== All experiments complete ==="
echo "Date: $(date)"

# Shutdown vLLM
echo "Shutting down vLLM servers..."
bash scripts/stop_vllm.sh 2>/dev/null || true
echo "Done."
