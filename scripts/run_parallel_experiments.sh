#!/bin/bash
# Run VEGA-KG experiments in parallel for maximum GPU utilization
# Strategy:
# - naive_rag uses only reader (GPU 1) → can run concurrently
# - multi-step agents use reasoning LLM (GPU 0) + reader (GPU 1) → run one at a time
# - Within each experiment, use --concurrent for multi-step agents

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$PROJECT_DIR/venv/bin/activate" ] && source "$PROJECT_DIR/venv/bin/activate"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"

RESULTS_DIR="$PROJECT_DIR/data/results"
mkdir -p "$RESULTS_DIR"

run_if_needed() {
    local dataset=$1
    local agent=$2
    local backend=$3
    local concurrent=${4:-1}
    local metrics_file="$RESULTS_DIR/metrics_${agent}_${backend}_${dataset}.json"

    if [ -f "$metrics_file" ]; then
        echo "[SKIP] $agent x $backend x $dataset (already done)"
        return 0
    fi

    echo "[$(date)] Running: $agent x $backend x $dataset (concurrent=$concurrent)"
    python run_experiment.py \
        --dataset "$dataset" \
        --agent "$agent" \
        --backend "$backend" \
        --concurrent "$concurrent" \
        >> "$RESULTS_DIR/experiment_${agent}_${backend}_${dataset}.log" 2>&1
    local exit_code=$?
    echo "[$(date)] Done: $agent x $backend x $dataset (exit=$exit_code)"
    return $exit_code
}

echo "=== Parallel VEGA-KG Experiment Suite ==="
echo "Start time: $(date)"
echo "GPU config: 80% memory utilization, max-model-len=8192"

AGENTS=("naive_rag" "react" "planrag" "autogen")
NON_KG_BACKENDS=("flat_chunk" "simpledoc" "multidocfusion" "ms_graphrag" "lightrag")

# Phase 1: Run all naive_rag experiments in parallel (they only use reader/GPU1)
echo ""
echo "--- Phase 1: All naive_rag experiments (parallel) ---"
for backend in "${NON_KG_BACKENDS[@]}"; do
    for dataset in m3docvqa frames; do
        run_if_needed "$dataset" "naive_rag" "$backend" 1 &
    done
done
# Also VEGA-KG backend naive_rag
for dataset in m3docvqa frames; do
    kg_file="$PROJECT_DIR/data/kg/${dataset}_kg.pkl"
    if [ -f "$kg_file" ]; then
        run_if_needed "$dataset" "naive_rag" "vega_kg" 1 &
    fi
done
echo "Waiting for all naive_rag experiments..."
wait
echo "[$(date)] Phase 1 complete"

# Phase 2: Run multi-step agents sequentially (they share reasoning LLM)
# But use --concurrent to process multiple samples in parallel within each experiment
echo ""
echo "--- Phase 2: Multi-step agents (concurrent samples) ---"
for agent in "react" "planrag" "autogen"; do
    for backend in "${NON_KG_BACKENDS[@]}"; do
        for dataset in m3docvqa frames; do
            run_if_needed "$dataset" "$agent" "$backend" 16
        done
    done
    # VEGA-KG backend
    for dataset in m3docvqa frames; do
        kg_file="$PROJECT_DIR/data/kg/${dataset}_kg.pkl"
        if [ -f "$kg_file" ]; then
            run_if_needed "$dataset" "$agent" "vega_kg" 16
        fi
    done
done

echo ""
echo "=== All experiments complete ==="
echo "End time: $(date)"

# Stop vLLM servers
echo ""
echo "--- Stopping vLLM servers ---"
bash "$PROJECT_DIR/scripts/stop_vllm.sh"

# Print summary
echo ""
echo "--- Results Summary ---"
python -c "
import json, glob
files = sorted(glob.glob('$RESULTS_DIR/metrics_*.json'))
for f in files:
    d = json.load(open(f))
    if 'error' not in d:
        print(f'{d[\"experiment\"]:>50s}: EM={d[\"EM\"]:.4f}  ANLS={d[\"ANLS\"]:.4f}  ROUGE={d[\"ROUGE-L\"]:.4f}  METEOR={d[\"METEOR\"]:.4f}')
    else:
        print(f'{d.get(\"experiment\",f):>50s}: ERROR - {d[\"error\"][:50]}')
"
