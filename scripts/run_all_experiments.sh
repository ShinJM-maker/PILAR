#!/bin/bash
# Run remaining VEGA-KG experiments (Table 1)
# Skips already-completed experiments by checking for existing metrics files

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$PROJECT_DIR/venv/bin/activate" ] && source "$PROJECT_DIR/venv/bin/activate"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"

RESULTS_DIR="$PROJECT_DIR/data/results"
mkdir -p "$RESULTS_DIR"

echo "=== VEGA-KG Experiment Suite ==="
echo "Start time: $(date)"

# Run experiment only if metrics file doesn't exist
run_if_needed() {
    local dataset=$1
    local agent=$2
    local backend=$3
    local metrics_file="$RESULTS_DIR/metrics_${agent}_${backend}_${dataset}.json"

    if [ -f "$metrics_file" ]; then
        echo "[SKIP] $agent x $backend x $dataset (already done)"
        return 0
    fi

    echo "[$(date)] Running: $agent x $backend x $dataset"
    python run_experiment.py \
        --dataset "$dataset" \
        --agent "$agent" \
        --backend "$backend" \
        >> "$RESULTS_DIR/experiment_${agent}_${backend}_${dataset}.log" 2>&1
    echo "[$(date)] Done: $agent x $backend x $dataset"
}

AGENTS=("naive_rag" "react" "planrag" "autogen")
NON_KG_BACKENDS=("flat_chunk" "simpledoc" "multidocfusion" "ms_graphrag" "lightrag")

# M3DocVQA non-KG backends
echo ""
echo "--- M3DocVQA (non-KG backends) ---"
for backend in "${NON_KG_BACKENDS[@]}"; do
    for agent in "${AGENTS[@]}"; do
        run_if_needed m3docvqa "$agent" "$backend"
    done
done

# Frames non-KG backends
echo ""
echo "--- Frames (non-KG backends) ---"
for backend in "${NON_KG_BACKENDS[@]}"; do
    for agent in "${AGENTS[@]}"; do
        run_if_needed frames "$agent" "$backend"
    done
done

# VEGA-KG backend (requires KG to be built)
echo ""
echo "--- VEGA-KG backend ---"
for dataset in m3docvqa frames; do
    kg_file="$PROJECT_DIR/data/kg/${dataset}_kg.pkl"
    if [ -f "$kg_file" ]; then
        for agent in "${AGENTS[@]}"; do
            run_if_needed "$dataset" "$agent" vega_kg
        done
    else
        echo "[SKIP] vega_kg x $dataset (KG not built: $kg_file)"
    fi
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
        print(f'{d[\"experiment\"]:>45s}: EM={d[\"EM\"]:.4f}  ANLS={d[\"ANLS\"]:.4f}  ROUGE={d[\"ROUGE-L\"]:.4f}  METEOR={d[\"METEOR\"]:.4f}')
    else:
        print(f'{d.get(\"experiment\",f):>45s}: ERROR - {d[\"error\"][:50]}')
"
