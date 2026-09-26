#!/bin/bash
# Setup virtual environment and install dependencies for VEGA-KG
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

echo "=== Setting up VEGA-KG environment ==="

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install PyTorch first (CUDA 12.x)
echo "Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install core dependencies
echo "Installing core dependencies..."
pip install \
    transformers>=4.45.0 \
    accelerate>=0.30.0 \
    sentence-transformers>=3.0.0 \
    PyMuPDF>=1.24.0 \
    pytesseract>=0.3.10 \
    Pillow>=10.0.0 \
    opencv-python>=4.8.0 \
    ultralytics>=8.2.0 \
    rank-bm25>=0.2.2 \
    networkx>=3.2 \
    rouge-score>=0.1.2 \
    nltk>=3.8.0 \
    pyyaml>=6.0 \
    tqdm>=4.66.0 \
    pandas>=2.1.0 \
    numpy>=1.24.0 \
    scikit-learn>=1.3.0 \
    rich>=13.0.0 \
    openai>=1.0.0

# Install vLLM (for model serving)
echo "Installing vLLM..."
pip install vllm>=0.6.0

# Install FAISS
echo "Installing FAISS..."
pip install faiss-gpu || pip install faiss-cpu

# Install baseline libraries (optional, may have conflicts)
echo "Installing baseline libraries..."
pip install graphrag || echo "graphrag install failed (optional)"
pip install lightrag-hku || echo "lightrag install failed (optional)"
pip install pyautogen || echo "pyautogen install failed (optional)"

# Download NLTK data
python3 -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('punkt_tab', quiet=True)"

# Verify Tesseract
if ! command -v tesseract &> /dev/null; then
    echo "WARNING: tesseract-ocr not found. Install with: sudo apt install tesseract-ocr"
fi

echo ""
echo "=== Setup complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo ""
echo "Next steps:"
echo "1. Start vLLM servers (see scripts/start_vllm.sh)"
echo "2. Run preprocessing: python -m src.preprocess.run_preprocess --dataset m3docvqa --limit 10"
echo "3. Build KG: python run_kg_build.py --dataset m3docvqa --limit 10"
echo "4. Run experiments: python run_experiment.py --dataset m3docvqa --agent naive_rag --backend flat_chunk --limit 10"
