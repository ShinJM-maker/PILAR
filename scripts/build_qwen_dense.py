"""Build dense indices using Qwen3-Embedding-0.6B.

BM25 indices are model-independent so we reuse existing ones.
Only rebuilds the dense (vector) index with the new embedding model.

Usage:
    PYTHONPATH=. python scripts/build_qwen_dense.py --dataset all
"""
import argparse
import logging
from pathlib import Path

from src.utils.io_utils import load_config, load_json
from src.retrieval.dense_index import DenseIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

QWEN_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--dataset", choices=["m3docvqa", "frames", "all"], default="all")
    args = parser.parse_args()

    config = load_config(args.config)
    indices_dir = Path(config["paths"]["indices"])

    datasets = ["m3docvqa", "frames"] if args.dataset == "all" else [args.dataset]

    for dataset in datasets:
        logger.info(f"\n=== Building Qwen3 dense index for {dataset} ===")

        # Load existing page chunks
        chunks_path = indices_dir / f"{dataset}_chunks.json"
        if not chunks_path.exists():
            logger.error(f"Chunks not found: {chunks_path}. Run build_indices.py first.")
            continue

        chunks = load_json(chunks_path)
        documents = [{"id": cid, "text": c["text"]} for cid, c in chunks.items()]
        logger.info(f"Loaded {len(documents)} page chunks")

        # Build dense index with Qwen3 embeddings
        dense = DenseIndex(model_name=QWEN_MODEL, device="cuda:0", batch_size=64)
        dense.build(documents)

        # Save with model-specific filename
        output_path = indices_dir / f"{dataset}_dense_qwen3.npz"
        dense.save(str(output_path))
        logger.info(f"Saved Qwen3 dense index to {output_path}")


if __name__ == "__main__":
    main()
