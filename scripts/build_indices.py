"""Pre-build BM25 and dense indices for retrieval.

Merges blocks into page-level chunks to reduce index size.

Usage:
    python scripts/build_indices.py --dataset m3docvqa
    python scripts/build_indices.py --dataset frames
"""
import argparse
import json
import logging
import pickle
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from src.utils.io_utils import load_config, load_json, save_json
from src.retrieval.bm25_index import BM25Index
from src.retrieval.dense_index import DenseIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_dataset_doc_ids(config: dict, dataset_name: str) -> set:
    if dataset_name == "m3docvqa":
        samples = load_json(config["paths"]["samples_m3docvqa"])
    else:
        samples = load_json(config["paths"]["samples_frames"])
    doc_ids = set()
    for s in samples:
        doc_ids.update(s.get("supporting_doc_ids", s.get("filtered_supporting_doc_ids", [])))
    return doc_ids


def build_page_chunks(config: dict, dataset_name: str):
    """Build page-level chunks from preprocessed blocks."""
    preprocessed_dir = Path(config["paths"]["preprocessed"])
    doc_ids = get_dataset_doc_ids(config, dataset_name)
    logger.info(f"Building page chunks for {len(doc_ids)} docs ({dataset_name})")

    # Merge blocks into page-level chunks
    chunks = {}
    block_to_chunk = {}  # maps block_id -> chunk_id

    for doc_id in tqdm(doc_ids, desc="Loading docs"):
        json_file = preprocessed_dir / f"{doc_id}.json"
        if not json_file.exists():
            continue
        doc_data = load_json(json_file)

        # Group blocks by page
        page_blocks = defaultdict(list)
        for block in doc_data.get("blocks", []):
            text = block.get("text", "").strip()
            if text:
                page_blocks[block.get("page", 0)].append(block)

        for page_num, blocks in page_blocks.items():
            chunk_id = f"{doc_id}_p{page_num:04d}"
            texts = [b["text"] for b in blocks if b.get("text", "").strip()]
            merged_text = "\n".join(texts)
            if len(merged_text) < 20:
                continue

            section_path = ""
            for b in blocks:
                if b.get("section_path"):
                    section_path = b["section_path"]
                    break

            chunks[chunk_id] = {
                "id": chunk_id,
                "doc_id": doc_id,
                "page": page_num,
                "text": merged_text,
                "section_path": section_path,
                "block_ids": [b["id"] for b in blocks],
            }

            for b in blocks:
                block_to_chunk[b["id"]] = chunk_id

    logger.info(f"Built {len(chunks)} page chunks from {len(doc_ids)} docs")
    return chunks, block_to_chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--dataset", choices=["m3docvqa", "frames", "all"], default="all")
    parser.add_argument("--skip-dense", action="store_true", help="Skip dense index (BM25 only)")
    args = parser.parse_args()

    config = load_config(args.config)
    indices_dir = Path(config["paths"]["indices"])
    indices_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["m3docvqa", "frames"] if args.dataset == "all" else [args.dataset]

    for dataset in datasets:
        logger.info(f"\n=== Building indices for {dataset} ===")

        chunks, block_to_chunk = build_page_chunks(config, dataset)

        # Save chunks
        save_json(chunks, indices_dir / f"{dataset}_chunks.json")
        save_json(block_to_chunk, indices_dir / f"{dataset}_block_to_chunk.json")

        # Build BM25
        documents = [{"id": cid, "text": c["text"]} for cid, c in chunks.items()]
        bm25 = BM25Index(k1=config["retrieval"]["bm25_k1"], b=config["retrieval"]["bm25_b"])
        bm25.build(documents)
        with open(indices_dir / f"{dataset}_bm25.pkl", "wb") as f:
            pickle.dump(bm25, f)
        logger.info(f"BM25 index: {len(documents)} documents")

        # Build dense index
        if not args.skip_dense:
            dense = DenseIndex(model_name=config["models"]["text_embedder"])
            dense.build(documents)
            dense.save(str(indices_dir / f"{dataset}_dense.npz"))
            logger.info(f"Dense index built")
        else:
            logger.info("Skipping dense index")

        logger.info(f"Indices saved to {indices_dir}")


if __name__ == "__main__":
    main()
