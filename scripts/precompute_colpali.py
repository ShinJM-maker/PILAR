"""Pre-compute ColPali page embeddings + query-page scores.

Must run BEFORE downgrading transformers (colpali needs transformers>=5).

Phase 1: Encode all page images → multi-vector embeddings
Phase 2: For each dataset query, compute MaxSim scores → save top-K per query

Usage:
    PYTHONPATH=. python scripts/precompute_colpali.py --dataset m3docvqa --device cuda:0
    PYTHONPATH=. python scripts/precompute_colpali.py --dataset frames --device cuda:0
"""
import argparse
import json
import logging
import time
from pathlib import Path

import fitz
import numpy as np
import torch
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor

from src.utils.io_utils import load_config, load_json, save_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

COLPALI_MODEL = "vidore/colpali-v1.2"
BATCH_SIZE = 8
DPI = 150
QUERY_TOP_K = 40


def render_page_to_pil(pdf_path: str, page_num: int) -> Image.Image:
    doc = fitz.open(pdf_path)
    if page_num >= len(doc):
        doc.close()
        return None
    page = doc[page_num]
    zoom = DPI / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def encode_pages(model, processor, chunks, pdf_dir, device):
    """Encode all page images with ColPali."""
    items = []
    for cid, chunk in chunks.items():
        doc_id = chunk.get("doc_id", "")
        page = chunk.get("page", 0)
        pdf_path = pdf_dir / f"{doc_id}.pdf"
        if pdf_path.exists():
            items.append((cid, str(pdf_path), page))

    logger.info(f"Encoding {len(items)} pages...")

    all_chunk_ids = []
    all_embeddings = []

    t0 = time.time()
    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i:i + BATCH_SIZE]

        images = []
        valid_cids = []
        for cid, pdf_path, page_num in batch:
            img = render_page_to_pil(pdf_path, page_num)
            if img is not None:
                images.append(img)
                valid_cids.append(cid)

        if not images:
            continue

        processed = processor.process_images(images)
        processed = {k: v.to(device) for k, v in processed.items()}

        with torch.no_grad():
            embeddings = model(**processed)

        for j, cid in enumerate(valid_cids):
            emb = embeddings[j].cpu().float().numpy()
            all_chunk_ids.append(cid)
            all_embeddings.append(emb)

        done = min(i + BATCH_SIZE, len(items))
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        if (done % 200 == 0) or done == len(items):
            logger.info(f"Pages: {done}/{len(items)} ({rate:.1f}/s, "
                        f"ETA {(len(items)-done)/rate:.0f}s)")

    elapsed = time.time() - t0
    logger.info(f"Encoded {len(all_chunk_ids)} pages in {elapsed:.1f}s")
    return all_chunk_ids, all_embeddings


def precompute_query_scores(model, processor, samples, chunk_ids,
                            offsets, flat_embs, device):
    """Compute ColPali MaxSim scores for all dataset queries.

    Uses 2-stage approach for efficiency:
    1. Mean-pooled doc embeddings → fast approximate top-200 candidates
    2. Exact MaxSim scoring on top-200 candidates only
    """
    logger.info(f"Pre-computing query scores for {len(samples)} queries...")

    n_docs = len(chunk_ids)
    offsets_np = offsets if isinstance(offsets, np.ndarray) else np.array(offsets)

    # Stage 1 prep: compute mean-pooled doc embeddings for fast pre-filtering
    logger.info("Computing mean-pooled doc embeddings for pre-filtering...")
    doc_mean_embs = np.zeros((n_docs, flat_embs.shape[1]), dtype=np.float32)
    for doc_idx in range(n_docs):
        start = offsets_np[doc_idx]
        end = offsets_np[doc_idx + 1]
        if start < end:
            doc_mean_embs[doc_idx] = flat_embs[start:end].mean(axis=0)

    doc_mean_t = torch.from_numpy(doc_mean_embs).to(device).to(torch.bfloat16)
    logger.info(f"Mean-pooled doc embeddings: {doc_mean_t.shape}")

    # Load patch embeddings per-doc on demand (keep on CPU to save GPU memory)
    # Only load candidate docs' patches to GPU for exact MaxSim
    PRE_FILTER_K = 200  # candidates for exact MaxSim

    query_scores = {}
    t0 = time.time()

    batch_size = 16
    sample_list = list(samples)
    for i in range(0, len(sample_list), batch_size):
        batch = sample_list[i:i + batch_size]
        queries = [s["question"] for s in batch]
        sids = [s["id"] for s in batch]

        q_inputs = processor.process_queries(queries)
        q_inputs = {k: v.to(device) for k, v in q_inputs.items()}
        with torch.no_grad():
            q_embs = model(**q_inputs)  # [batch, n_q, 128]

        for j, sid in enumerate(sids):
            q_emb = q_embs[j]  # [n_q, 128]

            # Stage 1: Fast pre-filter using mean-pooled embeddings
            # q_emb mean → [128], dot with doc_mean → [n_docs]
            q_mean = q_emb.mean(dim=0)  # [128]
            approx_scores = q_mean @ doc_mean_t.T  # [n_docs]
            _, cand_idxs = approx_scores.topk(min(PRE_FILTER_K, n_docs))

            # Stage 2: Exact MaxSim on candidates only
            scores = torch.zeros(PRE_FILTER_K, device=device)
            cand_list = cand_idxs.cpu().tolist()
            for k, doc_idx in enumerate(cand_list):
                start = int(offsets_np[doc_idx])
                end = int(offsets_np[doc_idx + 1])
                if start == end:
                    continue
                doc_patches = torch.from_numpy(
                    flat_embs[start:end]
                ).to(device).to(torch.bfloat16)  # [n_patches, 128]
                # MaxSim: for each query token, max sim across doc patches, then sum
                sim = q_emb @ doc_patches.T  # [n_q, n_patches]
                scores[k] = sim.max(dim=1).values.sum()

            # Top-K from candidates
            top_vals, top_local_idxs = scores.topk(min(QUERY_TOP_K, len(cand_list)))
            query_scores[sid] = [
                (chunk_ids[cand_list[idx.item()]], val.item())
                for idx, val in zip(top_local_idxs, top_vals)
            ]

        done = min(i + batch_size, len(sample_list))
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        if (done % 100 == 0) or done == len(sample_list):
            logger.info(f"Queries: {done}/{len(sample_list)} ({rate:.1f}/s)")

    elapsed = time.time() - t0
    logger.info(f"Pre-computed scores for {len(query_scores)} queries in {elapsed:.1f}s")
    return query_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["m3docvqa", "frames"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--phase2-only", action="store_true",
                        help="Skip Phase 1, load existing embeddings for Phase 2")
    args = parser.parse_args()

    config = load_config("config/default.yaml")
    dataset = args.dataset
    device = args.device
    pdf_dir = Path(config["paths"][f"pdf_{dataset}"])
    indices_dir = Path(config["paths"]["indices"])

    # Load chunks
    chunks = load_json(indices_dir / f"{dataset}_chunks.json")
    logger.info(f"Loaded {len(chunks)} chunks for {dataset}")

    # Load samples
    if dataset == "m3docvqa":
        samples = load_json(config["paths"]["samples_m3docvqa"])
    else:
        samples = load_json(config["paths"]["samples_frames"])
    for s in samples:
        if "supporting_doc_ids" not in s and "filtered_supporting_doc_ids" in s:
            s["supporting_doc_ids"] = s["filtered_supporting_doc_ids"]
    logger.info(f"Loaded {len(samples)} samples")

    # Load model
    logger.info(f"Loading ColPali model on {device}...")
    model = ColPali.from_pretrained(
        COLPALI_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    processor = ColPaliProcessor.from_pretrained(COLPALI_MODEL)
    model.eval()

    emb_path = indices_dir / f"{dataset}_colpali_embeddings.npz"
    ids_path = indices_dir / f"{dataset}_colpali_chunk_ids.json"

    if args.phase2_only:
        # Load existing Phase 1 results
        logger.info("Loading existing embeddings (--phase2-only)...")
        data = np.load(str(emb_path))
        flat_embs = data["embeddings"]
        offsets = data["offsets"]
        chunk_ids = load_json(ids_path)
        logger.info(f"Loaded {len(chunk_ids)} chunk embeddings, {flat_embs.shape[0]} patches")
    else:
        # Phase 1: Encode pages
        logger.info("=" * 60)
        logger.info("Phase 1: Encode page images")
        logger.info("=" * 60)

        chunk_ids, embeddings = encode_pages(model, processor, chunks, pdf_dir, device)

        # Save embeddings
        offsets = [0]
        for emb in embeddings:
            offsets.append(offsets[-1] + emb.shape[0])
        flat_embs = np.concatenate(embeddings, axis=0)
        offsets = np.array(offsets, dtype=np.int64)

        np.savez(str(emb_path), embeddings=flat_embs, offsets=offsets)
        save_json(chunk_ids, ids_path)
        logger.info(f"Saved: {flat_embs.shape[0]} patches for {len(chunk_ids)} pages")

    # Phase 2: Pre-compute query scores
    logger.info("=" * 60)
    logger.info("Phase 2: Pre-compute query-page scores")
    logger.info("=" * 60)

    scores_path = indices_dir / f"{dataset}_colpali_query_scores.json"
    query_scores = precompute_query_scores(
        model, processor, samples, chunk_ids,
        offsets, flat_embs, device,
    )
    save_json(query_scores, scores_path)
    logger.info(f"Saved query scores to {scores_path}")

    # Cleanup
    del model
    torch.cuda.empty_cache()
    logger.info("Done.")


if __name__ == "__main__":
    main()
