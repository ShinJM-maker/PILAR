"""Generate VLM page descriptions + ColPali visual embeddings for SimpleDoc baseline.

Two-phase preprocessing:
  Phase 1: ColPali visual embeddings (page images → multi-vector embeddings)
  Phase 2: VLM text descriptions (page images → natural language descriptions → dense index)

Usage:
    PYTHONPATH=. python scripts/generate_page_descriptions.py --dataset m3docvqa
    PYTHONPATH=. python scripts/generate_page_descriptions.py --dataset frames
    PYTHONPATH=. python scripts/generate_page_descriptions.py --dataset m3docvqa --phase colpali
    PYTHONPATH=. python scripts/generate_page_descriptions.py --dataset m3docvqa --phase description
"""
import argparse
import asyncio
import base64
import logging
import re
import time
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import torch
from PIL import Image

from src.utils.io_utils import load_config, load_json, save_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

DPI = 150
VLM_URL = "http://localhost:8001/v1"
VLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
CONCURRENT = 8
COLPALI_MODEL = "vidore/colpali-v1.2"
COLPALI_BATCH_SIZE = 8
COLPALI_DEVICE = "cuda:0"

DESC_PROMPT = """/no_think
Describe this document page in detail. Include:
- Main topic and key information
- Any tables, figures, charts with their content
- Important names, numbers, dates mentioned
- Layout structure (headers, sections)
Keep the description concise but informative (2-4 sentences)."""


def render_page_to_pil(pdf_path: str, page_num: int, dpi: int = DPI) -> Image.Image:
    """Render a PDF page to PIL Image."""
    doc = fitz.open(pdf_path)
    if page_num >= len(doc):
        doc.close()
        return None
    page = doc[page_num]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def render_page_to_b64(pdf_path: str, page_num: int, dpi: int = DPI) -> str:
    """Render a PDF page to base64 PNG."""
    import io
    img = render_page_to_pil(pdf_path, page_num, dpi)
    if img is None:
        return ""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ============================================================
# Phase 1: ColPali visual embeddings
# ============================================================
def run_colpali_phase(chunks, pdf_dir, output_dir, dataset, resume=False):
    """Generate ColPali multi-vector embeddings for all page images."""
    from colpali_engine.models import ColPali, ColPaliProcessor

    emb_path = output_dir / f"{dataset}_colpali_embeddings.npz"
    chunk_ids_path = output_dir / f"{dataset}_colpali_chunk_ids.json"

    # Check resume
    existing_ids = set()
    if resume and chunk_ids_path.exists():
        existing_ids = set(load_json(chunk_ids_path))
        logger.info(f"Resuming: {len(existing_ids)} existing embeddings")

    # Build work list
    items = []
    for cid, chunk in chunks.items():
        if cid in existing_ids:
            continue
        doc_id = chunk.get("doc_id", "")
        page = chunk.get("page", 0)
        pdf_path = pdf_dir / f"{doc_id}.pdf"
        if not pdf_path.exists():
            continue
        items.append((cid, str(pdf_path), page))

    logger.info(f"ColPali: {len(items)} pages to encode ({len(existing_ids)} already done)")

    if not items:
        logger.info("No new pages to encode")
        return

    # Load model
    logger.info(f"Loading ColPali model: {COLPALI_MODEL}")
    model = ColPali.from_pretrained(
        COLPALI_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=COLPALI_DEVICE,
    )
    processor = ColPaliProcessor.from_pretrained(COLPALI_MODEL)
    model.eval()

    # Process in batches
    all_chunk_ids = []
    all_embeddings = []  # list of np arrays, each [n_patches, 128]

    t0 = time.time()
    for i in range(0, len(items), COLPALI_BATCH_SIZE):
        batch = items[i:i + COLPALI_BATCH_SIZE]

        # Render pages
        images = []
        valid_cids = []
        for cid, pdf_path, page_num in batch:
            img = render_page_to_pil(pdf_path, page_num)
            if img is not None:
                images.append(img)
                valid_cids.append(cid)

        if not images:
            continue

        # Encode
        processed = processor.process_images(images)
        processed = {k: v.to(COLPALI_DEVICE) for k, v in processed.items()}

        with torch.no_grad():
            embeddings = model(**processed)  # [batch, n_patches, 128]

        # Store
        for j, cid in enumerate(valid_cids):
            emb = embeddings[j].cpu().float().numpy()  # [n_patches, 128]
            all_chunk_ids.append(cid)
            all_embeddings.append(emb)

        done = min(i + COLPALI_BATCH_SIZE, len(items))
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        if (done % 100 == 0) or done == len(items):
            logger.info(f"ColPali: {done}/{len(items)} ({rate:.1f} pages/s)")

    # Save: pack multi-vector embeddings
    # Since patches vary per image, store as flat array with offsets
    if all_embeddings:
        offsets = [0]
        flat_embs = []
        for emb in all_embeddings:
            flat_embs.append(emb)
            offsets.append(offsets[-1] + emb.shape[0])

        flat_embs = np.concatenate(flat_embs, axis=0)  # [total_patches, 128]
        offsets = np.array(offsets, dtype=np.int64)

        np.savez(
            str(emb_path),
            embeddings=flat_embs,
            offsets=offsets,
        )
        save_json(all_chunk_ids, chunk_ids_path)

        elapsed = time.time() - t0
        logger.info(f"ColPali: saved {len(all_chunk_ids)} embeddings "
                    f"({flat_embs.shape[0]} total patches) in {elapsed:.1f}s")

    # Cleanup GPU
    del model
    torch.cuda.empty_cache()


# ============================================================
# Phase 2: VLM text descriptions
# ============================================================
async def generate_descriptions_batch(items, vlm_url, vlm_model, concurrent):
    """Generate descriptions for a batch of (chunk_id, pdf_path, page_num) items."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=vlm_url, api_key="dummy")
    sem = asyncio.Semaphore(concurrent)
    results = {}

    async def process_one(chunk_id, pdf_path, page_num):
        async with sem:
            try:
                b64 = render_page_to_b64(str(pdf_path), page_num, DPI)
                if not b64:
                    return

                resp = await client.chat.completions.create(
                    model=vlm_model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                            {"type": "text", "text": DESC_PROMPT},
                        ],
                    }],
                    max_tokens=256,
                    temperature=0.0,
                )
                desc = resp.choices[0].message.content.strip()
                desc = re.sub(r"<think>.*?</think>", "", desc, flags=re.DOTALL).strip()
                results[chunk_id] = desc
            except Exception as e:
                logger.warning(f"Failed {chunk_id}: {e}")

    tasks = [process_one(cid, pp, pn) for cid, pp, pn in items]
    await asyncio.gather(*tasks)
    return results


def run_description_phase(chunks, pdf_dir, output_dir, dataset, resume=False):
    """Generate VLM text descriptions and build dense index."""
    from src.retrieval.dense_index import DenseIndex

    desc_path = output_dir / f"{dataset}_page_descriptions.json"
    desc_index_path = output_dir / f"{dataset}_desc_dense.npz"

    # Load existing
    existing_descs = {}
    if resume and desc_path.exists():
        existing_descs = load_json(desc_path)
        logger.info(f"Resuming: {len(existing_descs)} existing descriptions")

    # Build work list
    items = []
    for cid, chunk in chunks.items():
        if cid in existing_descs:
            continue
        doc_id = chunk.get("doc_id", "")
        page = chunk.get("page", 0)
        pdf_path = pdf_dir / f"{doc_id}.pdf"
        if not pdf_path.exists():
            continue
        items.append((cid, str(pdf_path), page))

    logger.info(f"Description: {len(items)} pages to describe ({len(existing_descs)} already done)")

    if items:
        t0 = time.time()
        all_descs = dict(existing_descs)
        batch_size = 32

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_descs = asyncio.run(
                generate_descriptions_batch(batch, VLM_URL, VLM_MODEL, CONCURRENT)
            )
            all_descs.update(batch_descs)

            done = min(i + batch_size, len(items))
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            logger.info(f"Description: {done}/{len(items)} ({rate:.1f} pages/s)")

            save_json(all_descs, desc_path)

        elapsed = time.time() - t0
        logger.info(f"Generated {len(all_descs) - len(existing_descs)} new descriptions "
                    f"in {elapsed:.1f}s")
    else:
        all_descs = existing_descs

    save_json(all_descs, desc_path)
    logger.info(f"Saved {len(all_descs)} descriptions to {desc_path}")

    # Build dense index
    logger.info("Building dense index from descriptions...")
    config = load_config("config/default.yaml")
    embedder_name = config["models"].get("text_embedder", "sentence-transformers/all-MiniLM-L6-v2")

    desc_docs = [{"id": cid, "text": desc} for cid, desc in all_descs.items() if desc]
    desc_index = DenseIndex(model_name=embedder_name, device="cpu")
    desc_index.build(desc_docs)
    desc_index.save(str(desc_index_path))
    logger.info(f"Saved description dense index ({len(desc_docs)} vectors) to {desc_index_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["m3docvqa", "frames"])
    parser.add_argument("--phase", choices=["colpali", "description", "all"], default="all")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed pages")
    parser.add_argument("--device", default="cuda:0", help="GPU device for ColPali")
    args = parser.parse_args()

    global COLPALI_DEVICE
    COLPALI_DEVICE = args.device

    config = load_config("config/default.yaml")
    dataset = args.dataset
    pdf_dir = Path(config["paths"][f"pdf_{dataset}"])
    indices_dir = Path(config["paths"]["indices"])
    indices_dir.mkdir(parents=True, exist_ok=True)

    # Load chunks
    chunks_path = indices_dir / f"{dataset}_chunks.json"
    chunks = load_json(chunks_path)
    logger.info(f"Loaded {len(chunks)} chunks for {dataset}")

    if args.phase in ("colpali", "all"):
        logger.info("=" * 60)
        logger.info("Phase 1: ColPali Visual Embeddings")
        logger.info("=" * 60)
        run_colpali_phase(chunks, pdf_dir, indices_dir, dataset, resume=args.resume)

    if args.phase in ("description", "all"):
        logger.info("=" * 60)
        logger.info("Phase 2: VLM Text Descriptions")
        logger.info("=" * 60)
        run_description_phase(chunks, pdf_dir, indices_dir, dataset, resume=args.resume)

    logger.info("Done.")


if __name__ == "__main__":
    main()
