"""Run the full preprocessing pipeline on all PDFs.

Usage:
    python -m src.preprocess.run_preprocess --dataset m3docvqa --limit 10
    python -m src.preprocess.run_preprocess --dataset all
"""
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

from src.preprocess.pdf_renderer import render_pdf_pages
from src.preprocess.block_extractor import BlockExtractor
from src.dhp.hierarchy_parser import HierarchyParser
from src.dhp.doc_card import build_doc_card
from src.utils.io_utils import load_config, save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def process_single_pdf(pdf_path: Path, config: dict,
                       extractor: BlockExtractor, parser: HierarchyParser) -> dict:
    """Process a single PDF: extract text blocks, build hierarchy."""
    doc_id = pdf_path.stem
    output_dir = Path(config["paths"]["preprocessed"])
    output_file = output_dir / f"{doc_id}.json"

    # Skip if already processed
    if output_file.exists():
        return {"doc_id": doc_id, "status": "skipped", "num_blocks": 0}

    try:
        # Extract blocks from each page using PyMuPDF
        all_blocks = []
        import fitz
        doc = fitz.open(str(pdf_path))
        num_pages = len(doc)
        doc.close()

        # Optionally render pages for visual crops
        render_images = config["preprocess"].get("render_images", False)
        page_images = {}
        if render_images:
            pages = render_pdf_pages(pdf_path, dpi=config["preprocess"]["dpi"])
            page_images = {pn: img for pn, img in pages}

        for page_num in range(num_pages):
            page_image = page_images.get(page_num, None)
            blocks = extractor.extract_from_pdf(
                str(pdf_path), doc_id, page_num, page_image
            )
            all_blocks.extend(blocks)

        # Build hierarchy
        all_blocks = parser.parse(all_blocks)

        # Build doc card
        doc_card = build_doc_card(all_blocks)

        # Save
        doc_data = {
            "doc_id": doc_id,
            "pdf_path": str(pdf_path),
            "num_pages": num_pages,
            "num_blocks": len(all_blocks),
            "doc_card": doc_card,
            "blocks": [b.to_dict() for b in all_blocks],
        }
        save_json(doc_data, output_file)

        return {"doc_id": doc_id, "status": "ok", "num_blocks": len(all_blocks)}

    except Exception as e:
        logger.error(f"Failed to process {pdf_path.name}: {e}")
        return {"doc_id": doc_id, "status": "error", "error": str(e), "num_blocks": 0}


def main():
    parser = argparse.ArgumentParser(description="Preprocess PDFs for VEGA-KG")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--dataset", choices=["m3docvqa", "frames", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config["paths"]["preprocessed"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect PDF paths
    pdf_paths = []
    if args.dataset in ("m3docvqa", "all"):
        m3_dir = Path(config["paths"]["pdf_m3docvqa"])
        pdf_paths.extend(sorted(m3_dir.glob("*.pdf")))
    if args.dataset in ("frames", "all"):
        f_dir = Path(config["paths"]["pdf_frames"])
        pdf_paths.extend(sorted(f_dir.glob("*.pdf")))

    if args.limit:
        pdf_paths = pdf_paths[:args.limit]

    logger.info(f"Processing {len(pdf_paths)} PDFs")

    # Initialize extractor (no layout detector for now — using PyMuPDF native)
    crop_dir = output_dir / "crops"
    extractor = BlockExtractor(layout_detector=None, crop_output_dir=crop_dir)
    hierarchy_parser = HierarchyParser(max_depth=config["dhp"]["max_depth"])

    # Process
    results = []
    total_blocks = 0
    for pdf_path in tqdm(pdf_paths, desc="Preprocessing PDFs"):
        result = process_single_pdf(pdf_path, config, extractor, hierarchy_parser)
        results.append(result)
        total_blocks += result.get("num_blocks", 0)

    # Summary
    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")
    logger.info(f"Done: {ok} processed, {skipped} skipped, {errors} errors, {total_blocks} total blocks")

    save_json(results, output_dir / "_summary.json")


if __name__ == "__main__":
    main()
