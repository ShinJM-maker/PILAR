"""Block extraction: merge layout detection + text extraction into unified Block objects."""
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image

from src.kg.schema import Block, BlockType
from src.preprocess.ocr_engine import extract_text_blocks_from_pdf_page, ocr_region, clean_ocr_text
from src.utils.image_utils import save_crop


# Font size thresholds for heading detection (PyMuPDF mode)
TITLE_FONT_THRESHOLD = 16.0
HEADER_FONT_THRESHOLD = 13.0


class BlockExtractor:
    """Extract blocks from documents using PyMuPDF text extraction + optional layout detection."""

    def __init__(self, layout_detector=None,
                 crop_output_dir: str | Path = None,
                 ocr_lang: str = "eng"):
        self.detector = layout_detector
        self.crop_dir = Path(crop_output_dir) if crop_output_dir else None
        if self.crop_dir:
            self.crop_dir.mkdir(parents=True, exist_ok=True)
        self.ocr_lang = ocr_lang

    def extract_from_pdf(self, pdf_path: str, doc_id: str,
                         page_num: int, page_image: Image.Image = None) -> List[Block]:
        """Extract blocks from a PDF page.

        Primary: PyMuPDF native text extraction (no OCR needed for digital PDFs).
        If layout_detector is available, uses it for block type classification.
        """
        # Get text blocks from PyMuPDF
        raw_blocks = extract_text_blocks_from_pdf_page(pdf_path, page_num)

        blocks = []
        for idx, raw in enumerate(raw_blocks):
            block_id = f"{doc_id}_p{page_num:04d}_b{idx:03d}"

            # Determine block type
            if self.detector and page_image:
                # Use layout detector for type classification
                block_type = self._classify_with_detector(raw, page_image)
            else:
                # Heuristic based on font size and content
                block_type = self._classify_heuristic(raw)

            # Save crop for visual elements
            image_path = None
            if block_type in (BlockType.TABLE, BlockType.FIGURE) and page_image and self.crop_dir:
                crop_path = self.crop_dir / doc_id / f"{block_id}.png"
                try:
                    # Scale bbox from PDF coordinates to image coordinates
                    bbox = self._scale_bbox(raw["bbox"], pdf_path, page_num, page_image)
                    image_path = save_crop(page_image, bbox, crop_path)
                except Exception:
                    pass

            block = Block(
                id=block_id,
                doc_id=doc_id,
                page=page_num,
                block_type=block_type,
                bbox=raw["bbox"],
                text=raw["text"],
                confidence=1.0,
                image_path=image_path,
            )
            blocks.append(block)

        return blocks

    def extract_blocks(self, image: Image.Image, doc_id: str, page_num: int) -> List[Block]:
        """Extract blocks from a page image (fallback for image-only documents)."""
        if self.detector:
            detections = self.detector.detect(image)
            blocks = []
            for idx, det in enumerate(detections):
                block_id = f"{doc_id}_p{page_num:04d}_b{idx:03d}"
                block_type = BlockType(det["block_type"])
                bbox = det["bbox"]
                text = ocr_region(image, bbox, lang=self.ocr_lang)

                image_path = None
                if block_type in (BlockType.TABLE, BlockType.FIGURE) and self.crop_dir:
                    crop_path = self.crop_dir / doc_id / f"{block_id}.png"
                    image_path = save_crop(image, bbox, crop_path)

                blocks.append(Block(
                    id=block_id, doc_id=doc_id, page=page_num,
                    block_type=block_type, bbox=bbox, text=text,
                    confidence=det["confidence"], image_path=image_path,
                ))
            return blocks
        else:
            # No detector, no PDF — minimal extraction
            return []

    def _classify_heuristic(self, raw_block: dict) -> BlockType:
        """Classify block type using font size heuristic."""
        text = raw_block.get("text", "").strip()
        font_size = raw_block.get("font_size", 12.0)
        block_type_hint = raw_block.get("block_type", "Paragraph")

        if block_type_hint == "Figure":
            return BlockType.FIGURE

        if not text:
            return BlockType.FIGURE

        # Font size based classification
        if font_size >= TITLE_FONT_THRESHOLD and len(text) < 200:
            return BlockType.TITLE
        elif font_size >= HEADER_FONT_THRESHOLD and len(text) < 200:
            return BlockType.HEADER

        # Content-based heuristics
        text_lower = text.lower()
        if text_lower.startswith(("table ", "fig.", "figure ")):
            return BlockType.CAPTION
        if len(text) < 30 and text.count("\t") >= 2:
            return BlockType.TABLE

        return BlockType.PARAGRAPH

    def _classify_with_detector(self, raw_block: dict, page_image: Image.Image) -> BlockType:
        """Use layout detector for block type classification."""
        # TODO: match detector output with raw_block bbox
        return self._classify_heuristic(raw_block)

    def _scale_bbox(self, bbox, pdf_path: str, page_num: int,
                    page_image: Image.Image):
        """Scale bbox from PDF coordinates to image pixel coordinates."""
        import fitz
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        pdf_w, pdf_h = page.rect.width, page.rect.height
        doc.close()

        img_w, img_h = page_image.size
        scale_x = img_w / pdf_w
        scale_y = img_h / pdf_h

        x0, y0, x1, y1 = bbox
        return (x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y)
