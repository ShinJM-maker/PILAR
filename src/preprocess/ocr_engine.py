"""OCR/text extraction engine. Uses PyMuPDF native text extraction (preferred)
with Tesseract as optional fallback for image-only PDFs."""
import re
from typing import Tuple, List, Optional
from PIL import Image

try:
    import pytesseract
    _HAS_TESSERACT = True
except (ImportError, Exception):
    _HAS_TESSERACT = False

try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False


def extract_text_from_pdf_page(pdf_path: str, page_num: int) -> str:
    """Extract text from a PDF page using PyMuPDF (no OCR needed for digital PDFs)."""
    if not _HAS_FITZ:
        raise ImportError("PyMuPDF is required.")
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    text = page.get_text("text")
    doc.close()
    return clean_ocr_text(text)


def extract_text_blocks_from_pdf_page(pdf_path: str, page_num: int) -> List[dict]:
    """Extract text blocks with bounding boxes from a PDF page using PyMuPDF.

    Returns:
        List of dicts with keys: bbox (x0, y0, x1, y1), text, block_type
    """
    if not _HAS_FITZ:
        raise ImportError("PyMuPDF is required.")
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    blocks = []
    # Get text blocks (type 0 = text, type 1 = image)
    for block in page.get_text("dict")["blocks"]:
        if block["type"] == 0:  # text block
            bbox = block["bbox"]  # (x0, y0, x1, y1)
            lines_text = []
            for line in block.get("lines", []):
                spans_text = []
                for span in line.get("spans", []):
                    spans_text.append(span["text"])
                lines_text.append(" ".join(spans_text))
            text = "\n".join(lines_text)
            if text.strip():
                blocks.append({
                    "bbox": tuple(bbox),
                    "text": clean_ocr_text(text),
                    "block_type": "Paragraph",  # default; will be refined by layout detector
                    "font_size": _avg_font_size(block),
                })
        elif block["type"] == 1:  # image block
            bbox = block["bbox"]
            blocks.append({
                "bbox": tuple(bbox),
                "text": "",
                "block_type": "Figure",
                "font_size": 0,
            })

    doc.close()

    # Sort by reading order
    blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    return blocks


def _avg_font_size(block: dict) -> float:
    """Get average font size from a PyMuPDF text block."""
    sizes = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            sizes.append(span.get("size", 12))
    return sum(sizes) / len(sizes) if sizes else 12.0


def ocr_region(image: Image.Image, bbox: Tuple[float, float, float, float],
               lang: str = "eng") -> str:
    """Run OCR on a cropped region of an image (Tesseract fallback)."""
    if not _HAS_TESSERACT:
        return ""  # gracefully return empty if no Tesseract

    x0, y0, x1, y1 = [int(c) for c in bbox]
    w, h = image.size
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return ""

    crop = image.crop((x0, y0, x1, y1))
    try:
        text = pytesseract.image_to_string(crop, lang=lang)
        return clean_ocr_text(text)
    except Exception:
        return ""


def ocr_full_page(image: Image.Image, lang: str = "eng") -> str:
    """Run OCR on a full page image (Tesseract fallback)."""
    if not _HAS_TESSERACT:
        return ""
    try:
        text = pytesseract.image_to_string(image, lang=lang)
        return clean_ocr_text(text)
    except Exception:
        return ""


def clean_ocr_text(text: str) -> str:
    """Normalize text: lowercase, remove control chars, normalize whitespace."""
    text = text.lower()
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
