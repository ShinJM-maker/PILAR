"""PDF to page image rendering using PyMuPDF."""
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import io


def render_pdf_pages(pdf_path: str | Path, dpi: int = 300) -> List[Tuple[int, Image.Image]]:
    """Render all pages of a PDF as PIL Images.

    Returns:
        List of (page_number, PIL.Image) tuples (0-indexed).
    """
    doc = fitz.open(str(pdf_path))
    pages = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        pages.append((page_num, img))
    doc.close()
    return pages


def render_and_save(pdf_path: str | Path, output_dir: str | Path,
                    dpi: int = 300, doc_id: str = None) -> List[str]:
    """Render PDF pages and save as PNG files.

    Returns:
        List of saved image paths.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if doc_id is None:
        doc_id = pdf_path.stem

    saved_paths = []
    pages = render_pdf_pages(pdf_path, dpi)
    for page_num, img in pages:
        out_path = output_dir / f"{doc_id}_page_{page_num:04d}.png"
        img.save(str(out_path))
        saved_paths.append(str(out_path))
    return saved_paths


def get_page_count(pdf_path: str | Path) -> int:
    doc = fitz.open(str(pdf_path))
    count = len(doc)
    doc.close()
    return count
