"""Doc_card construction for document-level summaries."""
from typing import List
from src.kg.schema import Block, BlockType


def build_doc_card(blocks: List[Block], max_length: int = 512) -> str:
    """Build a Doc_card summary from document blocks.

    Concatenates: Title + high-level section headers (depth 1-2).
    """
    title = ""
    headers = []

    for b in blocks:
        if b.block_type == BlockType.TITLE and not title:
            title = b.text.strip()
        elif b.block_type == BlockType.HEADER and b.depth <= 2:
            header_text = b.text.strip()
            if header_text and header_text not in headers:
                headers.append(header_text)

    parts = []
    if title:
        parts.append(f"Title: {title}")
    if headers:
        parts.append("Sections: " + " | ".join(headers))

    card = "\n".join(parts)
    if len(card) > max_length:
        card = card[:max_length]
    return card
