"""Serialize evidence packets into reader-facing prompt strings."""
from typing import List, Dict, Optional
from src.kg.schema import SupportUnit, BlockType
from src.utils.tokenizer_utils import count_tokens, IMAGE_TOKEN_COST


class EvidenceSerializer:
    """Convert a retrieved evidence packet into a serialized prompt string."""

    def __init__(self, token_budget: int = 16384, max_images: int = 4):
        self.token_budget = token_budget
        self.max_images = max_images

    def serialize(self, units: List[Dict], doc_meta: Dict = None) -> str:
        """Serialize a list of evidence units into a context string.

        Args:
            units: List of dicts with keys from SupportUnit.to_dict() plus
                   optional 'scope_path' and 'entity_anchors'.
            doc_meta: Optional document metadata.

        Returns:
            Serialized context string within token budget.
        """
        parts = []
        current_tokens = 0
        image_count = 0

        # Document metadata header
        if doc_meta:
            header = f"[DOC_META]\nID: {doc_meta.get('doc_id', 'N/A')} | Title: {doc_meta.get('title', 'N/A')}\n"
            header += "-" * 40 + "\n"
            parts.append(header)
            current_tokens += count_tokens(header)

        # Group units by scope/section
        current_scope = ""
        for unit in units:
            scope = unit.get("section_path", "")
            unit_type = unit.get("unit_type", "") or unit.get("block_type", "Text")
            unit_id = unit.get("id", "?")
            content = unit.get("content", "") or unit.get("text", "")

            # Scope header (only if changed)
            if scope and scope != current_scope:
                scope_line = f"\n[SCOPE] {scope}\n"
                scope_tokens = count_tokens(scope_line)
                if current_tokens + scope_tokens > self.token_budget:
                    break
                parts.append(scope_line)
                current_tokens += scope_tokens
                current_scope = scope

            # Unit content
            unit_header = f"\n[UNIT id={unit_id} | Type={unit_type}]\n"
            if scope:
                unit_header += f"Path: {scope}\n"

            # Handle visual units
            if unit_type in ("Table", "Figure") and unit.get("image_path"):
                if image_count < self.max_images:
                    unit_header += f"Visual: <|image_token_{image_count + 1}|>\n"
                    image_count += 1
                    current_tokens += IMAGE_TOKEN_COST

            unit_text = unit_header + f"Content: {content}\n"
            unit_tokens = count_tokens(unit_text)

            if current_tokens + unit_tokens > self.token_budget:
                # Try truncating content
                remaining = self.token_budget - current_tokens - count_tokens(unit_header) - 10
                if remaining > 50:
                    truncated = self._truncate_to_tokens(content, remaining)
                    unit_text = unit_header + f"Content: {truncated}\n"
                    parts.append(unit_text)
                break

            parts.append(unit_text)
            current_tokens += unit_tokens

        return "".join(parts)

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximately max_tokens."""
        # Rough estimate: 1 token ≈ 4 chars for English
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."
