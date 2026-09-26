"""v4 Boundary-aware snippet packer.

Instead of packing full pages, extracts relevant blocks (snippets) from each page
and packs them hierarchically by role: seed-core > seed-tail > adjacent-head/tail
> other-seed-core > expanded-core > fallback.
"""
import logging
import re
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

logger = logging.getLogger(__name__)


class SnippetPacker:
    """Boundary-aware snippet packer for VEGA-KG v4.

    Takes retriever output (pages with role metadata) and produces
    role-tagged snippets within token budget.
    """

    def __init__(self, preprocessed_dir: str, token_budget: int = 4500,
                 budget_ratios: Optional[Dict[str, float]] = None):
        """
        Args:
            preprocessed_dir: Path to data/preprocessed/ directory.
            token_budget: Total token budget for context.
            budget_ratios: Role-wise budget allocation.
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        self.token_budget = token_budget
        self.budget_ratios = budget_ratios or {
            "seed": 0.45,
            "adjacent": 0.25,
            "expanded": 0.10,
            "other": 0.20,
        }

        # Cache loaded doc blocks: doc_id -> {page_num -> [blocks]}
        self._doc_block_cache = {}

    # ---- Token estimation ----

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return int(len(text.split()) * 1.3)

    # ---- Block loading ----

    def _load_doc_blocks(self, doc_id: str) -> Dict[int, List[Dict]]:
        """Load blocks for a document, grouped by page. Cached."""
        if doc_id in self._doc_block_cache:
            return self._doc_block_cache[doc_id]

        doc_path = self.preprocessed_dir / f"{doc_id}.json"
        if not doc_path.exists():
            logger.warning(f"Preprocessed file not found: {doc_path}")
            self._doc_block_cache[doc_id] = {}
            return {}

        with open(doc_path) as f:
            doc_data = json.load(f)

        pages = defaultdict(list)
        for block in doc_data.get("blocks", []):
            text = block.get("text", "").strip()
            if not text:
                continue
            page_num = int(block.get("page", 0))
            pages[page_num].append({
                "id": block.get("id", ""),
                "text": text,
                "block_type": block.get("block_type", "Paragraph"),
                "bbox": block.get("bbox", ""),
                "section_path": block.get("section_path", ""),
            })

        self._doc_block_cache[doc_id] = dict(pages)
        return dict(pages)

    def clear_cache(self):
        """Clear block cache between queries."""
        self._doc_block_cache.clear()

    # ---- Paragraph merging ----

    MIN_PARAGRAPH_TOKENS = 40  # Minimum tokens per merged paragraph

    def _merge_blocks_to_paragraphs(self, blocks: List[Dict]) -> List[Dict]:
        """Merge consecutive fine-grained blocks into paragraph-level segments.

        PyMuPDF blocks are often single lines (~30 chars). This merges them
        into coherent paragraphs using section boundaries and minimum token count.
        """
        if not blocks:
            return []

        paragraphs = []
        current_texts = []
        current_section = blocks[0].get("section_path", "")
        current_types = []

        def flush():
            if current_texts:
                merged_text = "\n".join(current_texts)
                # Determine dominant block type
                dominant_type = max(set(current_types), key=current_types.count) if current_types else "Paragraph"
                paragraphs.append({
                    "text": merged_text,
                    "block_type": dominant_type,
                    "section_path": current_section,
                    "num_blocks": len(current_texts),
                })

        for block in blocks:
            block_section = block.get("section_path", "")
            block_type = block.get("block_type", "Paragraph")

            # Start new paragraph on section change or Header/Title block
            if (block_section != current_section and block_section) or \
               block_type in ("Header", "Title"):
                # Only flush if current paragraph is non-trivial
                if current_texts:
                    flush()
                    current_texts = []
                    current_types = []
                current_section = block_section

            current_texts.append(block["text"])
            current_types.append(block_type)

            # If accumulated enough tokens, flush
            merged = "\n".join(current_texts)
            if self._estimate_tokens(merged) >= self.MIN_PARAGRAPH_TOKENS:
                flush()
                current_texts = []
                current_types = []

        flush()
        return paragraphs

    # ---- Paragraph scoring ----

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _score_paragraphs(self, paragraphs: List[Dict], query: str) -> List[Tuple[Dict, float]]:
        """Score paragraphs by keyword overlap with query.

        Returns: list of (paragraph, score) sorted by score desc.
        """
        q_tokens = set(self._normalize_text(query).split())
        if not q_tokens:
            return [(p, 0.0) for p in paragraphs]

        scored = []
        for para in paragraphs:
            p_tokens = set(self._normalize_text(para["text"]).split())
            if not p_tokens:
                scored.append((para, 0.0))
                continue
            overlap = len(q_tokens & p_tokens)
            # Recall-oriented: how much of query is covered
            score = overlap / len(q_tokens)
            scored.append((para, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ---- Snippet extraction ----

    def _extract_snippets(
        self,
        page_chunk: Dict,
        query: str,
        role: str,
    ) -> List[Dict]:
        """Extract relevant paragraph-level snippets from a page based on role.

        Returns list of snippet dicts with keys:
            text, role_tag, doc_id, page, block_type, score
        """
        doc_id = page_chunk.get("doc_id", "")
        page_num = page_chunk.get("page", 0)
        if isinstance(page_num, str):
            page_num = int(page_num)

        doc_blocks = self._load_doc_blocks(doc_id)
        raw_blocks = doc_blocks.get(page_num, [])

        # Merge fine-grained blocks into paragraphs
        paragraphs = self._merge_blocks_to_paragraphs(raw_blocks)

        if not paragraphs:
            # Fallback: use full page text
            text = page_chunk.get("text", "")
            if text:
                return [{
                    "text": text,
                    "role_tag": f"{role}-full",
                    "doc_id": doc_id,
                    "page": page_num,
                    "block_type": "Page",
                    "score": 0.0,
                }]
            return []

        scored_paras = self._score_paragraphs(paragraphs, query)
        snippets = []

        if role == "seed":
            # Core: query-relevant paragraphs
            core_indices = set()
            for idx, (para, score) in enumerate(scored_paras):
                if score > 0:
                    snippets.append(self._make_snippet(
                        para, "seed-core", doc_id, page_num, score))
                    core_indices.add(id(para))

            # Tail: last paragraph(s) if not already in core
            for para in paragraphs[-2:]:
                if id(para) not in core_indices:
                    snippets.append(self._make_snippet(
                        para, "seed-tail", doc_id, page_num, 0.0))

            # Fallback: all paragraphs if nothing matched
            if not any(s["role_tag"] == "seed-core" for s in snippets):
                for para in paragraphs:
                    snippets.append(self._make_snippet(
                        para, "seed-core", doc_id, page_num, 0.0))

        elif role == "adjacent":
            # Head paragraphs
            for para in paragraphs[:2]:
                snippets.append(self._make_snippet(
                    para, "adjacent-head", doc_id, page_num, 0.0))
            # Tail paragraphs
            head_ids = {id(p) for p in paragraphs[:2]}
            for para in paragraphs[-2:]:
                if id(para) not in head_ids:
                    snippets.append(self._make_snippet(
                        para, "adjacent-tail", doc_id, page_num, 0.0))
            # Core: query-relevant not already included
            included_ids = head_ids | {id(p) for p in paragraphs[-2:]}
            for para, score in scored_paras:
                if score > 0 and id(para) not in included_ids:
                    snippets.append(self._make_snippet(
                        para, "adjacent-core", doc_id, page_num, score))

        elif role == "expanded":
            # Core only
            for para, score in scored_paras:
                if score > 0:
                    snippets.append(self._make_snippet(
                        para, "expanded-core", doc_id, page_num, score))
            if not snippets:
                for para in paragraphs[:2]:
                    snippets.append(self._make_snippet(
                        para, "expanded-core", doc_id, page_num, 0.0))

        else:  # "other"
            for para, score in scored_paras:
                if score > 0:
                    snippets.append(self._make_snippet(
                        para, "other-core", doc_id, page_num, score))
            if not snippets:
                for para in paragraphs[:2]:
                    snippets.append(self._make_snippet(
                        para, "other-core", doc_id, page_num, 0.0))

        return snippets

    @staticmethod
    def _make_snippet(para, role_tag, doc_id, page_num, score):
        return {
            "text": para["text"],
            "role_tag": role_tag,
            "doc_id": doc_id,
            "page": page_num,
            "block_type": para.get("block_type", "Paragraph"),
            "score": score,
        }

    # ---- Hierarchical packing ----

    def pack(
        self,
        pages: List[Dict],
        query: str,
    ) -> str:
        """Pack pages into boundary-aware snippets with role-based hierarchy.

        Args:
            pages: List of page chunks from retriever, each with '_role' field.
            query: The user query.

        Returns:
            Serialized context string with role-tagged snippets.
        """
        self.clear_cache()

        # Group pages by role
        role_pages = defaultdict(list)
        for page in pages:
            role = page.get("_role", "other")
            role_pages[role].append(page)

        # Extract all snippets per role
        role_snippets = defaultdict(list)
        for role, role_page_list in role_pages.items():
            for page in role_page_list:
                snippets = self._extract_snippets(page, query, role)
                role_snippets[role].extend(snippets)

        # Compute per-role budgets
        budgets = {}
        for role, ratio in self.budget_ratios.items():
            budgets[role] = int(self.token_budget * ratio)

        # Hierarchical packing order
        packing_order = [
            # (source_role, role_tag_filter, target_budget_key)
            ("seed", "seed-core", "seed"),
            ("seed", "seed-tail", "seed"),
            ("adjacent", "adjacent-head", "adjacent"),
            ("adjacent", "adjacent-tail", "adjacent"),
            ("adjacent", "adjacent-core", "adjacent"),
            ("seed", "seed-core", "seed"),  # remaining seed-core
            ("expanded", "expanded-core", "expanded"),
            ("other", "other-core", "other"),
        ]

        selected_snippets = []
        used_tokens = defaultdict(int)
        total_tokens = 0
        seen_texts = set()

        for source_role, tag_filter, budget_key in packing_order:
            budget_limit = budgets.get(budget_key, 0)
            candidates = [
                s for s in role_snippets.get(source_role, [])
                if s["role_tag"] == tag_filter
            ]
            # Sort by score descending
            candidates.sort(key=lambda x: x["score"], reverse=True)

            for snippet in candidates:
                text_hash = hash(snippet["text"])
                if text_hash in seen_texts:
                    continue
                est = self._estimate_tokens(snippet["text"])
                if used_tokens[budget_key] + est > budget_limit:
                    continue
                if total_tokens + est > self.token_budget:
                    continue
                selected_snippets.append(snippet)
                seen_texts.add(text_hash)
                used_tokens[budget_key] += est
                total_tokens += est

        # If budget not fully used, try filling with remaining snippets
        all_remaining = []
        for role, snips in role_snippets.items():
            for s in snips:
                if hash(s["text"]) not in seen_texts:
                    all_remaining.append(s)
        all_remaining.sort(key=lambda x: x["score"], reverse=True)

        for snippet in all_remaining:
            text_hash = hash(snippet["text"])
            if text_hash in seen_texts:
                continue
            est = self._estimate_tokens(snippet["text"])
            if total_tokens + est > self.token_budget:
                continue
            selected_snippets.append(snippet)
            seen_texts.add(text_hash)
            total_tokens += est

        # Serialize
        return self._serialize(selected_snippets)

    def _serialize(self, snippets: List[Dict]) -> str:
        """Serialize snippets into context string with role tags."""
        if not snippets:
            return ""

        parts = []
        for snippet in snippets:
            doc_id_short = snippet["doc_id"][:8] if snippet["doc_id"] else "?"
            header = (f"[Doc: {doc_id_short} | Page: {snippet['page']} | "
                      f"Role: {snippet['role_tag']}]")
            parts.append(f"{header}\n{snippet['text']}\n")

        return "\n".join(parts)

    def pack_full_page(self, pages: List[Dict], query: str) -> str:
        """B0 baseline: full-page packing (no snippet extraction).
        Uses role tags but packs entire page text.
        """
        parts = []
        total_tokens = 0

        for page in pages:
            text = page.get("text", "")
            if not text:
                continue
            est = self._estimate_tokens(text)
            if total_tokens + est > self.token_budget:
                break
            role = page.get("_role", "other")
            doc_id_short = page.get("doc_id", "?")[:8]
            page_num = page.get("page", 0)
            header = f"[Doc: {doc_id_short} | Page: {page_num} | Role: {role}]"
            parts.append(f"{header}\n{text}\n")
            total_tokens += est

        return "\n".join(parts)


def normalize_answer_v4(text: str) -> str:
    """Enhanced answer normalization for v4.

    - Whitespace normalization
    - Number format normalization (1,000 → 1000)
    - Unit normalization (percent → %, dollar → $)
    - Date format normalization
    - Strip trailing periods/punctuation
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.strip()

    # Strip quotes
    if len(text) >= 2 and text[0] == text[-1] and text[0] in '"\'':
        text = text[1:-1].strip()

    # Strip trailing period
    if text.endswith('.'):
        text = text[:-1].strip()

    # Number normalization: remove commas in numbers
    text = re.sub(r'(\d),(\d)', r'\1\2', text)

    # Unit normalization
    text = re.sub(r'\bpercent\b', '%', text, flags=re.IGNORECASE)
    text = re.sub(r'\bdollars?\b', '$', text, flags=re.IGNORECASE)
    text = re.sub(r'\bmillion\b', 'M', text, flags=re.IGNORECASE)
    text = re.sub(r'\bbillion\b', 'B', text, flags=re.IGNORECASE)

    # Whitespace normalization
    text = re.sub(r'\s+', ' ', text).strip()

    return text
