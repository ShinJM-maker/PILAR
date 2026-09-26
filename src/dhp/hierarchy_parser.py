"""Document Hierarchy Parsing: reconstruct section tree from layout blocks."""
import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from src.kg.schema import Block, BlockType

logger = logging.getLogger(__name__)

# Heading-level heuristic based on block type and position
HEADING_TYPES = {BlockType.TITLE, BlockType.HEADER}


class HierarchyParser:
    """Reconstruct document hierarchy tree from detected blocks.

    Uses a heuristic approach based on:
    1. Block types (Title > Header > Paragraph)
    2. Font size approximation from bbox height
    3. Reading order
    """

    def __init__(self, max_depth: int = 6):
        self.max_depth = max_depth

    def parse(self, blocks: List[Block]) -> List[Block]:
        """Build hierarchy tree by assigning parent-child relationships.

        Modifies blocks in-place (sets parent_id, children_ids, depth, section_path).
        Returns the same block list with hierarchy info populated.
        """
        if not blocks:
            return blocks

        # Step 1: Identify heading blocks and estimate their levels
        heading_blocks = []
        content_blocks = []
        for b in blocks:
            if b.block_type in HEADING_TYPES:
                heading_blocks.append(b)
            else:
                content_blocks.append(b)

        # Assign depth to headings based on type and font size proxy
        heading_depths = self._assign_heading_depths(heading_blocks)

        # Step 2: Build tree structure
        # Create a virtual root
        root_id = f"{blocks[0].doc_id}_root"
        block_map = {b.id: b for b in blocks}

        # Process all blocks in reading order (page, then y-position)
        all_sorted = sorted(blocks, key=lambda b: (b.page, b.bbox[1], b.bbox[0]))

        # Stack-based tree construction
        # Stack contains (block_id, depth) pairs
        stack: List[Tuple[str, int]] = [(root_id, -1)]
        current_heading_path: List[str] = []

        for block in all_sorted:
            if block.id in heading_depths:
                depth = heading_depths[block.id]
                block.depth = depth

                # Pop stack until we find a parent with lower depth
                while len(stack) > 1 and stack[-1][1] >= depth:
                    stack.pop()

                # Set parent
                parent_id = stack[-1][0]
                if parent_id != root_id:
                    block.parent_id = parent_id
                    block_map[parent_id].children_ids.append(block.id)

                stack.append((block.id, depth))

                # Update heading path
                current_heading_path = current_heading_path[:depth] + [block.text.strip()]

            else:
                # Content block: attach to current heading
                if len(stack) > 1:
                    parent_id = stack[-1][0]
                    block.parent_id = parent_id
                    block_map[parent_id].children_ids.append(block.id)
                    block.depth = stack[-1][1] + 1

            # Set section path
            block.section_path = " > ".join(current_heading_path) if current_heading_path else ""

        return blocks

    def _assign_heading_depths(self, headings: List[Block]) -> Dict[str, int]:
        """Assign depth levels to heading blocks.

        Strategy: Title=0, then rank remaining headers by bbox height (larger=higher level).
        """
        depths = {}
        if not headings:
            return depths

        # Title blocks are always depth 0
        non_title = []
        for h in headings:
            if h.block_type == BlockType.TITLE:
                depths[h.id] = 0
            else:
                non_title.append(h)

        if not non_title:
            return depths

        # Estimate font size from bbox height relative to text length
        height_ratios = []
        for h in non_title:
            bbox_height = h.bbox[3] - h.bbox[1]
            text_lines = max(1, h.text.count('\n') + 1)
            ratio = bbox_height / text_lines
            height_ratios.append((h, ratio))

        # Sort by height ratio descending, assign depths 1..max_depth
        height_ratios.sort(key=lambda x: x[1], reverse=True)

        if len(height_ratios) == 1:
            depths[height_ratios[0][0].id] = 1
        else:
            # Cluster into levels using quantiles
            ratios = [r for _, r in height_ratios]
            unique_levels = self._quantize_to_levels(ratios, max_levels=self.max_depth - 1)
            for (h, _), level in zip(height_ratios, unique_levels):
                depths[h.id] = level + 1  # +1 because Title is 0

        return depths

    def _quantize_to_levels(self, values: List[float], max_levels: int) -> List[int]:
        """Quantize a list of values into discrete levels."""
        if not values:
            return []
        if len(set(values)) == 1:
            return [0] * len(values)

        # Simple approach: rank-based quantization
        sorted_unique = sorted(set(values), reverse=True)
        n_levels = min(len(sorted_unique), max_levels)
        value_to_level = {}
        for i, v in enumerate(sorted_unique):
            level = min(i, n_levels - 1)
            value_to_level[v] = level

        return [value_to_level[v] for v in values]


def get_governing_header(block: Block, block_map: Dict[str, Block]) -> str:
    """Traverse up the tree to find the nearest heading ancestor."""
    current = block
    while current.parent_id:
        parent = block_map.get(current.parent_id)
        if parent is None:
            break
        if parent.block_type in HEADING_TYPES:
            return parent.text.strip()
        current = parent
    return ""


def get_section_path(block: Block, block_map: Dict[str, Block]) -> str:
    """Get full section path from root to this block's governing header."""
    path_parts = []
    current = block
    while current.parent_id:
        parent = block_map.get(current.parent_id)
        if parent is None:
            break
        if parent.block_type in HEADING_TYPES:
            path_parts.append(parent.text.strip())
        current = parent
    path_parts.reverse()
    return " > ".join(path_parts)
