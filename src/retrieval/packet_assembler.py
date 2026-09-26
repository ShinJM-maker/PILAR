"""Assertion-Centered Evidence Packet (ACEP) assembly — Algorithm 1 from paper."""
import logging
from typing import List, Dict, Set, Optional
import networkx as nx
from src.kg.schema import NodeType, EdgeType, SupportUnit, BlockType
from src.utils.tokenizer_utils import count_tokens, IMAGE_TOKEN_COST

logger = logging.getLogger(__name__)


class PacketAssembler:
    """Assemble evidence packets under strict token budget.

    Implements Algorithm 1 from the paper:
    Phase 1: Anchor & Ancestry (governing scope)
    Phase 2: Sibling Packing (structural neighbors)
    Phase 3: Semantic Packing (cross-section visual units)
    """

    def __init__(self, graph: nx.DiGraph, supports: Dict[str, SupportUnit],
                 token_budget: int = 16384, max_images: int = 4):
        self.graph = graph
        self.supports = supports
        self.token_budget = token_budget
        self.max_images = max_images

        # Pre-build sibling index: (doc_id, section_path) -> [support_ids]
        from collections import defaultdict
        self._sibling_index = defaultdict(list)
        for sid, support in supports.items():
            if support.doc_id and support.section_path:
                key = (support.doc_id, support.section_path)
                self._sibling_index[key].append(sid)
        logger.info(f"PacketAssembler: built sibling index with {len(self._sibling_index)} groups")

    def assemble(self, ranked_anchors: List[str],
                 similarity_fn=None) -> List[Dict]:
        """Assemble evidence packet from ranked anchor support units.

        Args:
            ranked_anchors: List of support unit IDs, ranked by retrieval score.
            similarity_fn: Optional function (unit_id, candidate_id) -> float
                          for semantic packing.

        Returns:
            List of unit dicts (serializable) in packing order.
        """
        packet: List[Dict] = []
        packet_ids: Set[str] = set()
        current_tokens = 0
        image_count = 0

        for anchor_id in ranked_anchors:
            if anchor_id not in self.supports:
                continue

            anchor = self.supports[anchor_id]

            # --- Phase 1: Anchor & Ancestry ---
            ancestry_units = self._get_ancestry(anchor_id)
            anchor_cost = self._estimate_tokens(anchor, include_image=True)
            ancestry_cost = sum(self._estimate_tokens(self.supports[uid])
                               for uid in ancestry_units if uid in self.supports)
            total_cost = anchor_cost + ancestry_cost

            if current_tokens + total_cost > self.token_budget:
                continue

            # Add anchor
            if anchor_id not in packet_ids:
                is_visual = anchor.unit_type in (BlockType.TABLE, BlockType.FIGURE)
                if is_visual and image_count >= self.max_images:
                    anchor_cost = self._estimate_tokens(anchor, include_image=False)
                else:
                    if is_visual:
                        image_count += 1

                packet.append(anchor.to_dict())
                packet_ids.add(anchor_id)
                current_tokens += anchor_cost

            # Add ancestry (scope carriers)
            for anc_id in ancestry_units:
                if anc_id in self.supports and anc_id not in packet_ids:
                    anc = self.supports[anc_id]
                    cost = self._estimate_tokens(anc)
                    if current_tokens + cost <= self.token_budget:
                        packet.append(anc.to_dict())
                        packet_ids.add(anc_id)
                        current_tokens += cost

            # --- Phase 2: Sibling Packing ---
            siblings = self._get_siblings(anchor_id)
            for sib_id in siblings:
                if sib_id in packet_ids or sib_id not in self.supports:
                    continue
                sib = self.supports[sib_id]
                cost = self._estimate_tokens(sib)
                if current_tokens + cost <= self.token_budget:
                    packet.append(sib.to_dict())
                    packet_ids.add(sib_id)
                    current_tokens += cost

            # --- Phase 3: Semantic Packing ---
            if similarity_fn is not None:
                candidates = self._get_semantic_candidates(anchor_id, similarity_fn)
                for cand_id in candidates:
                    if cand_id in packet_ids or cand_id not in self.supports:
                        continue
                    cand = self.supports[cand_id]
                    # Prefer visual units in semantic packing
                    if cand.unit_type not in (BlockType.TABLE, BlockType.FIGURE):
                        continue
                    is_visual = True
                    if image_count >= self.max_images:
                        cost = self._estimate_tokens(cand, include_image=False)
                    else:
                        cost = self._estimate_tokens(cand, include_image=True)
                        image_count += 1

                    # Include ancestry for semantic candidates too
                    anc_ids = self._get_ancestry(cand_id)
                    anc_cost = sum(self._estimate_tokens(self.supports[a])
                                   for a in anc_ids if a in self.supports and a not in packet_ids)

                    if current_tokens + cost + anc_cost <= self.token_budget:
                        packet.append(cand.to_dict())
                        packet_ids.add(cand_id)
                        current_tokens += cost
                        for a in anc_ids:
                            if a in self.supports and a not in packet_ids:
                                packet.append(self.supports[a].to_dict())
                                packet_ids.add(a)
                                current_tokens += self._estimate_tokens(self.supports[a])

        logger.debug(f"Assembled packet: {len(packet)} units, ~{current_tokens} tokens, "
                     f"{image_count} images")
        return packet

    def _get_ancestry(self, unit_id: str) -> List[str]:
        """Get scope carrier ancestry for a support unit."""
        ancestry = []
        for _, target, data in self.graph.edges(unit_id, data=True):
            if data.get("edge_type") == EdgeType.HAS_SCOPE.value:
                ancestry.append(target)
        return ancestry

    def _get_siblings(self, unit_id: str) -> List[str]:
        """Get sibling units (same parent section) using pre-built index."""
        unit = self.supports.get(unit_id)
        if not unit or not unit.doc_id or not unit.section_path:
            return []

        key = (unit.doc_id, unit.section_path)
        siblings = [sid for sid in self._sibling_index.get(key, [])
                     if sid != unit_id]
        return siblings[:5]

    def _get_semantic_candidates(self, anchor_id: str, similarity_fn,
                                  top_m: int = 5) -> List[str]:
        """Get top-M semantically similar units from same document."""
        anchor = self.supports.get(anchor_id)
        if not anchor:
            return []

        candidates = []
        for sid, support in self.supports.items():
            if sid == anchor_id or support.doc_id != anchor.doc_id:
                continue
            if support.unit_type in (BlockType.TABLE, BlockType.FIGURE):
                score = similarity_fn(anchor_id, sid)
                candidates.append((sid, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:top_m]]

    def _estimate_tokens(self, unit: SupportUnit, include_image: bool = False) -> int:
        """Estimate token cost of a support unit (fast approximation)."""
        # Use word count × 1.3 instead of full tokenizer for speed
        text_tokens = int(len(unit.content.split()) * 1.3) if unit.content else 0
        overhead = 30  # metadata, formatting
        image_tokens = IMAGE_TOKEN_COST if (include_image and unit.image_path) else 0
        return text_tokens + overhead + image_tokens
