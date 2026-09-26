"""KG seed retrieval: type-aware scoring for initial seed selection."""
import logging
from typing import List, Dict, Tuple

from src.retrieval.bm25_index import BM25Index
from src.retrieval.dense_index import DenseIndex
from src.kg.schema import NodeType

logger = logging.getLogger(__name__)


class KGSeedRetriever:
    """Retrieve seed nodes from the corpus-level KG.

    Implements type-aware scoring:
    - Text nodes: hybrid score (BM25 + dense)
    - Visual nodes: gamma * visual_score + (1-gamma) * hybrid_context_score
    """

    def __init__(self, bm25_index: BM25Index, dense_index: DenseIndex,
                 gamma_visual: float = 0.6):
        self.bm25 = bm25_index
        self.dense = dense_index
        self.gamma = gamma_visual
        self._node_types: Dict[str, str] = {}  # node_id -> type

    def set_node_types(self, node_types: Dict[str, str]):
        """Set node type mapping for type-aware scoring."""
        self._node_types = node_types

    def retrieve_seeds(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Retrieve top-k seed nodes from the KG.

        Returns:
            List of (node_id, score) tuples.
        """
        # Get BM25 scores
        bm25_results = {doc_id: score for doc_id, score in self.bm25.search(query, top_k * 3)}

        # Get dense scores
        dense_results = {doc_id: score for doc_id, score in self.dense.search(query, top_k * 3)}

        # Combine scores with type-aware weighting
        all_ids = set(bm25_results.keys()) | set(dense_results.keys())
        scored = []
        for node_id in all_ids:
            bm25_score = bm25_results.get(node_id, 0.0)
            dense_score = dense_results.get(node_id, 0.0)

            # Normalize BM25 scores to [0, 1] range
            max_bm25 = max(bm25_results.values()) if bm25_results else 1.0
            bm25_norm = bm25_score / max_bm25 if max_bm25 > 0 else 0.0

            # Hybrid score
            node_type = self._node_types.get(node_id, "text")
            if node_type in ("Table", "Figure"):
                # Visual nodes: heavier weight on dense similarity
                score = self.gamma * dense_score + (1 - self.gamma) * (0.5 * bm25_norm + 0.5 * dense_score)
            else:
                # Text nodes: balanced hybrid
                score = 0.4 * bm25_norm + 0.6 * dense_score

            scored.append((node_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
