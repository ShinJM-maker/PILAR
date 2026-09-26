"""BM25 sparse retrieval index."""
import logging
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Index:
    """BM25 index over KG node text fields."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._index = None
        self._doc_ids: List[str] = []
        self._doc_texts: List[str] = []

    def build(self, documents: List[Dict[str, str]]):
        """Build BM25 index from documents.

        Args:
            documents: List of {"id": ..., "text": ...} dicts.
        """
        self._doc_ids = [d["id"] for d in documents]
        self._doc_texts = [d["text"] for d in documents]

        # Tokenize
        tokenized = [text.lower().split() for text in self._doc_texts]
        self._index = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        logger.info(f"Built BM25 index with {len(documents)} documents")

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Search the index.

        Returns:
            List of (doc_id, score) tuples sorted by score descending.
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build() first.")

        tokenized_query = query.lower().split()
        scores = self._index.get_scores(tokenized_query)

        # Get top-k
        scored_docs = list(zip(self._doc_ids, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]
