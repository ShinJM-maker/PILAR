"""Dense retrieval index using sentence-transformers + FAISS."""
import logging
import threading
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DenseIndex:
    """Dense vector index for text and visual embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu", batch_size: int = 64,
                 revision: str | None = None):
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._index = None
        self._doc_ids: List[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    def _load_model(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                revision=self.revision,
            )
            logger.info(f"Loaded embedding model: {self.model_name} revision={self.revision}")

    def build(self, documents: List[Dict[str, str]], use_faiss: bool = True):
        """Build dense index from documents.

        Args:
            documents: List of {"id": ..., "text": ...} dicts.
            use_faiss: Use FAISS for efficient search. Falls back to numpy if unavailable.
        """
        self._load_model()
        self._doc_ids = [d["id"] for d in documents]
        texts = [d["text"] for d in documents]

        logger.info(f"Encoding {len(texts)} documents...")
        self._embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        if use_faiss:
            try:
                import faiss
                dim = self._embeddings.shape[1]
                self._index = faiss.IndexFlatIP(dim)  # Inner product (cosine with normalized vecs)
                self._index.add(self._embeddings.astype(np.float32))
                logger.info(f"Built FAISS index: {len(documents)} vectors, dim={dim}")
            except ImportError:
                logger.warning("FAISS not available, using numpy-based search")
                self._index = None
        else:
            self._index = None

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Search the index.

        Returns:
            List of (doc_id, score) tuples.
        """
        self._load_model()
        with self._lock:
            query_emb = self._model.encode([query], normalize_embeddings=True)

        if self._index is not None:
            import faiss
            scores, indices = self._index.search(query_emb.astype(np.float32), top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    results.append((self._doc_ids[idx], float(score)))
            return results
        else:
            # Numpy fallback
            scores = (query_emb @ self._embeddings.T)[0]
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [(self._doc_ids[i], float(scores[i])) for i in top_indices]

    def score_single(self, query: str, doc_id: str) -> float:
        """Compute query-document similarity for a single document.

        Returns cosine similarity (float). Returns 0.0 if doc_id not found.
        """
        if self._embeddings is None:
            return 0.0
        try:
            idx = self._doc_id_to_idx[doc_id]
        except (AttributeError, KeyError):
            # Build reverse index on first call
            if not hasattr(self, '_doc_id_to_idx'):
                self._doc_id_to_idx = {did: i for i, did in enumerate(self._doc_ids)}
            idx = self._doc_id_to_idx.get(doc_id)
            if idx is None:
                return 0.0

        self._load_model()
        with self._lock:
            query_emb = self._model.encode([query], normalize_embeddings=True)
        doc_emb = self._embeddings[idx:idx+1]
        return float((query_emb @ doc_emb.T)[0, 0])

    def score_many(self, query: str, doc_ids: List[str]) -> Dict[str, float]:
        """Score many documents against one query, encoding the query once.

        `score_single` re-encodes the query on every call, which is invisible while graph expansion
        proposes a handful of candidates but dominates runtime once it proposes hundreds. Same
        similarity, one encode.
        """
        if self._embeddings is None or not doc_ids:
            return {}
        if not hasattr(self, "_doc_id_to_idx"):
            self._doc_id_to_idx = {did: i for i, did in enumerate(self._doc_ids)}

        pairs = [(d, self._doc_id_to_idx[d]) for d in doc_ids if d in self._doc_id_to_idx]
        if not pairs:
            return {}

        self._load_model()
        with self._lock:
            query_emb = self._model.encode([query], normalize_embeddings=True)
        idxs = [i for _, i in pairs]
        sims = (query_emb @ self._embeddings[idxs].T)[0]
        return {d: float(s) for (d, _), s in zip(pairs, sims)}

    def search_batch(self, queries: List[str], top_k: int = 20) -> List[List[Tuple[str, float]]]:
        """Batch search."""
        self._load_model()
        query_embs = self._model.encode(queries, normalize_embeddings=True)

        all_results = []
        if self._index is not None:
            import faiss
            scores, indices = self._index.search(query_embs.astype(np.float32), top_k)
            for i in range(len(queries)):
                results = []
                for score, idx in zip(scores[i], indices[i]):
                    if idx >= 0:
                        results.append((self._doc_ids[idx], float(score)))
                all_results.append(results)
        else:
            all_scores = query_embs @ self._embeddings.T
            for i in range(len(queries)):
                top_indices = np.argsort(all_scores[i])[::-1][:top_k]
                all_results.append(
                    [(self._doc_ids[j], float(all_scores[i][j])) for j in top_indices]
                )
        return all_results

    def save(self, path: str):
        """Save embeddings and metadata."""
        np.savez(path, embeddings=self._embeddings, doc_ids=np.array(self._doc_ids))

    def load_embeddings(self, path: str):
        """Load pre-computed embeddings."""
        data = np.load(path, allow_pickle=True)
        self._embeddings = data["embeddings"]
        self._doc_ids = data["doc_ids"].tolist()
        try:
            import faiss
            dim = self._embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dim)
            self._index.add(self._embeddings.astype(np.float32))
        except ImportError:
            self._index = None
