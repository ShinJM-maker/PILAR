"""Closed predicate inventory and normalization."""
import numpy as np
from typing import List, Dict, Tuple

PREDICATE_INVENTORY = [
    "outperforms", "performs_worse_than", "equals", "contains", "part_of",
    "has_property", "has_value", "increases", "decreases", "located_in",
    "authored_by", "published_in", "affiliated_with", "founded_in",
    "occurred_on", "started_in", "ended_in", "member_of", "instance_of",
    "subclass_of", "related_to", "causes", "prevents", "requires",
    "produces", "uses", "connects_to", "compared_with", "measured_at",
    "ranked_at", "achieved", "won", "lost", "born_in", "died_in",
    "married_to", "child_of", "parent_of", "sibling_of", "succeeded_by",
    "preceded_by", "capital_of", "currency_of", "language_of",
    "population_of", "area_of", "manufactures", "designed_by",
    "operates", "serves", "employs", "revenue_of",
]

# Predicate clusters for normalization (synonyms map to canonical form)
PREDICATE_ALIASES = {
    "better_than": "outperforms",
    "worse_than": "performs_worse_than",
    "same_as": "equals",
    "includes": "contains",
    "has": "has_property",
    "is_part_of": "part_of",
    "written_by": "authored_by",
    "created_by": "authored_by",
    "located_at": "located_in",
    "based_in": "located_in",
    "is_a": "instance_of",
    "type_of": "subclass_of",
    "leads_to": "causes",
    "results_in": "causes",
    "used_by": "uses",
    "connected_to": "connects_to",
    "born_on": "born_in",
    "died_on": "died_in",
    "scored": "achieved",
    "surpasses": "outperforms",
    "exceeds": "outperforms",
}


def normalize_predicate(predicate: str) -> str:
    """Normalize a predicate string to canonical form."""
    pred = predicate.lower().strip().replace(" ", "_")
    if pred in PREDICATE_ALIASES:
        return PREDICATE_ALIASES[pred]
    if pred in PREDICATE_INVENTORY:
        return pred
    return pred  # keep as-is if not in inventory (will be post-filtered)


def is_valid_predicate(predicate: str) -> bool:
    """Check if a predicate is in the inventory or its aliases."""
    pred = predicate.lower().strip().replace(" ", "_")
    return pred in PREDICATE_INVENTORY or pred in PREDICATE_ALIASES


class PredicateNormalizer:
    """Post-hoc predicate normalization using embedding similarity."""

    def __init__(self, merge_threshold: float = 0.9):
        self.merge_threshold = merge_threshold
        self._embeddings = None
        self._model = None

    def _get_embeddings(self, predicates: List[str]) -> np.ndarray:
        """Get embeddings for predicate strings."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        # Convert underscore predicates to natural language
        texts = [p.replace("_", " ") for p in predicates]
        return self._model.encode(texts, normalize_embeddings=True)

    def cluster_predicates(self, predicates: List[str]) -> Dict[str, str]:
        """Cluster predicates by embedding similarity and map to canonical forms.

        Returns:
            Dict mapping each predicate to its canonical (cluster representative).
        """
        if not predicates:
            return {}

        unique_preds = list(set(predicates))
        embeddings = self._get_embeddings(unique_preds)

        # Compute pairwise cosine similarity
        sim_matrix = embeddings @ embeddings.T

        # Union-Find clustering
        parent = list(range(len(unique_preds)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                # Prefer inventory predicates as cluster heads
                if unique_preds[py] in PREDICATE_INVENTORY:
                    parent[px] = py
                else:
                    parent[py] = px

        for i in range(len(unique_preds)):
            for j in range(i + 1, len(unique_preds)):
                if sim_matrix[i][j] > self.merge_threshold:
                    union(i, j)

        # Build mapping
        mapping = {}
        for i, pred in enumerate(unique_preds):
            root_idx = find(i)
            canonical = unique_preds[root_idx]
            # Prefer inventory form
            canonical = normalize_predicate(canonical)
            mapping[pred] = canonical

        return mapping
