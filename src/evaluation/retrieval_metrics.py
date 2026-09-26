"""Retrieval evaluation metrics: Recall@K, MRR@K, Hit@K, All@K."""
from typing import List, Set, Dict


def recall_at_k(retrieved: List[str], gold: Set[str], k: int) -> float:
    """Recall@K: proportion of gold docs in top-K retrieved."""
    if not gold:
        return 0.0
    top_k = set(retrieved[:k])
    return len(gold & top_k) / len(gold)


def mrr_at_k(retrieved: List[str], gold: Set[str], k: int) -> float:
    """MRR@K: reciprocal rank of first relevant doc in top-K."""
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in gold:
            return 1.0 / (i + 1)
    return 0.0


def hit_at_k(retrieved: List[str], gold: Set[str], k: int) -> float:
    """Hit@K: 1 if any gold doc in top-K, else 0."""
    top_k = set(retrieved[:k])
    return 1.0 if gold & top_k else 0.0


def all_at_k(retrieved: List[str], gold: Set[str], k: int) -> float:
    """All@K: 1 if ALL gold docs are in top-K, else 0."""
    if not gold:
        return 1.0
    top_k = set(retrieved[:k])
    return 1.0 if gold.issubset(top_k) else 0.0


def evaluate_retrieval(all_retrieved: List[List[str]],
                       all_gold: List[Set[str]],
                       k_range: List[int] = None) -> Dict:
    """Evaluate retrieval over a dataset.

    Args:
        all_retrieved: List of retrieved doc ID lists per query.
        all_gold: List of gold doc ID sets per query.
        k_range: Range of K values to evaluate (default 1..10).

    Returns:
        Dict with averaged metrics over k_range.
    """
    if k_range is None:
        k_range = list(range(1, 11))

    n = len(all_retrieved)
    assert n == len(all_gold)

    results = {}
    for k in k_range:
        recall_scores = [recall_at_k(r, g, k) for r, g in zip(all_retrieved, all_gold)]
        mrr_scores = [mrr_at_k(r, g, k) for r, g in zip(all_retrieved, all_gold)]
        hit_scores = [hit_at_k(r, g, k) for r, g in zip(all_retrieved, all_gold)]
        all_scores = [all_at_k(r, g, k) for r, g in zip(all_retrieved, all_gold)]

        results[f"Recall@{k}"] = sum(recall_scores) / n if n > 0 else 0.0
        results[f"MRR@{k}"] = sum(mrr_scores) / n if n > 0 else 0.0
        results[f"Hit@{k}"] = sum(hit_scores) / n if n > 0 else 0.0
        results[f"All@{k}"] = sum(all_scores) / n if n > 0 else 0.0

    # Average across K range (Avg@1-10)
    for metric in ["Recall", "MRR", "Hit", "All"]:
        values = [results[f"{metric}@{k}"] for k in k_range]
        results[f"Avg_{metric}@1-{max(k_range)}"] = sum(values) / len(values) if values else 0.0

    return results
