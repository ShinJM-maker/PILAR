"""Grounding diagnostics: Support Coverage (SF) and Scope Coverage (SC)."""
from typing import List, Dict, Set, Callable


def support_coverage(retrieved_units: List[Dict],
                     gold_atoms: List[Dict],
                     align_fn: Callable = None) -> float:
    """Support Coverage (SF): fraction of gold evidence atoms covered by retrieved units.

    SF(q) = (1/|G_q|) * sum(covered(g, q)) for g in G_q

    Args:
        retrieved_units: List of retrieved evidence unit dicts.
        gold_atoms: List of gold evidence atom dicts with at least 'id' and 'content'.
        align_fn: Optional alignment function (retrieved, gold) -> bool.
                  Default: text overlap based.

    Returns:
        SF score in [0, 1].
    """
    if not gold_atoms:
        return 1.0

    if align_fn is None:
        align_fn = _default_align

    covered = 0
    for gold in gold_atoms:
        for retrieved in retrieved_units:
            if align_fn(retrieved, gold):
                covered += 1
                break

    return covered / len(gold_atoms)


def scope_coverage(retrieved_units: List[Dict],
                   gold_atoms: List[Dict],
                   align_fn: Callable = None) -> float:
    """Scope Coverage (SC): fraction of gold atoms that are both covered AND scope-covered.

    SC(q) = (1/|G_q|) * sum(scoped(g, q)) for g in G_q
    where scoped(g, q) requires both alignment AND scope carrier presence.

    Returns:
        SC score in [0, 1]. SC <= SF by construction.
    """
    if not gold_atoms:
        return 1.0

    if align_fn is None:
        align_fn = _default_align

    scoped = 0
    for gold in gold_atoms:
        gold_scope = gold.get("scope_carriers", [])
        for retrieved in retrieved_units:
            if align_fn(retrieved, gold):
                # Check if scope carrier is present
                if not gold_scope:
                    # No scope requirement -> auto-scoped
                    scoped += 1
                    break
                ret_scope = retrieved.get("section_path", "") or retrieved.get("governing_header", "")
                for scope_carrier in gold_scope:
                    if scope_carrier.lower() in ret_scope.lower():
                        scoped += 1
                        break
                else:
                    continue
                break

    return scoped / len(gold_atoms)


def _default_align(retrieved: Dict, gold: Dict) -> bool:
    """Default alignment: text overlap heuristic."""
    ret_content = (retrieved.get("content", "") or "").lower()
    gold_content = (gold.get("content", "") or "").lower()

    if not ret_content or not gold_content:
        return False

    # Check if gold content words overlap significantly with retrieved
    gold_words = set(gold_content.split())
    ret_words = set(ret_content.split())

    if not gold_words:
        return False

    overlap = len(gold_words & ret_words) / len(gold_words)
    return overlap >= 0.5


def evaluate_grounding(all_retrieved: List[List[Dict]],
                       all_gold: List[List[Dict]],
                       align_fn: Callable = None) -> Dict:
    """Evaluate SF and SC over a dataset.

    Returns:
        Dict with "SF" and "SC" averaged scores.
    """
    n = len(all_retrieved)
    assert n == len(all_gold)

    sf_scores = []
    sc_scores = []
    for retrieved, gold in zip(all_retrieved, all_gold):
        sf_scores.append(support_coverage(retrieved, gold, align_fn))
        sc_scores.append(scope_coverage(retrieved, gold, align_fn))

    return {
        "SF": sum(sf_scores) / n if n > 0 else 0.0,
        "SC": sum(sc_scores) / n if n > 0 else 0.0,
        "n": n,
    }
