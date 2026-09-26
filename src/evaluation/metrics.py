"""QA evaluation metrics: EM, ANLS, ROUGE-L, METEOR."""
import re
import string
from typing import List


def normalize_answer(text) -> str:
    """Normalize answer text for evaluation."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def exact_match(prediction: str, gold: str) -> float:
    """Exact match after normalization."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def anls_score(prediction: str, gold: str, threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity (ANLS).

    As used in MP-DocVQA: threshold-based implementation.
    """
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)

    if not gold_norm:
        return 1.0 if not pred_norm else 0.0

    dist = levenshtein_distance(pred_norm, gold_norm)
    max_len = max(len(pred_norm), len(gold_norm))

    if max_len == 0:
        return 1.0

    nls = dist / max_len
    score = 1.0 - nls if nls < threshold else 0.0
    return score


def rouge_l_score(prediction: str, gold: str) -> float:
    """ROUGE-L F1 score."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(gold, prediction)
        return scores["rougeL"].fmeasure
    except ImportError:
        # Fallback: simple LCS-based computation
        return _lcs_rouge_l(prediction, gold)


def _lcs_rouge_l(prediction: str, gold: str) -> float:
    """Simple ROUGE-L implementation via LCS."""
    pred_words = normalize_answer(prediction).split()
    gold_words = normalize_answer(gold).split()

    if not pred_words or not gold_words:
        return 0.0

    # LCS
    m, n = len(pred_words), len(gold_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_words[i-1] == gold_words[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs_len = dp[m][n]
    precision = lcs_len / m if m > 0 else 0
    recall = lcs_len / n if n > 0 else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def meteor_score_single(prediction, gold) -> float:
    """METEOR score for a single prediction-gold pair."""
    if not isinstance(prediction, str):
        prediction = str(prediction)
    if not isinstance(gold, str):
        gold = str(gold)
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score as nltk_meteor
        # Ensure wordnet is available
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt_tab', quiet=True)

        return nltk_meteor([gold.split()], prediction.split())
    except (ImportError, LookupError):
        return 0.0


def evaluate_qa(predictions: List[str], golds: List[str],
                anls_threshold: float = 0.5) -> dict:
    """Evaluate a list of predictions against gold answers.

    Returns:
        Dict with metric names as keys and average scores as values.
    """
    n = len(predictions)
    assert n == len(golds), f"Mismatched lengths: {n} vs {len(golds)}"

    em_scores = []
    anls_scores = []
    rouge_scores = []
    meteor_scores = []

    for pred, gold in zip(predictions, golds):
        em_scores.append(exact_match(pred, gold))
        anls_scores.append(anls_score(pred, gold, threshold=anls_threshold))
        rouge_scores.append(rouge_l_score(pred, gold))
        meteor_scores.append(meteor_score_single(pred, gold))

    return {
        "EM": sum(em_scores) / n if n > 0 else 0.0,
        "ANLS": sum(anls_scores) / n if n > 0 else 0.0,
        "ROUGE-L": sum(rouge_scores) / n if n > 0 else 0.0,
        "METEOR": sum(meteor_scores) / n if n > 0 else 0.0,
        "n": n,
    }
