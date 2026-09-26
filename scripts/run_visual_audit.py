"""Visual Answer-Source Audit — automated classification of visual query failures.

Classifies each visual query (n=140) into failure categories:
  - answer_present_verbatim_in_ocr: answer text appears verbatim in gold page OCR
  - answer_present_paraphrased_in_ocr: normalized answer matches in gold page text
  - answer_absent_from_ocr: answer not found in any gold page text
  - gold_page_retrieval_miss: gold page not found in index at all
  - reader_failed_despite_answer_present: gold text contains answer but reader got it wrong

Also checks: retrieval hit/miss, reader correct/wrong per condition.

Usage:
    PYTHONPATH=. python scripts/run_visual_audit.py
"""
import json
import logging
import random
import re
from collections import Counter
from pathlib import Path

from src.utils.io_utils import load_config, load_json, save_json
from src.evaluation.metrics import evaluate_qa, normalize_answer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DATASET = "m3docvqa"
LIMIT = 500


VISUAL_KEYWORDS = [
    "wearing", "worn", "wear", "dressed",
    "color", "colour",
    "facial hair", "beard", "moustache", "mustache",
    "what is shown", "what does it show", "what can be seen",
    "how many people", "how many persons",
    "what type of", "what kind of",
    "logo", "sign", "icon", "picture", "image", "photo", "photograph",
    "chart", "graph", "diagram", "figure",
    "looks like", "look like", "appearance",
    "holding", "held", "carries", "carrying",
    "sitting", "standing", "posing", "pointing",
    "background", "foreground",
    "left", "right", "top", "bottom",
    "wrist", "hand", "eye", "head", "hair", "face",
]
VISUAL_PATTERNS = [
    r"what is \w+ wearing",
    r"what is \w+ holding",
    r"what is on \w+ (head|face|wrist|hand|eyes|neck)",
    r"what position",
    r"what pose",
    r"what gesture",
    r"who is (taller|shorter|bigger|smaller)",
    r"how does .+ look",
]


def classify_query(question: str) -> str:
    q_lower = question.lower()
    cues = 0
    for kw in VISUAL_KEYWORDS:
        if kw in q_lower:
            cues += 1
    for pat in VISUAL_PATTERNS:
        if re.search(pat, q_lower):
            cues += 1
    words = question.split()
    if len(words) <= 8 and any(w.lower().startswith("what") for w in words[:2]):
        if any(w[0].isupper() for w in words[2:] if len(w) > 1):
            cues += 1
    return "visual" if cues >= 1 else "text"


def check_answer_in_text(answer: str, text: str) -> dict:
    """Check various levels of answer presence in text."""
    if not answer or not text:
        return {"verbatim": False, "normalized": False, "partial": False}

    answer_lower = answer.lower().strip()
    text_lower = text.lower()

    # Verbatim match
    verbatim = answer_lower in text_lower

    # Normalized match
    ans_norm = normalize_answer(answer)
    text_norm = normalize_answer(text)
    normalized = ans_norm in text_norm if (ans_norm and text_norm) else False

    # Partial match (for multi-word answers, check if most words appear)
    ans_words = set(ans_norm.split()) if ans_norm else set()
    text_words = set(text_norm.split()) if text_norm else set()
    if len(ans_words) > 0:
        overlap = len(ans_words & text_words) / len(ans_words)
        partial = overlap >= 0.5
    else:
        partial = False

    return {"verbatim": verbatim, "normalized": normalized, "partial": partial,
            "overlap_ratio": overlap if ans_words else 0.0}


def classify_failure(sample, page_chunks, norm_cache, reader_ab_pred):
    """Classify a visual sample into failure categories."""
    gold = str(sample.get("answer", ""))
    question = sample["question"]

    # Get gold page info
    supporting_docs = sample.get("supporting_doc_ids", [])
    gold_norm = normalize_answer(gold) if gold else ""

    # Find gold pages
    gold_page_ids = []
    if gold_norm:
        for cid, chunk in page_chunks.items():
            doc_id = chunk.get("doc_id", "")
            if supporting_docs and doc_id not in supporting_docs:
                continue
            text_norm = norm_cache.get(cid, "")
            if text_norm and gold_norm in text_norm:
                gold_page_ids.append(cid)

    has_gold_in_index = len(gold_page_ids) > 0

    # Check answer presence in gold page text
    gold_texts = []
    for cid in gold_page_ids:
        chunk = page_chunks.get(cid)
        if chunk:
            gold_texts.append(chunk.get("text", ""))
    combined_gold_text = "\n".join(gold_texts)

    answer_check = check_answer_in_text(gold, combined_gold_text) if combined_gold_text else {
        "verbatim": False, "normalized": False, "partial": False, "overlap_ratio": 0.0
    }

    # Reader predictions
    pred_r0 = reader_ab_pred.get("pred_R0", "")
    pred_r2 = reader_ab_pred.get("pred_R2", "")

    # Check correctness
    r0_correct = normalize_answer(pred_r0) == gold_norm if gold_norm else False
    r2_correct = normalize_answer(pred_r2) == gold_norm if gold_norm else False

    # Classify
    if not has_gold_in_index:
        primary_label = "gold_page_retrieval_miss"
    elif answer_check["normalized"]:
        if r2_correct:
            primary_label = "answer_present_reader_correct"
        else:
            primary_label = "reader_failed_despite_answer_present"
    elif answer_check["partial"]:
        primary_label = "answer_present_paraphrased_in_ocr"
    else:
        primary_label = "answer_absent_from_ocr"

    # Visual subcategory based on question keywords
    q_lower = question.lower()
    visual_subcategory = "other"
    if any(kw in q_lower for kw in ["wearing", "worn", "wear", "dressed"]):
        visual_subcategory = "clothing/appearance"
    elif any(kw in q_lower for kw in ["color", "colour"]):
        visual_subcategory = "color"
    elif any(kw in q_lower for kw in ["facial hair", "beard", "moustache", "mustache"]):
        visual_subcategory = "facial_feature"
    elif any(kw in q_lower for kw in ["logo", "sign", "icon"]):
        visual_subcategory = "logo/sign"
    elif any(kw in q_lower for kw in ["chart", "graph", "diagram", "figure", "table"]):
        visual_subcategory = "chart/table"
    elif any(kw in q_lower for kw in ["picture", "image", "photo", "photograph"]):
        visual_subcategory = "photo_reference"
    elif any(kw in q_lower for kw in ["holding", "sitting", "standing", "posing", "pointing"]):
        visual_subcategory = "pose/action"
    elif any(kw in q_lower for kw in ["background", "foreground", "left", "right", "top", "bottom"]):
        visual_subcategory = "spatial"
    elif any(kw in q_lower for kw in ["how many"]):
        visual_subcategory = "counting"
    elif any(kw in q_lower for kw in ["what type", "what kind"]):
        visual_subcategory = "type_classification"
    elif any(kw in q_lower for kw in ["head", "hair", "face", "eye", "hand", "wrist"]):
        visual_subcategory = "body_part"

    return {
        "id": sample.get("id", ""),
        "question": question,
        "gold": gold,
        "gold_page_ids": gold_page_ids,
        "has_gold_in_index": has_gold_in_index,
        "n_gold_pages": len(gold_page_ids),
        "answer_verbatim": answer_check["verbatim"],
        "answer_normalized": answer_check["normalized"],
        "answer_partial": answer_check["partial"],
        "answer_overlap_ratio": answer_check.get("overlap_ratio", 0.0),
        "primary_label": primary_label,
        "visual_subcategory": visual_subcategory,
        "pred_R0": pred_r0,
        "pred_R2": pred_r2,
        "r0_correct": r0_correct,
        "r2_correct": r2_correct,
        "gold_text_snippet": combined_gold_text[:300] if combined_gold_text else "",
    }


def main():
    config = load_config("config/default.yaml")

    from run_experiment import load_dataset

    samples = load_dataset(config, DATASET)
    random.seed(42)
    if LIMIT and LIMIT < len(samples):
        samples = random.sample(samples, LIMIT)
    logger.info(f"Loaded {len(samples)} samples")

    # Load page chunks
    chunks_path = Path(config["paths"]["indices"]) / f"{DATASET}_chunks.json"
    if not chunks_path.exists():
        chunks_path = Path(config["paths"]["preprocessed"]) / f"{DATASET}_page_chunks.json"
    page_chunks = load_json(chunks_path)
    logger.info(f"Loaded {len(page_chunks)} page chunks")

    # Build norm cache
    norm_cache = {}
    for cid, chunk in page_chunks.items():
        t = chunk.get("text", "")
        if t:
            norm_cache[cid] = normalize_answer(t)

    # Load reader_ab predictions
    reader_ab_path = Path(config["paths"]["results"]) / "reader_ab" / "predictions.json"
    reader_ab_preds = load_json(reader_ab_path)
    ab_by_id = {p["id"]: p for p in reader_ab_preds}

    # Filter visual queries
    visual_samples = [(s, ab_by_id.get(s.get("id", ""), {}))
                      for s in samples if classify_query(s["question"]) == "visual"]
    logger.info(f"Visual samples: {len(visual_samples)}")

    # Classify each
    results = []
    for s, ab_pred in visual_samples:
        r = classify_failure(s, page_chunks, norm_cache, ab_pred)
        results.append(r)

    # Summary statistics
    logger.info("\n" + "=" * 70)
    logger.info("VISUAL ANSWER-SOURCE AUDIT")
    logger.info("=" * 70)

    n = len(results)

    # Primary label distribution
    label_counts = Counter(r["primary_label"] for r in results)
    logger.info(f"\n=== Primary Classification (n={n}) ===")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = count / n * 100
        logger.info(f"  {label}: {count} ({pct:.1f}%)")

    # Answer presence breakdown
    has_gold = [r for r in results if r["has_gold_in_index"]]
    no_gold = [r for r in results if not r["has_gold_in_index"]]
    logger.info(f"\n=== Gold Page Availability ===")
    logger.info(f"  Gold in index: {len(has_gold)} ({len(has_gold)/n*100:.1f}%)")
    logger.info(f"  No gold: {len(no_gold)} ({len(no_gold)/n*100:.1f}%)")

    if has_gold:
        verbatim = sum(1 for r in has_gold if r["answer_verbatim"])
        normalized = sum(1 for r in has_gold if r["answer_normalized"])
        partial = sum(1 for r in has_gold if r["answer_partial"])
        absent = sum(1 for r in has_gold if not r["answer_partial"])
        logger.info(f"\n=== Answer in Gold Page Text (of {len(has_gold)} with gold) ===")
        logger.info(f"  Verbatim match: {verbatim} ({verbatim/len(has_gold)*100:.1f}%)")
        logger.info(f"  Normalized match: {normalized} ({normalized/len(has_gold)*100:.1f}%)")
        logger.info(f"  Partial match (≥50% words): {partial} ({partial/len(has_gold)*100:.1f}%)")
        logger.info(f"  Absent from OCR: {absent} ({absent/len(has_gold)*100:.1f}%)")

    # Reader performance by label
    logger.info(f"\n=== Reader Performance by Label ===")
    for label in sorted(label_counts.keys()):
        subset = [r for r in results if r["primary_label"] == label]
        r0_correct = sum(1 for r in subset if r["r0_correct"])
        r2_correct = sum(1 for r in subset if r["r2_correct"])
        logger.info(f"  {label} (n={len(subset)}):")
        logger.info(f"    R0 (current+final): {r0_correct}/{len(subset)} = {r0_correct/len(subset)*100:.1f}%")
        logger.info(f"    R2 (current+gold):  {r2_correct}/{len(subset)} = {r2_correct/len(subset)*100:.1f}%")

    # Visual subcategory breakdown
    subcat_counts = Counter(r["visual_subcategory"] for r in results)
    logger.info(f"\n=== Visual Subcategory Distribution ===")
    for subcat, count in sorted(subcat_counts.items(), key=lambda x: -x[1]):
        subset = [r for r in results if r["visual_subcategory"] == subcat]
        r0_correct = sum(1 for r in subset if r["r0_correct"])
        r2_correct = sum(1 for r in subset if r["r2_correct"])
        logger.info(f"  {subcat}: {count} ({count/n*100:.1f}%), "
                     f"R0={r0_correct}/{count} ({r0_correct/count*100:.0f}%), "
                     f"R2={r2_correct}/{count} ({r2_correct/count*100:.0f}%)")

    # Key insight: reader failure vs retrieval failure
    logger.info(f"\n=== Failure Decomposition ===")
    # Cases where answer IS in gold text but reader still wrong
    reader_failures = [r for r in results if r["answer_normalized"] and not r["r2_correct"]]
    # Cases where answer is NOT in any gold text
    ocr_absent = [r for r in results if r["has_gold_in_index"] and not r["answer_normalized"]]
    # Cases with no gold in index
    retrieval_miss = no_gold

    logger.info(f"  1. No gold page in index: {len(retrieval_miss)} ({len(retrieval_miss)/n*100:.1f}%)")
    logger.info(f"  2. Gold page found, but answer absent from OCR: {len(ocr_absent)} ({len(ocr_absent)/n*100:.1f}%)")
    logger.info(f"  3. Answer in OCR, reader failed (R2 wrong): {len(reader_failures)} ({len(reader_failures)/n*100:.1f}%)")
    answer_present_reader_ok = sum(1 for r in results if r["answer_normalized"] and r["r2_correct"])
    logger.info(f"  4. Answer in OCR, reader correct (R2 right): {answer_present_reader_ok} ({answer_present_reader_ok/n*100:.1f}%)")

    # The key question: what fraction of visual failures are solvable by better text retrieval vs need image?
    logger.info(f"\n=== Actionability Analysis ===")
    # Solvable by better text retrieval: answer IS in gold page text, just need to find the page
    solvable_retrieval = [r for r in results if r["answer_normalized"] and not r["r0_correct"]]
    # Needs image understanding: answer NOT in OCR text
    needs_image = [r for r in results if r["has_gold_in_index"] and not r["answer_partial"]]
    needs_image_or_missing = [r for r in results if not r["has_gold_in_index"] or not r["answer_partial"]]

    total_wrong_r0 = sum(1 for r in results if not r["r0_correct"])
    logger.info(f"  Total wrong (R0): {total_wrong_r0}")
    logger.info(f"  Of which:")
    logger.info(f"    Answer in OCR, solvable by retrieval+reader: {len(solvable_retrieval)}")
    logger.info(f"    Answer NOT in OCR (needs image or different modality): {len(needs_image)}")
    logger.info(f"    No gold page in index: {len(retrieval_miss)}")

    # Example cases
    logger.info(f"\n=== Example: reader_failed_despite_answer_present (up to 10) ===")
    for r in reader_failures[:10]:
        logger.info(f"  Q: {r['question'][:80]}")
        logger.info(f"    Gold: {r['gold']}, R2 pred: {r['pred_R2']}")
        logger.info(f"    Text snippet: {r['gold_text_snippet'][:120]}...")
        logger.info("")

    logger.info(f"\n=== Example: answer_absent_from_ocr (up to 10) ===")
    for r in ocr_absent[:10]:
        logger.info(f"  Q: {r['question'][:80]}")
        logger.info(f"    Gold: {r['gold']}")
        logger.info(f"    Text snippet: {r['gold_text_snippet'][:120]}...")
        logger.info("")

    # Print summary table
    print("\n" + "=" * 70)
    print("VISUAL ANSWER-SOURCE AUDIT SUMMARY")
    print("=" * 70)

    print(f"\nTotal visual queries: {n}")
    print(f"\n{'Category':<45} {'Count':>5} {'%':>6}")
    print("-" * 60)
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:<43} {count:>5} {count/n*100:>5.1f}%")

    print(f"\n{'Failure Decomposition':<45} {'Count':>5} {'%':>6}")
    print("-" * 60)
    print(f"  {'No gold in index':<43} {len(retrieval_miss):>5} {len(retrieval_miss)/n*100:>5.1f}%")
    print(f"  {'Gold found, answer absent from OCR':<43} {len(ocr_absent):>5} {len(ocr_absent)/n*100:>5.1f}%")
    print(f"  {'Answer in OCR, reader failed (R2)':<43} {len(reader_failures):>5} {len(reader_failures)/n*100:>5.1f}%")
    print(f"  {'Answer in OCR, reader correct (R2)':<43} {answer_present_reader_ok:>5} {answer_present_reader_ok/n*100:>5.1f}%")

    # Implications
    print("\nIMPLICATIONS")
    print("-" * 60)
    ans_in_ocr = sum(1 for r in results if r["answer_normalized"])
    ans_not_in_ocr = sum(1 for r in results if r["has_gold_in_index"] and not r["answer_normalized"])
    print(f"  Answer recoverable from OCR text: {ans_in_ocr}/{n} ({ans_in_ocr/n*100:.1f}%)")
    print(f"  Answer NOT in OCR (needs image): {ans_not_in_ocr}/{n} ({ans_not_in_ocr/n*100:.1f}%)")
    print(f"  No gold page at all: {len(retrieval_miss)}/{n} ({len(retrieval_miss)/n*100:.1f}%)")

    if ans_in_ocr > 0:
        reader_success_rate = answer_present_reader_ok / ans_in_ocr * 100
        print(f"\n  When answer IS in OCR:")
        print(f"    Reader success (R2): {answer_present_reader_ok}/{ans_in_ocr} = {reader_success_rate:.1f}%")
        print(f"    Reader failure (R2): {len(reader_failures)}/{ans_in_ocr} = {len(reader_failures)/ans_in_ocr*100:.1f}%")
        print(f"    → Stronger reader could recover up to {len(reader_failures)} more correct answers")

    if ans_not_in_ocr > 0:
        print(f"\n  When answer is NOT in OCR:")
        print(f"    These {ans_not_in_ocr} cases need either:")
        print(f"    - Better OCR / text extraction from images")
        print(f"    - Image-derived text descriptors for retrieval")
        print(f"    - Stronger VLM that can read images better")

    # Save
    output_dir = Path(config["paths"]["results"]) / "visual_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(results, output_dir / "audit_details.json")
    save_json({
        "n": n,
        "primary_labels": dict(label_counts),
        "subcategories": dict(subcat_counts),
        "gold_in_index": len(has_gold),
        "no_gold": len(no_gold),
        "answer_verbatim": sum(1 for r in results if r["answer_verbatim"]),
        "answer_normalized": sum(1 for r in results if r["answer_normalized"]),
        "answer_partial": sum(1 for r in results if r["answer_partial"]),
        "answer_absent_from_ocr": len(ocr_absent),
        "reader_failures": len(reader_failures),
        "retrieval_miss": len(retrieval_miss),
    }, output_dir / "audit_summary.json")
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
