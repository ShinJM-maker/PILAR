"""Stronger Reader A/B — O1-identical protocol.

Conditions:
  R0: current reader + final context (E011)
  R2: current reader + gold-page-only context (O1-identical)
  R1: stronger reader + final context (optional, needs --stronger-url)
  R3: stronger reader + gold-page-only (optional, needs --stronger-url)

Gold page identification uses EXACTLY the same logic as run_bottleneck_analysis.py:
  - normalize_answer text matching
  - sample["supporting_doc_ids"] top-level filter
  - build_oracle_context: raw text concatenation within token_budget

Results split by: all / text / visual query types.

Usage:
    PYTHONPATH=. python scripts/run_reader_ab.py
    PYTHONPATH=. python scripts/run_reader_ab.py --stronger-url http://localhost:8002/v1 --stronger-model Qwen/Qwen2.5-VL-72B
"""
import argparse
import json
import logging
import random
import re
import time
import asyncio
from collections import Counter
from copy import deepcopy
from pathlib import Path

from src.utils.io_utils import load_config, load_json, save_json
from src.evaluation.metrics import evaluate_qa, normalize_answer
from src.reader.prompt_templates import format_qa_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

DATASET = "m3docvqa"
LIMIT = 500
CONCURRENT = 16
EMBEDDER = "Qwen/Qwen3-Embedding-0.6B"

# Visual query keywords (same as visual routing script)
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


# --- O1-identical gold page functions (from run_bottleneck_analysis.py) ---

def find_gold_pages(sample, page_chunks, norm_cache):
    """O1-identical: top-level supporting_doc_ids filter."""
    gold = str(sample.get("answer", ""))
    if not gold:
        return []
    gold_norm = normalize_answer(gold)
    if not gold_norm:
        return []

    supporting_docs = sample.get("supporting_doc_ids", [])

    matches = []
    for cid, chunk in page_chunks.items():
        doc_id = chunk.get("doc_id", "")
        if supporting_docs and doc_id not in supporting_docs:
            continue
        text_norm = norm_cache.get(cid, "")
        if text_norm and gold_norm in text_norm:
            matches.append(cid)
    return matches


def estimate_tokens(text):
    return int(len(text.split()) * 1.3)


def build_oracle_context(page_chunks, chunk_ids, token_budget=4500):
    """O1-identical: raw text concatenation within budget."""
    parts = []
    total_tokens = 0
    for cid in chunk_ids:
        chunk = page_chunks.get(cid)
        if not chunk:
            continue
        text = chunk.get("text", "")
        if not text:
            continue
        est = estimate_tokens(text)
        if total_tokens + est > token_budget:
            break
        parts.append(text)
        total_tokens += est
    return "\n\n".join(parts)


def batch_vlm_call(base_url, model, prompts, concurrent=16, temperature=0.0, max_tokens=1024):
    """Batch async VLM calls to any OpenAI-compatible endpoint."""
    async def _batch():
        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url=base_url, api_key="dummy")
        sem = asyncio.Semaphore(concurrent)

        async def _call(prompt_text):
            if prompt_text is None:
                return ""
            async with sem:
                try:
                    messages = [{"role": "user", "content": prompt_text}]
                    response = await client.chat.completions.create(
                        model=model, messages=messages,
                        temperature=temperature, max_tokens=max_tokens,
                    )
                    from src.reader.llm_client import _strip_think_tags
                    return _strip_think_tags(response.choices[0].message.content)
                except Exception as e:
                    logger.error(f"VLM error: {e}")
                    return ""

        return await asyncio.gather(*[_call(p) for p in prompts])

    return asyncio.run(_batch())


def evaluate_split(all_data, pred_key, config):
    """Evaluate EM/ANLS split by all/text/visual."""
    results = {}
    for qt in ["all", "text", "visual"]:
        if qt == "all":
            subset = all_data
        else:
            subset = [d for d in all_data if d["query_type"] == qt]
        if not subset:
            continue

        preds = [d[pred_key] for d in subset]
        golds = [d["gold"] for d in subset]
        metrics = evaluate_qa(preds, golds,
                              anls_threshold=config["evaluation"]["anls_threshold"])
        metrics["n"] = len(subset)
        metrics["query_type"] = qt
        results[qt] = metrics
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stronger-url", default=None,
                        help="Base URL for stronger reader (e.g., http://localhost:8002/v1)")
    parser.add_argument("--stronger-model", default=None,
                        help="Model name for stronger reader")
    args = parser.parse_args()

    base_config = load_config("config/default.yaml")
    base_config["models"]["text_embedder"] = EMBEDDER

    from run_experiment import load_dataset, setup_backend, setup_agent

    samples = load_dataset(base_config, DATASET)
    random.seed(42)
    if LIMIT and LIMIT < len(samples):
        samples = random.sample(samples, LIMIT)
    logger.info(f"Loaded {len(samples)} samples")

    # Setup E011 config
    config = deepcopy(base_config)
    r = config["retrieval"]
    r["initial_multiplier"] = 2
    r["enable_local_bonus"] = True
    r["enable_expanded_gate"] = True
    r["adjacent_page_bonus"] = 0.04
    r["same_section_bonus"] = 0.00
    r["local_bonus_cap"] = 0.05
    r["expanded_dense_threshold"] = 0.35
    r["expanded_accept_epsilon"] = 0.03

    retriever, reader = setup_backend(config, "vega_kg", DATASET)
    agent = setup_agent(config, "naive_rag", retriever, reader)
    page_chunks = retriever.page_chunks
    vlm_client = reader.vlm

    # Current reader endpoint
    current_url = vlm_client.base_url
    current_model = vlm_client.model
    logger.info(f"Current reader: {current_model} @ {current_url}")

    output_dir = Path(base_config["paths"]["results"]) / "reader_ab"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build norm cache
    logger.info("Building norm cache...")
    t0 = time.time()
    norm_cache = {}
    for cid, chunk in page_chunks.items():
        t = chunk.get("text", "")
        if t:
            norm_cache[cid] = normalize_answer(t)
    logger.info(f"  Cache: {len(norm_cache)} entries ({time.time()-t0:.0f}s)")

    # ==========================================
    # PHASE 1: Retrieve + classify + find gold
    # ==========================================
    logger.info("=" * 60)
    logger.info("PHASE 1: Retrieve + classify + find gold pages")
    logger.info("=" * 60)

    all_data = []
    t0 = time.time()

    for idx, s in enumerate(samples):
        q = s["question"]
        gold = str(s["answer"]) if s["answer"] is not None else ""

        # Classify
        query_type = classify_query(q)

        # Retrieve (E011)
        evidence = agent.retriever.retrieve(q, top_k=agent.top_k)

        # Find gold pages (O1-identical)
        gold_page_ids = find_gold_pages(s, page_chunks, norm_cache)

        # Build contexts
        # Final context (for R0/R1)
        final_context = reader.serializer.serialize(evidence, None)

        # Gold-only context O1-identical (for R2/R3)
        if gold_page_ids:
            gold_context = build_oracle_context(page_chunks, gold_page_ids,
                                                token_budget=config["retrieval"].get("token_budget", 4500))
        else:
            gold_context = ""

        # Prompts
        prompt_final = format_qa_prompt(question=q, context=final_context)
        prompt_gold = format_qa_prompt(question=q, context=gold_context) if gold_context else None

        all_data.append({
            "idx": idx,
            "id": s.get("id", ""),
            "question": q,
            "gold": gold,
            "query_type": query_type,
            "gold_page_ids": gold_page_ids,
            "gold_in_index": len(gold_page_ids) > 0,
            "prompt_final": prompt_final,
            "prompt_gold": prompt_gold,
        })

        if (idx + 1) % 200 == 0:
            logger.info(f"  {idx+1}/{len(samples)}")

    t1 = time.time()
    logger.info(f"Phase 1 done: {t1-t0:.0f}s")

    # Stats
    type_counts = Counter(d["query_type"] for d in all_data)
    gold_found = sum(1 for d in all_data if d["gold_in_index"])
    logger.info(f"Query types: {dict(type_counts)}")
    logger.info(f"Gold found: {gold_found}/{len(all_data)}")

    # ==========================================
    # PHASE 2: R0 — current reader + final
    # ==========================================
    logger.info("\n" + "=" * 60)
    logger.info("R0: current reader + final context")
    logger.info("=" * 60)

    prompts_r0 = [d["prompt_final"] for d in all_data]
    t_start = time.time()
    answers_r0 = batch_vlm_call(current_url, current_model, prompts_r0, CONCURRENT)
    t_r0 = time.time() - t_start
    logger.info(f"  VLM done: {t_r0:.0f}s")

    for idx, d in enumerate(all_data):
        d["pred_R0"] = answers_r0[idx].strip() if answers_r0[idx] else ""

    results_r0 = evaluate_split(all_data, "pred_R0", config)
    for qt, m in results_r0.items():
        logger.info(f"  R0/{qt} (n={m['n']}): EM={m['EM']:.4f} ANLS={m['ANLS']:.4f}")
    save_json(results_r0, output_dir / "R0_metrics.json")

    # ==========================================
    # PHASE 3: R2 — current reader + gold only (O1-identical)
    # ==========================================
    logger.info("\n" + "=" * 60)
    logger.info("R2: current reader + gold-page-only (O1-identical)")
    logger.info("=" * 60)

    prompts_r2 = [d["prompt_gold"] for d in all_data]
    t_start = time.time()
    answers_r2 = batch_vlm_call(current_url, current_model, prompts_r2, CONCURRENT)
    t_r2 = time.time() - t_start
    logger.info(f"  VLM done: {t_r2:.0f}s")

    for idx, d in enumerate(all_data):
        d["pred_R2"] = answers_r2[idx].strip() if answers_r2[idx] else ""

    results_r2 = evaluate_split(all_data, "pred_R2", config)
    for qt, m in results_r2.items():
        logger.info(f"  R2/{qt} (n={m['n']}): EM={m['EM']:.4f} ANLS={m['ANLS']:.4f}")
    save_json(results_r2, output_dir / "R2_metrics.json")

    # Conditional: only samples with gold page
    logger.info("\nConditional (gold page exists):")
    has_gold = [d for d in all_data if d["gold_in_index"]]
    for qt in ["all", "text", "visual"]:
        if qt == "all":
            sub = has_gold
        else:
            sub = [d for d in has_gold if d["query_type"] == qt]
        if not sub:
            continue
        pr0 = [d["pred_R0"] for d in sub]
        pr2 = [d["pred_R2"] for d in sub]
        gs = [d["gold"] for d in sub]
        mr0 = evaluate_qa(pr0, gs, anls_threshold=config["evaluation"]["anls_threshold"])
        mr2 = evaluate_qa(pr2, gs, anls_threshold=config["evaluation"]["anls_threshold"])
        logger.info(f"  {qt} (n={len(sub)}): R0={mr0['EM']:.4f}, R2={mr2['EM']:.4f}, "
                     f"R2-R0={mr2['EM']-mr0['EM']:+.4f}")

    # ==========================================
    # PHASE 4: R1/R3 — stronger reader (optional)
    # ==========================================
    results_r1 = None
    results_r3 = None

    if args.stronger_url and args.stronger_model:
        logger.info("\n" + "=" * 60)
        logger.info(f"R1: stronger reader ({args.stronger_model}) + final context")
        logger.info("=" * 60)

        t_start = time.time()
        answers_r1 = batch_vlm_call(args.stronger_url, args.stronger_model,
                                     prompts_r0, CONCURRENT)
        t_r1 = time.time() - t_start
        logger.info(f"  VLM done: {t_r1:.0f}s")

        for idx, d in enumerate(all_data):
            d["pred_R1"] = answers_r1[idx].strip() if answers_r1[idx] else ""

        results_r1 = evaluate_split(all_data, "pred_R1", config)
        for qt, m in results_r1.items():
            logger.info(f"  R1/{qt} (n={m['n']}): EM={m['EM']:.4f} ANLS={m['ANLS']:.4f}")
        save_json(results_r1, output_dir / "R1_metrics.json")

        logger.info("\n" + "=" * 60)
        logger.info(f"R3: stronger reader ({args.stronger_model}) + gold-page-only")
        logger.info("=" * 60)

        t_start = time.time()
        answers_r3 = batch_vlm_call(args.stronger_url, args.stronger_model,
                                     prompts_r2, CONCURRENT)
        t_r3 = time.time() - t_start
        logger.info(f"  VLM done: {t_r3:.0f}s")

        for idx, d in enumerate(all_data):
            d["pred_R3"] = answers_r3[idx].strip() if answers_r3[idx] else ""

        results_r3 = evaluate_split(all_data, "pred_R3", config)
        for qt, m in results_r3.items():
            logger.info(f"  R3/{qt} (n={m['n']}): EM={m['EM']:.4f} ANLS={m['ANLS']:.4f}")
        save_json(results_r3, output_dir / "R3_metrics.json")
    else:
        logger.info("\nR1/R3 skipped — no --stronger-url/--stronger-model provided")

    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    print("\n" + "=" * 70)
    print("READER A/B RESULTS")
    print("=" * 70)

    conditions = [("R0", results_r0), ("R2", results_r2)]
    if results_r1:
        conditions.append(("R1", results_r1))
    if results_r3:
        conditions.append(("R3", results_r3))

    print(f"\n{'Cond':<6} {'Type':<8} {'n':>4} {'EM':>8} {'ANLS':>8} {'ROUGE-L':>8}")
    print("-" * 50)
    for cond_name, results in conditions:
        for qt in ["all", "text", "visual"]:
            if qt in results:
                m = results[qt]
                print(f"{cond_name:<6} {qt:<8} {m['n']:>4} "
                      f"{m['EM']:>8.4f} {m['ANLS']:>8.4f} {m['ROUGE-L']:>8.4f}")
        print()

    # Deltas
    print("DELTAS")
    print("-" * 50)
    for qt in ["all", "text", "visual"]:
        if qt in results_r0 and qt in results_r2:
            delta = results_r2[qt]["EM"] - results_r0[qt]["EM"]
            print(f"  R2-R0 ({qt}): {delta:+.4f}  (gold-only vs final, reader ceiling)")
    if results_r1:
        for qt in ["all", "text", "visual"]:
            if qt in results_r0 and qt in results_r1:
                delta = results_r1[qt]["EM"] - results_r0[qt]["EM"]
                print(f"  R1-R0 ({qt}): {delta:+.4f}  (stronger vs current, deploy gain)")
    if results_r3 and results_r2:
        for qt in ["all", "text", "visual"]:
            if qt in results_r2 and qt in results_r3:
                delta = results_r3[qt]["EM"] - results_r2[qt]["EM"]
                print(f"  R3-R2 ({qt}): {delta:+.4f}  (stronger vs current, pure reader gain)")

    # Decision rules
    if results_r1 and results_r3:
        print("\nDECISION RULES")
        print("-" * 50)
        for qt in ["all", "text", "visual"]:
            if all(qt in r for r in [results_r0, results_r1, results_r2, results_r3]):
                deploy_gain = results_r1[qt]["EM"] - results_r0[qt]["EM"]
                pure_gain = results_r3[qt]["EM"] - results_r2[qt]["EM"]
                if deploy_gain > 0.02 and pure_gain > 0.02:
                    verdict = "BOTH: reader upgrade high value"
                elif pure_gain > 0.02 and deploy_gain <= 0.02:
                    verdict = "RETRIEVAL MASKS: reader is better but retrieval hides it"
                elif deploy_gain > 0.02 and pure_gain <= 0.02:
                    verdict = "DEPLOY OK: reader helps even without perfect retrieval"
                else:
                    verdict = "LOW ROI: reader upgrade not impactful"
                print(f"  {qt}: deploy={deploy_gain:+.4f}, pure={pure_gain:+.4f} → {verdict}")

    # Save summary
    summary = {
        "R0": results_r0,
        "R2": results_r2,
        "current_reader": current_model,
        "gold_found": gold_found,
        "query_types": dict(type_counts),
    }
    if results_r1:
        summary["R1"] = results_r1
        summary["stronger_reader"] = args.stronger_model
    if results_r3:
        summary["R3"] = results_r3
    save_json(summary, output_dir / "reader_ab_summary.json")

    # Save per-sample predictions
    per_sample = []
    for d in all_data:
        entry = {
            "idx": d["idx"], "id": d["id"],
            "question": d["question"], "gold": d["gold"],
            "query_type": d["query_type"],
            "gold_in_index": d["gold_in_index"],
            "pred_R0": d.get("pred_R0", ""),
            "pred_R2": d.get("pred_R2", ""),
        }
        if "pred_R1" in d:
            entry["pred_R1"] = d["pred_R1"]
        if "pred_R3" in d:
            entry["pred_R3"] = d["pred_R3"]
        per_sample.append(entry)
    save_json(per_sample, output_dir / "predictions.json")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
