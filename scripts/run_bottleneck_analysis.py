"""Bottleneck analysis: flip analysis + oracle ceiling + stronger reader A/B.

Uses a common 500-sample subset (seed=42) across all experiments.

Phase 1: Generate retrieval metadata (selected pages, gold page identification)
Phase 2: Oracle ceiling experiments (O0-O4)
Phase 3: Flip analysis CSV generation

Usage:
    PYTHONPATH=. python scripts/run_bottleneck_analysis.py
"""
import json
import logging
import time
import random
import asyncio
import csv
import re
from pathlib import Path
from copy import deepcopy
from collections import defaultdict

from src.utils.io_utils import load_config, load_json, save_json
from src.evaluation.metrics import evaluate_qa, normalize_answer, exact_match

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

DATASET = "m3docvqa"
AGENT = "naive_rag"
LIMIT = 500
CONCURRENT = 16
EMBEDDER = "Qwen/Qwen3-Embedding-0.6B"


def make_final_config(base_config):
    """Apply E011 optimal settings."""
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
    return config


def make_e000_config(base_config):
    """E000: M=0, L=0, G=0."""
    config = deepcopy(base_config)
    r = config["retrieval"]
    r["initial_multiplier"] = 2
    r["enable_local_bonus"] = False
    r["enable_expanded_gate"] = False
    return config


def find_gold_pages(sample, page_chunks):
    """Find pages that contain the gold answer text.

    Returns list of (chunk_id, page_num, doc_id) for pages containing the answer.
    """
    gold = str(sample.get("answer", ""))
    if not gold:
        return []

    gold_norm = normalize_answer(gold)
    supporting_docs = sample.get("supporting_doc_ids", [])

    matches = []
    for cid, chunk in page_chunks.items():
        doc_id = chunk.get("doc_id", "")
        if supporting_docs and doc_id not in supporting_docs:
            continue

        text_norm = normalize_answer(chunk.get("text", ""))
        if gold_norm and gold_norm in text_norm:
            matches.append((cid, chunk.get("page", 0), doc_id))

    return matches


def get_adjacent_pages(chunk_id, page_chunks):
    """Get ±1 adjacent page chunk IDs."""
    try:
        prefix, page_part = chunk_id.rsplit("_p", 1)
        page_num = int(page_part)
    except (ValueError, AttributeError):
        return []
    result = []
    for offset in [-1, 1]:
        adj_id = f"{prefix}_p{page_num + offset:04d}"
        if adj_id in page_chunks:
            result.append(adj_id)
    return result


def estimate_tokens(text):
    return int(len(text.split()) * 1.3)


def build_oracle_context(page_chunks, chunk_ids, token_budget=4500):
    """Build context from specific chunk IDs."""
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


def batch_vlm_call(vlm_client, prompts, concurrent=16):
    """Batch VLM calls."""
    async def _batch():
        sem = asyncio.Semaphore(concurrent)
        async_client = vlm_client._get_async_client()

        async def _call(prompt_text):
            if prompt_text is None:
                return ""
            async with sem:
                try:
                    messages = [{"role": "user", "content": prompt_text}]
                    response = await async_client.chat.completions.create(
                        model=vlm_client.model, messages=messages,
                        temperature=vlm_client.temperature,
                        max_tokens=vlm_client.max_tokens,
                    )
                    from src.reader.llm_client import _strip_think_tags
                    return _strip_think_tags(response.choices[0].message.content)
                except Exception:
                    return ""

        return await asyncio.gather(*[_call(p) for p in prompts])

    return asyncio.run(_batch())


def main():
    base_config = load_config("config/default.yaml")
    base_config["models"]["text_embedder"] = EMBEDDER

    from run_experiment import load_dataset, setup_backend, setup_agent
    from src.reader.prompt_templates import format_qa_prompt

    # Load common subset
    samples = load_dataset(base_config, DATASET)
    random.seed(42)
    if LIMIT and LIMIT < len(samples):
        samples = random.sample(samples, LIMIT)
    logger.info(f"Loaded {len(samples)} samples")

    # =====================================================
    # PHASE 1: Retrieve with metadata for all conditions
    # =====================================================
    logger.info("=" * 60)
    logger.info("PHASE 1: Retrieval with metadata")
    logger.info("=" * 60)

    # Setup final config
    final_config = make_final_config(base_config)
    retriever_final, reader = setup_backend(final_config, "vega_kg", DATASET)
    agent_final = setup_agent(final_config, AGENT, retriever_final, reader)
    page_chunks = retriever_final.page_chunks

    # Setup E000 config
    e000_config = make_e000_config(base_config)
    retriever_e000, _ = setup_backend(e000_config, "vega_kg", DATASET)
    agent_e000 = setup_agent(e000_config, AGENT, retriever_e000, reader)

    # Setup page_only
    from run_experiment import _setup_vega_page_only
    retriever_po = _setup_vega_page_only(final_config, DATASET)

    vlm_client = reader.vlm

    # Retrieve for all 3 conditions + collect metadata
    retrieval_data = []  # per-sample metadata

    logger.info("Retrieving for final, E000, page_only...")
    t0 = time.time()

    for idx, s in enumerate(samples):
        q = s["question"]

        # Final retrieval
        evidence_final = agent_final.retriever.retrieve(q, top_k=agent_final.top_k)
        final_page_ids = [e.get("id", "") for e in evidence_final]
        final_roles = {e.get("id", ""): e.get("_role", "other") for e in evidence_final}

        # E000 retrieval
        evidence_e000 = agent_e000.retriever.retrieve(q, top_k=agent_e000.top_k)
        e000_page_ids = [e.get("id", "") for e in evidence_e000]

        # Page-only retrieval
        evidence_po = retriever_po.retrieve(q, top_k=20)
        po_page_ids = [e.get("id", "") for e in evidence_po]

        # Find gold pages
        gold_pages = find_gold_pages(s, page_chunks)
        gold_page_ids = [gp[0] for gp in gold_pages]
        gold_doc_ids = list(set(gp[2] for gp in gold_pages))

        # Gold adjacent pages
        gold_adj_ids = []
        for gp_id in gold_page_ids:
            for adj_id in get_adjacent_pages(gp_id, page_chunks):
                if adj_id not in gold_adj_ids and adj_id not in gold_page_ids:
                    gold_adj_ids.append(adj_id)

        # Contexts for oracle conditions
        context_final = reader.serializer.serialize(evidence_final, None)
        prompt_final = format_qa_prompt(question=q, context=context_final)

        # O1: gold page only
        o1_context = build_oracle_context(page_chunks, gold_page_ids)
        prompt_o1 = format_qa_prompt(question=q, context=o1_context) if o1_context else None

        # O2: gold page + oracle adjacent
        o2_ids = gold_page_ids + gold_adj_ids
        o2_context = build_oracle_context(page_chunks, o2_ids)
        prompt_o2 = format_qa_prompt(question=q, context=o2_context) if o2_context else None

        # O3: final + gold page forced
        o3_evidence = list(evidence_final)
        for gp_id in gold_page_ids:
            if gp_id not in final_page_ids:
                chunk = page_chunks.get(gp_id)
                if chunk:
                    o3_evidence.insert(0, chunk)  # prepend gold page
        o3_context = reader.serializer.serialize(o3_evidence, None)
        prompt_o3 = format_qa_prompt(question=q, context=o3_context)

        # O4: final + gold page + oracle adjacent forced
        o4_evidence = list(evidence_final)
        for gp_id in gold_page_ids + gold_adj_ids:
            if gp_id not in final_page_ids:
                chunk = page_chunks.get(gp_id)
                if chunk:
                    o4_evidence.insert(0, chunk)
        o4_context = reader.serializer.serialize(o4_evidence, None)
        prompt_o4 = format_qa_prompt(question=q, context=o4_context)

        retrieval_data.append({
            "idx": idx,
            "id": s.get("id", ""),
            "question": q,
            "gold": str(s["answer"]) if s["answer"] is not None else "",
            "gold_page_ids": gold_page_ids,
            "gold_doc_ids": gold_doc_ids,
            "gold_adj_ids": gold_adj_ids,
            "final_page_ids": final_page_ids,
            "final_roles": final_roles,
            "e000_page_ids": e000_page_ids,
            "po_page_ids": po_page_ids,
            "gold_in_final": any(gp in final_page_ids for gp in gold_page_ids),
            "gold_in_e000": any(gp in e000_page_ids for gp in gold_page_ids),
            "gold_in_po": any(gp in po_page_ids for gp in gold_page_ids),
            "gold_adj_in_final": any(adj in final_page_ids for adj in gold_adj_ids),
            "prompts": {
                "final": prompt_final,
                "o1": prompt_o1,
                "o2": prompt_o2,
                "o3": prompt_o3,
                "o4": prompt_o4,
            },
            "metadata_type": s.get("metadata", {}).get("type", ""),
            "metadata_modalities": s.get("metadata", {}).get("modalities", []),
        })

        if (idx + 1) % 100 == 0:
            logger.info(f"  Retrieval metadata: {idx+1}/{len(samples)}")

    t_ret = time.time() - t0
    logger.info(f"Phase 1 done: {t_ret:.0f}s")

    # Gold page coverage stats
    n_gold_found = sum(1 for d in retrieval_data if d["gold_page_ids"])
    n_gold_in_final = sum(1 for d in retrieval_data if d["gold_in_final"])
    n_gold_in_e000 = sum(1 for d in retrieval_data if d["gold_in_e000"])
    logger.info(f"Gold page found in chunks: {n_gold_found}/{len(samples)}")
    logger.info(f"Gold page in final selected: {n_gold_in_final}/{len(samples)}")
    logger.info(f"Gold page in E000 selected: {n_gold_in_e000}/{len(samples)}")

    # =====================================================
    # PHASE 2: Oracle ceiling experiments (O0-O4)
    # =====================================================
    logger.info("=" * 60)
    logger.info("PHASE 2: Oracle ceiling experiments")
    logger.info("=" * 60)

    oracle_conditions = ["final", "o1", "o2", "o3", "o4"]
    oracle_results = {}

    for cond in oracle_conditions:
        cond_label = {"final": "O0", "o1": "O1", "o2": "O2", "o3": "O3", "o4": "O4"}[cond]
        logger.info(f"Running {cond_label}...")

        prompts = [d["prompts"][cond] for d in retrieval_data]

        t1 = time.time()
        answers = batch_vlm_call(vlm_client, prompts, CONCURRENT)
        t_vlm = time.time() - t1
        logger.info(f"  VLM done: {t_vlm:.0f}s")

        predictions = []
        golds = []
        for idx, d in enumerate(retrieval_data):
            pred = answers[idx].strip() if answers[idx] else ""
            predictions.append(pred)
            golds.append(d["gold"])
            d[f"pred_{cond}"] = pred

        metrics = evaluate_qa(predictions, golds,
                              anls_threshold=base_config["evaluation"]["anls_threshold"])
        metrics["condition"] = cond_label
        oracle_results[cond_label] = metrics
        logger.info(f"  {cond_label}: EM={metrics['EM']:.4f} ANLS={metrics['ANLS']:.4f} "
                     f"ROUGE-L={metrics['ROUGE-L']:.4f}")

    # =====================================================
    # PHASE 3: Flip analysis
    # =====================================================
    logger.info("=" * 60)
    logger.info("PHASE 3: Flip analysis")
    logger.info("=" * 60)

    # Load E000 and page_only predictions (need VLM for these too)
    # E000 predictions
    logger.info("Running E000 predictions...")
    e000_prompts = []
    for idx, d in enumerate(retrieval_data):
        evidence_e000 = [page_chunks[pid] for pid in d["e000_page_ids"] if pid in page_chunks]
        context = reader.serializer.serialize(evidence_e000, None)
        prompt = format_qa_prompt(question=d["question"], context=context)
        e000_prompts.append(prompt)

    e000_answers = batch_vlm_call(vlm_client, e000_prompts, CONCURRENT)
    for idx, d in enumerate(retrieval_data):
        d["pred_e000"] = e000_answers[idx].strip() if e000_answers[idx] else ""

    # Page-only predictions
    logger.info("Running page_only predictions...")
    po_prompts = []
    for idx, d in enumerate(retrieval_data):
        evidence_po = [page_chunks[pid] for pid in d["po_page_ids"] if pid in page_chunks]
        context = reader.serializer.serialize(evidence_po, None)
        prompt = format_qa_prompt(question=d["question"], context=context)
        po_prompts.append(prompt)

    po_answers = batch_vlm_call(vlm_client, po_prompts, CONCURRENT)
    for idx, d in enumerate(retrieval_data):
        d["pred_po"] = po_answers[idx].strip() if po_answers[idx] else ""

    # =====================================================
    # PHASE 4: Generate flip analysis CSV
    # =====================================================
    logger.info("=" * 60)
    logger.info("PHASE 4: Generate analysis outputs")
    logger.info("=" * 60)

    output_dir = Path(base_config["paths"]["results"]) / "bottleneck"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Flip analysis rows
    flip_rows = []
    for d in retrieval_data:
        gold = d["gold"]

        em_final = exact_match(d["pred_final"], gold)
        em_e000 = exact_match(d["pred_e000"], gold)
        em_po = exact_match(d["pred_po"], gold)
        em_o1 = exact_match(d.get("pred_o1", ""), gold)
        em_o2 = exact_match(d.get("pred_o2", ""), gold)

        # Determine bucket for E000 vs final
        if em_e000 == 0 and em_final == 1:
            bucket_e000 = "B+"
        elif em_e000 == 1 and em_final == 0:
            bucket_e000 = "B-"
        elif em_e000 == 0 and em_final == 0:
            if d["gold_in_final"]:
                bucket_e000 = "Bhard"
            else:
                bucket_e000 = "Bmiss"
        else:
            bucket_e000 = "Bboth"  # both correct

        # Determine bucket for page_only vs final
        if em_po == 0 and em_final == 1:
            bucket_po = "B+"
        elif em_po == 1 and em_final == 0:
            bucket_po = "B-"
        elif em_po == 0 and em_final == 0:
            if d["gold_in_final"]:
                bucket_po = "Bhard"
            else:
                bucket_po = "Bmiss"
        else:
            bucket_po = "Bboth"

        # Auto-label hints
        auto_label = ""
        if bucket_e000 == "B+" or bucket_po == "B+":
            # Check if any gold adjacent is in final but not in baseline
            if d["gold_adj_in_final"]:
                auto_label = "adjacent_rescue"
        if not d["gold_page_ids"]:
            auto_label = "gold_page_not_found"
        elif not d["gold_in_final"]:
            auto_label = "gold_page_missing"
        elif em_final == 0 and d["gold_in_final"]:
            if em_o1 == 1 or em_o2 == 1:
                auto_label = "reader_fail_on_present_evidence"
            else:
                auto_label = "reader_fail_even_oracle"

        flip_rows.append({
            "qid": d["id"],
            "question": d["question"],
            "gold_answer": gold,
            "pred_final": d["pred_final"],
            "pred_e000": d["pred_e000"],
            "pred_po": d["pred_po"],
            "pred_o1": d.get("pred_o1", ""),
            "pred_o2": d.get("pred_o2", ""),
            "em_final": em_final,
            "em_e000": em_e000,
            "em_po": em_po,
            "em_o1": exact_match(d.get("pred_o1", ""), gold),
            "em_o2": exact_match(d.get("pred_o2", ""), gold),
            "bucket_e000_vs_final": bucket_e000,
            "bucket_po_vs_final": bucket_po,
            "gold_page_ids": ";".join(d["gold_page_ids"]),
            "gold_doc_ids": ";".join(d["gold_doc_ids"]),
            "gold_in_final": d["gold_in_final"],
            "gold_in_e000": d["gold_in_e000"],
            "gold_in_po": d["gold_in_po"],
            "gold_adj_in_final": d["gold_adj_in_final"],
            "final_page_ids": ";".join(d["final_page_ids"][:5]),
            "metadata_type": d["metadata_type"],
            "auto_label": auto_label,
            "primary_label": "",  # for manual labeling
            "secondary_label": "",
            "notes": "",
        })

    # Write CSV
    csv_path = output_dir / "flip_analysis.csv"
    fieldnames = list(flip_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flip_rows)
    logger.info(f"Flip analysis CSV: {csv_path}")

    # =====================================================
    # PHASE 5: Summary statistics
    # =====================================================
    logger.info("=" * 60)
    logger.info("PHASE 5: Summary")
    logger.info("=" * 60)

    # Oracle results
    print("\n" + "=" * 70)
    print("ORACLE CEILING RESULTS")
    print("=" * 70)
    print(f"{'ID':<5} {'EM':>8} {'ANLS':>8} {'ROUGE-L':>8}")
    print("-" * 35)
    for cond_label in ["O0", "O1", "O2", "O3", "O4"]:
        m = oracle_results[cond_label]
        print(f"{cond_label:<5} {m['EM']:>8.4f} {m['ANLS']:>8.4f} {m['ROUGE-L']:>8.4f}")

    print("\nORACLE GAPS")
    print("-" * 50)
    gaps = {
        "Retrieval gap (O1-O0)": oracle_results["O1"]["EM"] - oracle_results["O0"]["EM"],
        "Boundary gap (O2-O1)": oracle_results["O2"]["EM"] - oracle_results["O1"]["EM"],
        "Packing gap (O3-O0)": oracle_results["O3"]["EM"] - oracle_results["O0"]["EM"],
        "Local miss gap (O4-O3)": oracle_results["O4"]["EM"] - oracle_results["O3"]["EM"],
    }
    for label, val in gaps.items():
        print(f"  {label}: {val:+.4f}")

    # Flip analysis bucket counts
    print("\n" + "=" * 70)
    print("FLIP ANALYSIS: E000 vs Final")
    print("=" * 70)
    buckets_e000 = defaultdict(int)
    for r in flip_rows:
        buckets_e000[r["bucket_e000_vs_final"]] += 1
    for bucket in ["B+", "B-", "Bhard", "Bmiss", "Bboth"]:
        count = buckets_e000.get(bucket, 0)
        print(f"  {bucket:<6}: {count:>4} ({100*count/len(flip_rows):.1f}%)")

    print("\nFLIP ANALYSIS: PageOnly vs Final")
    print("=" * 70)
    buckets_po = defaultdict(int)
    for r in flip_rows:
        buckets_po[r["bucket_po_vs_final"]] += 1
    for bucket in ["B+", "B-", "Bhard", "Bmiss", "Bboth"]:
        count = buckets_po.get(bucket, 0)
        print(f"  {bucket:<6}: {count:>4} ({100*count/len(flip_rows):.1f}%)")

    # Auto-label distribution
    print("\nAUTO-LABEL DISTRIBUTION")
    print("=" * 70)
    auto_labels = defaultdict(int)
    for r in flip_rows:
        if r["auto_label"]:
            auto_labels[r["auto_label"]] += 1
    for label, count in sorted(auto_labels.items(), key=lambda x: -x[1]):
        print(f"  {label:<40}: {count:>4} ({100*count/len(flip_rows):.1f}%)")

    # Gold page coverage
    print("\nGOLD PAGE COVERAGE")
    print("=" * 70)
    print(f"  Gold page found in chunks: {n_gold_found}/{len(samples)} ({100*n_gold_found/len(samples):.1f}%)")
    print(f"  Gold page in final:  {n_gold_in_final}/{len(samples)} ({100*n_gold_in_final/len(samples):.1f}%)")
    print(f"  Gold page in E000:   {n_gold_in_e000}/{len(samples)} ({100*n_gold_in_e000/len(samples):.1f}%)")

    # E000 vs final EM comparison
    em_e000_total = sum(r["em_e000"] for r in flip_rows) / len(flip_rows)
    em_final_total = sum(r["em_final"] for r in flip_rows) / len(flip_rows)
    em_po_total = sum(r["em_po"] for r in flip_rows) / len(flip_rows)
    print(f"\n  EM E000:      {em_e000_total:.4f}")
    print(f"  EM Final:     {em_final_total:.4f}")
    print(f"  EM PageOnly:  {em_po_total:.4f}")

    # Save all data
    save_json(oracle_results, output_dir / "oracle_results.json")
    save_json({
        "bucket_e000_vs_final": dict(buckets_e000),
        "bucket_po_vs_final": dict(buckets_po),
        "auto_labels": dict(auto_labels),
        "gold_coverage": {
            "found": n_gold_found,
            "in_final": n_gold_in_final,
            "in_e000": n_gold_in_e000,
            "total": len(samples),
        },
        "em_scores": {
            "e000": em_e000_total,
            "final": em_final_total,
            "page_only": em_po_total,
        },
        "oracle_gaps": gaps,
    }, output_dir / "summary.json")

    # Save per-sample retrieval data (without prompts to save space)
    compact_data = []
    for d in retrieval_data:
        compact = {k: v for k, v in d.items() if k != "prompts"}
        compact["pred_final"] = d.get("pred_final", "")
        compact["pred_e000"] = d.get("pred_e000", "")
        compact["pred_po"] = d.get("pred_po", "")
        compact["pred_o1"] = d.get("pred_o1", "")
        compact["pred_o2"] = d.get("pred_o2", "")
        compact_data.append(compact)
    save_json(compact_data, output_dir / "retrieval_metadata.json")

    logger.info(f"\nAll outputs saved to {output_dir}")
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
