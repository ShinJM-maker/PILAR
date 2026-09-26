"""Selection/packing bottleneck experiments.

A. Full-page pruning sweep (P1-P8, Pcur)
B. Rank-aware diagnostics (R0-R4 classification)
C. Score-only prior ablation (S0-S3)

Uses same 500-sample subset (seed=42) as bottleneck analysis.

Usage:
    PYTHONPATH=. python scripts/run_selection_experiments.py
"""
import json
import logging
import time
import random
import asyncio
import csv
from pathlib import Path
from copy import deepcopy
from collections import Counter, defaultdict

from src.utils.io_utils import load_config, load_json, save_json
from src.evaluation.metrics import evaluate_qa, normalize_answer, exact_match

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

DATASET = "m3docvqa"
AGENT = "naive_rag"
LIMIT = 500
CONCURRENT = 16
EMBEDDER = "Qwen/Qwen3-Embedding-0.6B"


def make_final_config(base_config):
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


def batch_vlm_call(vlm_client, prompts, concurrent=16):
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


def estimate_tokens(text):
    return int(len(text.split()) * 1.3)


def find_gold_pages(sample, page_chunks):
    """Find gold pages by text matching."""
    gold = str(sample["answer"]) if sample["answer"] is not None else ""
    if not gold:
        return []

    norm_gold = normalize_answer(gold)
    if not norm_gold:
        return []

    supporting_doc_ids = set()
    if sample.get("metadata", {}).get("supporting_doc_ids"):
        supporting_doc_ids = set(sample["metadata"]["supporting_doc_ids"])

    results = []
    for chunk_id, chunk in page_chunks.items():
        text = chunk.get("text", "")
        if not text:
            continue
        doc_id = chunk.get("doc_id", "")
        if supporting_doc_ids and doc_id not in supporting_doc_ids:
            continue
        if norm_gold in normalize_answer(text):
            page_num = chunk.get("page", 0)
            results.append((chunk_id, page_num, doc_id))

    return results


def retrieve_with_internals(retriever, query, top_k=20):
    """Call retrieve but also extract internal stage data.

    Returns: (evidence_list, internals_dict)
    internals_dict has: base_scored, final_scored, ranked, seed_chunk_ids, etc.
    """
    from src.retrieval.bm25_index import BM25Index
    from src.retrieval.dense_index import DenseIndex
    from collections import Counter

    r = retriever

    # Step 1: Base hybrid retrieval
    top_n = top_k * r.initial_multiplier
    bm25_results = {did: s for did, s in r.page_bm25.search(query, top_n)}
    dense_results = {did: s for did, s in r.page_dense.search(query, top_n)}

    all_ids = set(bm25_results.keys()) | set(dense_results.keys())
    max_bm25 = max(bm25_results.values()) if bm25_results else 1.0

    base_scored = {}
    for chunk_id in all_ids:
        bm25_norm = bm25_results.get(chunk_id, 0.0) / max_bm25 if max_bm25 > 0 else 0.0
        dense_score = dense_results.get(chunk_id, 0.0)
        base_scored[chunk_id] = 0.4 * bm25_norm + 0.6 * dense_score

    if not base_scored:
        return [], {"base_scored": {}, "final_scored": {}, "ranked": []}

    # Min-max normalize
    base_scored = r._minmax_normalize(base_scored)
    base_candidate_ids = set(base_scored.keys())

    # Step 2: Seed selection
    base_ranked = sorted(base_scored.items(), key=lambda x: x[1], reverse=True)
    best_score = base_ranked[0][1]

    seeds = []
    for cid, sc in base_ranked[:r.seed_top_n]:
        if sc >= r.seed_gate_ratio * best_score:
            seeds.append((cid, sc))

    seed_chunk_ids = {cid for cid, _ in seeds}
    seed_scores = {cid: sc for cid, sc in seeds}
    seed_doc_ids = {
        r._chunk_to_doc_id[cid]
        for cid in seed_chunk_ids
        if cid in r._chunk_to_doc_id
    }

    # Step 3: Local continuity bonus
    local_bonus = Counter()
    if r.enable_local_bonus:
        local_bonus = r._compute_local_bonus(
            seed_chunk_ids=seed_chunk_ids,
            base_candidate_ids=base_candidate_ids,
            best_score=best_score,
            seed_scores=seed_scores,
        )

    # Step 4: KG expansion
    matched_entity_ids = r._match_query_entities(query)
    alias_votes = r._expand_alias_pages(
        matched_entity_ids=matched_entity_ids,
        seed_doc_ids=seed_doc_ids,
        seed_chunk_ids=seed_chunk_ids,
    )
    assertion_votes = r._expand_graph_pages(
        seed_chunk_ids=seed_chunk_ids,
        seed_doc_ids=seed_doc_ids,
    )

    # Step 5: Gate expanded
    kg_expanded_ids = (set(alias_votes.keys()) | set(assertion_votes.keys()))
    new_expanded_ids = kg_expanded_ids - base_candidate_ids
    expanded_direct_scores = {}
    gated_new_expanded = set()

    if r.enable_expanded_gate and new_expanded_ids:
        expanded_direct_scores = r._gate_expanded_pages(
            query=query,
            expanded_chunk_ids=kg_expanded_ids,
            base_candidate_ids=base_candidate_ids,
        )
        gated_new_expanded = set(expanded_direct_scores.keys())
    elif not r.enable_expanded_gate:
        gated_new_expanded = new_expanded_ids

    # Step 6: Final score
    all_candidate_ids = base_candidate_ids | gated_new_expanded

    final_scored = {}
    for chunk_id in all_candidate_ids:
        base = base_scored.get(chunk_id, 0.0)
        if chunk_id in gated_new_expanded:
            if r.enable_expanded_gate and chunk_id in expanded_direct_scores:
                base = expanded_direct_scores[chunk_id] * 0.45
            else:
                base = 0.0
        bonus = 0.0
        if r.enable_local_bonus:
            bonus += local_bonus.get(chunk_id, 0.0)
        kg_bonus = 0.0
        if alias_votes.get(chunk_id, 0) > 0:
            kg_bonus += r.alias_bonus
        if assertion_votes.get(chunk_id, 0) > 0:
            kg_bonus += r.assertion_bonus
        kg_bonus = min(kg_bonus, r.max_total_bonus)
        bonus += kg_bonus
        final_scored[chunk_id] = base + bonus

    # Step 7: Packing (get ranked list)
    if r.enable_expanded_gate:
        ranked = r._pack_conditional(
            final_scored=final_scored,
            base_candidate_ids=base_candidate_ids,
            top_k=top_k,
        )
    else:
        ranked = r._pack_base_first(
            final_scored=final_scored,
            base_candidate_ids=base_candidate_ids,
            top_k=top_k,
        )

    # Step 8-9: Build output with roles
    adjacent_chunk_ids = set()
    for seed_id in seed_chunk_ids:
        for adj_id in r._adjacent_chunk_ids(seed_id):
            if adj_id not in seed_chunk_ids:
                adjacent_chunk_ids.add(adj_id)

    packet = []
    for chunk_id, score in ranked:
        chunk = r.page_chunks.get(chunk_id)
        if chunk:
            chunk_with_role = dict(chunk)
            if chunk_id in seed_chunk_ids:
                chunk_with_role["_role"] = "seed"
            elif chunk_id in adjacent_chunk_ids:
                chunk_with_role["_role"] = "adjacent"
            elif chunk_id not in base_candidate_ids:
                chunk_with_role["_role"] = "expanded"
            else:
                chunk_with_role["_role"] = "other"
            chunk_with_role["_score"] = score
            packet.append(chunk_with_role)

    internals = {
        "base_scored": base_scored,
        "base_ranked": [(cid, sc) for cid, sc in base_ranked],
        "final_scored": final_scored,
        "ranked": ranked,
        "seed_chunk_ids": seed_chunk_ids,
        "local_bonus": dict(local_bonus),
        "alias_votes": dict(alias_votes),
        "assertion_votes": dict(assertion_votes),
        "base_candidate_ids": base_candidate_ids,
        "gated_new_expanded": gated_new_expanded,
        "adjacent_chunk_ids": adjacent_chunk_ids,
    }

    return packet, internals


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

    # Setup
    final_config = make_final_config(base_config)
    retriever_final, reader = setup_backend(final_config, "vega_kg", DATASET)
    agent_final = setup_agent(final_config, AGENT, retriever_final, reader)
    page_chunks = retriever_final.page_chunks
    vlm_client = reader.vlm

    output_dir = Path(base_config["paths"]["results"]) / "selection_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # PHASE 1: Retrieve with internal stage data
    # =====================================================
    logger.info("=" * 60)
    logger.info("PHASE 1: Retrieval with internals (B: rank diagnostics)")
    logger.info("=" * 60)

    all_data = []
    t0 = time.time()

    for idx, s in enumerate(samples):
        q = s["question"]
        gold = str(s["answer"]) if s["answer"] is not None else ""

        evidence, internals = retrieve_with_internals(
            retriever_final, q, top_k=agent_final.top_k
        )

        gold_pages = find_gold_pages(s, page_chunks)
        gold_page_ids = [gp[0] for gp in gold_pages]

        # Rank diagnostics per gold page
        base_ranked_ids = [cid for cid, _ in internals["base_ranked"]]
        final_ranked_ids = [cid for cid, _ in internals["ranked"]]
        packed_ids = [e.get("id", "") for e in evidence]

        # Find gold page rank in each stage
        def find_rank(page_id, ranked_list):
            for i, cid in enumerate(ranked_list):
                if cid == page_id:
                    return i + 1  # 1-indexed
            return -1

        gold_in_index = len(gold_page_ids) > 0
        gold_rank_base = -1
        gold_rank_final = -1
        gold_rank_packed = -1
        gold_in_base_top40 = False
        gold_in_final = False
        gold_in_packed = False
        gold_base_score = 0.0
        gold_final_score = 0.0
        gold_local_bonus = 0.0
        gold_kg_bonus = 0.0
        gold_promoted_by_local = False
        gold_promoted_by_kg = False

        for gp_id in gold_page_ids:
            r_base = find_rank(gp_id, base_ranked_ids)
            r_final = find_rank(gp_id, final_ranked_ids)
            r_packed = find_rank(gp_id, packed_ids)

            if r_base > 0 and (gold_rank_base < 0 or r_base < gold_rank_base):
                gold_rank_base = r_base
                gold_base_score = internals["base_scored"].get(gp_id, 0.0)
                gold_final_score = internals["final_scored"].get(gp_id, 0.0)
                gold_local_bonus = internals["local_bonus"].get(gp_id, 0.0)
                alias_v = internals["alias_votes"].get(gp_id, 0)
                assert_v = internals["assertion_votes"].get(gp_id, 0)
                gold_kg_bonus = min(
                    (0.02 if alias_v > 0 else 0) + (0.02 if assert_v > 0 else 0),
                    0.04
                )
                gold_promoted_by_local = gold_local_bonus > 0
                gold_promoted_by_kg = gold_kg_bonus > 0

            if r_base > 0 and r_base <= 40:
                gold_in_base_top40 = True
            if r_final > 0:
                gold_in_final = True
                if gold_rank_final < 0 or r_final < gold_rank_final:
                    gold_rank_final = r_final
            if r_packed > 0:
                gold_in_packed = True
                if gold_rank_packed < 0 or r_packed < gold_rank_packed:
                    gold_rank_packed = r_packed

        # R0-R4 classification
        if not gold_in_index:
            r_case = "R0_not_in_index"
        elif not gold_in_base_top40:
            r_case = "R1_not_in_base40"
        elif not gold_in_final:
            r_case = "R2_not_in_final"
        elif not gold_in_packed:
            r_case = "R3_not_in_packed"
        else:
            r_case = "R4_in_packed"

        # Serialize context for different page counts
        all_data.append({
            "idx": idx,
            "id": s.get("id", ""),
            "question": q,
            "gold": gold,
            "gold_page_ids": gold_page_ids,
            "evidence": evidence,  # full evidence list
            "internals": internals,
            # Rank diagnostics
            "gold_in_index": gold_in_index,
            "gold_in_base_top40": gold_in_base_top40,
            "gold_in_final": gold_in_final,
            "gold_in_packed": gold_in_packed,
            "gold_rank_base": gold_rank_base,
            "gold_rank_final": gold_rank_final,
            "gold_rank_packed": gold_rank_packed,
            "gold_base_score": gold_base_score,
            "gold_final_score": gold_final_score,
            "gold_local_bonus": gold_local_bonus,
            "gold_kg_bonus": gold_kg_bonus,
            "gold_promoted_by_local": gold_promoted_by_local,
            "gold_promoted_by_kg": gold_promoted_by_kg,
            "r_case": r_case,
            "num_packed_pages": len(evidence),
            "num_packed_tokens": sum(estimate_tokens(e.get("text", "")) for e in evidence),
        })

        if (idx + 1) % 100 == 0:
            logger.info(f"  Phase 1: {idx+1}/{len(samples)}")

    t1 = time.time()
    logger.info(f"Phase 1 done: {t1-t0:.0f}s")

    # =====================================================
    # PHASE 1B: Rank diagnostics summary
    # =====================================================
    logger.info("=" * 60)
    logger.info("PHASE 1B: Rank diagnostics")
    logger.info("=" * 60)

    # R0-R4 distribution
    r_cases = Counter(d["r_case"] for d in all_data)
    for case in ["R0_not_in_index", "R1_not_in_base40", "R2_not_in_final",
                  "R3_not_in_packed", "R4_in_packed"]:
        n = r_cases.get(case, 0)
        logger.info(f"  {case}: {n} ({n/len(all_data)*100:.1f}%)")

    # Recall@K for final
    for k in [1, 3, 5, 10, 20]:
        recall = sum(1 for d in all_data
                     if d["gold_rank_final"] > 0 and d["gold_rank_final"] <= k) / len(all_data)
        logger.info(f"  FinalRecall@{k}: {recall:.4f}")

    # MRR
    mrr_base = sum(1.0/d["gold_rank_base"] for d in all_data
                   if d["gold_rank_base"] > 0) / len(all_data)
    mrr_final = sum(1.0/d["gold_rank_final"] for d in all_data
                    if d["gold_rank_final"] > 0) / len(all_data)
    logger.info(f"  MRR_base: {mrr_base:.4f}")
    logger.info(f"  MRR_final: {mrr_final:.4f}")

    # Gold page packed position mean (among those packed)
    packed_positions = [d["gold_rank_packed"] for d in all_data if d["gold_rank_packed"] > 0]
    if packed_positions:
        logger.info(f"  PackedPosMean: {sum(packed_positions)/len(packed_positions):.2f}")

    # Avg pages/tokens
    avg_pages = sum(d["num_packed_pages"] for d in all_data) / len(all_data)
    avg_tokens = sum(d["num_packed_tokens"] for d in all_data) / len(all_data)
    logger.info(f"  Avg packed pages: {avg_pages:.1f}")
    logger.info(f"  Avg packed tokens: {avg_tokens:.0f}")

    # Local/KG prior effectiveness
    n_promoted_local = sum(1 for d in all_data if d["gold_promoted_by_local"])
    n_promoted_kg = sum(1 for d in all_data if d["gold_promoted_by_kg"])
    logger.info(f"  Gold promoted by local: {n_promoted_local} ({n_promoted_local/len(all_data)*100:.1f}%)")
    logger.info(f"  Gold promoted by KG: {n_promoted_kg} ({n_promoted_kg/len(all_data)*100:.1f}%)")

    # Save rank diagnostics
    rank_diagnostics = {
        "r_cases": dict(r_cases),
        "recall_at_k": {},
        "mrr_base": mrr_base,
        "mrr_final": mrr_final,
        "packed_pos_mean": sum(packed_positions)/len(packed_positions) if packed_positions else 0,
        "avg_packed_pages": avg_pages,
        "avg_packed_tokens": avg_tokens,
        "gold_promoted_local": n_promoted_local,
        "gold_promoted_kg": n_promoted_kg,
    }
    for k in [1, 3, 5, 10, 20]:
        recall = sum(1 for d in all_data
                     if d["gold_rank_final"] > 0 and d["gold_rank_final"] <= k) / len(all_data)
        rank_diagnostics["recall_at_k"][f"@{k}"] = recall

    save_json(rank_diagnostics, output_dir / "rank_diagnostics.json")
    logger.info(f"Saved rank_diagnostics.json")

    # =====================================================
    # PHASE 2: Full-page pruning sweep (A)
    # =====================================================
    logger.info("=" * 60)
    logger.info("PHASE 2: Full-page pruning sweep")
    logger.info("=" * 60)

    page_caps = [1, 2, 3, 4, 5, 8, 0]  # 0 = current (no cap)
    pruning_results = {}

    for cap in page_caps:
        cap_label = f"P{cap}" if cap > 0 else "Pcur"
        logger.info(f"  Building prompts for {cap_label}...")

        prompts = []
        cap_gold_included = 0
        cap_tokens_total = 0

        for d in all_data:
            evidence = d["evidence"]
            if cap > 0:
                evidence_capped = evidence[:cap]
            else:
                evidence_capped = evidence

            context = reader.serializer.serialize(evidence_capped, None)
            prompt = format_qa_prompt(question=d["question"], context=context)
            prompts.append(prompt)

            # Check gold inclusion
            capped_ids = [e.get("id", "") for e in evidence_capped]
            if any(gp in capped_ids for gp in d["gold_page_ids"]):
                cap_gold_included += 1
            cap_tokens_total += sum(estimate_tokens(e.get("text", "")) for e in evidence_capped)

        avg_cap_tokens = cap_tokens_total / len(all_data)
        avg_cap_pages = cap if cap > 0 else avg_pages

        logger.info(f"  {cap_label}: VLM batch ({len(prompts)} prompts, "
                     f"avg {avg_cap_tokens:.0f} tokens, gold in {cap_gold_included}/{len(all_data)})...")

        t_start = time.time()
        answers = batch_vlm_call(vlm_client, prompts, CONCURRENT)
        t_vlm = time.time() - t_start

        predictions = [a.strip() if a else "" for a in answers]
        golds = [d["gold"] for d in all_data]

        metrics = evaluate_qa(predictions, golds,
                              anls_threshold=base_config["evaluation"]["anls_threshold"])
        metrics["condition"] = cap_label
        metrics["page_cap"] = cap if cap > 0 else "none"
        metrics["avg_tokens"] = avg_cap_tokens
        metrics["avg_pages"] = avg_cap_pages
        metrics["gold_included"] = cap_gold_included
        metrics["vlm_time"] = t_vlm

        pruning_results[cap_label] = metrics
        logger.info(f"  {cap_label}: EM={metrics['EM']:.4f} ANLS={metrics['ANLS']:.4f} "
                     f"ROUGE-L={metrics['ROUGE-L']:.4f} ({t_vlm:.0f}s)")

        # Save per-condition predictions
        save_json(metrics, output_dir / f"metrics_{cap_label}.json")

    save_json(pruning_results, output_dir / "pruning_sweep_summary.json")

    # =====================================================
    # PHASE 3: Score-only prior ablation (C)
    # =====================================================
    logger.info("=" * 60)
    logger.info("PHASE 3: Score-only prior ablation")
    logger.info("=" * 60)

    # Use best page cap from Phase 2 (or if unclear, use top-3 as default)
    best_cap_label = max(pruning_results.keys(), key=lambda k: pruning_results[k]["EM"])
    best_cap = pruning_results[best_cap_label].get("page_cap", "none")
    if best_cap == "none":
        best_cap = 0
    else:
        best_cap = int(best_cap)
    logger.info(f"Best page cap from pruning: {best_cap_label} (cap={best_cap})")

    # S0: current final_score for both shortlist and pack
    # S1: final_score for shortlist, base_score for pack ordering
    # S2: S1 + base_score threshold guard (base top-10 or score >= 0.20)
    # S3: top-1 uses final_score, rest uses base_score

    prior_conditions = {
        "S0": {"pack_score": "final", "guard": None, "anchor_only": False},
        "S1": {"pack_score": "base", "guard": None, "anchor_only": False},
        "S2": {"pack_score": "base", "guard": "top10_or_score020", "anchor_only": False},
        "S3": {"pack_score": "hybrid", "guard": None, "anchor_only": True},
    }

    prior_results = {}

    for cond_id, cond_cfg in prior_conditions.items():
        logger.info(f"  Building {cond_id}...")

        prompts = []
        cond_gold_included = 0

        for d in all_data:
            internals = d["internals"]
            evidence = d["evidence"]

            # Re-rank based on condition
            if cond_cfg["pack_score"] == "final":
                # S0: current behavior
                reordered = evidence
            elif cond_cfg["pack_score"] == "base":
                # S1/S2: reorder by base_score
                base_scored = internals["base_scored"]

                if cond_cfg["guard"] == "top10_or_score020":
                    # S2: only include pages that are in base top-10 or score >= 0.20
                    base_ranked_ids = [cid for cid, _ in internals["base_ranked"][:10]]
                    allowed = set(base_ranked_ids)
                    for cid, sc in internals["base_scored"].items():
                        if sc >= 0.20:
                            allowed.add(cid)

                    reordered = []
                    for e in evidence:
                        eid = e.get("id", "")
                        if eid in allowed:
                            reordered.append(e)
                    # Sort by base score
                    reordered.sort(key=lambda e: base_scored.get(e.get("id", ""), 0), reverse=True)
                else:
                    # S1: just reorder by base_score
                    reordered = sorted(evidence,
                                       key=lambda e: base_scored.get(e.get("id", ""), 0),
                                       reverse=True)
            elif cond_cfg["pack_score"] == "hybrid":
                # S3: top-1 by final_score, rest by base_score
                if evidence:
                    anchor = [evidence[0]]  # top-1 from final ranking
                    rest = sorted(evidence[1:],
                                  key=lambda e: internals["base_scored"].get(e.get("id", ""), 0),
                                  reverse=True)
                    reordered = anchor + rest
                else:
                    reordered = evidence

            # Apply page cap
            if best_cap > 0:
                reordered = reordered[:best_cap]

            context = reader.serializer.serialize(reordered, None)
            prompt = format_qa_prompt(question=d["question"], context=context)
            prompts.append(prompt)

            capped_ids = [e.get("id", "") for e in reordered]
            if any(gp in capped_ids for gp in d["gold_page_ids"]):
                cond_gold_included += 1

        logger.info(f"  {cond_id}: VLM batch (gold in {cond_gold_included}/{len(all_data)})...")

        t_start = time.time()
        answers = batch_vlm_call(vlm_client, prompts, CONCURRENT)
        t_vlm = time.time() - t_start

        predictions = [a.strip() if a else "" for a in answers]
        golds = [d["gold"] for d in all_data]

        metrics = evaluate_qa(predictions, golds,
                              anls_threshold=base_config["evaluation"]["anls_threshold"])
        metrics["condition"] = cond_id
        metrics["gold_included"] = cond_gold_included
        metrics["vlm_time"] = t_vlm
        metrics["page_cap"] = best_cap if best_cap > 0 else "none"

        prior_results[cond_id] = metrics
        logger.info(f"  {cond_id}: EM={metrics['EM']:.4f} ANLS={metrics['ANLS']:.4f} "
                     f"ROUGE-L={metrics['ROUGE-L']:.4f} ({t_vlm:.0f}s)")

        save_json(metrics, output_dir / f"metrics_{cond_id}.json")

    save_json(prior_results, output_dir / "prior_ablation_summary.json")

    # =====================================================
    # FINAL SUMMARY
    # =====================================================
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)

    print("\n" + "=" * 70)
    print("A. FULL-PAGE PRUNING SWEEP")
    print("=" * 70)
    print(f"{'ID':<6} {'Pages':>6} {'AvgTok':>7} {'GoldIn':>7} "
          f"{'EM':>8} {'ANLS':>8} {'ROUGE-L':>8}")
    print("-" * 55)
    for cap in page_caps:
        label = f"P{cap}" if cap > 0 else "Pcur"
        m = pruning_results[label]
        pages_str = str(cap) if cap > 0 else "all"
        print(f"{label:<6} {pages_str:>6} {m['avg_tokens']:>7.0f} {m['gold_included']:>7} "
              f"{m['EM']:>8.4f} {m['ANLS']:>8.4f} {m['ROUGE-L']:>8.4f}")

    # Best pruning
    best_pruning = max(pruning_results.items(), key=lambda x: x[1]["EM"])
    print(f"\nBest: {best_pruning[0]} (EM={best_pruning[1]['EM']:.4f})")

    print("\n" + "=" * 70)
    print("B. RANK DIAGNOSTICS")
    print("=" * 70)
    for case in ["R0_not_in_index", "R1_not_in_base40", "R2_not_in_final",
                  "R3_not_in_packed", "R4_in_packed"]:
        n = r_cases.get(case, 0)
        print(f"  {case:<25}: {n:>4} ({n/len(all_data)*100:.1f}%)")

    print(f"\n  MRR_base:  {mrr_base:.4f}")
    print(f"  MRR_final: {mrr_final:.4f}")
    print(f"  MRR gain:  {mrr_final - mrr_base:+.4f}")

    for k in [1, 3, 5, 10, 20]:
        r_val = rank_diagnostics["recall_at_k"][f"@{k}"]
        print(f"  FinalRecall@{k}: {r_val:.4f}")

    print(f"\n  Avg packed pages: {avg_pages:.1f}")
    print(f"  Avg packed tokens: {avg_tokens:.0f}")
    print(f"  Gold promoted by local: {n_promoted_local} ({n_promoted_local/len(all_data)*100:.1f}%)")
    print(f"  Gold promoted by KG: {n_promoted_kg} ({n_promoted_kg/len(all_data)*100:.1f}%)")

    print("\n" + "=" * 70)
    print("C. SCORE-ONLY PRIOR ABLATION")
    print("=" * 70)
    print(f"{'ID':<6} {'GoldIn':>7} {'EM':>8} {'ANLS':>8} {'ROUGE-L':>8}")
    print("-" * 40)
    for cond_id in ["S0", "S1", "S2", "S3"]:
        m = prior_results[cond_id]
        print(f"{cond_id:<6} {m['gold_included']:>7} "
              f"{m['EM']:>8.4f} {m['ANLS']:>8.4f} {m['ROUGE-L']:>8.4f}")

    # Incremental effects
    print(f"\nPrior effects (vs S0):")
    for cond_id in ["S1", "S2", "S3"]:
        delta = prior_results[cond_id]["EM"] - prior_results["S0"]["EM"]
        print(f"  S0→{cond_id}: {delta:+.4f}")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
