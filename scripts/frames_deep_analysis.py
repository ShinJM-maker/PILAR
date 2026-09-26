"""Frames deep analysis: retrieval sweep + oracle + support-chain coverage.

Steps:
0. Hop distribution verification (fixed)
1. Retrieval sweep F0-F3 (vega_kg with 4 configs, 2+hop focused)
2. Gold oracle O0-O2 (current / gold page / gold support chain)
3. Support-chain coverage diagnostics (no VLM needed)

Usage:
    PYTHONPATH=. python scripts/frames_deep_analysis.py
"""
import json
import pickle
import time
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")

from src.utils.io_utils import load_config, load_json, save_json
from src.evaluation.metrics import exact_match, anls_score, rouge_l_score, meteor_score_single
from src.reader.llm_client import VLLMClient
from src.reader.prompt_templates import format_qa_prompt
from src.retrieval.dense_index import DenseIndex

RESULTS_DIR = Path("data/results")
OUTPUT_DIR = RESULTS_DIR / "frames_deep_analysis"


def load_frames_samples():
    samples = load_json("dataset/frames/samples.json")
    for s in samples:
        if "supporting_doc_ids" not in s and "filtered_supporting_doc_ids" in s:
            s["supporting_doc_ids"] = s["filtered_supporting_doc_ids"]
    return samples


def evidence_page_to_chunk_id(ep: str) -> str:
    """Map evidence page 'DocName_N' (1-indexed) to chunk 'DocName_p{N-1:04d}'."""
    parts = ep.rsplit("_", 1)
    if len(parts) != 2:
        return None
    doc_name, page_str = parts
    try:
        page_num = int(page_str) - 1  # 1-indexed to 0-indexed
        return f"{doc_name}_p{page_num:04d}"
    except ValueError:
        return None


def compute_metrics(results, hop_filter=None):
    ems, anlss, rouges, meteors = [], [], [], []
    for r in results:
        if hop_filter is not None and r.get("hops", 0) not in hop_filter:
            continue
        pred = str(r.get("prediction", ""))
        gold = str(r.get("gold", ""))
        ems.append(exact_match(pred, gold))
        anlss.append(anls_score(pred, gold))
        rouges.append(rouge_l_score(pred, gold))
        meteors.append(meteor_score_single(pred, gold))
    n = len(ems)
    if n == 0:
        return {"EM": 0, "ANLS": 0, "ROUGE-L": 0, "METEOR": 0, "n": 0}
    return {
        "EM": sum(ems) / n,
        "ANLS": sum(anlss) / n,
        "ROUGE-L": sum(rouges) / n,
        "METEOR": sum(meteors) / n,
        "n": n,
    }


def compute_hop_metrics(results):
    """Compute metrics by hop bucket: 2-hop, 3-hop, 4+hop, 2+hop."""
    buckets = {
        "2-hop": {2},
        "3-hop": {3},
        "4+hop": {4, 5},
        "2+hop": {2, 3, 4, 5},
    }
    out = {}
    for name, hops in buckets.items():
        out[name] = compute_metrics(results, hop_filter=hops)
    return out


# ============================================================
# STEP 2: Retrieval sweep F0-F3
# ============================================================
def run_retrieval_sweep(config, samples, reader, chunks):
    """Run F0-F3 retrieval configs with vega_kg on all samples."""
    from run_experiment import _load_chunks_and_indices
    from src.kg.graph_builder import HybridKGBuilder
    from src.retrieval.retriever import VEGAKGRetriever

    # Load KG once
    kg_path = Path(config["paths"]["kg"]) / "frames_kg.pkl"
    kg = HybridKGBuilder.load(str(kg_path))
    print(f"  KG loaded: {len(kg.supports)} supports, {len(kg.entities)} entities")

    data = _load_chunks_and_indices(config, "frames")
    page_chunks, page_bm25, page_dense = data["chunks"], data["bm25"], data["dense"]
    entities_dict = {eid: e.to_dict() for eid, e in kg.entities.items()}
    assertions_dict = {aid: a.to_dict() for aid, a in kg.assertions.items()}

    sweep_configs = {
        "F0": {"initial_multiplier": 2, "adjacent_page_bonus": 0.04, "same_section_bonus": 0.00},
        "F1": {"initial_multiplier": 2, "adjacent_page_bonus": 0.00, "same_section_bonus": 0.00},
        "F2": {"initial_multiplier": 3, "adjacent_page_bonus": 0.00, "same_section_bonus": 0.00},
        "F3": {"initial_multiplier": 4, "adjacent_page_bonus": 0.00, "same_section_bonus": 0.00},
    }

    all_sweep_results = {}

    for sweep_id, overrides in sweep_configs.items():
        print(f"\n  --- {sweep_id}: {overrides} ---")

        # Check if F0 results already exist (= existing qwen3emb predictions)
        if sweep_id == "F0":
            existing = RESULTS_DIR / "predictions_naive_rag_vega_kg_frames_qwen3emb.json"
            if existing.exists():
                print(f"  Reusing existing F0 from {existing}")
                preds = load_json(existing)
                results = []
                for p in preds:
                    sid = p["id"]
                    s_meta = {s["id"]: s for s in samples}.get(sid)
                    hops = s_meta.get("num_hops", 0) if s_meta else 0
                    results.append({
                        "id": sid,
                        "question": p.get("question", ""),
                        "gold": str(p.get("gold", "")),
                        "prediction": str(p.get("prediction", "")),
                        "hops": hops,
                    })
                all_sweep_results[sweep_id] = results
                hop_m = compute_hop_metrics(results)
                for bname, m in hop_m.items():
                    print(f"    {bname}: EM={m['EM']:.3f}, ANLS={m['ANLS']:.3f} (n={m['n']})")
                continue

        # Build retriever with overrides
        ret_config = dict(config["retrieval"])
        ret_config.update(overrides)

        retriever = VEGAKGRetriever(
            kg.graph, kg.supports,
            page_bm25, page_dense, page_chunks,
            entities_dict, assertions_dict,
            ret_config,
        )

        # Phase 1: Retrieve
        t0 = time.time()
        samples_with_context = []
        for s in samples:
            query = s["question"]
            evidence = retriever.retrieve(query, top_k=config["retrieval"]["seed_top_k"])
            if isinstance(evidence, list):
                if evidence and isinstance(evidence[0], dict):
                    context = "\n\n".join(e.get("text", "") for e in evidence)
                else:
                    context = "\n\n".join(str(e) for e in evidence)
            else:
                context = str(evidence)
            samples_with_context.append({
                "id": s["id"],
                "question": query,
                "gold": str(s["answer"]),
                "context": context,
                "hops": s.get("num_hops", 0),
            })
        ret_time = time.time() - t0
        print(f"    Retrieval: {ret_time:.1f}s")

        # Phase 2: Batch VLM
        t1 = time.time()
        prompts = [format_qa_prompt(item["question"], item["context"]) for item in samples_with_context]
        responses = reader.generate_batch(prompts)
        vlm_time = time.time() - t1
        print(f"    VLM: {vlm_time:.1f}s")

        results = []
        for item, resp in zip(samples_with_context, responses):
            results.append({
                "id": item["id"],
                "question": item["question"],
                "gold": item["gold"],
                "prediction": resp.strip(),
                "hops": item["hops"],
            })

        all_sweep_results[sweep_id] = results
        save_json(results, OUTPUT_DIR / f"predictions_{sweep_id}.json")

        hop_m = compute_hop_metrics(results)
        for bname, m in hop_m.items():
            print(f"    {bname}: EM={m['EM']:.3f}, ANLS={m['ANLS']:.3f} (n={m['n']})")

    return all_sweep_results


# ============================================================
# STEP 3: Gold oracle experiments O0-O2
# ============================================================
def run_oracle_experiments(config, samples, reader, chunks):
    """O0: current retrieval, O1: gold answer page, O2: all gold support pages."""
    from run_experiment import _load_chunks_and_indices

    data = _load_chunks_and_indices(config, "frames")
    page_chunks = data["chunks"]

    id_to_sample = {s["id"]: s for s in samples}

    # O0: load existing predictions
    existing = RESULTS_DIR / "predictions_naive_rag_vega_kg_frames_qwen3emb.json"
    o0_results = []
    if existing.exists():
        preds = load_json(existing)
        for p in preds:
            s = id_to_sample.get(p["id"])
            hops = s.get("num_hops", 0) if s else 0
            o0_results.append({
                "id": p["id"],
                "question": p.get("question", ""),
                "gold": str(p.get("gold", "")),
                "prediction": str(p.get("prediction", "")),
                "hops": hops,
            })

    # Build O1 and O2 contexts
    o1_items = []  # gold answer page only (first evidence page)
    o2_items = []  # all gold supporting pages

    for s in samples:
        hops = s.get("num_hops", 0)
        evidence_pages = s.get("evidence_pages", [])
        unique_pages = s.get("unique_evidence_pages", evidence_pages)

        # O1: first evidence page (answer page)
        o1_context = ""
        if evidence_pages:
            chunk_id = evidence_page_to_chunk_id(evidence_pages[0])
            if chunk_id and chunk_id in page_chunks:
                o1_context = page_chunks[chunk_id]["text"]

        # O2: all unique evidence pages
        o2_texts = []
        for ep in unique_pages:
            chunk_id = evidence_page_to_chunk_id(ep)
            if chunk_id and chunk_id in page_chunks:
                o2_texts.append(page_chunks[chunk_id]["text"])
        o2_context = "\n\n".join(o2_texts)

        o1_items.append({
            "id": s["id"],
            "question": s["question"],
            "gold": str(s["answer"]),
            "context": o1_context,
            "hops": hops,
        })
        o2_items.append({
            "id": s["id"],
            "question": s["question"],
            "gold": str(s["answer"]),
            "context": o2_context,
            "hops": hops,
        })

    # Run O1
    print("\n  --- O1: Gold answer page only ---")
    t0 = time.time()
    prompts = [format_qa_prompt(item["question"], item["context"]) for item in o1_items]
    responses = reader.generate_batch(prompts)
    print(f"    VLM: {time.time()-t0:.1f}s")

    o1_results = []
    for item, resp in zip(o1_items, responses):
        o1_results.append({
            "id": item["id"], "question": item["question"],
            "gold": item["gold"], "prediction": resp.strip(), "hops": item["hops"],
        })
    save_json(o1_results, OUTPUT_DIR / "predictions_O1_gold_answer_page.json")

    # Run O2
    print("\n  --- O2: All gold supporting pages ---")
    t0 = time.time()
    prompts = [format_qa_prompt(item["question"], item["context"]) for item in o2_items]
    responses = reader.generate_batch(prompts)
    print(f"    VLM: {time.time()-t0:.1f}s")

    o2_results = []
    for item, resp in zip(o2_items, responses):
        o2_results.append({
            "id": item["id"], "question": item["question"],
            "gold": item["gold"], "prediction": resp.strip(), "hops": item["hops"],
        })
    save_json(o2_results, OUTPUT_DIR / "predictions_O2_gold_support_chain.json")

    return {"O0": o0_results, "O1": o1_results, "O2": o2_results}


# ============================================================
# STEP 4: Support-chain coverage diagnostics (no VLM)
# ============================================================
def run_coverage_diagnostics(config, samples, chunks):
    """Compute support-chain coverage metrics for flat_chunk, vega_kg, lightrag."""
    from run_experiment import _load_chunks_and_indices, _setup_vega_kg

    data = _load_chunks_and_indices(config, "frames")
    page_chunks = data["chunks"]

    # Setup retrievers
    from src.retrieval.retriever import FlatChunkRetriever
    from src.retrieval.baseline_backends import LightRAGRetriever

    bm25, dense = data["bm25"], data["dense"]
    token_budget = config["retrieval"]["token_budget"]

    retrievers = {
        "flat_chunk": FlatChunkRetriever(bm25, dense, page_chunks, token_budget),
        "vega_kg": _setup_vega_kg(config, "frames"),
    }

    # lightrag
    kg_dir = Path(config["paths"]["kg"])
    checkpoint = kg_dir / "frames_checkpoint.json"
    assertions = []
    if checkpoint.exists():
        ckpt = load_json(str(checkpoint))
        assertions = ckpt.get("assertions", [])
    retrievers["lightrag"] = LightRAGRetriever(bm25, dense, page_chunks, assertions, token_budget)

    coverage_results = {}

    for backend_name, retriever in retrievers.items():
        print(f"\n  --- Coverage: {backend_name} ---")
        t0 = time.time()

        per_sample = []
        for s in samples:
            hops = s.get("num_hops", 0)
            query = s["question"]

            # Get retrieved chunk IDs
            evidence = retriever.retrieve(query, top_k=config["retrieval"]["seed_top_k"])
            if isinstance(evidence, list):
                if evidence and isinstance(evidence[0], dict):
                    retrieved_ids = [e.get("id", "") for e in evidence]
                else:
                    retrieved_ids = []
            else:
                retrieved_ids = []

            retrieved_set = set(retrieved_ids[:10])
            retrieved_set_20 = set(retrieved_ids[:20])

            # Gold evidence pages -> chunk IDs
            gold_pages = s.get("unique_evidence_pages", s.get("evidence_pages", []))
            gold_chunk_ids = set()
            for ep in gold_pages:
                cid = evidence_page_to_chunk_id(ep)
                if cid and cid in page_chunks:
                    gold_chunk_ids.add(cid)

            # Gold supporting docs
            gold_docs = set(s.get("supporting_doc_ids", []))

            # Retrieved docs
            retrieved_docs = set()
            for rid in retrieved_ids[:10]:
                parts = rid.rsplit("_p", 1)
                if len(parts) == 2:
                    retrieved_docs.add(parts[0])

            retrieved_docs_20 = set()
            for rid in retrieved_ids[:20]:
                parts = rid.rsplit("_p", 1)
                if len(parts) == 2:
                    retrieved_docs_20.add(parts[0])

            # Metrics
            # answer_doc_recall@10: is the first evidence page's doc retrieved?
            answer_page = s.get("evidence_pages", [""])[0] if s.get("evidence_pages") else ""
            answer_chunk = evidence_page_to_chunk_id(answer_page)
            answer_doc = answer_page.rsplit("_", 1)[0] if "_" in answer_page else ""
            answer_doc_hit = 1 if answer_doc in retrieved_docs else 0

            # any_support_doc_recall@10
            if gold_docs:
                any_support_hit = 1 if (gold_docs & retrieved_docs) else 0
            else:
                any_support_hit = 0

            # all_support_docs_covered@10
            if gold_docs:
                # Only count docs that are in the index
                indexable_gold_docs = set()
                for gd in gold_docs:
                    if any(k.startswith(gd + "_p") for k in page_chunks):
                        indexable_gold_docs.add(gd)
                if indexable_gold_docs:
                    all_covered_10 = 1 if indexable_gold_docs.issubset(retrieved_docs) else 0
                    all_covered_20 = 1 if indexable_gold_docs.issubset(retrieved_docs_20) else 0
                else:
                    all_covered_10 = 0
                    all_covered_20 = 0
            else:
                all_covered_10 = 0
                all_covered_20 = 0

            # non_answer_support_doc_recall@10
            non_answer_gold = gold_docs - {answer_doc}
            indexable_non_answer = set()
            for gd in non_answer_gold:
                if any(k.startswith(gd + "_p") for k in page_chunks):
                    indexable_non_answer.add(gd)
            if indexable_non_answer:
                non_answer_recall = len(indexable_non_answer & retrieved_docs) / len(indexable_non_answer)
            else:
                non_answer_recall = 0

            # support_diversity@10 (unique docs in top-10)
            diversity = len(retrieved_docs)

            # same_doc_concentration@10 (fraction of top-10 from most common doc)
            doc_counts = defaultdict(int)
            for rid in retrieved_ids[:10]:
                parts = rid.rsplit("_p", 1)
                if len(parts) == 2:
                    doc_counts[parts[0]] += 1
            max_conc = max(doc_counts.values()) / max(len(retrieved_ids[:10]), 1) if doc_counts else 0

            # gold_page_recall@10
            if gold_chunk_ids:
                gold_page_recall = len(gold_chunk_ids & retrieved_set) / len(gold_chunk_ids)
            else:
                gold_page_recall = 0

            per_sample.append({
                "id": s["id"],
                "hops": hops,
                "n_gold_docs": len(gold_docs),
                "n_indexable_gold_docs": len(indexable_gold_docs) if gold_docs else 0,
                "n_gold_pages": len(gold_chunk_ids),
                "answer_doc_recall": answer_doc_hit,
                "any_support_doc_recall": any_support_hit,
                "all_support_docs_covered_10": all_covered_10,
                "all_support_docs_covered_20": all_covered_20,
                "non_answer_support_doc_recall": non_answer_recall,
                "support_diversity": diversity,
                "same_doc_concentration": max_conc,
                "gold_page_recall": gold_page_recall,
            })

        elapsed = time.time() - t0
        print(f"    Retrieval time: {elapsed:.1f}s")

        # Aggregate by hop bucket
        hop_buckets = {
            "2-hop": {2}, "3-hop": {3}, "4+hop": {4, 5}, "2+hop": {2, 3, 4, 5}, "all": {1, 2, 3, 4, 5},
        }
        metrics_keys = [
            "answer_doc_recall", "any_support_doc_recall",
            "all_support_docs_covered_10", "all_support_docs_covered_20",
            "non_answer_support_doc_recall", "support_diversity",
            "same_doc_concentration", "gold_page_recall",
        ]

        backend_summary = {}
        for bname, hops_set in hop_buckets.items():
            bucket = [p for p in per_sample if p["hops"] in hops_set]
            if not bucket:
                continue
            agg = {"n": len(bucket)}
            for mk in metrics_keys:
                vals = [p[mk] for p in bucket]
                agg[mk] = sum(vals) / len(vals)
            backend_summary[bname] = agg

        coverage_results[backend_name] = backend_summary
        save_json(per_sample, OUTPUT_DIR / f"coverage_{backend_name}.json")

        # Print
        for bname, agg in backend_summary.items():
            print(f"    {bname} (n={agg['n']}): "
                  f"ans_doc={agg['answer_doc_recall']:.3f}, "
                  f"any_sup={agg['any_support_doc_recall']:.3f}, "
                  f"all_sup@10={agg['all_support_docs_covered_10']:.3f}, "
                  f"all_sup@20={agg['all_support_docs_covered_20']:.3f}, "
                  f"non_ans_recall={agg['non_answer_support_doc_recall']:.3f}, "
                  f"diversity={agg['support_diversity']:.1f}, "
                  f"concentration={agg['same_doc_concentration']:.3f}")

    return coverage_results


def main():
    config = load_config("config/default.yaml")
    config["models"]["text_embedder"] = "Qwen/Qwen3-Embedding-0.6B"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = load_frames_samples()

    # Load chunks
    chunks = load_json(f"data/indices/frames_chunks.json")

    # Step 0: Hop distribution
    print("=" * 70)
    print("STEP 0: Hop Distribution Verification")
    print("=" * 70)
    hop_dist = defaultdict(int)
    for s in samples:
        hop_dist[s.get("num_hops", 0)] += 1
    for h in sorted(hop_dist):
        print(f"  {h}-hop: {hop_dist[h]} ({hop_dist[h]/len(samples)*100:.1f}%)")
    n_2plus = sum(v for k, v in hop_dist.items() if k >= 2)
    print(f"  2+hop: {n_2plus} ({n_2plus/len(samples)*100:.1f}%)")

    # Check indexable evidence coverage
    total_ev = 0
    indexable_ev = 0
    for s in samples:
        for ep in s.get("unique_evidence_pages", []):
            total_ev += 1
            cid = evidence_page_to_chunk_id(ep)
            if cid and cid in chunks:
                indexable_ev += 1
    print(f"\n  Evidence page coverage: {indexable_ev}/{total_ev} "
          f"({indexable_ev/total_ev*100:.1f}%) pages indexable")

    # Setup reader (shared across experiments)
    reader = VLLMClient(
        base_url="http://localhost:8001/v1",
        model=config["models"]["reader"],
        max_tokens=512,
    )

    # Step 4: Coverage diagnostics (no VLM, run first)
    print("\n" + "=" * 70)
    print("STEP 4: Support-chain Coverage Diagnostics")
    print("=" * 70)
    coverage_results = run_coverage_diagnostics(config, samples, chunks)
    save_json(coverage_results, OUTPUT_DIR / "coverage_summary.json")

    # Step 2: Retrieval sweep
    print("\n" + "=" * 70)
    print("STEP 2: Retrieval Sweep F0-F3")
    print("=" * 70)
    sweep_results = run_retrieval_sweep(config, samples, reader, chunks)

    # Step 3: Oracle experiments
    print("\n" + "=" * 70)
    print("STEP 3: Gold Oracle Experiments O0-O2")
    print("=" * 70)
    oracle_results = run_oracle_experiments(config, samples, reader, chunks)

    # ============================================================
    # FINAL REPORT
    # ============================================================
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    # Sweep table
    print("\n--- Retrieval Sweep (vega_kg, 2+hop focused) ---")
    header = f"{'Config':<6s}  {'2-hop EM':>10s}  {'3-hop EM':>10s}  {'4+hop EM':>10s}  {'2+hop EM':>10s}  {'2+hop ANLS':>12s}"
    print(header)
    print("-" * len(header))
    for sweep_id in ["F0", "F1", "F2", "F3"]:
        if sweep_id not in sweep_results:
            continue
        hm = compute_hop_metrics(sweep_results[sweep_id])
        row = f"{sweep_id:<6s}"
        for bk in ["2-hop", "3-hop", "4+hop", "2+hop"]:
            m = hm.get(bk, {"EM": 0, "n": 0})
            row += f"  {m['EM']:.3f}({m['n']:>3d})"
        m2p = hm.get("2+hop", {"ANLS": 0})
        row += f"  {m2p['ANLS']:.4f}"
        print(row)

    # Oracle table
    print("\n--- Gold Oracle (2+hop focused) ---")
    header = f"{'Oracle':<6s}  {'2-hop EM':>10s}  {'3-hop EM':>10s}  {'4+hop EM':>10s}  {'2+hop EM':>10s}  {'2+hop ANLS':>12s}"
    print(header)
    print("-" * len(header))
    for oid in ["O0", "O1", "O2"]:
        if oid not in oracle_results:
            continue
        hm = compute_hop_metrics(oracle_results[oid])
        row = f"{oid:<6s}"
        for bk in ["2-hop", "3-hop", "4+hop", "2+hop"]:
            m = hm.get(bk, {"EM": 0, "n": 0})
            row += f"  {m['EM']:.3f}({m['n']:>3d})"
        m2p = hm.get("2+hop", {"ANLS": 0})
        row += f"  {m2p['ANLS']:.4f}"
        print(row)

    # Coverage table
    print("\n--- Support-chain Coverage (2+hop) ---")
    header = (f"{'Backend':<14s}  {'ans_doc':>8s}  {'any_sup':>8s}  {'all@10':>8s}  "
              f"{'all@20':>8s}  {'non_ans':>8s}  {'divers':>7s}  {'conc':>6s}  {'pg_rec':>7s}")
    print(header)
    print("-" * len(header))
    for backend in ["flat_chunk", "vega_kg", "lightrag"]:
        if backend not in coverage_results:
            continue
        agg = coverage_results[backend].get("2+hop", {})
        if not agg:
            continue
        print(f"{backend:<14s}  "
              f"{agg.get('answer_doc_recall', 0):.3f}     "
              f"{agg.get('any_support_doc_recall', 0):.3f}     "
              f"{agg.get('all_support_docs_covered_10', 0):.3f}     "
              f"{agg.get('all_support_docs_covered_20', 0):.3f}     "
              f"{agg.get('non_answer_support_doc_recall', 0):.3f}     "
              f"{agg.get('support_diversity', 0):.1f}     "
              f"{agg.get('same_doc_concentration', 0):.3f}  "
              f"{agg.get('gold_page_recall', 0):.3f}")

    save_json({
        "sweep": {k: compute_hop_metrics(v) for k, v in sweep_results.items()},
        "oracle": {k: compute_hop_metrics(v) for k, v in oracle_results.items()},
        "coverage": coverage_results,
    }, OUTPUT_DIR / "final_summary.json")

    print(f"\nAll results saved to {OUTPUT_DIR}")

    from src.utils.io_utils import shutdown_vllm
    shutdown_vllm()


if __name__ == "__main__":
    main()
