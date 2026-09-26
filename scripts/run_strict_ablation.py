"""Run strict matched-interface ablation for Table 2.

3 missing rows:
  - Page-only + local prior (local=on, KG=off)
  - Text-KG only (local=off, visual=off, gate=off)
  - Text-KG + local prior (local=on, visual=off, gate=off)

All share: naive_rag, M3DocVQA 500-sample subset (seed=42), qwen3emb.

Usage:
    PYTHONPATH=. python scripts/run_strict_ablation.py
"""
import json
import pickle
import random
import time
import sys
import copy
import networkx as nx
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, ".")

from src.utils.io_utils import load_config, load_json, save_json
from src.evaluation.metrics import exact_match, anls_score, rouge_l_score, meteor_score_single
from src.reader.llm_client import VLLMClient
from src.reader.prompt_templates import format_qa_prompt
from src.kg.graph_builder import HybridKGBuilder
from src.kg.schema import Modality, NodeType
from src.retrieval.retriever import VEGAKGRetriever
from src.retrieval.dense_index import DenseIndex

RESULTS_DIR = Path("data/results")
OUTPUT_DIR = RESULTS_DIR / "strict_ablation"


def compute_metrics(results):
    ems, anlss, rouges, meteors = [], [], [], []
    for r in results:
        pred = str(r.get("prediction", ""))
        gold = str(r.get("gold", ""))
        ems.append(exact_match(pred, gold))
        anlss.append(anls_score(pred, gold))
        rouges.append(rouge_l_score(pred, gold))
        meteors.append(meteor_score_single(pred, gold))
    n = len(results)
    return {
        "EM": sum(ems) / n if n else 0,
        "ANLS": sum(anlss) / n if n else 0,
        "ROUGE-L": sum(rouges) / n if n else 0,
        "METEOR": sum(meteors) / n if n else 0,
        "n": n,
    }


def filter_visual_assertions(kg):
    """Create a KG copy with visual assertions removed.
    Returns (filtered_graph, filtered_supports, filtered_entities, filtered_assertions_dict).
    """
    # Identify visual assertion IDs
    visual_aids = set()
    for aid, assertion in kg.assertions.items():
        if assertion.modality == Modality.VISUAL:
            visual_aids.add(aid)

    print(f"  Filtering: {len(visual_aids)} visual / {len(kg.assertions) - len(visual_aids)} text assertions")

    # Filter assertions dict (for retriever init)
    text_assertions = {}
    for aid, assertion in kg.assertions.items():
        if aid not in visual_aids:
            text_assertions[aid] = assertion

    # Filter supports: remove supports that are ONLY referenced by visual assertions
    # Build support -> assertion mapping
    support_to_assertions = defaultdict(set)
    for aid, assertion in kg.assertions.items():
        for sid in assertion.source_support_ids:
            support_to_assertions[sid].add(aid)

    # Keep supports that have at least one text assertion
    text_supports = {}
    for sid, sup in kg.supports.items():
        referencing_aids = support_to_assertions.get(sid, set())
        if referencing_aids - visual_aids:  # has at least one non-visual assertion
            text_supports[sid] = sup
        elif not referencing_aids:  # no assertions reference it
            text_supports[sid] = sup

    # Filter graph: remove visual assertion nodes and their edges
    filtered_graph = kg.graph.copy()
    for aid in visual_aids:
        if aid in filtered_graph:
            filtered_graph.remove_node(aid)

    print(f"  Text-only KG: {len(text_assertions)} assertions, "
          f"{len(text_supports)} supports, "
          f"{filtered_graph.number_of_nodes()} nodes, "
          f"{filtered_graph.number_of_edges()} edges")

    return filtered_graph, text_supports, text_assertions


def run_ablation(name, retriever, samples, reader, config):
    """Run a single ablation condition."""
    print(f"\n{'='*60}")
    print(f"Ablation: {name}")
    print(f"{'='*60}")

    t0 = time.time()

    # Phase 1: Retrieve
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
        })

    ret_time = time.time() - t0
    print(f"  Retrieval: {ret_time:.1f}s")

    # Phase 2: Batch VLM
    t1 = time.time()
    prompts = [format_qa_prompt(item["question"], item["context"]) for item in samples_with_context]
    responses = reader.generate_batch(prompts)
    vlm_time = time.time() - t1
    print(f"  VLM: {vlm_time:.1f}s")

    results = []
    for item, resp in zip(samples_with_context, responses):
        results.append({
            "id": item["id"],
            "question": item["question"],
            "gold": item["gold"],
            "prediction": resp.strip(),
        })

    metrics = compute_metrics(results)
    elapsed = time.time() - t0
    metrics["time_seconds"] = elapsed

    print(f"  EM={metrics['EM']:.4f}, ANLS={metrics['ANLS']:.4f}, "
          f"ROUGE-L={metrics['ROUGE-L']:.4f}, METEOR={metrics['METEOR']:.4f}")

    save_json(results, OUTPUT_DIR / f"predictions_{name}.json")
    save_json(metrics, OUTPUT_DIR / f"metrics_{name}.json")

    return metrics


def main():
    config = load_config("config/default.yaml")
    config["models"]["text_embedder"] = "Qwen/Qwen3-Embedding-0.6B"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset (500 random samples, seed=42)
    samples = load_json(config["paths"]["samples_m3docvqa"])
    for s in samples:
        if "supporting_doc_ids" not in s and "filtered_supporting_doc_ids" in s:
            s["supporting_doc_ids"] = s["filtered_supporting_doc_ids"]
    random.seed(42)
    samples = random.sample(samples, min(500, len(samples)))
    print(f"Loaded {len(samples)} samples")

    # Load KG
    kg_path = Path(config["paths"]["kg"]) / "m3docvqa_kg.pkl"
    kg = HybridKGBuilder.load(str(kg_path))
    print(f"KG: {len(kg.supports)} supports, {len(kg.assertions)} assertions, {len(kg.entities)} entities")

    # Load page-level indices
    from run_experiment import _load_chunks_and_indices
    data = _load_chunks_and_indices(config, "m3docvqa")
    page_chunks, page_bm25, page_dense = data["chunks"], data["bm25"], data["dense"]

    # Full entities/assertions dicts
    entities_dict = {eid: e.to_dict() for eid, e in kg.entities.items()}
    full_assertions_dict = {aid: a.to_dict() for aid, a in kg.assertions.items()}

    # Filter text-only assertions
    text_graph, text_supports, text_assertions_obj = filter_visual_assertions(kg)
    text_assertions_dict = {aid: a.to_dict() for aid, a in text_assertions_obj.items()}

    # Setup reader
    reader = VLLMClient(
        base_url="http://localhost:8001/v1",
        model=config["models"]["reader"],
        max_tokens=512,
    )

    all_metrics = {}

    # ============================================================
    # Row 2: Page-only + local prior
    # KG expansion disabled: max_matched_entities=0, expansion_max_nodes=0
    # Local bonus ON: adjacent_page_bonus=0.04
    # Gate OFF
    # ============================================================
    page_local_config = dict(config["retrieval"])
    page_local_config.update({
        "enable_local_bonus": True,
        "adjacent_page_bonus": 0.04,
        "same_section_bonus": 0.00,
        "max_matched_entities": 0,
        "expansion_max_nodes": 0,
        "alias_bonus": 0.0,
        "assertion_bonus": 0.0,
        "max_total_bonus": 0.0,
        "enable_expanded_gate": False,
    })

    # Use full KG objects (but they won't matter since expansion is disabled)
    page_local_retriever = VEGAKGRetriever(
        kg.graph, kg.supports,
        page_bm25, page_dense, page_chunks,
        entities_dict, full_assertions_dict,
        page_local_config,
    )
    all_metrics["page_local"] = run_ablation(
        "page_local", page_local_retriever, samples, reader, config
    )

    # ============================================================
    # Row 3: Text-KG only (no local, no visual, no gate)
    # Local OFF, Visual OFF, Gate OFF
    # ============================================================
    text_kg_config = dict(config["retrieval"])
    text_kg_config.update({
        "enable_local_bonus": False,
        "adjacent_page_bonus": 0.0,
        "same_section_bonus": 0.0,
        "enable_expanded_gate": False,
    })

    text_kg_retriever = VEGAKGRetriever(
        text_graph, text_supports,
        page_bm25, page_dense, page_chunks,
        entities_dict, text_assertions_dict,
        text_kg_config,
    )
    all_metrics["text_kg_only"] = run_ablation(
        "text_kg_only", text_kg_retriever, samples, reader, config
    )

    # ============================================================
    # Row 4: Text-KG + local prior (no visual, no gate)
    # Local ON, Visual OFF, Gate OFF
    # ============================================================
    text_kg_local_config = dict(config["retrieval"])
    text_kg_local_config.update({
        "enable_local_bonus": True,
        "adjacent_page_bonus": 0.04,
        "same_section_bonus": 0.00,
        "enable_expanded_gate": False,
    })

    text_kg_local_retriever = VEGAKGRetriever(
        text_graph, text_supports,
        page_bm25, page_dense, page_chunks,
        entities_dict, text_assertions_dict,
        text_kg_local_config,
    )
    all_metrics["text_kg_local"] = run_ablation(
        "text_kg_local", text_kg_local_retriever, samples, reader, config
    )

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("STRICT ABLATION SUMMARY")
    print("=" * 70)

    # Include existing values for complete table
    existing = {
        "page_only": {"EM": 0.320, "ANLS": 0.344, "ROUGE-L": 0.373},
        "hybrid_only": {"EM": 0.328, "ANLS": 0.356, "ROUGE-L": 0.382},
        "hybrid_local": {"EM": 0.334, "ANLS": 0.361, "ROUGE-L": 0.389},
        "final": {"EM": 0.336, "ANLS": 0.363, "ROUGE-L": 0.391},
    }

    rows = [
        ("Page-only", existing["page_only"], "No", "No", "No", "No"),
        ("Page-only + local", all_metrics["page_local"], "Yes", "No", "No", "No"),
        ("Text-KG only", all_metrics["text_kg_only"], "No", "Yes", "No", "No"),
        ("Text-KG + local", all_metrics["text_kg_local"], "Yes", "Yes", "No", "No"),
        ("Text+Visual-KG only", existing["hybrid_only"], "No", "Yes", "Yes", "No"),
        ("Text+Visual-KG + local", existing["hybrid_local"], "Yes", "Yes", "Yes", "No"),
        ("Final (+ gate)", existing["final"], "Yes", "Yes", "Yes", "Yes"),
    ]

    header = f"{'Variant':<26s} {'Local':>5s} {'TxtKG':>5s} {'Vis':>5s} {'Gate':>5s}  {'EM':>6s}  {'ANLS':>6s}  {'ROUGE':>6s}"
    print(header)
    print("-" * len(header))
    for name, m, local, tkg, vis, gate in rows:
        em = m.get("EM", 0)
        anls = m.get("ANLS", 0)
        rouge = m.get("ROUGE-L", 0)
        print(f"{name:<26s} {local:>5s} {tkg:>5s} {vis:>5s} {gate:>5s}  {em:.3f}   {anls:.3f}   {rouge:.3f}")

    # Incremental deltas
    print("\n--- Incremental Effects ---")
    p = existing["page_only"]
    pl = all_metrics["page_local"]
    tk = all_metrics["text_kg_only"]
    tkl = all_metrics["text_kg_local"]
    ho = existing["hybrid_only"]
    hl = existing["hybrid_local"]
    f = existing["final"]

    print(f"  Local prior effect (page-only → page+local):  ΔEM={pl['EM']-p['EM']:+.3f}, ΔANLS={pl['ANLS']-p['ANLS']:+.3f}")
    print(f"  Text-KG effect (page-only → text-KG):         ΔEM={tk['EM']-p['EM']:+.3f}, ΔANLS={tk['ANLS']-p['ANLS']:+.3f}")
    print(f"  Visual assertion (text-KG → text+vis-KG):      ΔEM={ho['EM']-tk['EM']:+.3f}, ΔANLS={ho['ANLS']-tk['ANLS']:+.3f}")
    print(f"  Visual assertion (text-KG+L → text+vis-KG+L):  ΔEM={hl['EM']-tkl['EM']:+.3f}, ΔANLS={hl['ANLS']-tkl['ANLS']:+.3f}")
    print(f"  Gate effect (text+vis+L → final):              ΔEM={f['EM']-hl['EM']:+.3f}, ΔANLS={f['ANLS']-hl['ANLS']:+.3f}")
    print(f"  Full KG effect (page-only → final):            ΔEM={f['EM']-p['EM']:+.3f}, ΔANLS={f['ANLS']-p['ANLS']:+.3f}")

    # LaTeX macro values (multiply by 100 for percentages)
    print("\n--- LaTeX Macro Values ---")
    print(f"  \\MMKGAblPageLocalEM{{{pl['EM']*100:.1f}}}")
    print(f"  \\MMKGAblPageLocalANLS{{{pl['ANLS']*100:.1f}}}")
    print(f"  \\MMKGAblPageLocalRouge{{{pl['ROUGE-L']*100:.1f}}}")
    print(f"  \\MMKGAblTextKGOnlyEM{{{tk['EM']*100:.1f}}}")
    print(f"  \\MMKGAblTextKGOnlyANLS{{{tk['ANLS']*100:.1f}}}")
    print(f"  \\MMKGAblTextKGOnlyRouge{{{tk['ROUGE-L']*100:.1f}}}")
    print(f"  \\MMKGAblTextKGLocalEM{{{tkl['EM']*100:.1f}}}")
    print(f"  \\MMKGAblTextKGLocalANLS{{{tkl['ANLS']*100:.1f}}}")
    print(f"  \\MMKGAblTextKGLocalRouge{{{tkl['ROUGE-L']*100:.1f}}}")

    save_json(all_metrics, OUTPUT_DIR / "ablation_summary.json")
    print(f"\nAll results saved to {OUTPUT_DIR}")

    from src.utils.io_utils import shutdown_vllm
    shutdown_vllm()


if __name__ == "__main__":
    main()
