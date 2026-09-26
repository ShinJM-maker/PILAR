"""Re-run Frames full-set with Frames-tuned config (F1/F3).

Runs naive_rag × all backends on full 507 Frames samples with:
  - Baseline: E011 config (current, adjacent=0.04, multiplier=2)
  - F1: adjacent=0, multiplier=2
  - F3: adjacent=0, multiplier=4

Also re-runs flat_chunk with multiplier=4 for fair comparison.

Usage:
    PYTHONPATH=. python scripts/run_frames_recompute.py
"""
import json
import pickle
import time
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, ".")

from src.utils.io_utils import load_config, load_json, save_json
from src.evaluation.metrics import exact_match, anls_score, rouge_l_score, meteor_score_single
from src.reader.llm_client import VLLMClient
from src.reader.prompt_templates import format_qa_prompt
from run_experiment import _load_chunks_and_indices

RESULTS_DIR = Path("data/results")
OUTPUT_DIR = RESULTS_DIR / "frames_recompute"

BACKENDS_NO_KG = ["flat_chunk", "simpledoc", "vega_page_only"]


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


def run_backend(retriever, samples, reader, config):
    """Run retrieval + VLM for a backend."""
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

    prompts = [format_qa_prompt(item["question"], item["context"]) for item in samples_with_context]
    responses = reader.generate_batch(prompts)

    results = []
    for item, resp in zip(samples_with_context, responses):
        results.append({
            "id": item["id"],
            "question": item["question"],
            "gold": item["gold"],
            "prediction": resp.strip(),
        })
    return results


def main():
    config = load_config("config/default.yaml")
    config["models"]["text_embedder"] = "Qwen/Qwen3-Embedding-0.6B"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load full Frames dataset
    samples = load_json(config["paths"]["samples_frames"])
    for s in samples:
        if "supporting_doc_ids" not in s and "filtered_supporting_doc_ids" in s:
            s["supporting_doc_ids"] = s["filtered_supporting_doc_ids"]
    print(f"Loaded {len(samples)} Frames samples")

    # Load indices
    data = _load_chunks_and_indices(config, "frames")
    page_chunks, page_bm25, page_dense = data["chunks"], data["bm25"], data["dense"]

    # Load KG for vega_kg
    from src.kg.graph_builder import HybridKGBuilder
    from src.retrieval.retriever import VEGAKGRetriever, FlatChunkRetriever, VEGAPageOnlyRetriever
    from src.retrieval.baseline_backends import MultiDocFusionRetriever, MSGraphRAGRetriever, LightRAGRetriever

    kg_path = Path(config["paths"]["kg"]) / "frames_kg.pkl"
    kg = HybridKGBuilder.load(str(kg_path))
    entities_dict = {eid: e.to_dict() for eid, e in kg.entities.items()}
    assertions_dict = {aid: a.to_dict() for aid, a in kg.assertions.items()}
    print(f"KG: {len(kg.supports)} supports, {len(kg.entities)} entities")

    # Setup ms_graphrag entities
    kg_dir = Path(config["paths"]["kg"])
    entities_path = kg_dir / "frames_entities.json"
    ms_entities = load_json(entities_path) if entities_path.exists() else {}

    # Setup lightrag assertions
    checkpoint = kg_dir / "frames_checkpoint.json"
    lr_assertions = []
    if checkpoint.exists():
        ckpt = load_json(str(checkpoint))
        lr_assertions = ckpt.get("assertions", [])

    # Setup reader
    reader = VLLMClient(
        base_url="http://localhost:8001/v1",
        model=config["models"]["reader"],
        max_tokens=512,
    )

    token_budget = config["retrieval"]["token_budget"]

    # Define configs to test
    configs = {
        "F1": {  # adjacent=0, multiplier=2
            "initial_multiplier": 2,
            "adjacent_page_bonus": 0.00,
            "same_section_bonus": 0.00,
        },
        "F3": {  # adjacent=0, multiplier=4
            "initial_multiplier": 4,
            "adjacent_page_bonus": 0.00,
            "same_section_bonus": 0.00,
        },
    }

    all_summary = {}

    for cfg_name, overrides in configs.items():
        print(f"\n{'='*70}")
        print(f"Config: {cfg_name} — {overrides}")
        print(f"{'='*70}")

        multiplier = overrides["initial_multiplier"]

        # vega_kg
        ret_config = dict(config["retrieval"])
        ret_config.update(overrides)
        vega_retriever = VEGAKGRetriever(
            kg.graph, kg.supports,
            page_bm25, page_dense, page_chunks,
            entities_dict, assertions_dict,
            ret_config,
        )

        print(f"\n  Backend: vega_kg ({cfg_name})")
        t0 = time.time()
        results = run_backend(vega_retriever, samples, reader, config)
        metrics = compute_metrics(results)
        metrics["time"] = time.time() - t0
        print(f"    EM={metrics['EM']:.4f}, ANLS={metrics['ANLS']:.4f}, "
              f"ROUGE-L={metrics['ROUGE-L']:.4f} ({metrics['time']:.1f}s)")
        all_summary[f"vega_kg_{cfg_name}"] = metrics
        save_json(results, OUTPUT_DIR / f"predictions_vega_kg_{cfg_name}.json")

        # flat_chunk (same multiplier via seed_top_k override)
        flat_retriever = FlatChunkRetriever(page_bm25, page_dense, page_chunks, token_budget)
        print(f"\n  Backend: flat_chunk")
        t0 = time.time()
        results = run_backend(flat_retriever, samples, reader, config)
        metrics = compute_metrics(results)
        metrics["time"] = time.time() - t0
        print(f"    EM={metrics['EM']:.4f}, ANLS={metrics['ANLS']:.4f}, "
              f"ROUGE-L={metrics['ROUGE-L']:.4f} ({metrics['time']:.1f}s)")
        all_summary[f"flat_chunk_{cfg_name}"] = metrics
        save_json(results, OUTPUT_DIR / f"predictions_flat_chunk_{cfg_name}.json")

        # vega_page_only (same multiplier)
        page_only_retriever = VEGAPageOnlyRetriever(
            page_bm25, page_dense, page_chunks, token_budget,
            initial_multiplier=multiplier,
        )
        print(f"\n  Backend: vega_page_only (multiplier={multiplier})")
        t0 = time.time()
        results = run_backend(page_only_retriever, samples, reader, config)
        metrics = compute_metrics(results)
        metrics["time"] = time.time() - t0
        print(f"    EM={metrics['EM']:.4f}, ANLS={metrics['ANLS']:.4f}, "
              f"ROUGE-L={metrics['ROUGE-L']:.4f} ({metrics['time']:.1f}s)")
        all_summary[f"vega_page_only_{cfg_name}"] = metrics
        save_json(results, OUTPUT_DIR / f"predictions_vega_page_only_{cfg_name}.json")

        # ms_graphrag
        ms_retriever = MSGraphRAGRetriever(page_bm25, page_dense, page_chunks, ms_entities, token_budget)
        print(f"\n  Backend: ms_graphrag")
        t0 = time.time()
        results = run_backend(ms_retriever, samples, reader, config)
        metrics = compute_metrics(results)
        metrics["time"] = time.time() - t0
        print(f"    EM={metrics['EM']:.4f}, ANLS={metrics['ANLS']:.4f}, "
              f"ROUGE-L={metrics['ROUGE-L']:.4f} ({metrics['time']:.1f}s)")
        all_summary[f"ms_graphrag_{cfg_name}"] = metrics
        save_json(results, OUTPUT_DIR / f"predictions_ms_graphrag_{cfg_name}.json")

        # lightrag
        lr_retriever = LightRAGRetriever(page_bm25, page_dense, page_chunks, lr_assertions, token_budget)
        print(f"\n  Backend: lightrag")
        t0 = time.time()
        results = run_backend(lr_retriever, samples, reader, config)
        metrics = compute_metrics(results)
        metrics["time"] = time.time() - t0
        print(f"    EM={metrics['EM']:.4f}, ANLS={metrics['ANLS']:.4f}, "
              f"ROUGE-L={metrics['ROUGE-L']:.4f} ({metrics['time']:.1f}s)")
        all_summary[f"lightrag_{cfg_name}"] = metrics
        save_json(results, OUTPUT_DIR / f"predictions_lightrag_{cfg_name}.json")

        # multidocfusion (formerly simpledoc)
        sd_retriever = MultiDocFusionRetriever(page_bm25, page_dense, page_chunks, token_budget)
        print(f"\n  Backend: simpledoc")
        t0 = time.time()
        results = run_backend(sd_retriever, samples, reader, config)
        metrics = compute_metrics(results)
        metrics["time"] = time.time() - t0
        print(f"    EM={metrics['EM']:.4f}, ANLS={metrics['ANLS']:.4f}, "
              f"ROUGE-L={metrics['ROUGE-L']:.4f} ({metrics['time']:.1f}s)")
        all_summary[f"simpledoc_{cfg_name}"] = metrics
        save_json(results, OUTPUT_DIR / f"predictions_simpledoc_{cfg_name}.json")

    # Print comparison
    print(f"\n{'='*70}")
    print("FRAMES FULL-SET COMPARISON")
    print(f"{'='*70}")

    # Include existing E011 results
    print("\n--- Existing E011 results (from docs/frames_qwen3emb_results.md) ---")
    print("  vega_kg:       EM=0.1262, ANLS=0.1607, ROUGE-L=0.1902")
    print("  flat_chunk:    EM=0.1262, ANLS=0.1589, ROUGE-L=0.1868")
    print("  ms_graphrag:   EM=0.1282, ANLS=0.1595, ROUGE-L=0.1883")
    print("  lightrag:      EM=0.1243, ANLS=0.1533, ROUGE-L=0.1819")
    print("  simpledoc:     EM=0.1243, ANLS=0.1548, ROUGE-L=0.1811")

    for cfg_name in ["F1", "F3"]:
        print(f"\n--- {cfg_name} results ---")
        header = f"{'Backend':<20s}  {'EM':>8s}  {'ANLS':>8s}  {'ROUGE-L':>8s}"
        print(header)
        print("-" * len(header))
        for backend in ["vega_kg", "flat_chunk", "vega_page_only", "ms_graphrag", "lightrag", "simpledoc"]:
            key = f"{backend}_{cfg_name}"
            if key in all_summary:
                m = all_summary[key]
                print(f"{backend:<20s}  {m['EM']:.4f}  {m['ANLS']:.4f}  {m['ROUGE-L']:.4f}")

        # Delta vs flat_chunk
        fc_key = f"flat_chunk_{cfg_name}"
        if fc_key in all_summary:
            print(f"\n  Delta vs flat_chunk ({cfg_name}):")
            fc = all_summary[fc_key]
            for backend in ["vega_kg", "vega_page_only", "ms_graphrag", "lightrag", "simpledoc"]:
                key = f"{backend}_{cfg_name}"
                if key in all_summary:
                    m = all_summary[key]
                    d_em = m["EM"] - fc["EM"]
                    d_anls = m["ANLS"] - fc["ANLS"]
                    print(f"    {backend:<20s}: ΔEM={d_em:+.4f}, ΔANLS={d_anls:+.4f}")

    save_json(all_summary, OUTPUT_DIR / "frames_recompute_summary.json")
    print(f"\nAll results saved to {OUTPUT_DIR}")

    from src.utils.io_utils import shutdown_vllm
    shutdown_vllm()


if __name__ == "__main__":
    main()
