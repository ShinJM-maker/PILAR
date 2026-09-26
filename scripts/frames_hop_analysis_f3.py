"""Frames hop-level analysis with F3 config (adj=0, mult=4).

Loads F3 predictions for all 6 backends, joins with samples for hop info,
and computes metrics by hop bucket.

Usage:
    PYTHONPATH=. python scripts/frames_hop_analysis_f3.py
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")

from src.utils.io_utils import load_json, save_json
from src.evaluation.metrics import exact_match, anls_score, rouge_l_score, meteor_score_single

RESULTS_DIR = Path("data/results/frames_recompute")
OUTPUT_DIR = Path("data/results/frames_hop_f3")

BACKENDS = ["vega_kg", "flat_chunk", "vega_page_only", "ms_graphrag", "lightrag", "simpledoc"]


def compute_metrics_list(results):
    ems, anlss, rouges, meteors = [], [], [], []
    for r in results:
        pred = str(r.get("prediction", ""))
        gold = str(r.get("gold", ""))
        ems.append(exact_match(pred, gold))
        anlss.append(anls_score(pred, gold))
        rouges.append(rouge_l_score(pred, gold))
        meteors.append(meteor_score_single(pred, gold))
    n = len(results)
    if n == 0:
        return {"EM": 0, "ANLS": 0, "ROUGE-L": 0, "METEOR": 0, "n": 0}
    return {
        "EM": sum(ems) / n,
        "ANLS": sum(anlss) / n,
        "ROUGE-L": sum(rouges) / n,
        "METEOR": sum(meteors) / n,
        "n": n,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load samples for hop info
    samples = load_json("dataset/frames/samples.json")
    id_to_hops = {}
    for s in samples:
        id_to_hops[s["id"]] = s.get("num_hops", 0)

    print(f"Loaded {len(samples)} samples")

    # Hop distribution
    hop_dist = defaultdict(int)
    for h in id_to_hops.values():
        hop_dist[h] += 1
    for h in sorted(hop_dist):
        print(f"  {h}-hop: {hop_dist[h]} ({hop_dist[h]/len(samples)*100:.1f}%)")

    hop_buckets = {
        "1-hop": {1},
        "2-hop": {2},
        "3-hop": {3},
        "4-hop": {4},
        "5-hop": {5},
        "4+hop": {4, 5},
        "2+hop": {2, 3, 4, 5},
        "all": {1, 2, 3, 4, 5},
    }

    all_results = {}

    for backend in BACKENDS:
        # Load F3 predictions
        pred_path = RESULTS_DIR / f"predictions_{backend}_F3.json"
        if not pred_path.exists():
            print(f"  SKIP {backend} (no F3 predictions)")
            continue

        preds = load_json(pred_path)

        # Join hop info
        for p in preds:
            p["hops"] = id_to_hops.get(p["id"], 0)

        # Compute per-hop metrics
        backend_metrics = {}
        for bucket_name, hops_set in hop_buckets.items():
            filtered = [p for p in preds if p["hops"] in hops_set]
            m = compute_metrics_list(filtered)
            backend_metrics[bucket_name] = m

        all_results[backend] = backend_metrics

        print(f"\n  {backend}:")
        for bucket_name in ["1-hop", "2-hop", "3-hop", "4-hop", "5-hop", "2+hop", "all"]:
            m = backend_metrics[bucket_name]
            print(f"    {bucket_name:>6s}: EM={m['EM']:.3f}, ANLS={m['ANLS']:.3f}, "
                  f"ROUGE-L={m['ROUGE-L']:.3f} (n={m['n']})")

    # Also load E011 predictions for comparison
    print("\n\n--- E011 (old config) comparison ---")
    e011_results = {}
    for backend in BACKENDS:
        pred_path = Path(f"data/results/predictions_naive_rag_{backend}_frames_qwen3emb.json")
        if not pred_path.exists():
            continue
        preds = load_json(pred_path)
        for p in preds:
            p["hops"] = id_to_hops.get(p["id"], 0)

        backend_metrics = {}
        for bucket_name, hops_set in hop_buckets.items():
            filtered = [p for p in preds if p["hops"] in hops_set]
            backend_metrics[bucket_name] = compute_metrics_list(filtered)

        e011_results[backend] = backend_metrics

    # Print comparison table
    print("\n" + "=" * 90)
    print("HOP-LEVEL COMPARISON: E011 vs F3 (EM)")
    print("=" * 90)

    for hop_name in ["1-hop", "2-hop", "3-hop", "4-hop", "5-hop", "2+hop", "all"]:
        print(f"\n  --- {hop_name} ---")
        header = f"  {'Backend':<18s}  {'E011 EM':>8s}  {'F3 EM':>8s}  {'ΔEM':>8s}  {'E011 ANLS':>10s}  {'F3 ANLS':>10s}  {'ΔANLS':>8s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for backend in BACKENDS:
            if backend not in all_results or backend not in e011_results:
                continue
            f3_m = all_results[backend].get(hop_name, {"EM": 0, "ANLS": 0, "n": 0})
            e011_m = e011_results[backend].get(hop_name, {"EM": 0, "ANLS": 0, "n": 0})
            d_em = f3_m["EM"] - e011_m["EM"]
            d_anls = f3_m["ANLS"] - e011_m["ANLS"]
            print(f"  {backend:<18s}  {e011_m['EM']:.3f}     {f3_m['EM']:.3f}     {d_em:+.3f}     "
                  f"{e011_m['ANLS']:.4f}      {f3_m['ANLS']:.4f}      {d_anls:+.4f}")

    # Delta vs flat_chunk (F3)
    print("\n" + "=" * 90)
    print("DELTA VS FLAT_CHUNK (F3)")
    print("=" * 90)

    if "flat_chunk" in all_results:
        for hop_name in ["1-hop", "2-hop", "3-hop", "4-hop", "5-hop", "2+hop", "all"]:
            fc = all_results["flat_chunk"].get(hop_name, {"EM": 0, "ANLS": 0})
            print(f"\n  --- {hop_name} (flat_chunk EM={fc['EM']:.3f}) ---")
            for backend in BACKENDS:
                if backend == "flat_chunk" or backend not in all_results:
                    continue
                m = all_results[backend].get(hop_name, {"EM": 0, "ANLS": 0})
                d_em = m["EM"] - fc["EM"]
                d_anls = m["ANLS"] - fc["ANLS"]
                print(f"    {backend:<18s}: ΔEM={d_em:+.3f}, ΔANLS={d_anls:+.4f}")

    save_json({"f3": all_results, "e011": e011_results}, OUTPUT_DIR / "hop_analysis_summary.json")
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
