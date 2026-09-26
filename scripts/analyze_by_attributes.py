"""Analyze existing experiment results by query attributes (hop, modality, type).

Uses prediction files + dataset metadata to break down EM by:
- Number of hops (1, 2, 3+)
- Modality (text, table, image)
- Question type (simple vs composed)
- Number of supporting docs
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")
from src.evaluation.metrics import normalize_answer

RESULTS_DIR = Path("data/results")
DATASET_PATH = Path("dataset/m3docvqa/samples.json")


def count_hops(qtype: str) -> int:
    if qtype in ("TextQ", "TableQ", "ImageQ", "ImageListQ"):
        return 1
    inner = re.findall(r"[A-Za-z]+Q", qtype)
    return len(inner) if inner else 1


def get_modality_group(modalities: list) -> str:
    """Classify into: text-only, table-only, image-only, multi-modal."""
    s = set(modalities)
    if s == {"text"}:
        return "text-only"
    elif s == {"table"}:
        return "table-only"
    elif s <= {"image", "imagelist"}:
        return "image-only"
    else:
        return "multi-modal"


def get_type_group(qtype: str) -> str:
    if qtype in ("TextQ", "TableQ", "ImageQ", "ImageListQ"):
        return "simple"
    elif qtype.startswith("Compose"):
        return "compose"
    elif qtype.startswith("Compare"):
        return "compare"
    elif qtype.startswith("Intersect"):
        return "intersect"
    return "other"


def compute_em(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def main():
    # Load dataset metadata
    samples = json.load(open(DATASET_PATH))
    id_to_meta = {}
    for s in samples:
        meta = s.get("metadata", {})
        qtype = meta.get("type", "unknown")
        modalities = meta.get("modalities", [])
        n_docs = len(s.get("supporting_doc_ids", []))
        id_to_meta[s["id"]] = {
            "hops": count_hops(qtype),
            "type": qtype,
            "type_group": get_type_group(qtype),
            "modalities": modalities,
            "modality_group": get_modality_group(modalities),
            "n_docs": n_docs,
        }

    # Experiments to analyze
    experiments = [
        ("naive_rag_vega_kg", "data/results/predictions_naive_rag_vega_kg_m3docvqa_qwen3emb.json"),
        ("naive_rag_flat_chunk", "data/results/predictions_naive_rag_flat_chunk_m3docvqa_qwen3emb.json"),
        ("naive_rag_ms_graphrag", "data/results/predictions_naive_rag_ms_graphrag_m3docvqa_qwen3emb.json"),
        ("naive_rag_lightrag", "data/results/predictions_naive_rag_lightrag_m3docvqa_qwen3emb.json"),
        ("naive_rag_simpledoc", "data/results/predictions_naive_rag_simpledoc_m3docvqa_qwen3emb.json"),
        ("react_vega_kg", "data/results/predictions_react_vega_kg_m3docvqa_qwen3emb.json"),
        ("react_flat_chunk", "data/results/predictions_react_flat_chunk_m3docvqa_qwen3emb.json"),
        ("autogen_vega_kg", "data/results/predictions_autogen_vega_kg_m3docvqa_qwen3emb.json"),
        ("autogen_flat_chunk", "data/results/predictions_autogen_flat_chunk_m3docvqa_qwen3emb.json"),
    ]

    all_results = {}

    for exp_name, pred_path in experiments:
        if not Path(pred_path).exists():
            print(f"SKIP: {pred_path} not found")
            continue

        preds = json.load(open(pred_path))

        # Group by attributes
        groups = {
            "by_hop": defaultdict(list),
            "by_modality": defaultdict(list),
            "by_type_group": defaultdict(list),
            "by_n_docs": defaultdict(list),
            "by_type": defaultdict(list),
        }

        for p in preds:
            qid = p["id"]
            meta = id_to_meta.get(qid)
            if meta is None:
                continue

            em = compute_em(str(p.get("prediction", "")), str(p.get("gold", "")))

            groups["by_hop"][meta["hops"]].append(em)
            groups["by_modality"][meta["modality_group"]].append(em)
            groups["by_type_group"][meta["type_group"]].append(em)
            groups["by_n_docs"][meta["n_docs"]].append(em)
            groups["by_type"][meta["type"]].append(em)

        # Compute means
        result = {}
        for group_name, group_data in groups.items():
            result[group_name] = {}
            for key in sorted(group_data.keys()):
                vals = group_data[key]
                result[group_name][str(key)] = {
                    "n": len(vals),
                    "EM": sum(vals) / len(vals) if vals else 0,
                }

        all_results[exp_name] = result

    # Save raw results
    output_dir = RESULTS_DIR / "attribute_analysis"
    output_dir.mkdir(exist_ok=True)
    json.dump(all_results, open(output_dir / "analysis_by_attributes.json", "w"), indent=2)

    # Print tables
    print("=" * 80)
    print("ANALYSIS BY HOP COUNT")
    print("=" * 80)

    hop_keys = ["1", "2", "3", "4"]
    header = f"{'Experiment':<30s}" + "".join(f"{'  ' + h + '-hop':>12s}" for h in hop_keys)
    print(header)
    print("-" * len(header))
    for exp_name in all_results:
        row = f"{exp_name:<30s}"
        for h in hop_keys:
            d = all_results[exp_name]["by_hop"].get(h, {"n": 0, "EM": 0})
            if d["n"] > 0:
                row += f"  {d['EM']:.3f}({d['n']:>3d})"
            else:
                row += "            "
        print(row)

    # Delta table (vega_kg - flat_chunk)
    print("\n" + "=" * 80)
    print("VEGA_KG vs FLAT_CHUNK DELTA BY HOP (naive_rag)")
    print("=" * 80)
    if "naive_rag_vega_kg" in all_results and "naive_rag_flat_chunk" in all_results:
        vk = all_results["naive_rag_vega_kg"]["by_hop"]
        fc = all_results["naive_rag_flat_chunk"]["by_hop"]
        for h in hop_keys:
            if h in vk and h in fc:
                delta = vk[h]["EM"] - fc[h]["EM"]
                print(f"  {h}-hop: vega_kg={vk[h]['EM']:.3f} flat_chunk={fc[h]['EM']:.3f} delta={delta:+.3f} (n={vk[h]['n']})")

    print("\n" + "=" * 80)
    print("ANALYSIS BY MODALITY")
    print("=" * 80)

    mod_keys = ["text-only", "table-only", "image-only", "multi-modal"]
    header = f"{'Experiment':<30s}" + "".join(f"{'  ' + m:>16s}" for m in mod_keys)
    print(header)
    print("-" * len(header))
    for exp_name in all_results:
        row = f"{exp_name:<30s}"
        for m in mod_keys:
            d = all_results[exp_name]["by_modality"].get(m, {"n": 0, "EM": 0})
            if d["n"] > 0:
                row += f"  {d['EM']:.3f}({d['n']:>3d})"
            else:
                row += "                "
        print(row)

    # Delta by modality
    print("\n" + "=" * 80)
    print("VEGA_KG vs FLAT_CHUNK DELTA BY MODALITY (naive_rag)")
    print("=" * 80)
    if "naive_rag_vega_kg" in all_results and "naive_rag_flat_chunk" in all_results:
        vk = all_results["naive_rag_vega_kg"]["by_modality"]
        fc = all_results["naive_rag_flat_chunk"]["by_modality"]
        for m in mod_keys:
            if m in vk and m in fc:
                delta = vk[m]["EM"] - fc[m]["EM"]
                print(f"  {m:>14s}: vega_kg={vk[m]['EM']:.3f} flat_chunk={fc[m]['EM']:.3f} delta={delta:+.3f} (n={vk[m]['n']})")

    print("\n" + "=" * 80)
    print("ANALYSIS BY TYPE GROUP")
    print("=" * 80)

    tg_keys = ["simple", "compose", "compare", "intersect"]
    header = f"{'Experiment':<30s}" + "".join(f"{'  ' + t:>14s}" for t in tg_keys)
    print(header)
    print("-" * len(header))
    for exp_name in all_results:
        row = f"{exp_name:<30s}"
        for t in tg_keys:
            d = all_results[exp_name]["by_type_group"].get(t, {"n": 0, "EM": 0})
            if d["n"] > 0:
                row += f"  {d['EM']:.3f}({d['n']:>3d})"
            else:
                row += "              "
        print(row)

    # Delta by type group
    print("\n" + "=" * 80)
    print("VEGA_KG vs FLAT_CHUNK DELTA BY TYPE GROUP (naive_rag)")
    print("=" * 80)
    if "naive_rag_vega_kg" in all_results and "naive_rag_flat_chunk" in all_results:
        vk = all_results["naive_rag_vega_kg"]["by_type_group"]
        fc = all_results["naive_rag_flat_chunk"]["by_type_group"]
        for t in tg_keys:
            if t in vk and t in fc:
                delta = vk[t]["EM"] - fc[t]["EM"]
                print(f"  {t:>10s}: vega_kg={vk[t]['EM']:.3f} flat_chunk={fc[t]['EM']:.3f} delta={delta:+.3f} (n={vk[t]['n']})")

    print("\n" + "=" * 80)
    print("ANALYSIS BY #SUPPORTING DOCS")
    print("=" * 80)

    doc_keys = ["1", "2", "3"]
    header = f"{'Experiment':<30s}" + "".join(f"{'  ' + d + '-doc':>12s}" for d in doc_keys) + "      4+-doc"
    print(header)
    print("-" * len(header))
    for exp_name in all_results:
        row = f"{exp_name:<30s}"
        for d in doc_keys:
            data = all_results[exp_name]["by_n_docs"].get(d, {"n": 0, "EM": 0})
            if data["n"] > 0:
                row += f"  {data['EM']:.3f}({data['n']:>3d})"
            else:
                row += "            "
        # 4+ docs combined
        n4plus = 0
        em4plus = 0
        for dk, dv in all_results[exp_name]["by_n_docs"].items():
            if int(dk) >= 4:
                n4plus += dv["n"]
                em4plus += dv["EM"] * dv["n"]
        if n4plus > 0:
            row += f"  {em4plus/n4plus:.3f}({n4plus:>3d})"
        print(row)

    # Cross-agent comparison by hop (vega_kg only)
    print("\n" + "=" * 80)
    print("CROSS-AGENT COMPARISON BY HOP (vega_kg backend)")
    print("=" * 80)
    vk_agents = [e for e in all_results if "vega_kg" in e]
    header = f"{'Agent+Backend':<30s}" + "".join(f"{'  ' + h + '-hop':>12s}" for h in hop_keys)
    print(header)
    print("-" * len(header))
    for exp_name in vk_agents:
        row = f"{exp_name:<30s}"
        for h in hop_keys:
            d = all_results[exp_name]["by_hop"].get(h, {"n": 0, "EM": 0})
            if d["n"] > 0:
                row += f"  {d['EM']:.3f}({d['n']:>3d})"
            else:
                row += "            "
        print(row)

    print("\nDone. Results saved to data/results/attribute_analysis/analysis_by_attributes.json")


if __name__ == "__main__":
    main()
