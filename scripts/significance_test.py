"""
VEGA-KG Statistical Significance Tests
=======================================
Usage:
    PYTHONPATH=. python scripts/significance_test.py
    PYTHONPATH=. python scripts/significance_test.py --pred_dir data/sig_test_inputs

Two modes:
  1. Without --pred_dir: auto-converts existing prediction JSONs → .jsonl, then runs tests
  2. With --pred_dir: reads pre-built .jsonl files directly

Each .jsonl file: one JSON object per line with {"qid": str, "em": int, "anls": float}

Statistical methods:
  - Bootstrap 95% CI (paired, 10K resamples)
  - Approximate Randomization Test (two-sided, 10K permutations) for p-values
  - Win/Tie/Loss counts (EM)
"""

import json
import re
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


# ============================================================
# Scoring functions (for converting prediction files)
# ============================================================

def normalize_answer(s):
    if not isinstance(s, str):
        s = str(s)
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def compute_em(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))


def compute_anls(pred, gold, tau=0.5):
    pred_n, gold_n = normalize_answer(pred), normalize_answer(gold)
    if not gold_n:
        return 1.0 if not pred_n else 0.0
    m, n = len(pred_n), len(gold_n)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1,
                        prev + (0 if pred_n[i - 1] == gold_n[j - 1] else 1))
            prev = temp
    nl = dp[n] / max(m, n) if max(m, n) > 0 else 0.0
    return 1.0 - nl if nl < tau else 0.0


# ============================================================
# I/O
# ============================================================

def load_scores(path):
    """Load per-question scores from a .jsonl file."""
    scores = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            scores[obj["qid"]] = {"em": obj["em"], "anls": obj["anls"]}
    return scores


def prediction_json_to_jsonl(src_path, dst_path):
    """Convert prediction JSON (list of {id, prediction, gold}) → .jsonl ({qid, em, anls})."""
    with open(src_path) as f:
        data = json.load(f)
    with open(dst_path, "w") as f:
        for item in data:
            qid = item["id"]
            pred = item.get("prediction", "")
            gold = item.get("gold", "")
            em = compute_em(pred, gold)
            anls = compute_anls(pred, gold)
            f.write(json.dumps({"qid": qid, "em": em, "anls": anls}) + "\n")
    return len(data)


def align_scores(scores_a, scores_b):
    """Align two score dicts by shared qids, return numpy arrays."""
    shared = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    assert len(shared) > 0, "No shared question IDs found"
    a_em = np.array([scores_a[q]["em"] for q in shared])
    b_em = np.array([scores_b[q]["em"] for q in shared])
    a_anls = np.array([scores_a[q]["anls"] for q in shared])
    b_anls = np.array([scores_b[q]["anls"] for q in shared])
    return a_em, b_em, a_anls, b_anls, shared


# ============================================================
#  Bootstrap Confidence Interval
# ============================================================

def bootstrap_ci(scores_a, scores_b, n_bootstrap=10000, alpha=0.05, seed=42):
    """
    Paired bootstrap CI for mean(scores_a) - mean(scores_b).
    Returns: (observed_diff, lower, upper) in percentage points.
    """
    rng = np.random.RandomState(seed)
    n = len(scores_a)
    observed = np.mean(scores_a) - np.mean(scores_b)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        diffs[i] = np.mean(scores_a[idx]) - np.mean(scores_b[idx])
    lower = np.percentile(diffs, 100 * alpha / 2)
    upper = np.percentile(diffs, 100 * (1 - alpha / 2))
    return observed * 100, lower * 100, upper * 100


# ============================================================
#  Approximate Randomization Test
# ============================================================

def approx_randomization(scores_a, scores_b, n_perm=10000, seed=42):
    """
    Two-sided approximate randomization test.
    Returns: p-value
    """
    rng = np.random.RandomState(seed)
    observed = abs(np.mean(scores_a) - np.mean(scores_b))
    n = len(scores_a)
    count = 0
    for _ in range(n_perm):
        mask = rng.randint(0, 2, size=n).astype(bool)
        perm_a = np.where(mask, scores_a, scores_b)
        perm_b = np.where(mask, scores_b, scores_a)
        if abs(np.mean(perm_a) - np.mean(perm_b)) >= observed:
            count += 1
    return count / n_perm


# ============================================================
#  Win / Tie / Loss
# ============================================================

def win_tie_loss(scores_a, scores_b):
    """Per-question win/tie/loss counts (system A vs B on EM)."""
    wins = int(np.sum((scores_a == 1) & (scores_b == 0)))
    losses = int(np.sum((scores_a == 0) & (scores_b == 1)))
    ties = int(len(scores_a) - wins - losses)
    return wins, ties, losses


# ============================================================
#  Run single comparison
# ============================================================

def run_comparison(name, scores_a_dict, scores_b_dict, label_a="System A", label_b="System B"):
    """Run all tests for a pair of systems."""
    a_em, b_em, a_anls, b_anls, shared = align_scores(scores_a_dict, scores_b_dict)
    n = len(shared)

    # EM
    diff_em, lo_em, hi_em = bootstrap_ci(a_em, b_em)
    p_em = approx_randomization(a_em, b_em)
    w, t, l = win_tie_loss(a_em, b_em)

    # ANLS
    diff_anls, lo_anls, hi_anls = bootstrap_ci(a_anls, b_anls)
    p_anls = approx_randomization(a_anls, b_anls)

    def sig(p):
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return "(n.s.)"

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  A = {label_a}  |  B = {label_b}")
    print(f"  n = {n} questions")
    print(f"{'='*60}")

    print(f"\n  EM:")
    print(f"    System A: {np.mean(a_em)*100:.1f}  |  System B: {np.mean(b_em)*100:.1f}")
    print(f"    Delta (A-B): {diff_em:+.1f}")
    print(f"    95% CI:      [{lo_em:+.1f}, {hi_em:+.1f}]")
    print(f"    p-value:     {p_em:.4f} {sig(p_em)}")
    print(f"    Win/Tie/Loss (A vs B): {w}/{t}/{l}")

    print(f"\n  ANLS:")
    print(f"    System A: {np.mean(a_anls)*100:.1f}  |  System B: {np.mean(b_anls)*100:.1f}")
    print(f"    Delta (A-B): {diff_anls:+.1f}")
    print(f"    95% CI:      [{lo_anls:+.1f}, {hi_anls:+.1f}]")
    print(f"    p-value:     {p_anls:.4f} {sig(p_anls)}")

    return {
        "name": name, "label_a": label_a, "label_b": label_b, "n": n,
        "em_a": float(np.mean(a_em)), "em_b": float(np.mean(b_em)),
        "em_diff": diff_em, "em_ci": (lo_em, hi_em), "em_p": p_em,
        "em_wtl": (w, t, l),
        "anls_a": float(np.mean(a_anls)), "anls_b": float(np.mean(b_anls)),
        "anls_diff": diff_anls, "anls_ci": (lo_anls, hi_anls), "anls_p": p_anls,
    }


# ============================================================
#  Convert existing predictions → .jsonl
# ============================================================

def build_jsonl_inputs(results_dir: Path, output_dir: Path):
    """Convert existing prediction JSON files to .jsonl format for significance testing."""
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_recompute_dir = results_dir / "frames_recompute"
    ablation_dir = results_dir / "strict_ablation"

    converted = {}

    # --- Table 1: M3DocVQA qwen3emb (500 samples) ---
    t1_dir = output_dir / "table1"
    t1_dir.mkdir(exist_ok=True)
    for agent in ["naive_rag", "react", "planrag", "autogen"]:
        for backend in ["flat_chunk", "vega_kg", "ms_graphrag", "lightrag", "simpledoc",
                        "vega_page_only"]:
            src = results_dir / f"predictions_{agent}_{backend}_m3docvqa_qwen3emb.json"
            if src.exists():
                dst = t1_dir / f"{agent}_{backend}_m3docvqa.jsonl"
                n = prediction_json_to_jsonl(src, dst)
                converted[f"t1:{agent}_{backend}_m3d"] = dst
                print(f"  [T1] {agent}/{backend} m3docvqa: {n} → {dst.name}")

    # --- Table 1: Frames F3 (naive_rag recompute) ---
    for backend in ["flat_chunk", "vega_kg", "vega_page_only", "ms_graphrag",
                    "lightrag", "simpledoc"]:
        src = frames_recompute_dir / f"predictions_{backend}_F3.json"
        if src.exists():
            dst = t1_dir / f"naive_rag_{backend}_frames.jsonl"
            n = prediction_json_to_jsonl(src, dst)
            converted[f"t1:naive_rag_{backend}_frames"] = dst
            print(f"  [T1] naive_rag/{backend} frames F3: {n} → {dst.name}")

    # --- Table 1: Frames E011 (react/planrag/autogen) ---
    for agent in ["react", "planrag", "autogen"]:
        for backend in ["flat_chunk", "vega_kg"]:
            src = results_dir / f"predictions_{agent}_{backend}_frames.json"
            if src.exists():
                dst = t1_dir / f"{agent}_{backend}_frames.jsonl"
                n = prediction_json_to_jsonl(src, dst)
                converted[f"t1:{agent}_{backend}_frames"] = dst
                print(f"  [T1] {agent}/{backend} frames E011: {n} → {dst.name}")

    # --- Table 2: Ablation ---
    t2_dir = output_dir / "table2_ablation"
    t2_dir.mkdir(exist_ok=True)

    # page_only = naive_rag/vega_page_only qwen3emb
    src = results_dir / "predictions_naive_rag_vega_page_only_m3docvqa_qwen3emb.json"
    if src.exists():
        dst = t2_dir / "page_only.jsonl"
        n = prediction_json_to_jsonl(src, dst)
        converted["t2:page_only"] = dst
        print(f"  [T2] page_only (qwen3emb): {n} → {dst.name}")

    # vega_kg final = naive_rag/vega_kg qwen3emb
    src = results_dir / "predictions_naive_rag_vega_kg_m3docvqa_qwen3emb.json"
    if src.exists():
        dst = t2_dir / "final_vegakg.jsonl"
        n = prediction_json_to_jsonl(src, dst)
        converted["t2:final_vegakg"] = dst
        print(f"  [T2] final_vegakg (qwen3emb): {n} → {dst.name}")

    # strict ablation variants (already 500 samples)
    ablation_map = {
        "page_local": "page_local.jsonl",
        "text_kg_only": "textkg_only.jsonl",
        "text_kg_local": "textkg_local.jsonl",
    }
    for variant, dst_name in ablation_map.items():
        src = ablation_dir / f"predictions_{variant}.json"
        if src.exists():
            dst = t2_dir / dst_name
            n = prediction_json_to_jsonl(src, dst)
            converted[f"t2:{variant}"] = dst
            print(f"  [T2] {variant}: {n} → {dst.name}")

    return converted


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, default=None,
                        help="Directory with pre-built .jsonl files. "
                             "If omitted, auto-converts from data/results/")
    args = parser.parse_args()

    results_dir = Path("data/results")

    if args.pred_dir:
        pred_dir = Path(args.pred_dir)
    else:
        pred_dir = results_dir / "sig_test_inputs"
        print("=" * 60)
        print("  Phase 1: Converting prediction JSONs → .jsonl")
        print("=" * 60)
        paths = build_jsonl_inputs(results_dir, pred_dir)

    # --------------------------------------------------------
    # Load all .jsonl files
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Phase 2: Loading .jsonl scores")
    print("=" * 60)

    systems = {}
    t1_dir = pred_dir / "table1"
    t2_dir = pred_dir / "table2_ablation"
    bottleneck_dir = pred_dir / "bottleneck"

    # Table 1 systems
    if t1_dir.exists():
        for f in sorted(t1_dir.glob("*.jsonl")):
            key = f.stem  # e.g. "naive_rag_flat_chunk_m3docvqa"
            systems[key] = load_scores(f)
            print(f"  {key}: {len(systems[key])} questions")

    # Table 2 ablation systems
    if t2_dir.exists():
        for f in sorted(t2_dir.glob("*.jsonl")):
            key = f"abl_{f.stem}"  # e.g. "abl_page_only"
            systems[key] = load_scores(f)
            print(f"  {key}: {len(systems[key])} questions")

    # Bottleneck systems
    if bottleneck_dir.exists():
        for f in sorted(bottleneck_dir.glob("*.jsonl")):
            key = f"bot_{f.stem}"
            systems[key] = load_scores(f)
            print(f"  {key}: {len(systems[key])} questions")

    print(f"\n  Total systems: {len(systems)}")

    # --------------------------------------------------------
    # Comparisons
    # --------------------------------------------------------
    all_results = []

    print("\n" + "=" * 60)
    print("  Phase 3: Significance Tests")
    print("=" * 60)

    # === Table 1: flat_chunk vs vega_kg per agent (M3DocVQA, qwen3emb) ===
    for agent in ["naive_rag", "react", "planrag", "autogen"]:
        a_key = f"{agent}_vega_kg_m3docvqa"
        b_key = f"{agent}_flat_chunk_m3docvqa"
        if a_key in systems and b_key in systems:
            r = run_comparison(
                f"Table1: {agent} — VEGA-KG vs No-KG (M3DocVQA, qwen3emb)",
                systems[a_key], systems[b_key],
                f"{agent}/vega_kg", f"{agent}/flat_chunk"
            )
            all_results.append(r)

    # === Table 1: Pooled M3DocVQA ===
    pooled_vega = {}
    pooled_flat = {}
    for agent in ["naive_rag", "react", "planrag", "autogen"]:
        for qid, v in systems.get(f"{agent}_vega_kg_m3docvqa", {}).items():
            pooled_vega[f"{agent}_{qid}"] = v
        for qid, v in systems.get(f"{agent}_flat_chunk_m3docvqa", {}).items():
            pooled_flat[f"{agent}_{qid}"] = v
    if pooled_vega and pooled_flat:
        r = run_comparison(
            "Table1: ALL agents pooled — VEGA-KG vs No-KG (M3DocVQA)",
            pooled_vega, pooled_flat,
            "all/vega_kg", "all/flat_chunk"
        )
        all_results.append(r)

    # === Table 1: Frames — flat_chunk vs vega_kg per agent ===
    for agent in ["naive_rag", "react", "planrag", "autogen"]:
        a_key = f"{agent}_vega_kg_frames"
        b_key = f"{agent}_flat_chunk_frames"
        if a_key in systems and b_key in systems:
            config = "F3" if agent == "naive_rag" else "E011"
            r = run_comparison(
                f"Table1: {agent} — VEGA-KG vs No-KG (Frames, {config})",
                systems[a_key], systems[b_key],
                f"{agent}/vega_kg", f"{agent}/flat_chunk"
            )
            all_results.append(r)

    # === Table 1: Pooled Frames ===
    pooled_vega_f = {}
    pooled_flat_f = {}
    for agent in ["naive_rag", "react", "planrag", "autogen"]:
        for qid, v in systems.get(f"{agent}_vega_kg_frames", {}).items():
            pooled_vega_f[f"{agent}_{qid}"] = v
        for qid, v in systems.get(f"{agent}_flat_chunk_frames", {}).items():
            pooled_flat_f[f"{agent}_{qid}"] = v
    if pooled_vega_f and pooled_flat_f:
        r = run_comparison(
            "Table1: ALL agents pooled — VEGA-KG vs No-KG (Frames)",
            pooled_vega_f, pooled_flat_f,
            "all/vega_kg", "all/flat_chunk"
        )
        all_results.append(r)

    # === Table 2 Ablation: Text-KG only vs Page-only ===
    if "abl_textkg_only" in systems and "abl_page_only" in systems:
        r = run_comparison(
            "Ablation: Text-KG only vs Page-only (KG entity effect)",
            systems["abl_textkg_only"], systems["abl_page_only"],
            "text_kg_only", "page_only"
        )
        all_results.append(r)

    # === Table 2 Ablation: Text-KG+Local vs Page-only ===
    if "abl_textkg_local" in systems and "abl_page_only" in systems:
        r = run_comparison(
            "Ablation: Text-KG+Local vs Page-only (KG+locality effect)",
            systems["abl_textkg_local"], systems["abl_page_only"],
            "text_kg_local", "page_only"
        )
        all_results.append(r)

    # === Table 2 Ablation: Text-KG+Local vs Text-KG only (locality) ===
    if "abl_textkg_local" in systems and "abl_textkg_only" in systems:
        r = run_comparison(
            "Ablation: Text-KG+Local vs Text-KG only (locality effect)",
            systems["abl_textkg_local"], systems["abl_textkg_only"],
            "text_kg_local", "text_kg_only"
        )
        all_results.append(r)

    # === Table 2 Ablation: Page-local vs Page-only (locality without KG) ===
    if "abl_page_local" in systems and "abl_page_only" in systems:
        r = run_comparison(
            "Ablation: Page+Local vs Page-only (locality without KG)",
            systems["abl_page_local"], systems["abl_page_only"],
            "page_local", "page_only"
        )
        all_results.append(r)

    # === Table 2 Ablation: Final VEGA-KG vs Page-only ===
    if "abl_final_vegakg" in systems and "abl_page_only" in systems:
        r = run_comparison(
            "Ablation: Final VEGA-KG vs Page-only (full pipeline)",
            systems["abl_final_vegakg"], systems["abl_page_only"],
            "vega_kg (final)", "page_only"
        )
        all_results.append(r)

    # === Bottleneck: Gold-page text vs Retrieved ===
    if "bot_gold_page_text" in systems and "bot_retrieved" in systems:
        r = run_comparison(
            "Bottleneck: Gold-page text vs Retrieved (retrieval headroom)",
            systems["bot_gold_page_text"], systems["bot_retrieved"],
            "gold_page", "retrieved"
        )
        all_results.append(r)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print(f"\n\n{'='*60}")
    print("  SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'Comparison':<55} {'ΔEM':>6} {'95% CI':>18} {'p-val':>8} {'W/T/L':>12}")
    print("-" * 102)
    for r in all_results:
        name = r["name"][:55]
        d = r["em_diff"]
        ci = f"[{r['em_ci'][0]:+.1f}, {r['em_ci'][1]:+.1f}]"
        p = r["em_p"]
        w, t, l = r["em_wtl"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{name:<55} {d:+.1f} {ci:>18} {p:>7.4f}{sig:>1} {w:>3}/{t:>4}/{l:>3}")

    # --------------------------------------------------------
    # LaTeX-ready summary
    # --------------------------------------------------------
    print(f"\n\n{'='*60}")
    print("  LaTeX-ready summary (paste into paper)")
    print(f"{'='*60}\n")
    for r in all_results:
        ci = r["em_ci"]
        sig = ""
        if r["em_p"] < 0.001: sig = "$^{***}$"
        elif r["em_p"] < 0.01: sig = "$^{**}$"
        elif r["em_p"] < 0.05: sig = "$^{*}$"
        print(f"% {r['name']}")
        print(f"%   ΔEM = {r['em_diff']:+.1f} (95\\% CI: [{ci[0]:+.1f}, {ci[1]:+.1f}], "
              f"p={r['em_p']:.3f}){sig}")
        print()

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------
    output_path = results_dir / "significance_tests.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
